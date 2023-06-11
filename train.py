from dataclasses import dataclass
import math
import chainlightning as cl
from chainlightning.torch.seed import SeedContext
from chainlightning.track.aim import AimWriter
from chainlightning.scenario.datamodule import ImageFolderModule

import torch
import typing as t
from data import get_metadata, fix_legacy_dict
from torch import Tensor, nn
from main import GaussianDiffusion, sample_N
from torchvision import datasets, transforms
from aim import Run
from unets import UNet, UNetModel
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from ema_pytorch import EMA


@dataclass
class ArgumentsCD:
    past_dataset: str
    present_dataset: str
    diffusion_steps: int
    sampling_steps: int
    init_pretrained: str
    sample_frequency: int
    ema_beta: float = 0.99
    device: str = "cuda"


class CatastrophicDiffusion(cl.Module):
    def __init__(self, args: ArgumentsCD):
        super().__init__()
        self.args = args
        meta_past = get_metadata(args.past_dataset)
        meta_present = get_metadata(args.present_dataset)

        assert meta_past.image_size == meta_present.image_size
        assert meta_past.num_channels == meta_present.num_channels
        self.image_size = meta_past.image_size
        self.num_channels = meta_past.num_channels

        self.u_net: UNetModel = UNet(
            self.image_size,
            self.num_channels,
            self.num_channels,
            num_classes=meta_past.num_classes,
        )
        with open(args.init_pretrained, "rb") as f:
            self.u_net.load_state_dict(fix_legacy_dict(torch.load(f)), strict=True)

        self.u_net_ema = EMA(
            self.u_net,
            beta=args.ema_beta,
            update_after_step=0,
            update_every=1,
            min_value=args.ema_beta,
        )
        self.u_net_ema.update()

        # SAMPLING ------------------------------------------------------------
        self.past_embed = self.u_net.label_emb
        self.past_embed.requires_grad_(False)
        self.present_embed = nn.Embedding(
            meta_present.num_classes, self.past_embed.weight.shape[1]
        )
        self.diffusion = GaussianDiffusion(args.diffusion_steps, args.device)

        self.sample_number = 100

        with SeedContext(42):
            self.sample_past_y = torch.randint(
                0, meta_past.num_classes, (self.sample_number,)
            )
            self.sample_present_y = torch.randint(
                0, meta_present.num_classes, (self.sample_number,)
            )
            self.sample_seed = torch.randn(
                self.sample_number, self.num_channels, self.image_size, self.image_size
            )

    def forward(
        self,
        x,
        timesteps,
        y: t.Optional[Tensor] = None,
        use_ema: bool = False,
    ):
        return self.u_net.forward(
            x, timesteps=timesteps, y_embedding=self.present_embed(y)
        )

    def training_step(self, batch):
        x: Tensor = batch[0]
        y: Tensor = batch[1]
        batch_size = x.shape[0]
        # Original images must be in [0, 1]
        assert (x.max().item() <= 1) and (0 <= x.min().item())
        # Images are transformed to [-1, 1]
        x = 2 * x - 1

        time_step = torch.randint(
            self.diffusion.timesteps, (batch_size,), dtype=torch.int64
        )
        time_step = time_step.to(self.device)

        xt, eps = self.diffusion.sample_from_forward_process(x, time_step)
        pred_eps = self(xt, time_step, y=y)

        loss = ((pred_eps - eps) ** 2).mean()
        self.track.now("train_loss", loss)
        self.track.now("ema_decay", self.u_net_ema.get_current_decay())
        return loss

    def past_embedding(self):
        return self.past_embed(self.sample_past_y.to(self.device))

    def present_embedding(self):
        return self.present_embed(self.sample_present_y.to(self.device))

    def pil_grid(self, images, row):
        return to_pil_image(make_grid(images, nrow=row, normalize=True, range=(-1, 1)))

    @cl.on_event(cl.Event.BeforeTrainBatch)
    @SeedContext(42)
    @torch.no_grad()
    def sample_ema(self):
        if self.clock.train_batch % self.args.sample_frequency != 0:
            return

        self.u_net_ema.update()
        self.u_net_ema.eval()

        y_embedding = torch.concat(
            [self.past_embedding(), self.present_embedding()], dim=0
        )
        x = torch.concat([self.sample_seed, self.sample_seed], dim=0).to(self.device)

        sampled_images = sample_N(
            self.sample_number * 2,
            self.u_net_ema,
            self.diffusion,
            self.device,
            y_embedding,
            x,
            self.args.sampling_steps,
            self.num_channels,
            self.image_size,
        )
        # Split into past and present
        past_img = sampled_images[: self.sample_number]
        present_img = sampled_images[self.sample_number :]
        row = int(math.ceil(math.sqrt(self.sample_number)))
        self.track.figure(
            "EMA_Present", self.pil_grid(present_img, row), self.clock.global_step
        )
        self.track.figure(
            "EMA_Past", self.pil_grid(past_img, row), self.clock.global_step
        )
        self.u_net_ema.train()

    @cl.on_event(cl.Event.BeforeTrainBatch)
    @SeedContext(42)
    @torch.no_grad()
    def sample_model(self):
        if self.clock.train_batch % self.args.sample_frequency != 0:
            return

        y_embedding = torch.concat(
            [self.past_embedding(), self.present_embedding()], dim=0
        )
        x = torch.concat([self.sample_seed, self.sample_seed], dim=0).to(self.device)

        sampled_images = sample_N(
            self.sample_number * 2,
            self.u_net,
            self.diffusion,
            self.device,
            y_embedding,
            x,
            self.args.sampling_steps,
            self.num_channels,
            self.image_size,
        )
        # Split into past and present
        past_img = sampled_images[: self.sample_number]
        present_img = sampled_images[self.sample_number :]
        row = int(math.ceil(math.sqrt(self.sample_number)))
        self.track.figure(
            "Present", self.pil_grid(present_img, row), self.clock.global_step
        )
        self.track.figure("Past", self.pil_grid(past_img, row), self.clock.global_step)
        self.u_net_ema.train()

    @cl.on_event(cl.Event.AfterTrainEpoch)
    def checkpoint(self):
        torch.save(
            self.u_net_ema.state_dict(),
            f"trained_models/my_checkpoints/ema_UNet_{self.args.past_dataset}-epoch_{self.clock.train_epoch}.pt",
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        parameters = list(self.u_net.parameters())
        parameters += list(self.present_embed.parameters())
        return torch.optim.AdamW(parameters, lr=1e-4)

    def hparams(self) -> t.Dict[str, t.Any]:
        return {}


def main():
    args = ArgumentsCD(
        past_dataset="cars",
        present_dataset="afhq",
        init_pretrained="trained_models/UNet_cars-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt",
        diffusion_steps=1000,
        sampling_steps=20,
        sample_frequency=20,
    )

    # CUDA clear cache to avoid out of memory error
    torch.cuda.empty_cache()

    model = CatastrophicDiffusion(args)
    run = Run(experiment="CatastrophicDiffusionTest")
    aim = AimWriter(run)

    afhq_transform = transforms.Compose(
        [
            transforms.Resize(74),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageFolderModule(
        "/local/scratch/antonlee/datasets/afhq/train",
        "/local/scratch/antonlee/datasets/afhq/val",
        class_count=3,
        task_count=1,
        num_workers=5,
        train_batch_size=64,
        eval_batch_size=64,
        train_transform=afhq_transform,
        eval_transform=afhq_transform,
    )
    cl.Trainer(256, aim, "cuda").fit(model, dataset)


if __name__ == "__main__":
    main()
