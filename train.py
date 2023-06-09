from dataclasses import dataclass
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
from unets import UNet
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image


@dataclass
class ArgumentsCD:
    past_dataset: str
    present_dataset: str
    diffusion_steps: int
    sampling_steps: int
    init_pretrained: str
    sample_frequency: int
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

        self.u_net = UNet(
            self.image_size,
            self.num_channels,
            self.num_channels,
            num_classes=meta_past.num_classes,
        )
        with open(args.init_pretrained, "rb") as f:
            self.u_net.load_state_dict(fix_legacy_dict(torch.load(f)), strict=True)

        # self.ema_state_dict =

        # SAMPLING ------------------------------------------------------------
        self.past_embed = self.u_net.label_emb
        self.present_embed = nn.Embedding(
            meta_present.num_classes, self.past_embed.weight.shape[1]
        )
        self.active_embed: nn.Embedding = self.present_embed
        self.diffusion = GaussianDiffusion(args.diffusion_steps, args.device)

        self.sample_number = 100
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
    ):
        y_embed = self.active_embed(y)
        return self.u_net.forward(x, timesteps, y_embedding=y_embed)

    def training_step(self, batch):
        x, y = batch
        self.active_embed = self.present_embed
        time_step = torch.randint_like(y, self.diffusion.timesteps).long()

        # eps or epsilon is the noise that is added to the image
        x_t, eps = self.diffusion.sample_from_forward_process(x, time_step)
        pred_eps = self(x_t, time_step, y)

        loss = ((pred_eps - eps) ** 2).mean()

        self.track.now("train_loss", loss)
        return loss

    @cl.on_event(cl.Event.BeforeTrainBatch)
    @SeedContext(42)
    def sample_past(self):
        if self.event == cl.Event.BeforeTrainBatch:
            if self.clock.train_step % self.args.sample_frequency != 0:
                return
        print("Sampling past")
        self.active_embed = self.past_embed

        sampled_images = sample_N(
            self.sample_number,
            self,
            self.diffusion,
            self.device,
            self.sample_past_y.to(self.device),
            self.sample_seed.to(self.device),
            self.args.sampling_steps,
            self.num_channels,
            self.image_size,
        )

        img = to_pil_image(
            make_grid(sampled_images, nrow=10, normalize=True, range=(-1, 1))
        )
        self.track.figure("PastSampledImage", img)

    @cl.on_event(cl.Event.BeforeTrainBatch)
    @SeedContext(42)
    def sample_present(self):
        if self.event == cl.Event.BeforeTrainBatch:
            if self.clock.train_step % self.args.sample_frequency != 0:
                return

        print("Sampling Present")
        self.active_embed = self.present_embed
        sampled_images = sample_N(
            self.sample_number,
            self,
            self.diffusion,
            self.device,
            self.sample_present_y.to(self.device),
            self.sample_seed.to(self.device),
            self.args.sampling_steps,
            self.num_channels,
            self.image_size,
        )

        img = to_pil_image(
            make_grid(sampled_images, nrow=10, normalize=True, range=(-1, 1))
        )
        self.track.figure("PresentSampledImage", img)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def hparams(self) -> t.Dict[str, t.Any]:
        return {}


def main():
    args = ArgumentsCD(
        past_dataset="cars",
        present_dataset="afhq",
        init_pretrained="trained_models/UNet_cars-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt",
        diffusion_steps=1000,
        sampling_steps=20,
        sample_frequency=10,
    )

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
        train_batch_size=32,
        eval_batch_size=32,
        train_transform=afhq_transform,
        eval_transform=afhq_transform,
    )
    cl.Trainer(100, aim, "cuda").fit(model, dataset)


if __name__ == "__main__":
    main()
