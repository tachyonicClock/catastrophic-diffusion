import typing as t
import os
import cv2
import copy
import math
import numpy as np
from time import time
from tqdm import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from arguments import Arguments

from data import get_metadata, get_dataset, fix_legacy_dict
import unets


def unsqueeze3x(x):
    return x[..., None, None, None]


class GaussianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values
    (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta,
        beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps)
        in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label
                      for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very
            small number of sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, j in zip(
            tqdm(np.arange(timesteps)[::-1], desc="Sampling"), new_timesteps[::-1]
        ):
            with torch.no_grad():
                current_t = torch.tensor([j] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps
                # (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
        return final


class loss_logger:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.loss = []
        self.start_time = time()
        self.ema_loss = None
        self.ema_w = 0.9

    def log(self, v, display=False):
        self.loss.append(v)
        if self.ema_loss is None:
            self.ema_loss = v
        else:
            self.ema_loss = self.ema_w * self.ema_loss + (1 - self.ema_w) * v

        if display:
            print(
                f"Steps: {len(self.loss)}/{self.max_steps}"
                + f"\t loss (ema): {self.ema_loss:.3f}"
                + f"\t Time elapsed: {(time() - self.start_time)/3600:.3f} hr"
            )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer,
    logger: t.Any,
    lrs: t.Optional[torch.optim.lr_scheduler._LRScheduler],
    args: Arguments,
    ema_state_dict,
):
    model.train()
    for step, (images, labels) in enumerate(dataloader):
        assert (images.max().item() <= 1) and (0 <= images.min().item())

        # must use [-1, 1] pixel range for images
        images, labels = (
            2 * images.to(args.device) - 1,
            labels.to(args.device) if args.class_cond else None,
        )
        t = torch.randint(diffusion.timesteps, (len(images),), dtype=torch.int64).to(
            args.device
        )
        xt, eps = diffusion.sample_from_forward_process(images, t)
        pred_eps = model(xt, t, y=labels)

        loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lrs is not None:
            lrs.step()

        # update ema_dict
        if args.local_rank == 0:
            new_dict = model.state_dict()
            for k, v in ema_state_dict.items():
                ema_state_dict[k] = (
                    args.ema_w * ema_state_dict[k] + (1 - args.ema_w) * new_dict[k]
                )
            logger.log(loss.item(), display=not step % 100)


def sample_N_images(
    N: int,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    args: Arguments,
    xT: t.Optional[torch.Tensor] = None,
    sampling_steps: int = 250,
    batch_size: int = 64,
    num_channels: int = 3,
    image_size: int = 32,
    num_classes: t.Optional[int] = None,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for
                     class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, labels, num_samples = [], [], 0

    num_processes = 1
    group = None
    if args.local_rank > 0:
        num_processes, group = dist.get_world_size(), dist.group.WORLD

    while num_samples < N:
        if xT is None:
            xT = (
                torch.randn(batch_size, num_channels, image_size, image_size)
                .float()
                .to(args.device)
            )
        if args.class_cond and num_classes is not None:
            y = torch.randint(num_classes, (len(xT),), dtype=torch.int64).to(
                args.device
            )
        else:
            y = None
        gen_images = diffusion.sample_from_reverse_process(
            model, xT, sampling_steps, {"y": y}, args.ddim
        )
        samples_list = [torch.zeros_like(gen_images) for _ in range(num_processes)]
        if args.class_cond and y is not None:
            labels_list = [torch.zeros_like(y) for _ in range(num_processes)]

            if group is None:
                labels_list = [y]
                labels.append(y.detach().cpu().numpy())
            else:
                dist.all_gather(labels_list, y, group)
                labels.append(torch.cat(labels_list).detach().cpu().numpy())

        if group is None:
            samples_list = [gen_images]
        else:
            dist.all_gather(samples_list, gen_images, group)

        samples.append(torch.cat(samples_list).detach().cpu().numpy())
        num_samples += len(xT) * num_processes
    samples = np.concatenate(samples).transpose(0, 2, 3, 1)[:N]
    samples = (127.5 * (samples + 1)).astype(np.uint8)
    return (samples, np.concatenate(labels) if args.class_cond else None)


def sample_N(
    N: int,
    model: unets.UNetModel,
    diffusion: GaussianDiffusion,
    device: torch.device,
    y_embedding: torch.Tensor,
    xT: t.Optional[torch.Tensor] = None,
    sampling_steps: int = 250,
    num_channels: int = 3,
    image_size: int = 32,
):
    """use this function to sample any number of images from a given
        diffusion model and diffusion process.

    Args:
        N : Number of images
        model : Diffusion model
        diffusion : Diffusion process
        xT : Starting instantiation of noise vector.
        sampling_steps : Number of sampling steps.
        batch_size : Batch-size for sampling.
        num_channels : Number of channels in the image.
        image_size : Image size (assuming square images).
        num_classes : Number of classes in the dataset (needed for
                     class-conditioned models)
        args : All args from the argparser.

    Returns: Numpy array with N images and corresponding labels.
    """
    samples, num_samples = [], 0
    N = len(xT) if xT is not None else N

    if xT is None:
        x = torch.randn(N, num_channels, image_size, image_size)
        x = x.to(device)
    else:
        x = xT[num_samples : num_samples + N]

    gen_images = diffusion.sample_from_reverse_process(
        model, x, sampling_steps, {"y_embedding": y_embedding}, False
    )

    samples.append(gen_images.detach().cpu())
    return torch.cat(samples)


def main():
    args = Arguments().parse_args()

    metadata = get_metadata(args.dataset)

    args.device = args.device or "cuda:{}".format(args.local_rank)

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    if args.local_rank == 0:
        print(args)

    print(metadata)

    # Creat model and diffusion process
    model: unets.UNetModel = unets.__dict__[args.arch](
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=metadata.num_classes if args.class_cond else None,
    ).to(args.device)
    if args.local_rank == 0:
        print(
            "We are assuming that model input/ouput pixel range is [-1, 1]. "
            + "Please adhere to it."
        )
    diffusion = GaussianDiffusion(args.diffusion_steps, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # load pre-trained model
    if args.pretrained_ckpt:
        print(f"Loading pretrained model from {args.pretrained_ckpt}")
        d = fix_legacy_dict(torch.load(args.pretrained_ckpt, map_location=args.device))
        dm = model.state_dict()
        if args.delete_keys:
            for k in args.delete_keys:
                print(
                    f"Deleting key {k} becuase its shape in ckpt ({d[k].shape}) "
                    + f"doesn't match with shape in model ({dm[k].shape})"
                )
                del d[k]

        if dm["label_emb.weight"].shape != d["label_emb.weight"].shape:
            # Shape mismatch for label embedding. Reinitialize it.
            d["label_emb.weight"] = model.label_emb.weight

        model.load_state_dict(d, strict=False)
        print(
            "Mismatched keys in ckpt and model: ",
            set(d.keys()) ^ set(dm.keys()),
        )
        print(f"Loaded pretrained model from {args.pretrained_ckpt}")

    # distributed training
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        if args.local_rank == 0:
            print(f"Using distributed training on {ngpus} gpus.")
        args.batch_size = args.batch_size // ngpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # sampling
    if args.sampling_only:
        sampled_images, labels = sample_N_images(
            args.num_sampled_images,
            model,
            diffusion,
            args,
            None,
            args.sampling_steps,
            args.batch_size,
            metadata.num_channels,
            metadata.image_size,
            metadata.num_classes,
        )

        np.savez(
            os.path.join(
                args.save_samples,
                f"{args.arch}_{args.dataset}-{args.sampling_steps}"
                + f"-sampling_steps-{len(sampled_images)}_images"
                + f"-class_condn_{args.class_cond}.npz",
            ),
            sampled_images,
            labels,
        )
        return

    # Load dataset
    train_set = get_dataset(args.dataset, args.data_dir, metadata)
    sampler = DistributedSampler(train_set) if ngpus > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    if args.local_rank == 0:
        print(
            f"Training dataset loaded: Number of batches: {len(train_loader)}, "
            + f"Number of images: {len(train_set)}"
        )
    logger = loss_logger(len(train_loader) * args.epochs)

    # ema model
    ema_state_dict = copy.deepcopy(model.state_dict())

    # lets start training the model
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            train_loader,
            diffusion,
            optimizer,
            logger,
            None,
            args,
            ema_state_dict,
        )
        if not epoch % 1:
            sampled_images, _ = sample_N_images(
                64,
                model,
                diffusion,
                args,
                None,
                args.sampling_steps,
                args.batch_size,
                metadata.num_channels,
                metadata.image_size,
                metadata.num_classes,
            )
            if args.local_rank == 0:
                cv2.imwrite(
                    os.path.join(
                        args.save_samples,
                        f"{args.arch}_{args.dataset}-{args.diffusion_steps}"
                        + f"_steps-{args.sampling_steps}-sampling_steps-"
                        + "class_condn_{args.class_cond}.png",
                    ),
                    np.concatenate(sampled_images, axis=1)[:, :, ::-1],
                )
        if args.local_rank == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_samples,
                    f"{args.arch}_{args.dataset}-epoch_{args.epochs}-"
                    + f"timesteps_{args.diffusion_steps}-class_condn_"
                    + f"{args.class_cond}.pt",
                ),
            )
            torch.save(
                ema_state_dict,
                os.path.join(
                    args.save_checkpoints,
                    f"{args.arch}_{args.dataset}-epoch_{args.epochs}-timesteps_"
                    + f"{args.diffusion_steps}-class_condn_{args.class_cond}_"
                    + f"ema_{args.ema_w}.pt",
                ),
            )


if __name__ == "__main__":
    main()
