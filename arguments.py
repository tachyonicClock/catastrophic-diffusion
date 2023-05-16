import typing as t
from tap import Tap


class Arguments(Tap):
    """Demonstration of catastrophic forgetting on a diffusion model"""

    # DIFFUSION MODEL ---------------------------------------------------------
    arch: str
    """Neural network architecture"""
    class_cond: bool = False
    """Whether to use class conditional generation in training and sampling"""
    diffusion_steps: int = 1000
    """Number of steps in the diffusion process during training"""
    sampling_steps: int = 250
    """Number of steps in the diffusion process during sampling"""
    ddim: bool = False
    """Sampling using DDIM update step"""

    # DATASET -----------------------------------------------------------------
    dataset: t.Optional[str] = None
    """Dataset to train on"""
    data_dir: t.Optional[str] = None
    """Path to directory containing data"""

    # OPTIMIZER ---------------------------------------------------------------
    batch_size: int = 128
    """Batch size per GPU"""
    lr: float = 1e-4
    """Learning rate"""
    epochs: int = 500
    """Number of epochs to train for"""
    ema_w: float = 0.9995
    """Exponential moving average loss weight"""

    # SAMPLING/FINETUNING -----------------------------------------------------
    pretrained_ckpt: str = None
    """Path to pretrained model checkpoint"""
    delete_keys: t.List[str] = []
    """List of keys to delete from pretrained model state dict"""
    sampling_only: bool = False
    """No training, just sample images (will save them in `save_dir`)"""
    num_sampled_images: int = 50000
    """Number of images required to sample from the model"""

    # MISC --------------------------------------------------------------------
    save_samples: str = "./samples"
    """Path to directory to save samples"""
    save_checkpoints: str = "./models"
    """Path to directory to save checkpoints"""
    local_rank: int = 0
    """Local rank of GPU for distributed training"""
    seed: int = 42
    """Random seed"""
    device: t.Optional[str] = None
