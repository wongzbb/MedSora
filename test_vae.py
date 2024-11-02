# Copyright (c) 2024, Zhenbin Wang.


import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
import argparse
from loguru import logger
import os
import imageio

from autoencoders import AutoencoderKLCogVideoX as VAE
from load_data import get_dataset
from torch.cuda.amp import GradScaler
from einops import rearrange
from omegaconf import OmegaConf


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def find_model(model_name):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint: 
        checkpoint = checkpoint["ema"]
    return checkpoint

def find_model_model(model_name):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    checkpoint = checkpoint["model"]
    return checkpoint

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logger.add(f"{logging_dir}/log"+f"_{dist.get_rank()}.txt", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    return logger



from torch import inf
from typing import Union, Iterable
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def main(config, args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    assert config.vae_sample_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    assert config.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    vae = VAE().to(device)
    pretrain_vae_state_dict = find_model_model(config.vae_test_ckpt)
    vae.load_state_dict(pretrain_vae_state_dict)
    vae.eval()

    train_dataset = get_dataset(config)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=config.vae_global_seed
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(config.vae_sample_batch_size // dist.get_world_size()), 
        shuffle=False, 
        sampler=sampler, 
        num_workers=config.num_workers, 
        drop_last=True, 
        pin_memory=True,
        ) # When using a DistributedSampler, you should set shuffle to False.

    if rank == 0:
        logger.info(f"Dataset contains {len(train_dataset)}.")

    len_loader = len(train_loader)

    item = 0.0
    os.makedirs(config.vae_test_results_dir, exist_ok=True)

    for video_data in train_loader:
        print(f'item: {item/len_loader*100:.2f}%')
        item+=1
        x = video_data['video'].to(device, non_blocking=True)  #B,24,3,128,128

        img = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()   #B*24,3,128,128
        patch_size = 8
        w, h = (
            img.shape[-2] - img.shape[-2] % patch_size,
            img.shape[-1] - img.shape[-1] % patch_size,
        )
        img = img[:, :, :w, :h]
        if config.dataset == "ucf101_img":
            image_name = video_data['image_name']
            image_names = []
            for caption in image_name:
                single_caption = [int(item) for item in caption.split('=====')]
                image_names.append(torch.as_tensor(single_caption))


        b, _, _, _, _ = x.shape
        x_ = x[:,:-1,:,:,:]
        x = rearrange(x[:,:-1,:,:,:], 'b f c h w -> b c f h w').contiguous()
        
        with torch.no_grad(): 
            x = vae.encode(x).latent_dist.sample().mul_(1.15258426)

        with torch.no_grad(): 
            samples = vae.decode(x / 1.15258426).sample

        samples = rearrange(samples, 'b c f h w -> b f c h w', b=b).contiguous()  #torch.Size([1, 24, 4, 16, 16])
        samples = samples[:,0:config.num_frames,:,:,:]

        video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = config.vae_test_results_dir + str(item) + '_sample' + '.mp4'
        imageio.mimwrite(video_save_path, video_, fps=6, quality=10)

        x_ = x_[:,0:config.num_frames,:,:,:]
        video_ = ((x_[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = config.vae_test_results_dir + str(item) + '_origin' + '.mp4'
        imageio.mimwrite(video_save_path, video_, fps=6, quality=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")

    args = parser.parse_args()
    main(OmegaConf.load(args.config), args)
