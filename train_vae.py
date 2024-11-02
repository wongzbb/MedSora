# Copyright (c) 2024, Zhenbin Wang.
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
from loguru import logger
import os
import imageio
import lpips
from diffusers.models import AutoencoderKLCogVideoX
from autoencoders import AutoencoderKLCogVideoX as VAE
from load_data import get_dataset
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange
from omegaconf import OmegaConf
import math
from autoencoders import FocalFrequencyLoss

# torch._dynamo.config.suppress_errors = True

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

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

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(config, args):
    """
    Trains a new VAE model.
    """
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    assert config.vae_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(config.vae_results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{config.vae_results_dir}/*"))
        model_string_name = "vae"  
        experiment_dir = f"{config.vae_results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)

        logger.info(f"VAE directory created at {experiment_dir}")
        OmegaConf.save(config, os.path.join(experiment_dir, 'config.yaml'))
    else:
        logger = create_logger(None)

    # Create model:
    assert config.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    vae_pretrained_dict = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae").state_dict()

    vae = VAE().to(device)
    vae_dict = vae.state_dict()

    vae_pretrained_dict = {k: v for k, v in vae_pretrained_dict.items() if k in vae_dict and vae_dict[k].shape == v.shape}
    vae_dict.update(vae_pretrained_dict)

    vae.load_state_dict(vae_dict)

    if config.vae_init_from_pretrain_ckpt:
        model_state_dict_ = find_model_model(config.vae_pretrain_ckpt_path)
        vae.load_state_dict(model_state_dict_)

    for param in vae.parameters():
        param.requires_grad = False
    # for param in vae.encoder.frequency_compensation.parameters():
    #     param.requires_grad = True
    # for param in vae.decoder.frequency_compensation.parameters():
    #     param.requires_grad = True
    # for param in vae.decoder.up_freq_compensations.parameters():
    #     param.requires_grad = True

    g_params = list(vae.encoder.frequency_compensation.parameters()) + list(vae.decoder.frequency_compensation.parameters()) + \
                list(vae.decoder.up_freq_compensations.parameters())
    opt_g = torch.optim.Adam(g_params, lr=config.vae_lr, betas=(0.5, 0.9))

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
        batch_size=int(config.vae_batch_size // dist.get_world_size()), 
        shuffle=False, 
        sampler=sampler, 
        num_workers=config.num_workers, 
        drop_last=True, 
        pin_memory=True,
        ) # When using a DistributedSampler, you should set shuffle to False.

    if rank == 0:
        logger.info(f"Dataset contains {len(train_dataset)}.")

    focal_freq_loss = FocalFrequencyLoss(
        loss_weight=1.0,   # You can adjust the weight of the loss
        alpha=1.0,         # Adjust the scaling factor for the weight matrix
        patch_factor=1,    # Patch factor for cropping images into patches
        ave_spectrum=False, # Whether to average the spectrum over the batch
        log_matrix=False,  # Whether to apply logarithm adjustment to the spectrum weight matrix
        batch_matrix=False # Whether to use batch-based statistics for the weight matrix
    )

    lpips_vgg = lpips.LPIPS(net='alex').to(device).eval()

    len_loader = len(train_loader)

    for epoch in range(config.vae_epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        item = 0
        for video_data in train_loader:
            item+=1
            x = video_data['video'].to(device, non_blocking=True) 

            # Manipulating an (img) to ensure its dimensions are compatible with a given patch size
            img = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()   #B*24,3,128,128
            patch_size = 8
            w, h = (
                img.shape[-2] - img.shape[-2] % patch_size,
                img.shape[-1] - img.shape[-1] % patch_size,
            )
            img = img[:, :, :w, :h]

            video_name = video_data['video_name']
            if config.dataset == "ucf101_img":
                image_name = video_data['image_name']
                image_names = []
                for caption in image_name:
                    single_caption = [int(item) for item in caption.split('=====')]
                    image_names.append(torch.as_tensor(single_caption))

            b, _, _, _, _ = x.shape
            x = torch.cat([x[:,:config.num_frames-1,:,:,:], x[:,config.num_frames:,:,:,:]], dim=1)
            x_ = x
            x = rearrange(x, 'b f c h w -> b c f h w').contiguous()
            x = vae.encode(x).latent_dist.sample().mul_(1.15258426)

            samples = vae.decode(x / 1.15258426).sample
            samples = rearrange(samples, 'b c f h w -> b f c h w', b=b).contiguous()  #torch.Size([1, 24, 4, 16, 16])
            samples = samples[:,0:config.num_frames,:,:,:]

            x_ = x_[:,0:config.num_frames,:,:,:]

            opt_g.zero_grad()
            
            loss_l1 = (samples-x_).abs().mean()
            loss_lpips = lpips_vgg(rearrange(samples, 'b f c h w -> (b f) c h w', b=b).contiguous(), rearrange(x_, 'b f c h w -> (b f) c h w', b=b).contiguous()).mean() * 0.1
            loss_ffl = focal_freq_loss(rearrange(samples, 'b f c h w -> b (f c) h w', b=b).contiguous(), rearrange(x_, 'b f c h w -> b (f c) h w', b=b).contiguous()).mean() * 0.1
            loss = loss_l1 + loss_lpips * config.vae_lpips_weight + loss_ffl * config.vae_ffl_weight

            logger.info(f"Epoch {epoch}, Progress {item/len_loader*100:.4f}%, loss: {loss.item():.7f}, \
                        loss_ffl: {loss_ffl.item():.7f}, loss_lpips: {loss_lpips.item():.7f}, loss_l1: {loss_l1.item():.7f}, ")

            loss.backward()
            opt_g.step()
            opt_g.zero_grad()


        checkpoint = {
            "model": vae.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")

    args = parser.parse_args()
    main(OmegaConf.load(args.config), args)
