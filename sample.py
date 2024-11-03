"""
Sample new video from a pre-trained MedSora.
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from diffusion import create_diffusion
import argparse
from omegaconf import OmegaConf
from einops import rearrange
import imageio

import lpips
from autoencoders import AutoencoderKLCogVideoX as VAE
from models import MedSora_models

def find_model_model(model_name):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    checkpoint = checkpoint["model"]
    return checkpoint

def find_model(model_name, config):
    """
    Finds a pre-trained model. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if config.load_ckpt_type in checkpoint: 
        checkpoint = checkpoint[config.load_ckpt_type]
    return checkpoint


def main(config, args):
    # Setup PyTorch:
    torch.manual_seed(config.global_seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    dist.init_process_group("nccl")

    # Load model:
    latent_size = config.image_size // 8
    model = MedSora_models[args.model](
        in_channels=16,
        latent_num_frames=4,
        latent_num_all=6,
        input_size=latent_size,
        dt_rank=config.dt_rank,
        d_state=config.d_state,
        use_image_num = config.use_image_num,
        use_covariance=config.use_covariance,
        use_local_attention=config.use_local_attention,
        use_local_cov=args.use_local_cov,
    ).to(device)

    # Load checkpoint:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path, config)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    # Load diffusion:
    diffusion = create_diffusion(str(config.num_sampling_steps))

    vae = VAE().to(device)
    pretrain_vae_state_dict = find_model_model(config.vae_weight_path)
    vae.load_state_dict(pretrain_vae_state_dict)
    vae.eval()

    for i in range(config.create_video_num):
        print(f"Creating video {i}...")
        z = torch.randn(1, 6, 16, latent_size, latent_size, device=device)
        sample_fn = model.forward
        model_kwargs = dict(frame_space=torch.randn(1, config.frame_num_in_video, 512, device=device))

        if config.sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        elif config.sample_method == 'ddpm':
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        
        b, f, c, h, w = samples.shape
        samples = rearrange(samples, 'b f c h w -> b c f h w')
        samples = vae.decode(samples / 1.15258426).sample
        samples = rearrange(samples, 'b c f h w -> b f c h w', b=b)

        samples = samples[:,0:config.num_frames-1,:,:,:]

        if not os.path.exists(config.save_video_path):
            os.makedirs(config.save_video_path)

        video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = os.path.join(config.save_video_path, str(i)+'_our_sample' + '.mp4')
        imageio.mimwrite(video_save_path, video_, fps=6, quality=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MedSora_models.keys()), default="MedSora-B/2")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--use-local-cov", action="store_true", help="")

    args = parser.parse_args()
    main(OmegaConf.load(args.config), args)

