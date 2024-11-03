"""
A minimal training script for MedSora using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
from loguru import logger
import os
import wandb
from models import vit_small
from diffusion import create_diffusion
from autoencoders import AutoencoderKLCogVideoX as VAE
from load_data import get_dataset
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange
from omegaconf import OmegaConf
from tools import SophiaG
from models import RAFT
from models import InputPadder
from models import flow_to_image
from models import MedSora_models

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

def get_layer_outputs(model, layer_names, input_tensor):
    outputs = {}
    def hook_fn(module, input, output):
        if module.__class__.__name__ in layer_names:
            outputs[module.__class__.__name__] = output
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    model(input_tensor)
    for hook in hooks:
        hook.remove()
    return outputs

def load_dino_model(device, pretrained_path):
    model = vit_small(
            patch_size=8, num_classes=0
        )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    state_dict = torch.load(pretrained_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_path, msg
        )
    )
    return model

from torch import inf
from typing import Union, Iterable
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]
def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad = True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(config, args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(config.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{config.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  

        experiment_dir = f"{config.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)

        if args.wandb:
            wandb.init(project='MedSora_'+args.model.replace('/','_'))
            wandb.config = {"learning_rate": config.lr, 
                            "epochs": args.epochs, 
                            "batch_size": args.global_batch_size,
                            "dt-rank": config.dt_rank,
                            "autocast": args.autocast,
                            "margin": config.margin,
                            "save-path": experiment_dir,
                            "autocast": args.autocast,
                            }

        logger.info(f"Experiment directory created at {experiment_dir}")
        OmegaConf.save(config, os.path.join(experiment_dir, 'config.yaml'))
    else:
        logger = create_logger(None)

    # Create model:
    assert config.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
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
    )

    if config.init_from_pretrain_ckpt:
        #load model
        model_state_dict_ = find_model_model(config.pretrain_ckpt_path)
        model.load_state_dict(model_state_dict_)
        #load ema
        ema = deepcopy(model).to(device)
        ema_state_dict_ = find_model(config.pretrain_ckpt_path)
        ema.load_state_dict(ema_state_dict_)
        # log
        logger.info(f"Loaded pretrain model from {config.pretrain_ckpt_path}")
    else:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank])

    if config.use_compile:
        model = torch.compile(model)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule, see ./diffusion/__init__.py

    vae = VAE().to(device)
    pretrain_vae_state_dict = find_model_model(config.vae_weight_path)
    vae.load_state_dict(pretrain_vae_state_dict)
    vae.eval()

    # load the pretrained flow model
    flow_model = torch.nn.DataParallel(RAFT(args))
    flow_model.load_state_dict(torch.load(config.flow_model_path))
    flow_model = flow_model.module
    flow_model.to(device)
    requires_grad(flow_model, False)
    flow_model.eval()

    # load the pretrained dino model
    dino_model = load_dino_model(device=device, pretrained_path=config.dino_weights_path)
    requires_grad(dino_model, False)
    dino_model.eval()
    
    if rank == 0:
        logger.info(f"MedSora Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Use half-precision training? {args.autocast}")
        logger.info(f"Use covariance to guide training? {config.use_covariance}")
        logger.info(f"Use local attention to guide training? {config.use_local_attention}")
        logger.info(f"Use margin? {config.use_margin}")

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
        batch_size=int(args.global_batch_size // dist.get_world_size()), 
        shuffle=False, 
        sampler=sampler, 
        num_workers=config.num_workers, 
        drop_last=True, 
        pin_memory=True,
        ) 

    if rank == 0:
        logger.info(f"Dataset contains {len(train_dataset)}.")

    if config.SophiaG:
        opt = SophiaG(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    if config.init_from_pretrain_ckpt:
        train_steps = config.init_train_steps
    else:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    noise_running_loss = 0
    cov_running_loss = 0
    cov_running_loss_ = 0

    if rank == 0:
        logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")

        item = 0
        for video_data in train_loader:
            item+=1
            x = video_data['video'].to(device, non_blocking=True)  #B,24,3,128,128
            batch_size = x.shape[0]

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

            
            with torch.no_grad():
                flow_tensor_up_list = []

                for i1, i2 in zip(x[:,:config.num_frames-1,:,:,:], x[:,1:config.num_frames,:,:,:]):
                    padder = InputPadder(i1.shape)
                    i1, i2 = padder.pad(i1, i2)
                    _, flow_up = flow_model(i1, i2, iters=20, test_mode=True)
                    f = flow_up.shape[0]
                    flow_up_f = []

                    for i in range(f):
                        flow_up_i = flow_to_image(flow_up[i].permute(1,2,0).cpu().numpy())
                        flow_up_f.append(flow_up_i)

                    flow_up_np = np.stack(flow_up_f, axis=0)
                    flow_up = torch.as_tensor(flow_up_np).permute(0,3,1,2).to(device)
                    flow_tensor_up_list.append(flow_up)

                flow_tensor_up = torch.stack(flow_tensor_up_list)

            x_video = rearrange(x[:,:config.num_frames-1,:,:,:], 'b f c h w -> b c f h w').contiguous()
            with torch.no_grad(): 
                x_video = vae.encode(x_video).latent_dist.sample().mul_(1.15258426)
            x_video = rearrange(x_video, 'b c f h w -> b f c h w').contiguous()

            x_image = rearrange(x[:,config.num_frames:,:,:,:], 'b f c h w -> b c f h w').contiguous()
            with torch.no_grad(): 
                x_image = vae.encode(x_image).latent_dist.sample().mul_(1.15258426)
            x_image = rearrange(x_image, 'b c f h w -> b f c h w').contiguous()

            x = torch.cat((x_video, x_image), dim=1)

            flow_tensor_up = rearrange(flow_tensor_up, 'b f c h w -> (b f) c h w').contiguous()
            with torch.no_grad(): 
                layer_output = dino_model.get_special_layers(flow_tensor_up.float(), [])

            batch_size = x.shape[0]
            layer_output = [rearrange(item[:, 1:, :], '(b f) l d -> b f l d', b=batch_size).contiguous() for item in layer_output]

            model_kwargs = dict(frame_space=layer_output)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            if args.autocast:
                with autocast():
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            else:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            noise_loss = loss_dict["loss"].mean()

            if config.use_wt:
                cov_loss_dict = loss_dict["cov"] * (1.0/(t+1))
                cov_loss_ = cov_loss_dict.mean()
            else:
                cov_loss_ = loss_dict["cov"].mean()

            if config.use_margin:
                if config.use_wt:
                    cov_loss = torch.max(cov_loss_-config.wt_margin, torch.zeros_like(cov_loss_).to(cov_loss_.device))
                else:
                    cov_loss = torch.max(cov_loss_-config.margin, torch.zeros_like(cov_loss_).to(cov_loss_.device))
            else:
                cov_loss = cov_loss_

            if config.use_covariance:
                if config.use_magic:
                    loss = noise_loss + config.balance_alpha* (cov_loss / (cov_loss.detach()/noise_loss.detach()))
                else:
                    loss = noise_loss + config.balance_alpha * cov_loss
            else:
                loss = noise_loss

            if rank == 0 and args.wandb:
                wandb.log({"loss": loss.item()})
                wandb.log({"noise_loss": noise_loss.item()})
                wandb.log({"cov_loss": cov_loss.item()})
                wandb.log({"cov_loss_": cov_loss_.item()})

            if torch.isnan(loss).any():  #important
                logger.info(f"nan......      ignore losses......")
                continue

            if args.autocast:
                with autocast():
                    scaler.scale(loss).backward()
            else:
                loss.backward()
            
             #clip 
            if config.use_clip_norm and config.start_clip_epoch <= epoch:
                gradient_norm = clip_grad_norm_(model.module.parameters(), config.clip_max_norm, clip_grad=True)
            else:
                gradient_norm = 1.0

            if train_steps % config.accumulation_steps == 0:
                if args.autocast:
                    with autocast():
                        scaler.step(opt)
                        scaler.update()
                        update_ema(ema, model.module)
                        opt.zero_grad()
                else:
                    opt.step()
                    update_ema(ema, model.module)
                    opt.zero_grad()

            # Log loss values:
            running_loss += loss.item()
            noise_running_loss += noise_loss.item()
            cov_running_loss += cov_loss.item()
            cov_running_loss_ += cov_loss_.item()
            log_steps += 1
            train_steps += 1
            if train_steps % config.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                epoch_isfinish = int(args.global_batch_size // dist.get_world_size()) * item / len(train_dataset) * 100
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_noise_loss = torch.tensor(noise_running_loss / log_steps, device=device)
                avg_cov_loss = torch.tensor(cov_running_loss / log_steps, device=device)
                avg_cov_loss_ = torch.tensor(cov_running_loss_ / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_noise_loss, op=dist.ReduceOp.SUM)
                avg_noise_loss = avg_noise_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_cov_loss, op=dist.ReduceOp.SUM)
                avg_cov_loss = avg_cov_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_cov_loss_, op=dist.ReduceOp.SUM)
                avg_cov_loss_ = avg_cov_loss_.item() / dist.get_world_size()
                if rank == 0:
                    logger.info(f"({epoch_isfinish:.1f}%) (step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Noise Loss: {avg_noise_loss:.4f}, Cov Loss_: {avg_cov_loss_:.4f}, Cov Loss**: {avg_cov_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                noise_running_loss = 0
                cov_running_loss = 0
                cov_running_loss_ = 0
                log_steps = 0
                start_time = time()

            # Save MedSora checkpoint:
            if train_steps % config.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if rank == 0:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if rank == 0 and args.wandb:
        wandb.finish()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MedSora_models.keys()), default="MedSora-B")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.")
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.")
    parser.add_argument("--use-local-cov", action="store_true", help="Use the local covariance to guide training")
    parser.add_argument('--small', action='store_true', help='use small raft model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--config", type=str, default="")

    args = parser.parse_args()
    main(OmegaConf.load(args.config), args)
