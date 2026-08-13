import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json

from wan_utils.dataset import TextDataset, TextImagePairDataset
from wan_utils.misc import set_seed
from wan_utils.camera_trajectory import parse_trajectory
from dykv_cases import DYKV_CASES, resolve_dykv_case

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=20, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--sp_size", type=int, default=1, help="Sequence parallel size (1=disabled)")
parser.add_argument("--trajectory", type=str, default=None, help="Camera trajectory string (e.g., 'w*19' for camera control)")
parser.add_argument("--trajectory_path", type=str, default=None, help="Path to trajectory file (one trajectory string per line, aligned with data_path)")
parser.add_argument("--dykv", action="store_true", help="Enable the complete dyKV long-horizon memory preset")
parser.add_argument(
    "--dykv-case",
    choices=tuple(DYKV_CASES),
    default=None,
    help="Registered experiment preset (defaults to baseline, or yaw_intrinsics with --dykv)",
)
args = parser.parse_args()

# Initialize distributed inference
# IMPORTANT: distributed init MUST happen before importing pipeline modules,
# because causal_model.py checks for CleanCode SP infra at import time.
if args.sp_size > 1:
    # SP mode requires torchrun with nproc_per_node >= sp_size
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size_env >= args.sp_size, (
        f"SP requires at least {args.sp_size} processes, but WORLD_SIZE={world_size_env}. "
        f"Launch with: torchrun --nproc_per_node={args.sp_size} inference.py ... --sp_size {args.sp_size}"
    )
    from wan_utils.distributed import launch_distributed_job, get_sp_seed_offset
    launch_distributed_job(backend="nccl", sp_size=args.sp_size)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
elif "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

# Seed: under SP, ranks in the same SP group must share the same seed
if args.sp_size > 1:
    set_seed(args.seed + get_sp_seed_offset())
else:
    set_seed(args.seed)

# Refresh gpu device handle (demo_utils.memory.gpu is captured at import time
# before distributed init sets the correct CUDA device)
from demo_utils import memory as _mem
_mem.gpu = torch.device(f'cuda:{local_rank}')
gpu = _mem.gpu

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

dykv_case = resolve_dykv_case(args.dykv_case, enabled=args.dykv)

# Every registered case uses the same fixed four-frame sink. In the baseline,
# the remaining 16 frames are rolling local context. dyKV narrows the live
# cache to 4 sink + 8 local and supplies the other 8 frames via retrieval.
config.sink_mode = dykv_case.sink_mode
config.sink_frames = dykv_case.sink_frames
config.model_kwargs.sink_size = dykv_case.sink_frames

if args.dykv:
    chunk_frames = int(config.get("num_frame_per_block", 1))
    if chunk_frames != 4:
        raise ValueError("dyKV currently targets the four-frame causal minWM checkpoint")
    config.dykv_enabled = True
    config.dykv_case = dykv_case.name
    config.dykv_memory_frames = 8
    config.dykv_retrieval_frames = dykv_case.retrieval_frames
    config.dykv_compression_mode = dykv_case.compression_mode
    config.dykv_compression_keep_ratio = dykv_case.compression_keep_ratio
    config.dykv_packing_mode = dykv_case.packing_mode
    config.dykv_retrieval_fov_source = dykv_case.retrieval_fov_source
    config.dykv_compression_fov_source = dykv_case.compression_fov_source
    # Fixed contiguous 4 + 8 + 8 layout: the live cache physically holds the
    # four-frame sink plus the eight-frame local region (recent + current).
    # Retrieved K/V lives in the CPU bank until attention materialization.
    config.model_kwargs.local_attn_size = dykv_case.sink_frames + dykv_case.local_frames
    config.model_kwargs.dykv_enabled = True
    config.model_kwargs.dykv_memory_frames = 8
    config.model_kwargs.dykv_local_frames = 8
    config.model_kwargs.dykv_rope_train_frames = 20
else:
    config.dykv_enabled = False
    config.dykv_case = "baseline"
    config.model_kwargs.local_attn_size = dykv_case.sink_frames + dykv_case.local_frames
    config.model_kwargs.dykv_enabled = False

# Import pipeline AFTER distributed init so causal_model.py sees CleanCode SP infra
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
    BidirectionalDiffusionInferencePipeline,
    BidirectionalInferencePipeline,
)

# Initialize pipeline
is_causal = config.get('causal', True)

if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    if is_causal:
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = BidirectionalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    if is_causal:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)
    else:
        pipeline = BidirectionalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator_ema'
    try:
        gen_sd = state_dict[key]
    except:
        key = 'generator'
        gen_sd = state_dict[key]
    
    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized() and args.sp_size <= 1:
    # Standard DP: split prompts across ranks
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
elif dist.is_initialized() and args.sp_size > 1:
    # SP mode: use SP-aware sampler so ranks in the same SP group get the same data
    from wan_utils.distributed import get_sp_data_sampler
    sampler = get_sp_data_sampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
    output_manifest_path = os.path.join(args.output_folder, "generation_manifest.jsonl")
    open(output_manifest_path, "w", encoding="utf-8").close()
    dykv_events_path = os.path.join(args.output_folder, "dykv_summaries.jsonl")
    if args.dykv:
        open(dykv_events_path, "w", encoding="utf-8").close()

if dist.is_initialized():
    dist.barrier()

# Load per-prompt trajectory list if provided
trajectory_list = None
if args.trajectory_path:
    with open(args.trajectory_path, encoding="utf-8") as _f:
        trajectory_list = [line.strip() for line in _f if line.strip()]
    assert len(trajectory_list) >= num_prompts, (
        f"trajectory_path has {len(trajectory_list)} lines but need >= {num_prompts} prompts"
    )

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


# Latency bookkeeping (rank 0 only; first prompt is recorded as None to skip warmup).
# All pipelines expose `last_chunk0_latency`: time from sampling start to the first
# denoised latent ready, EXCLUDING VAE decode — matches HY15 latency definition.
chunk0_latencies = []


def record_generation(prompt_index, prompt, trajectory, output_path, status):
    if local_rank != 0:
        return
    row = {
        "prompt_index": int(prompt_index),
        "prompt": prompt,
        "trajectory": trajectory or "",
        "dykv_case": str(config.dykv_case),
        "dykv_packing_mode": str(getattr(config, "dykv_packing_mode", "none")),
        "dykv_retrieval_frames": int(
            getattr(config, "dykv_retrieval_frames", 8)
        ),
        "dykv_compression_keep_ratio": float(
            getattr(config, "dykv_compression_keep_ratio", 0.5)
        ),
        "sink_mode": str(config.sink_mode),
        "sink_frames": int(config.sink_frames),
        "output_path": os.path.abspath(output_path),
        "status": status,
    }
    with open(output_manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    prompt = batch['prompts'][0]

    # Resolve camera and the final output name before checking resumable output.
    viewmats = None
    Ks = None
    traj_str = None
    if trajectory_list:
        traj_str = trajectory_list[idx]
    elif args.trajectory:
        traj_str = args.trajectory
    if traj_str:
        import numpy as np
        viewmats_np = parse_trajectory(traj_str)
        if len(viewmats_np) != args.num_output_frames:
            raise ValueError(
                f"trajectory for prompt {idx} has {len(viewmats_np)} poses, "
                f"but --num_output_frames={args.num_output_frames}"
            )
        fx, fy, cx, cy = 0.5, 0.5, 0.5, 0.5
        Ks_np = np.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]] * len(viewmats_np), dtype=np.float32)
        viewmats = torch.from_numpy(viewmats_np).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        Ks = torch.from_numpy(Ks_np).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    traj_suffix = "_" + traj_str.replace("*", "").replace(",", "") if traj_str else ""
    safe_prompt = prompt[:100].replace("/", "_").replace("\\", "_")
    output_path = os.path.join(
        args.output_folder, f'{idx:05d}_{safe_prompt}{traj_suffix}.mp4'
    )

    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            record_generation(idx, prompt, traj_str, output_path, "skipped_exists")
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt] 
        sampled_noise = torch.randn(
            [1, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            record_generation(idx, prompt, traj_str, output_path, "skipped_exists")
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # Generate frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        viewmats=viewmats,
        Ks=Ks
    )
    if local_rank == 0 and args.dykv:
        dykv_record = {
            "prompt_index": int(idx),
            "prompt": prompt,
            "trajectory": traj_str or "",
            "case": str(config.dykv_case),
            "sink_mode": str(config.sink_mode),
            "sink_frames": int(config.sink_frames),
            "summary": getattr(pipeline, "last_dykv_summary", {}),
        }
        with open(dykv_events_path, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(dykv_record, ensure_ascii=False) + "\n")

    # Record latency on rank 0; first prompt is warmup → None.
    # All pipelines stop the timer before VAE decode (see pipeline.last_chunk0_latency).
    if local_rank == 0:
        sample_lat = getattr(pipeline, "last_chunk0_latency", None)
        if len(chunk0_latencies) >= 1:
            chunk0_latencies.append(sample_lat)
        else:
            chunk0_latencies.append(None)

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if not (args.sp_size > 1 and local_rank != 0):
        write_video(output_path, video[0], fps=16)
        record_generation(idx, prompt, traj_str, output_path, "generated")
    if dist.is_initialized():
        dist.barrier()


# Aggregate latency on rank 0 (drop the first prompt's warmup).
if local_rank == 0:
    valid = [v for v in chunk0_latencies[1:] if v is not None]
    if valid:
        print(f"[timing] rank0 chunk0 latency excl. decode (from 2nd prompt): "
              f"avg={sum(valid)/len(valid):.3f}s over {len(valid)} samples")

       
