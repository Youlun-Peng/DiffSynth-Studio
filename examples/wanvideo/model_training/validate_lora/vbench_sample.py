import os
import json
import torch
import random
import argparse
import multiprocessing as mp

from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


# -------------------------------------------------------------
# 模型加载（每个 GPU 的进程都会跑一次）
# -------------------------------------------------------------
def load_pipeline(device_id):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=f"cuda:{device_id}",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="Wan2.2_VAE.pth",
                        offload_device="cpu"),
        ],
    )

    pipe.load_lora(pipe.dit, "models/train/Experiment02/epoch-20.safetensors", alpha=1)
    pipe.enable_vram_management()

    return pipe


# -------------------------------------------------------------
# 子进程执行的采样任务
# -------------------------------------------------------------
def worker_task(device_id, task_list, save_dir, return_dict):

    # 每个子进程独立随机种子
    torch.manual_seed(device_id * 10000 + 123)
    random.seed(device_id * 10000 + 456)

    # 加载本 GPU 的 pipeline
    pipe = load_pipeline(device_id)

    # input image（你可自行修改）
    input_image = VideoData("data/nuscenes_dataset/nuscenes/video0001.mp4",
                            height=480, width=832)[0]

    local_seed_log = {}

    for (prompt, index) in task_list:

        save_path = os.path.join(save_dir, f"{prompt}-{index}.mp4")
        if os.path.exists(save_path):
            print(f"[GPU {device_id}] SKIP (already exists) -> {save_path}")
            continue
            
        seed_this = random.randint(0, 2**31 - 1)

        # 记录 seed
        local_seed_log.setdefault(prompt, []).append(seed_this)

        print(f"[GPU {device_id}] prompt={prompt}, idx={index}, seed={seed_this}")

        video = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=None,
            num_frames=81,
            seed=seed_this,
            tiled=True,
        )

        # safe_prompt = prompt.replace("/", "_").replace(" ", "_")

        save_video(video, save_path, fps=16, quality=5)
        print(f"[GPU {device_id}] saved -> {save_path}")

    return_dict[device_id] = local_seed_log


# -------------------------------------------------------------
# 主控函数：将任务分配给多个 GPU
# -------------------------------------------------------------
def main(args):

    # 读取 prompt
    if args.mode == "dimension":
        dims = args.dimensions.split(",")
    else:
        dims = ["all"]

    prompts = []
    for dim in dims:
        if dim == "all":
            file = "../vbench/prompts/all_dimension.txt"
        else:
            file = f"./prompts/prompts_per_dimension/{dim}.txt"

        with open(file, "r") as f:
            p = [line.strip() for line in f.readlines()]
        prompts.extend(p)

    print(f"[INFO] Total prompts: {len(prompts)}")

    # 构建任务列表
    tasks = []
    for dim in dims:
        num_samples = 25 if dim == "temporal_flickering" else 5

        # 所有 prompt 使用相同次数（保持 VBench 规范）
        for prompt in prompts:
            for i in range(num_samples):
                tasks.append((prompt, i))

    print(f"[INFO] Total video samples to generate: {len(tasks)}")

    # 创建输出目录
    os.makedirs(args.save_path, exist_ok=True)

    # 可使用的 GPU 数量
    num_gpus = args.num_gpus
    print(f"[INFO] Using {num_gpus} GPUs")

    # 将任务平均分配给 num_gpus 个进程
    chunks = [tasks[i::num_gpus] for i in range(num_gpus)]

    # manager dict 用于返回 seed log
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_task,
                       args=(gpu_id, chunks[gpu_id], args.save_path, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并所有子进程 seed log
    seed_log = {}
    for d in return_dict.values():
        for prompt, seeds in d.items():
            seed_log.setdefault(prompt, []).extend(seeds)

    # 保存 seed log
    with open(os.path.join(args.save_path, "seed_log.json"), "w") as f:
        json.dump(seed_log, f, indent=2)

    print("[DONE] Multi-GPU VBench sampling completed.")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="dimension",
                        choices=["dimension", "all"])
    parser.add_argument("--dimensions", type=str,
                        default="object_class,overall_consistency")
    parser.add_argument("--save_path", type=str, default="vbench_videos")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="使用的 GPU 数量（从 0 开始分配）")

    args = parser.parse_args()
    main(args)
