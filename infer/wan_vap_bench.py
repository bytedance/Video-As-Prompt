 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

import os
import random
from pathlib import Path
from typing import List

import torch
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import tempfile, shutil

from transformers import CLIPVisionModel
from diffusers import (
    AutoencoderKLWan,
    WanImageToVideoMOTPipeline,
    WanTransformer3DMOTModel,
)
from diffusers.utils import export_to_video, load_video


def export_via_tmp(frames, final_path, fps):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    local_tmp_dir = os.environ.get("LOCAL_TMP", "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
    suffix = final_path.suffix or ".mp4"

    fd, tmp_path = tempfile.mkstemp(prefix="vid_", suffix=suffix, dir=local_tmp_dir)
    os.close(fd)
    try:
        export_to_video(frames, tmp_path, fps=fps)

        partial = final_path.with_suffix(final_path.suffix + ".partial")
        shutil.copyfile(tmp_path, partial)
        os.replace(partial, final_path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

def select_frames(video_frames: List[PIL.Image.Image], num: int, mode: str) -> List[PIL.Image.Image]:
    if len(video_frames) == 0:
        return []
    if mode == "first":
        return video_frames[:num]
    if mode == "evenly":
        import torch as _torch
        idx = _torch.linspace(0, len(video_frames) - 1, num).long().tolist()
        return [video_frames[i] for i in idx]
    if mode == "random":
        if len(video_frames) <= num:
            return video_frames
        import random as _random
        start = _random.randint(0, len(video_frames) - num)
        return video_frames[start:start+num]
    return video_frames

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    model_id = "ckpts/Video-As-Prompt-Wan2.1-14B"
    output_root = "outputs_infer"

    set_global_seed(42)
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DMOTModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)

    pipe = WanImageToVideoMOTPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=image_encoder,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    benchmark = pd.read_csv("data/VAP-Data/benchmark.csv")
    data_dir = "data/VAP-Data"

    benchmark['video_paths'] = benchmark['video_paths'].apply(lambda x: os.path.join(data_dir, x))
    benchmark['ref_video_path'] = benchmark['ref_video_path'].apply(lambda x: os.path.join(data_dir, x))

    for row_idx, row in benchmark.iterrows():
        tar_video_path = row["video_paths"]
        ref_video_path = row["ref_video_path"]

        ref_video = load_video(ref_video_path)
        tar_video = load_video(tar_video_path)
        image = tar_video[0]
        ref_frames = select_frames(ref_video, num=49, mode="evenly")
        tar_frames = select_frames(tar_video, num=49, mode="evenly")

        output_frames = pipe(
            image=image,
            ref_videos=[ref_frames],
            prompt=row["tar_video_caption"],
            prompt_mot_ref=[
                row["ref_video_caption"]
            ],
            height=480,
            width=832,
            num_frames=49,
            frames_selection="evenly",
            guidance_scale=5.0,
            output_type="pil",
            # generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]

        # Concat [ref | target | ours]
        concat_frames = []
        for i in range(len(output_frames)):
            w, h = output_frames[i].size
            resized_tgt = tar_frames[i].resize((w, h), PIL.Image.LANCZOS)
            resized_ref = ref_frames[i].resize((w, h), PIL.Image.LANCZOS)
            img = PIL.Image.new("RGB", (w * 3, h))
            img.paste(resized_ref.convert("RGB"), (0, 0))
            img.paste(resized_tgt, (w, 0))
            img.paste(output_frames[i], (2 * w, 0))
            concat_frames.append(img)

        export_via_tmp(concat_frames, os.path.join(output_root, f"wan_vap_bench_{row_idx}.mp4"), fps=16)