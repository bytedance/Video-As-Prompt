 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

import os
import random
from pathlib import Path
from typing import List

import torch
import numpy as np
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

    target_image_path = "assets/images/demo/woman-7.jpg"
    ref_video_path = "assets/videos/demo/man-534.mp4"

    print(f"Loading from {model_checkpoint}")
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

    # pipe = WanImageToVideoMOTPipeline.from_pretrained(
    #     model_id,
    #     vae=vae,
    #     image_encoder=image_encoder,
    #     transformer=transformer,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    # )
    # # offload base on module
    # # pipe.enable_model_cpu_offload()
    # # offload base on layer
    # pipe.enable_sequential_cpu_offload()

    ref_video = load_video(ref_video_path)
    image = Image.open(target_image_path).convert("RGB")

    ref_frames = select_frames(ref_video, num=49, mode= "evenly")

    output_frames = pipe(
        image=image,
        ref_videos=[ref_frames],
        prompt="A young woman with curly hair, wearing a green hijab and a floral dress, plays a violin in front of a vintage green car on a tree-lined street. She executes a swift counter-clockwise turn to face the camera. During the turn, a brilliant shower of golden, sparkling particles erupts and momentarily obscures her figure. As the particles fade, she is revealed to have seamlessly transformed into a Labubu toy character. This new figure, now with the toy's signature large ears, big eyes, and toothy grin, maintains the original pose and continues playing the violin. The character's clothing—the green hijab, floral dress, and black overcoat—remains identical to the woman's. Throughout this transition, the camera stays static, and the street-side environment remains completely consistent.",
        prompt_mot_ref=[
            "A man stands with his back to the camera on a dirt path overlooking sun-drenched, rolling green tea plantations. He wears a blue and green plaid shirt, dark pants, and white shoes. As he turns to face the camera and spreads his arms, a brief, magical burst of sparkling golden light particles envelops him. Through this shimmer, he seamlessly transforms into a Labubu toy character. His head morphs into the iconic large, furry-eared head of the toy, featuring a wide grin with pointed teeth and red cheek markings. The character retains the man's original plaid shirt and clothing, which now fit its stylized, cartoonish body. The camera remains static throughout the transformation, positioned low among the tea bushes, maintaining a consistent view of the subject and the expansive scenery."
        ],
        height=480,
        width=832,
        num_frames=49,
        frames_selection="evenly",
        guidance_scale=5.0,
        output_type="pil",
        # generator=torch.Generator(device="cuda").manual_seed(seed),
    ).frames[0]

    export_via_tmp(output_frames, os.path.join(output_root, "wan_vap.mp4"), fps=16)
