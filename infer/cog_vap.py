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

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXImageToVideoMOTPipeline,
    CogVideoXTransformer3DMOTModel,
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
    model_id = "ckpts/Video-As-Prompt-CogVideoX-5B"
    output_root = "outputs_infer"

    target_image_path = "assets/images/demo/animal-2.jpg"
    ref_video_path = "assets/videos/demo/object-725.mp4"

    set_global_seed(42)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = CogVideoXTransformer3DMOTModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = CogVideoXImageToVideoMOTPipeline.from_pretrained(
        model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    ref_video = load_video(ref_video_path)
    image = Image.open(target_image_path).convert("RGB")

    ref_frames = select_frames(ref_video, num=49, mode="evenly")
    
    output_frames = pipe(
        image=image,
        ref_videos=[ref_frames],
        prompt="A chestnut-colored horse stands on a grassy hill against a backdrop of distant, snow-dusted mountains. The horse begins to inflate, its defined, muscular body swelling and rounding into a smooth, balloon-like form while retaining its rich, brown hide color. Without changing its orientation, the now-buoyant horse lifts silently from the ground. It begins a steady vertical ascent, rising straight up and eventually floating out of the top of the frame. The camera remains completely static throughout the entire sequence, holding a fixed shot on the landscape as the horse transforms and departs, ensuring the verdant hill and mountain range in the background stay perfectly still.",
        prompt_mot_ref=[
            "A hand holds up a single beige sneaker decorated with gold calligraphy and floral illustrations, with small green plants tucked inside. The sneaker immediately begins to inflate like a balloon, its shape distorting as the decorative details stretch and warp across the expanding surface. It rapidly transforms into a perfectly smooth, matte beige sphere, inheriting the primary color from the original shoe. Once the transformation is complete, the new balloon-like object quickly ascends, moving straight up and exiting the top of the frame. The camera remains completely static and the plain white background is unchanged throughout the entire sequence."
        ],
        height=480,
        width=720,
        num_frames=49,
        frames_selection="evenly",
        use_dynamic_cfg=True,
        # generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    export_via_tmp(output_frames, os.path.join(output_root, "cog_vap.mp4"), fps=16)