# Copyright (c) 2025 The Wan Team and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.
#
# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/finetrainers/blob/main/LICENSE.
#
# This modified file is released under the same license.
import math
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ..attention import FeedForward
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import FP32LayerNorm

import einops

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnCrossMOTProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        # mot
        num_mot_ref=1,
    ) -> torch.Tensor:

        # 512 is the context length of the text encoder, hardcoded for now
        image_context_length = encoder_hidden_states.shape[1] - 512 * num_mot_ref
        encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
        encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        key_img = attn.norm_added_k(key_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)

        key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states_img = F.scaled_dot_product_attention(
                einops.rearrange(query, 'b h (n l) c -> (b n) h l c', n=num_mot_ref), 
                einops.rearrange(key_img, 'b h (n l) c -> (b n) h l c', n=num_mot_ref), 
                einops.rearrange(value_img, 'b h (n l) c -> (b n) h l c', n=num_mot_ref), 
                attn_mask=None, dropout_p=0.0, is_causal=False
            )
        hidden_states_img = einops.rearrange(hidden_states_img, '(b n) h l c -> b h (n l) c', n=num_mot_ref)

        hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
        hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            einops.rearrange(query, 'b h (n l) c -> (b n) h l c', n=num_mot_ref), 
            einops.rearrange(key, 'b h (n l) c -> (b n) h l c', n=num_mot_ref), 
            einops.rearrange(value, 'b h (n l) c -> (b n) h l c', n=num_mot_ref), 
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = einops.rearrange(hidden_states, '(b n) h l c -> b h (n l) c', n=num_mot_ref)

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnMOTProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnMOTProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        # mot
        is_before_attn=True,
        query: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_before_attn:
            encoder_hidden_states = hidden_states

            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if rotary_emb is not None:

                def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                    dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                    x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                    return x_out.type_as(hidden_states)

                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)

            return query, key, value, attention_mask

        else:
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanTimeTextImageEmbeddingRef(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep_list_mot_ref: List[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype

        temb = []
        timestep_proj = []
        for timestep in timestep_list_mot_ref:
            # print(f"timestep: {timestep} in timestep_list_mot_ref:{timestep_list_mot_ref}")
            timestep = self.timesteps_proj(timestep)

            
            if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
                timestep = timestep.to(time_embedder_dtype)
            tmp_temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
            tmp_timestep_proj = self.time_proj(self.act_fn(tmp_temb))

            temb.append(tmp_temb)
            timestep_proj.append(tmp_timestep_proj)
        temb = torch.cat(temb, dim=0)
        timestep_proj = torch.cat(timestep_proj, dim=0)

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=freqs_dtype
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class WanRotaryPosEmbedRef(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim, self.h_dim, self.w_dim = t_dim, h_dim, w_dim
        self.theta = theta
        self.freqs = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = []
        for dim_idx, dim in enumerate([self.t_dim, self.h_dim, self.w_dim]):
            if dim_idx == 0:
                positions = torch.arange(-hidden_states.shape[2+dim_idx], self.max_seq_len)                
                freq = get_1d_rotary_pos_embed(
                    dim, positions, self.theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
                )
            else:
                freq = get_1d_rotary_pos_embed(
                    dim, self.max_seq_len, self.theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
                )
            
            freq = freq[:self.max_seq_len]
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        # mot
        with_mot_ref: bool = False,
        _block_idx: int = 0,
        dim_mot_ref: Optional[int] = None,
    ):
        super().__init__()
        self.with_mot_ref = with_mot_ref
        self._block_idx = _block_idx
        self.dim_mot_ref = dim_mot_ref

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0() if not self.with_mot_ref else WanAttnMOTProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0() if not self.with_mot_ref else WanAttnCrossMOTProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)


        # mot
        if self.with_mot_ref:
            # 1. Self-attention
            self.norm1_mot_ref = FP32LayerNorm(dim if dim_mot_ref is None else dim_mot_ref, eps, elementwise_affine=False)
            self.attn1_mot_ref = Attention(
                query_dim=dim if dim_mot_ref is None else dim_mot_ref,
                heads=num_heads,
                kv_heads=num_heads,
                dim_head=dim // num_heads if dim_mot_ref is None else dim_mot_ref // num_heads,
                qk_norm=qk_norm,
                eps=eps,
                bias=True,
                cross_attention_dim=None,
                out_bias=True,
                processor=WanAttnMOTProcessor2_0(),
            )

            # 2. Cross-attention
            self.attn2_mot_ref = Attention(
                query_dim=dim if dim_mot_ref is None else dim_mot_ref,
                heads=num_heads,
                kv_heads=num_heads,
                dim_head=dim // num_heads if dim_mot_ref is None else dim_mot_ref // num_heads,
                qk_norm=qk_norm,
                eps=eps,
                bias=True,
                cross_attention_dim=None,
                out_bias=True,
                added_kv_proj_dim=added_kv_proj_dim,
                added_proj_bias=True,
                processor=WanAttnCrossMOTProcessor2_0(),
            )
            self.norm2_mot_ref = FP32LayerNorm(dim if dim_mot_ref is None else dim_mot_ref, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

            # 3. Feed-forward
            self.ffn_mot_ref = FeedForward(dim if dim_mot_ref is None else dim_mot_ref, inner_dim=ffn_dim, activation_fn="gelu-approximate")
            self.norm3_mot_ref = FP32LayerNorm(dim if dim_mot_ref is None else dim_mot_ref, eps, elementwise_affine=False)

            self.scale_shift_table_mot_ref = nn.Parameter(torch.randn(1, 6, dim if dim_mot_ref is None else dim_mot_ref) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        # mot
        hidden_states_mot_ref: Optional[torch.Tensor] = None,
        encoder_hidden_states_mot_ref: Optional[torch.Tensor] = None,
        temb_mot_ref: Optional[torch.Tensor] = None,
        rotary_emb_mot_ref: Optional[torch.Tensor] = None,
        num_mot_ref: Optional[int] = None,
    ) -> torch.Tensor:

        if not self.with_mot_ref:

            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

            # 1. Self-attention
            norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
            attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

            # 2. Cross-attention
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = hidden_states + attn_output

            # 3. Feed-forward
            norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
                hidden_states
            )
            ff_output = self.ffn(norm_hidden_states)
            hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        
        else:
            

            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

            # BUG: temb_mot_ref only support 1
            assert num_mot_ref == 1
            scale_params_mot_ref = self.scale_shift_table_mot_ref + temb_mot_ref.float()
            
            scale_params_mot_ref = einops.rearrange(scale_params_mot_ref, '(b n) t c -> b n t c', n=num_mot_ref)

            shift_msa_mot_ref, scale_msa_mot_ref, gate_msa_mot_ref, c_shift_msa_mot_ref, c_scale_msa_mot_ref, c_gate_msa_mot_ref = scale_params_mot_ref.chunk(6, dim=2)


            # 1. Self-attention
            norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
            norm_hidden_states_mot_ref = einops.rearrange(self.norm1_mot_ref(hidden_states_mot_ref.float()), 'b (n t) c -> b n t c', n=num_mot_ref)
            norm_hidden_states_mot_ref = (norm_hidden_states_mot_ref * (1 + scale_msa_mot_ref) + shift_msa_mot_ref).type_as(hidden_states_mot_ref)
            norm_hidden_states_mot_ref = einops.rearrange(norm_hidden_states_mot_ref, 'b n t c -> b (n t) c', n=num_mot_ref)

            query, key, value, _ = self.attn1(
                hidden_states=norm_hidden_states, 
                rotary_emb=rotary_emb,
                is_before_attn=True,
            )

            query_mot_ref, key_mot_ref, value_mot_ref, _ = self.attn1_mot_ref(
                hidden_states=norm_hidden_states_mot_ref, 
                rotary_emb=rotary_emb_mot_ref,
                is_before_attn=True,
            )

            tmp_hidden_states = F.scaled_dot_product_attention(
                torch.cat([query, query_mot_ref], dim=-2),
                torch.cat([key, key_mot_ref], dim=-2),
                torch.cat([value, value_mot_ref], dim=-2),
                attn_mask=None, 
                dropout_p=0.0, 
                is_causal=False
            )

            tmp_hidden_states = tmp_hidden_states.type_as(query)
            attn_output, attn_output_mot_ref = torch.split(tmp_hidden_states, [query.shape[-2], query_mot_ref.shape[-2]], dim=-2)

            attn_output = self.attn1(
                hidden_states=attn_output,
                is_before_attn=False
            )
            attn_output_mot_ref = self.attn1_mot_ref(
                hidden_states=attn_output_mot_ref,
                is_before_attn=False
            )

            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

            attn_output_mot_ref = einops.rearrange(attn_output_mot_ref, 'b (n t) c -> b n t c', n=num_mot_ref)
            attn_output_mot_ref = attn_output_mot_ref * gate_msa_mot_ref
            attn_output_mot_ref = einops.rearrange(attn_output_mot_ref, 'b n t c -> b (n t) c', n=num_mot_ref)
            hidden_states_mot_ref = (hidden_states_mot_ref.float() + attn_output_mot_ref).type_as(hidden_states_mot_ref)



            # 2. Cross-attention
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            norm_hidden_states_mot_ref = self.norm2_mot_ref(hidden_states_mot_ref.float()).type_as(hidden_states_mot_ref)

            attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)

            attn_output_mot_ref = self.attn2_mot_ref(hidden_states=norm_hidden_states_mot_ref, encoder_hidden_states=encoder_hidden_states_mot_ref)

            hidden_states = hidden_states + attn_output
            hidden_states_mot_ref = hidden_states_mot_ref + attn_output_mot_ref


            # 3. Feed-forward
            norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
                hidden_states
            )
            ff_output = self.ffn(norm_hidden_states)
            hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

            
            norm_hidden_states_mot_ref = einops.rearrange(self.norm3_mot_ref(hidden_states_mot_ref.float()), 'b (n t) c -> b n t c', n=num_mot_ref)
            norm_hidden_states_mot_ref = (norm_hidden_states_mot_ref * (1 + c_scale_msa_mot_ref) + c_shift_msa_mot_ref).type_as(hidden_states_mot_ref)
            norm_hidden_states_mot_ref = einops.rearrange(norm_hidden_states_mot_ref, 'b n t c -> b (n t) c', n=num_mot_ref)

            ff_output_mot_ref = self.ffn_mot_ref(norm_hidden_states_mot_ref)

            ff_output_mot_ref = einops.rearrange(ff_output_mot_ref, 'b (n t) c -> b n t c', n=num_mot_ref)
            ff_output_mot_ref = ff_output_mot_ref.float() * c_gate_msa_mot_ref
            ff_output_mot_ref = einops.rearrange(ff_output_mot_ref, 'b n t c -> b (n t) c', n=num_mot_ref)

            hidden_states_mot_ref = (hidden_states_mot_ref.float() + ff_output_mot_ref).type_as(hidden_states_mot_ref)

        return hidden_states, hidden_states_mot_ref


class WanTransformer3DMOTModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        # mot
        block_idx_with_mot_ref: List[int] = [0, 10, 20],
        attention_head_dim_mot_ref: Optional[int] = None,
        supported_effect_types: Optional[List[str]] = None,
        num_ref_embeddings: Optional[int] = None,
        reference_train_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        if attention_head_dim_mot_ref is not None:
            inner_dim_mot_ref = num_attention_heads * attention_head_dim_mot_ref
        else:
            inner_dim_mot_ref = None

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        # 1-mot. Patch & position embedding
        self.rope_mot_ref = WanRotaryPosEmbedRef(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mot_ref = nn.Conv3d(in_channels, inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        # 2-mot. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder_mot_ref = WanTimeTextImageEmbeddingRef(
            dim=inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6 if inner_dim_mot_ref is None else inner_dim_mot_ref * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        print(f"block_idx_with_mot_ref: {block_idx_with_mot_ref}")
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_attention_heads,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                    # mot
                    with_mot_ref=block_idx in block_idx_with_mot_ref, 
                    _block_idx=block_idx,
                    dim_mot_ref=inner_dim_mot_ref,
                )
                for block_idx in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.reference_train_mode = reference_train_mode
        if self.reference_train_mode in ["reference_independent"]:
            # 4-mot. Output norm & projection
            self.norm_out_mot_ref = FP32LayerNorm(
                inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref, 
                eps, 
                elementwise_affine=False
            )
            self.proj_out_mot_ref = nn.Linear(
                inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref, 
                out_channels * math.prod(patch_size)
            )
            if inner_dim_mot_ref is None:
                self.scale_shift_table_mot_ref = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
            else:
                self.scale_shift_table_mot_ref = nn.Parameter(torch.randn(1, 2, inner_dim_mot_ref) / inner_dim_mot_ref**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        # mot
        num_mot_ref: int = 1,
        hidden_states_mot_ref: Optional[torch.Tensor] = None,
        timestep_list_mot_ref: Union[List[int], List[float], List[torch.LongTensor]] = None,
        encoder_hidden_states_mot_ref: Optional[torch.Tensor] = None,
        encoder_hidden_states_image_mot_ref: Optional[torch.Tensor] = None,
        effect_types: Optional[List[str]] = None,
        reference_train_mode: Optional[str] = None,
        
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        # mot
        rotary_emb_mot_ref = self.rope_mot_ref(hidden_states_mot_ref)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        # mot
        hidden_states_mot_ref = self.patch_embedding_mot_ref(hidden_states_mot_ref)
        hidden_states_mot_ref = hidden_states_mot_ref.flatten(2).transpose(1, 2)

        # print(f"hidden_states: {hidden_states.shape}; hidden_states_mot_ref:{hidden_states_mot_ref.shape}")

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        # mot
        temb_mot_ref, timestep_proj_mot_ref, encoder_hidden_states_mot_ref, encoder_hidden_states_image_mot_ref = self.condition_embedder_mot_ref(
            timestep_list_mot_ref, encoder_hidden_states_mot_ref, encoder_hidden_states_image_mot_ref
        )
        timestep_proj_mot_ref = timestep_proj_mot_ref.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
            # mot
            encoder_hidden_states_mot_ref = torch.concat([encoder_hidden_states_image_mot_ref, encoder_hidden_states_mot_ref], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states, hidden_states_mot_ref = self._gradient_checkpointing_func(
                    block, 
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=timestep_proj,
                    rotary_emb=rotary_emb,
                    # mot
                    hidden_states_mot_ref=hidden_states_mot_ref,
                    encoder_hidden_states_mot_ref=encoder_hidden_states_mot_ref,
                    temb_mot_ref=timestep_proj_mot_ref,
                    rotary_emb_mot_ref=rotary_emb_mot_ref,
                    num_mot_ref=num_mot_ref,
                )
        else:
            for block in self.blocks:
                hidden_states, hidden_states_mot_ref = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=timestep_proj,
                    rotary_emb=rotary_emb,
                    # mot
                    hidden_states_mot_ref=hidden_states_mot_ref,
                    encoder_hidden_states_mot_ref=encoder_hidden_states_mot_ref,
                    temb_mot_ref=timestep_proj_mot_ref,
                    rotary_emb_mot_ref=rotary_emb_mot_ref,
                    num_mot_ref=num_mot_ref,
                )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


        if self.reference_train_mode in ["reference_independent"]:
            # 5-mot. Output norm, projection & unpatchify
            shift_mot_ref, scale_mot_ref = (self.scale_shift_table_mot_ref + temb_mot_ref.unsqueeze(1)).chunk(2, dim=1)

            shift_mot_ref = shift_mot_ref.to(hidden_states.device)
            scale_mot_ref = scale_mot_ref.to(hidden_states.device)

            hidden_states_mot_ref = (self.norm_out_mot_ref(hidden_states_mot_ref.float()) * (1 + scale_mot_ref) + shift_mot_ref).type_as(hidden_states_mot_ref)
            hidden_states_mot_ref = self.proj_out_mot_ref(hidden_states)

            hidden_states_mot_ref = hidden_states_mot_ref.reshape(
                batch_size, post_patch_num_frames * num_mot_ref, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            )
            hidden_states_mot_ref = hidden_states_mot_ref.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output_mot_ref = hidden_states_mot_ref.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        else:
            output_mot_ref = None

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict and output_mot_ref is None:
            return (output,)
        elif not return_dict and output_mot_ref is not None:
            return (output, output_mot_ref)
        elif return_dict and output_mot_ref is None:
            return Transformer2DModelOutput(sample=output)
        elif return_dict and output_mot_ref is not None:
            return Transformer2DModelOutput(sample=output, sample_mot_ref=output_mot_ref)

