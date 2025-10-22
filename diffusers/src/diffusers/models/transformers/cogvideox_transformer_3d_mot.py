# Copyright (c) 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.
#
# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/finetrainers/blob/main/LICENSE.
#
# This modified file is released under the same license.


from typing import Any, Dict, Optional, Tuple, Union, List

import torch
from torch import nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, FeedForward
from ..attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0, CogVideoXAttnMOTProcessor2_0
from ..cache_utils import CacheMixin
from ..embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        # mot
        with_mot_ref: bool = False,
        _block_idx: int = 0,
        dim_mot_ref: Optional[int] = None,
        # ablation
        ablation_single_encoder: bool = False,
        ablation_residual_addition: bool = False,
    ):
        super().__init__()
        self.with_mot_ref = with_mot_ref
        self._block_idx = _block_idx
        self.dim_mot_ref = dim_mot_ref

        self.ablation_single_encoder = ablation_single_encoder
        self.ablation_residual_addition = ablation_residual_addition

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0() if (not self.with_mot_ref or self.ablation_single_encoder or self.ablation_residual_addition) else CogVideoXAttnMOTProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        if self.with_mot_ref:
            # 1. Self Attention
            self.norm1_mot_ref = CogVideoXLayerNormZero(time_embed_dim, dim if dim_mot_ref is None else dim_mot_ref, norm_elementwise_affine, norm_eps, bias=True)

            self.attn1_mot_ref = Attention(
                query_dim=dim if dim_mot_ref is None else dim_mot_ref,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                qk_norm="layer_norm" if qk_norm else None,
                eps=1e-6,
                bias=attention_bias,
                out_bias=attention_out_bias,
                processor=CogVideoXAttnMOTProcessor2_0() if (not self.ablation_single_encoder and not self.ablation_residual_addition) else CogVideoXAttnProcessor2_0(),
            )

            # 2. Feed Forward
            self.norm2_mot_ref = CogVideoXLayerNormZero(time_embed_dim, dim if dim_mot_ref is None else dim_mot_ref, norm_elementwise_affine, norm_eps, bias=True)

            self.ff_mot_ref = FeedForward(
                dim if dim_mot_ref is None else dim_mot_ref,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        # mot
        hidden_states_mot_ref: Optional[torch.Tensor] = None,
        encoder_hidden_states_mot_ref: Optional[torch.Tensor] = None,
        temb_mot_ref: Optional[torch.Tensor] = None,
        temb_list_mot_ref: Optional[List[torch.Tensor]] = None,
        image_rotary_emb_mot_ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        if not self.with_mot_ref:
            text_seq_length = encoder_hidden_states.size(1)
            attention_kwargs = attention_kwargs or {}

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
                hidden_states, encoder_hidden_states, temb
            )

            # attention
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )

            hidden_states = hidden_states + gate_msa * attn_hidden_states
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
                hidden_states, encoder_hidden_states, temb
            )

            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)

            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

            return hidden_states, encoder_hidden_states, hidden_states_mot_ref, encoder_hidden_states_mot_ref
        
        elif self.ablation_single_encoder and not self.ablation_residual_addition and self.with_mot_ref:
            text_seq_length = encoder_hidden_states.size(1)
            video_seq_length = hidden_states.size(1)
            attention_kwargs = attention_kwargs or {}

            ################################
            # reference encoder begin
            ################################

            # norm & modulate
            norm_hidden_states_mot_ref, norm_encoder_hidden_states_mot_ref, gate_msa_mot_ref, enc_gate_msa_mot_ref = self.norm1_mot_ref(
                hidden_states_mot_ref, encoder_hidden_states_mot_ref, temb_mot_ref
            )

            # attention
            attn_hidden_states_mot_ref, attn_encoder_hidden_states_mot_ref = self.attn1_mot_ref(
                hidden_states=norm_hidden_states_mot_ref,
                encoder_hidden_states=norm_encoder_hidden_states_mot_ref,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )

            hidden_states_mot_ref = hidden_states_mot_ref + gate_msa_mot_ref * attn_hidden_states_mot_ref
            encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref + enc_gate_msa_mot_ref * attn_encoder_hidden_states_mot_ref

            # norm & modulate
            norm_hidden_states_mot_ref, norm_encoder_hidden_states_mot_ref, gate_ff_mot_ref, enc_gate_ff_mot_ref = self.norm2_mot_ref(
                hidden_states_mot_ref, encoder_hidden_states_mot_ref, temb_mot_ref
            )

            # feed-forward
            norm_hidden_states_mot_ref = torch.cat([norm_encoder_hidden_states_mot_ref, norm_hidden_states_mot_ref], dim=1)
            ff_output_mot_ref = self.ff_mot_ref(norm_hidden_states_mot_ref)

            hidden_states_mot_ref = hidden_states_mot_ref + gate_ff_mot_ref * ff_output_mot_ref[:, text_seq_length:]
            encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref + enc_gate_ff_mot_ref * ff_output_mot_ref[:, :text_seq_length]


            ################################
            # reference encoder end
            ################################


            hidden_states = torch.cat([hidden_states, hidden_states_mot_ref], dim=1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_mot_ref], dim=1)

            tmp_image_rotary_emb = (
                torch.cat([image_rotary_emb[0], image_rotary_emb_mot_ref[0]], dim=0),
                torch.cat([image_rotary_emb[1], image_rotary_emb_mot_ref[1]], dim=0)
            )

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
                hidden_states, encoder_hidden_states, temb
            )

            # attention
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=tmp_image_rotary_emb,
                **attention_kwargs,
            )

            
            attn_hidden_states = attn_hidden_states[:, :video_seq_length]
            attn_encoder_hidden_states = attn_encoder_hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, :video_seq_length]
            encoder_hidden_states = encoder_hidden_states[:, :text_seq_length]


            hidden_states = hidden_states + gate_msa * attn_hidden_states
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
                hidden_states, encoder_hidden_states, temb
            )

            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)

            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

            return hidden_states, encoder_hidden_states, hidden_states_mot_ref, encoder_hidden_states_mot_ref

        elif self.ablation_residual_addition and not self.ablation_single_encoder and self.with_mot_ref:
            text_seq_length = encoder_hidden_states.size(1)
            video_seq_length = hidden_states.size(1)
            attention_kwargs = attention_kwargs or {}

            ################################
            # reference encoder begin
            ################################

            # norm & modulate
            norm_hidden_states_mot_ref, norm_encoder_hidden_states_mot_ref, gate_msa_mot_ref, enc_gate_msa_mot_ref = self.norm1_mot_ref(
                hidden_states_mot_ref, encoder_hidden_states_mot_ref, temb_mot_ref
            )

            # attention
            attn_hidden_states_mot_ref, attn_encoder_hidden_states_mot_ref = self.attn1_mot_ref(
                hidden_states=norm_hidden_states_mot_ref,
                encoder_hidden_states=norm_encoder_hidden_states_mot_ref,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )

            hidden_states_mot_ref = hidden_states_mot_ref + gate_msa_mot_ref * attn_hidden_states_mot_ref
            encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref + enc_gate_msa_mot_ref * attn_encoder_hidden_states_mot_ref

            # norm & modulate
            norm_hidden_states_mot_ref, norm_encoder_hidden_states_mot_ref, gate_ff_mot_ref, enc_gate_ff_mot_ref = self.norm2_mot_ref(
                hidden_states_mot_ref, encoder_hidden_states_mot_ref, temb_mot_ref
            )

            # feed-forward
            norm_hidden_states_mot_ref = torch.cat([norm_encoder_hidden_states_mot_ref, norm_hidden_states_mot_ref], dim=1)
            ff_output_mot_ref = self.ff_mot_ref(norm_hidden_states_mot_ref)

            hidden_states_mot_ref = hidden_states_mot_ref + gate_ff_mot_ref * ff_output_mot_ref[:, text_seq_length:]
            encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref + enc_gate_ff_mot_ref * ff_output_mot_ref[:, :text_seq_length]


            ################################
            # reference encoder end
            ################################


            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
                hidden_states, encoder_hidden_states, temb
            )

            # attention
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )

            hidden_states = hidden_states + gate_msa * attn_hidden_states
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
                hidden_states, encoder_hidden_states, temb
            )

            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)

            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

            ################################
            # residual add
            ################################
            hidden_states = hidden_states + hidden_states_mot_ref
            encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_mot_ref
            ################################
            # residual end
            ################################

            return hidden_states, encoder_hidden_states, hidden_states_mot_ref, encoder_hidden_states_mot_ref

        elif not self.ablation_single_encoder and not self.ablation_residual_addition and self.with_mot_ref:
            batch_size, hidden_dim = hidden_states.size(0), hidden_states.size(-1)
            video_seq_length = hidden_states.size(-2)
            video_seq_length_mot_ref = hidden_states_mot_ref.size(-2)
            text_seq_length = encoder_hidden_states.size(-2)
            text_seq_length_mot_ref = encoder_hidden_states_mot_ref.size(-2)
            num_mot_ref = int(video_seq_length_mot_ref // video_seq_length)
            attention_kwargs = attention_kwargs or {}

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
                hidden_states, encoder_hidden_states, temb
            )

            if temb_list_mot_ref is None and temb_mot_ref is not None:
                norm_hidden_states_mot_ref, norm_encoder_hidden_states_mot_ref, gate_msa_mot_ref, enc_gate_msa_mot_ref = self.norm1_mot_ref(
                    hidden_states_mot_ref, encoder_hidden_states_mot_ref, temb_mot_ref
                )
            elif temb_list_mot_ref is not None and temb_mot_ref is None:

                norm_hidden_states_mot_ref, norm_encoder_hidden_states_mot_ref, gate_msa_mot_ref, enc_gate_msa_mot_ref = self.norm1_mot_ref(
                    hidden_states_mot_ref.reshape(batch_size * num_mot_ref, video_seq_length, hidden_dim), 
                    encoder_hidden_states_mot_ref.reshape(batch_size * num_mot_ref, text_seq_length, hidden_dim), 
                    torch.cat(temb_list_mot_ref, dim=0)
                )
                norm_hidden_states_mot_ref = norm_hidden_states_mot_ref.reshape(batch_size, num_mot_ref * video_seq_length, hidden_dim)
                norm_encoder_hidden_states_mot_ref = norm_encoder_hidden_states_mot_ref.reshape(batch_size, num_mot_ref * text_seq_length, hidden_dim)
            else:
                raise NotImplementedError("Not supprted for temb_list_mot_ref is not None and temb_mot_ref is not None or both are None")

            # attention

            query, key, value, attention_mask = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                is_before_attn=True,
                is_ref_video=False,
                **attention_kwargs,
            )
            query_mot_ref, key_mot_ref, value_mot_ref, attention_mask_mot_ref = self.attn1_mot_ref(
                hidden_states=norm_hidden_states_mot_ref,
                encoder_hidden_states=norm_encoder_hidden_states_mot_ref,
                image_rotary_emb=image_rotary_emb_mot_ref,
                is_before_attn=True,
                is_ref_video=True,
                **attention_kwargs,
            )

            tmp_hidden_states = F.scaled_dot_product_attention(
                torch.cat([query, query_mot_ref], dim=-2), 
                torch.cat([key, key_mot_ref], dim=-2), 
                torch.cat([value, value_mot_ref], dim=-2), 
                attn_mask=None, 
                dropout_p=0.0, 
                is_causal=False
            )

            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=tmp_hidden_states[..., :video_seq_length + text_seq_length, :],
                is_before_attn=False,
                text_seq_length=text_seq_length,
            )
            attn_hidden_states_mot_ref, attn_encoder_hidden_states_mot_ref = self.attn1_mot_ref(
                hidden_states=tmp_hidden_states[..., video_seq_length + text_seq_length:, :],
                is_before_attn=False,
                text_seq_length=text_seq_length_mot_ref,
            )


            hidden_states = hidden_states + gate_msa * attn_hidden_states
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

            # norm & modulate
            norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
                hidden_states, encoder_hidden_states, temb
            )

            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(norm_hidden_states)

            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

            # mot

            # norm & modulate
            
            if temb_list_mot_ref is None and temb_mot_ref is not None:
                hidden_states_mot_ref = hidden_states_mot_ref + gate_msa_mot_ref * attn_hidden_states_mot_ref
                encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref + enc_gate_msa_mot_ref * attn_encoder_hidden_states_mot_ref

                norm_hidden_states_mot_ref, norm_encoder_hidden_states_mot_ref, gate_ff_mot_ref, enc_gate_ff_mot_ref = self.norm2_mot_ref(
                    hidden_states_mot_ref, encoder_hidden_states_mot_ref, temb_mot_ref
                )
            elif temb_list_mot_ref is not None and temb_mot_ref is None:
                hidden_states_mot_ref = hidden_states_mot_ref.reshape(batch_size, num_mot_ref, video_seq_length, hidden_dim) + \
                    gate_msa_mot_ref.reshape(batch_size, num_mot_ref, 1, hidden_dim) * \
                        attn_hidden_states_mot_ref.reshape(batch_size, num_mot_ref, video_seq_length, hidden_dim)
                hidden_states_mot_ref = hidden_states_mot_ref.reshape(batch_size, -1, hidden_dim)

                encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref.reshape(batch_size, num_mot_ref, text_seq_length, hidden_dim) + \
                    enc_gate_msa_mot_ref.reshape(batch_size, num_mot_ref, 1, hidden_dim) * \
                        attn_encoder_hidden_states_mot_ref.reshape(batch_size, num_mot_ref, text_seq_length, hidden_dim)
                encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref.reshape(batch_size, -1, hidden_dim)

                norm_hidden_states_mot_ref, norm_encoder_hidden_states_mot_ref, gate_ff_mot_ref, enc_gate_ff_mot_ref = self.norm2_mot_ref(
                    hidden_states_mot_ref.reshape(batch_size * num_mot_ref, video_seq_length, hidden_dim), 
                    encoder_hidden_states_mot_ref.reshape(batch_size * num_mot_ref, text_seq_length, hidden_dim), 
                    torch.cat(temb_list_mot_ref, dim=0)
                )
                norm_hidden_states_mot_ref = norm_hidden_states_mot_ref.reshape(batch_size, num_mot_ref * video_seq_length, hidden_dim)
                norm_encoder_hidden_states_mot_ref = norm_encoder_hidden_states_mot_ref.reshape(batch_size, num_mot_ref * text_seq_length, hidden_dim)

            else:
                raise NotImplementedError("Not supprted for temb_list_mot_ref is not None and temb_mot_ref is not None or both are None")

            # feed-forward
            norm_hidden_states_mot_ref = torch.cat([norm_encoder_hidden_states_mot_ref, norm_hidden_states_mot_ref], dim=1)
            ff_output_mot_ref = self.ff_mot_ref(norm_hidden_states_mot_ref)


            if temb_list_mot_ref is None and temb_mot_ref is not None:
                hidden_states_mot_ref = hidden_states_mot_ref + gate_ff_mot_ref * ff_output_mot_ref[:, text_seq_length_mot_ref:]
                encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref + enc_gate_ff_mot_ref * ff_output_mot_ref[:, :text_seq_length_mot_ref]
            elif temb_list_mot_ref is not None and temb_mot_ref is None:

                hidden_states_mot_ref = hidden_states_mot_ref.reshape(batch_size, num_mot_ref, video_seq_length, hidden_dim) + \
                    gate_ff_mot_ref.reshape(batch_size, num_mot_ref, 1, hidden_dim) * \
                        ff_output_mot_ref[:, text_seq_length_mot_ref:].reshape(batch_size, num_mot_ref, video_seq_length, hidden_dim)
                hidden_states_mot_ref = hidden_states_mot_ref.reshape(batch_size, -1, hidden_dim)

                encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref.reshape(batch_size, num_mot_ref, text_seq_length, hidden_dim) + \
                    enc_gate_ff_mot_ref.reshape(batch_size, num_mot_ref, 1, hidden_dim) * \
                        ff_output_mot_ref[:, :text_seq_length_mot_ref].reshape(batch_size, num_mot_ref, text_seq_length, hidden_dim)
                encoder_hidden_states_mot_ref = encoder_hidden_states_mot_ref.reshape(batch_size, -1, hidden_dim)

            return hidden_states, encoder_hidden_states, hidden_states_mot_ref, encoder_hidden_states_mot_ref
        else:
            raise ValueError(f"ablation_single_encoder: {self.ablation_single_encoder}, ablation_residual_addition: {self.ablation_residual_addition}, self.with_mot_ref: {self.with_mot_ref}")

class CogVideoXTransformer3DMOTModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        # mot
        block_idx_with_mot_ref: List[int] = [0, 10, 20],
        attention_head_dim_mot_ref: Optional[int] = None,
        supported_effect_types: Optional[List[str]] = None,
        num_ref_embeddings: Optional[int] = None,
        reference_train_mode: Optional[str] = None,
        # ablation
        ablation_single_encoder: bool = False,
        ablation_residual_addition: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if attention_head_dim_mot_ref is not None:
            inner_dim_mot_ref = num_attention_heads * attention_head_dim_mot_ref
        else:
            inner_dim_mot_ref = None

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # mot
        self.patch_embed_mot_ref = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout_mot_ref = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # mot
        self.time_proj_mot_ref = Timesteps(inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref, flip_sin_to_cos, freq_shift)
        self.time_embedding_mot_ref = TimestepEmbedding(inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref, time_embed_dim, timestep_activation_fn)



        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs


        # 3. Define spatio-temporal transformers blocks
        print(f"block_idx_with_mot_ref: {block_idx_with_mot_ref}")
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    # mot
                    with_mot_ref=i in block_idx_with_mot_ref, 
                    _block_idx=i,
                    dim_mot_ref=inner_dim_mot_ref,
                    # ablation
                    ablation_single_encoder=ablation_single_encoder,
                    ablation_residual_addition=ablation_residual_addition,
                )
                for i in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)


        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        # mot
        self.reference_train_mode = reference_train_mode
        if self.reference_train_mode in ["reference_independent"]:
            
            self.norm_final_mot_ref = nn.LayerNorm(inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref, norm_eps, norm_elementwise_affine)

            self.norm_out_mot_ref = AdaLayerNorm(
                embedding_dim=time_embed_dim,
                output_dim=2 * inner_dim if inner_dim_mot_ref is None else 2 * inner_dim_mot_ref,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                chunk_dim=1,
            )
            
            self.proj_out_mot_ref = nn.Linear(inner_dim if inner_dim_mot_ref is None else inner_dim_mot_ref, output_dim)


        self.supported_effect_types = supported_effect_types or []
        self.effect_embed_dim = inner_dim_mot_ref or inner_dim
        
        if self.supported_effect_types:
            # print(f"supported_effect_types: {supported_effect_types}")
            self.effect_embeddings = nn.ParameterDict({
                effect_type: nn.Parameter(torch.randn(1, 1, self.effect_embed_dim))
                for effect_type in self.supported_effect_types
            })
            for effect_embed in self.effect_embeddings.values():
                nn.init.normal_(effect_embed, std=0.02)
        else:
            self.effect_embeddings = None

        self.num_ref_embeddings = num_ref_embeddings
        self.ref_embed_dim = inner_dim_mot_ref or inner_dim

        if self.num_ref_embeddings:
            # print(f"num_ref_embeddings: {num_ref_embeddings}")
            self.ref_embeddings = nn.ParameterDict({
                f"ref_{ref_idx}": nn.Parameter(torch.randn(1, 1, self.ref_embed_dim))
                for ref_idx in range(self.num_ref_embeddings)
            })
            for ref_embed in self.ref_embeddings.values():
                nn.init.normal_(ref_embed, std=0.02)
        else:
            self.ref_embeddings = None

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        # mot
        num_mot_ref: int = 1,
        hidden_states_mot_ref: Optional[torch.Tensor] = None,
        encoder_hidden_states_mot_ref: Optional[torch.Tensor] = None,
        image_rotary_emb_mot_ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        effect_types: Optional[List[str]] = None,
        reference_train_mode: Optional[str] = None,
        timestep_list_mot_ref: Union[List[int], List[float], List[torch.LongTensor]] = None,
    ):
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

        batch_size, num_frames, channels, height, width = hidden_states.shape
        num_text_tokens = encoder_hidden_states.shape[-2]

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # mot
        if timestep_list_mot_ref is not None:
            emb_list_mot_ref = []
            for timestep_mot_ref in timestep_list_mot_ref:
                timesteps_mot_ref = timestep_mot_ref
                t_emb_mot_ref = self.time_proj_mot_ref(timesteps_mot_ref)
                t_emb_mot_ref = t_emb_mot_ref.to(dtype=hidden_states.dtype)
                emb_mot_ref = self.time_embedding_mot_ref(t_emb_mot_ref, timestep_cond)
                emb_list_mot_ref.append(emb_mot_ref)
            emb_mot_ref = None
            # print(f"emb_list_mot_ref: {len(emb_list_mot_ref)}-{emb_list_mot_ref[0].shape}, timestep_list_mot_ref: {timestep_list_mot_ref}")
        else:
            timesteps_mot_ref = timestep
            t_emb_mot_ref = self.time_proj_mot_ref(timesteps_mot_ref)
            t_emb_mot_ref = t_emb_mot_ref.to(dtype=hidden_states.dtype)
            emb_mot_ref = self.time_embedding_mot_ref(t_emb_mot_ref, timestep_cond)
            emb_list_mot_ref = None
            # print(f"emb_mot_ref: {emb_mot_ref.shape}, timestep: {timestep}")

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb
            # mot
            if emb_list_mot_ref is None:
                emb_mot_ref = emb_mot_ref + ofs_emb
            else:
                emb_list_mot_ref = [emb_list_mot_ref_item + ofs_emb for emb_list_mot_ref_item in emb_list_mot_ref]
                
        

        assert hidden_states_mot_ref.shape[1] // hidden_states.shape[1] == num_mot_ref, f"hidden_states_mot_ref.shape[1]: {hidden_states_mot_ref.shape}, hidden_states.shape[1]: {hidden_states.shape}"
        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # mot
        hidden_states_mot_ref_list = []
        encoder_hidden_states_mot_ref_list = []
        for i in range(num_mot_ref):
            hidden_states_mot_ref_i = self.patch_embed_mot_ref(encoder_hidden_states_mot_ref[:, i*num_text_tokens:(i+1)*num_text_tokens], hidden_states_mot_ref[:, i*num_frames:(i+1)*num_frames])
            hidden_states_mot_ref_i = self.embedding_dropout_mot_ref(hidden_states_mot_ref_i)

            if self.ref_embeddings is not None:
                ref_embed = self.ref_embeddings[f"ref_{int(num_mot_ref - i - 1)}"]  # [1, 1, D]
                ref_embed = ref_embed.expand(
                    hidden_states_mot_ref_i.shape[0], 
                    hidden_states_mot_ref_i.shape[1], 
                    self.ref_embed_dim
                )
                hidden_states_mot_ref_i = hidden_states_mot_ref_i + ref_embed

            if self.effect_embeddings is not None and effect_types is not None and i < len(effect_types):
                effect_type = effect_types[i]
                if effect_type in self.effect_embeddings:
                    effect_embed = self.effect_embeddings[effect_type]  # [1, 1, D]
                    effect_embed = effect_embed.expand(
                        hidden_states_mot_ref_i.shape[0], 
                        hidden_states_mot_ref_i.shape[1], 
                        self.effect_embed_dim
                    )
                    hidden_states_mot_ref_i = hidden_states_mot_ref_i + effect_embed
                else:
                    raise ValueError(f"{effect_type} is not supported in {self.effect_embeddings.keys()}")
            
            encoder_hidden_states_mot_ref_list.append(hidden_states_mot_ref_i[:, :text_seq_length])
            hidden_states_mot_ref_list.append(hidden_states_mot_ref_i[:, text_seq_length:])
        hidden_states_mot_ref = torch.cat(hidden_states_mot_ref_list, dim=1)
        encoder_hidden_states_mot_ref = torch.cat(encoder_hidden_states_mot_ref_list, dim=1)

        # HACK: DPO
        if hidden_states.shape[0] == 2 and emb.shape[0] == 1 and emb_mot_ref is not None and emb_mot_ref.shape[0] == 1:
            emb = emb.unsqueeze(1).expand(-1, 2, -1).reshape(2, -1)
            emb_mot_ref = emb_mot_ref.unsqueeze(1).expand(-1, 2, -1).reshape(2, -1)

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states, hidden_states_mot_ref, encoder_hidden_states_mot_ref = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                    # mot
                    hidden_states_mot_ref=hidden_states_mot_ref,
                    encoder_hidden_states_mot_ref=encoder_hidden_states_mot_ref,
                    temb_mot_ref=emb_mot_ref,
                    temb_list_mot_ref=emb_list_mot_ref,
                    image_rotary_emb_mot_ref=image_rotary_emb_mot_ref,
                )
            else:
                hidden_states, encoder_hidden_states, hidden_states_mot_ref, encoder_hidden_states_mot_ref = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    # mot
                    hidden_states_mot_ref=hidden_states_mot_ref,
                    encoder_hidden_states_mot_ref=encoder_hidden_states_mot_ref,
                    temb_mot_ref=emb_mot_ref,
                    temb_list_mot_ref=emb_list_mot_ref,
                    image_rotary_emb_mot_ref=image_rotary_emb_mot_ref,
                )

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)


        if self.reference_train_mode in ["reference_independent"]:
            hidden_states_mot_ref = self.norm_final_mot_ref(hidden_states_mot_ref)

            # 4. Final block
            if emb_mot_ref is not None and emb_list_mot_ref is None: 
                hidden_states_mot_ref = self.norm_out_mot_ref(hidden_states_mot_ref, temb=emb_mot_ref)
            elif emb_mot_ref is None and emb_list_mot_ref is not None:
                hidden_states_mot_ref = self.norm_out_mot_ref(
                    hidden_states_mot_ref.reshape(batch_size*num_mot_ref, hidden_states.shape[-2], hidden_states_mot_ref.shape[-1]), 
                    temb=torch.cat(emb_list_mot_ref, dim=0)
                )
            else:
                raise ValueError("emb_mot_ref and emb_list_mot_ref cannot be both None or both Non-None")
            hidden_states_mot_ref = self.proj_out_mot_ref(hidden_states_mot_ref)

            # 5. Unpatchify
            p = self.config.patch_size
            p_t = self.config.patch_size_t

            if p_t is None:
                output_mot_ref = hidden_states_mot_ref.reshape(batch_size, num_frames * num_mot_ref, height // p, width // p, -1, p, p)
                output_mot_ref = output_mot_ref.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            else:
                output_mot_ref = hidden_states_mot_ref.reshape(
                    batch_size, (num_frames * num_mot_ref + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                )
                output_mot_ref = output_mot_ref.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
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

