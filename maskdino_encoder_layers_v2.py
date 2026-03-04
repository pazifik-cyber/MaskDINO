# Copyright (c) OpenMMLab. All rights reserved.
# ViT-5 Transformer Optimization for MaskDINO Encoder (Pixel Decoder)
# Stage 1+2: RMSNorm + Pre-Norm + SwiGLU FFN + LayerScale
#
# Changes vs original maskdino_encoder_layers.py:
#   [MSDeformAttnTransformerEncoderOnly → ViT5MSDeformAttnTransformerEncoderOnly]
#     1. Post-Norm → Pre-Norm
#     2. nn.LayerNorm → RMSNorm
#     3. ReLU FFN → SwiGLU FFN
#     4. Bare residual → LayerScale residual (init=1e-4)
#
#   [MaskDINOEncoder → ViT5MaskDINOEncoder]
#     5. self.transformer → ViT5MSDeformAttnTransformerEncoderOnly
#     All FPN / mask_features / lateral/output convs are inherited unchanged.
#
# Design note on reference points:
#   The original encoder uses mmdet's DeformableDetrTransformerEncoder which
#   computes normalized grid reference points internally. We replicate that
#   logic in ViT5MSDeformAttnEncoderLayer so we can apply Pre-Norm around it.

import warnings
from collections import namedtuple
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import caffe2_xavier_init
from torch import Tensor
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.init import normal_

from mmdet.models.layers import SinePositionalEncoding

# ViT-5 primitives
from .vit5_transformer_utils import RMSNorm, SwiGLU, LayerScale

# Base encoder – we subclass MaskDINOEncoder so FPN/conv layers are inherited
from .maskdino_encoder_layers import MaskDINOEncoder


# =============================================================================
# ViT5MSDeformAttnEncoderLayer
# One encoder layer: Deformable Self-Attn + SwiGLU FFN, both Pre-Norm + LayerScale
# =============================================================================

class ViT5MSDeformAttnEncoderLayer(nn.Module):
    """Single encoder layer with ViT-5 Transformer improvements.

    Architecture:
        x = x + LayerScale_sa( MSDeformAttn( RMSNorm(x) ) )
        x = x + LayerScale_ffn( SwiGLU( RMSNorm(x) ) )

    Args:
        d_model (int): Feature dimension. Default: 256.
        d_ffn (int): SwiGLU hidden dimension. Default: 1024.
        dropout (float): Dropout probability. Default: 0.1.
        num_feature_levels (int): Number of feature scales. Default: 4.
        n_heads (int): Attention heads. Default: 8.
        n_points (int): Deformable sampling points per head per level. Default: 4.
        init_values (float): LayerScale init value. Default: 1e-4.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        num_feature_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        init_values: float = 1e-4,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Deformable Multi-Scale Self-Attention
        # ------------------------------------------------------------------
        self.attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_levels=num_feature_levels,
            num_heads=n_heads,
            num_points=n_points,
            dropout=dropout,
        )
        # Stage 1: Pre-Norm with RMSNorm
        self.norm_attn = RMSNorm(d_model)
        # Stage 2: LayerScale
        self.ls_attn = LayerScale(d_model, init_values=init_values)

        # ------------------------------------------------------------------
        # Feed-Forward Network: SwiGLU
        # ------------------------------------------------------------------
        # Stage 2: SwiGLU replaces Linear→ReLU→Linear
        self.ffn = SwiGLU(
            in_features=d_model,
            hidden_features=d_ffn,
            out_features=d_model,
            dropout=dropout,
        )
        # Stage 1: Pre-Norm with RMSNorm
        self.norm_ffn = RMSNorm(d_model)
        # Stage 2: LayerScale
        self.ls_ffn = LayerScale(d_model, init_values=init_values)

    def forward(
        self,
        query: Tensor,                         # [bs, sum(HiWi), d_model]
        query_pos: Tensor,                     # [bs, sum(HiWi), d_model]
        reference_points: Tensor,              # [bs, sum(HiWi), n_levels, 2]
        spatial_shapes: Tensor,                # [n_levels, 2]
        level_start_index: Tensor,             # [n_levels]
        key_padding_mask: Tensor = None,       # [bs, sum(HiWi)]
    ) -> Tensor:
        """
        Pre-Norm forward.

        For the attention:
            - identity=zeros_like(normed) disables the internal residual of
              MultiScaleDeformableAttention so we can apply LayerScale ourselves.
        """
        # ==================================================================
        # 1. Deformable Self-Attention  (Pre-Norm + LayerScale)
        # ==================================================================
        # Pre-Norm: normalize features before attention.
        # query_pos is passed separately (NOT pre-added to normed).
        # mmcv's MultiScaleDeformableAttention internally computes
        #   (normed + query_pos) for sampling-offset prediction,
        #   but uses normed (without pos) as the actual value content.
        # This matches the Pre-Norm convention: pos is only for guiding
        # where to look, not mixed into the normalized feature stream.
        normed = self.norm_attn(query)

        # identity=zeros → disables internal residual; we apply LayerScale.
        attn_out = self.attn(
            query=normed,
            value=normed,
            identity=torch.zeros_like(normed),
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        # LayerScale + residual
        query = query + self.ls_attn(attn_out)

        # ==================================================================
        # 2. SwiGLU FFN  (Pre-Norm + LayerScale)
        # ==================================================================
        normed = self.norm_ffn(query)
        ffn_out = self.ffn(normed)
        query = query + self.ls_ffn(ffn_out)

        return query


# =============================================================================
# ViT5MSDeformAttnTransformerEncoderOnly
# Drop-in replacement for MSDeformAttnTransformerEncoderOnly
# =============================================================================

class ViT5MSDeformAttnTransformerEncoderOnly(nn.Module):
    """Multi-scale deformable attention encoder with ViT-5 improvements.

    Drop-in replacement for MSDeformAttnTransformerEncoderOnly.
    Same interface: forward(srcs, masks, pos_embeds) → (memory, spatial_shapes, level_start_index)

    Args:
        d_model (int): Feature dimension. Default: 256.
        nhead (int): Attention heads. Default: 8.
        num_encoder_layers (int): Number of encoder layers. Default: 6.
        dim_feedforward (int): SwiGLU hidden dim. Default: 1024.
        dropout (float): Dropout probability. Default: 0.1.
        num_feature_levels (int): Number of feature scales. Default: 4.
        enc_n_points (int): Deformable sampling points. Default: 4.
        init_values (float): LayerScale init value. Default: 1e-4.
        swiglu_ffn_ratio (float): Scale factor on dim_feedforward for SwiGLU.
            Use 2/3 to match standard FFN param count. Default: 1.0.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_feature_levels: int = 4,
        enc_n_points: int = 4,
        init_values: float = 1e-4,
        swiglu_ffn_ratio: float = 1.0,
        # act_cfg: accepted for API compatibility with MaskDINOEncoder call-site
        # (original MSDeformAttnTransformerEncoderOnly uses it for the ReLU FFN).
        # Ignored here because the FFN is replaced by SwiGLU with fixed SiLU activation.
        act_cfg=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        swiglu_hidden = int(dim_feedforward * swiglu_ffn_ratio)

        # Stack of ViT5 encoder layers
        self.layers = nn.ModuleList([
            ViT5MSDeformAttnEncoderLayer(
                d_model=d_model,
                d_ffn=swiglu_hidden,
                dropout=dropout,
                num_feature_levels=num_feature_levels,
                n_heads=nhead,
                n_points=enc_n_points,
                init_values=init_values,
            )
            for _ in range(num_encoder_layers)
        ])

        # Level embedding: distinguishes features from different resolution levels
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        normal_(self.level_embed)

    @staticmethod
    def get_reference_points(
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Compute normalized 2D grid reference points for encoder self-attention.

        Each token attends to positions across all feature levels, with reference
        points at the center of each spatial location normalized by valid_ratios.

        Returns:
            Tensor: [bs, sum(HiWi), n_levels, 2] reference points in [0,1] range.
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij',
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [bs, H_*W_, 2]
            reference_points_list.append(ref)

        # [bs, sum(HiWi), 2]
        reference_points = torch.cat(reference_points_list, dim=1)
        # Expand to all levels: [bs, sum(HiWi), n_levels, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Compute valid (non-padded) ratio for H and W dimensions."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], dim=1)
        valid_W = torch.sum(~mask[:, 0, :], dim=1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], dim=-1)

    def forward(
        self,
        srcs: List[Tensor],
        masks: List[Tensor],
        pos_embeds: List[Tensor],
    ):
        """
        Args:
            srcs: List of feature maps [bs, c, hi, wi] per level.
            masks: List of padding masks [bs, hi, wi] per level.
            pos_embeds: List of positional embeddings [bs, c, hi, wi] per level.

        Returns:
            memory: [bs, sum(HiWi), d_model] enhanced flat feature sequence.
            spatial_shapes: [n_levels, 2] (H, W) per level.
            level_start_index: [n_levels] start index in the flat sequence.
        """
        # ------------------------------------------------------------------
        # Handle missing masks (e.g., images divisible by 32)
        # ------------------------------------------------------------------
        enable_mask = 0
        if masks is not None:
            for src in srcs:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [
                torch.zeros(
                    (src.size(0), src.size(2), src.size(3)),
                    device=src.device,
                    dtype=torch.bool,
                )
                for src in srcs
            ]

        # ------------------------------------------------------------------
        # Flatten and concatenate all feature levels
        # ------------------------------------------------------------------
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))

            # [bs, h*w, c]
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)

            # Add level embedding to distinguish scales
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, dim=1)           # [bs, sum(HiWi), c]
        mask_flatten = torch.cat(mask_flatten, dim=1)         # [bs, sum(HiWi)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], dim=1
        )  # [bs, n_levels, 2]

        # ------------------------------------------------------------------
        # Compute encoder reference points (grid-based, all levels)
        # ------------------------------------------------------------------
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src_flatten.device
        )  # [bs, sum(HiWi), n_levels, 2]

        # ------------------------------------------------------------------
        # Forward through ViT5 encoder layers
        # ------------------------------------------------------------------
        output = src_flatten


        print(f"[DEBUG] src_flatten shape: {src_flatten.shape}")
        print(f"[DEBUG] spatial_shapes: {spatial_shapes}")
        print(f"[DEBUG] level_start_index: {level_start_index}")
        print(f"[DEBUG] num_value from src: {src_flatten.shape[1]}")
        print(f"[DEBUG] expected from spatial_shapes: {spatial_shapes.prod(1).sum()}")
        print(f"[DEBUG] len(srcs): {len(srcs)}")

        for layer in self.layers:
            output = layer(
                query=output,
                query_pos=lvl_pos_embed_flatten,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=mask_flatten,
            )

        return output, spatial_shapes, level_start_index


# =============================================================================
# ViT5MaskDINOEncoder
# Subclass of MaskDINOEncoder; swaps out only the Transformer encoder.
# All FPN/lateral/output convs are inherited unchanged.
# =============================================================================

class ViT5MaskDINOEncoder(MaskDINOEncoder):
    """MaskDINO Pixel Decoder with ViT-5 Transformer encoder (Stage 1 + 2).

    Inherits all convolutional components (FPN, lateral convs, output convs,
    mask_features projection) from MaskDINOEncoder unchanged.

    Only the internal Transformer encoder (MSDeformAttnTransformerEncoderOnly)
    is replaced with ViT5MSDeformAttnTransformerEncoderOnly.

    Args:
        init_values (float): LayerScale init value. Default: 1e-4.
        swiglu_ffn_ratio (float): SwiGLU hidden dim ratio vs dim_feedforward.
            Default: 1.0 (same hidden size, ~50% more params vs standard FFN).
            Use 2/3 to match standard FFN param count.
        All other args: same as MaskDINOEncoder.
    """

    def __init__(
        self,
        # ViT-5 specific
        init_values: float = 1e-4,
        swiglu_ffn_ratio: float = 1.0,
        # MaskDINOEncoder args (forwarded as-is)
        in_channels=None,
        in_strides=None,
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_enc_layers: int = 6,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm_cfg=dict(type='GN', num_groups=32),
        transformer_in_features: List[str] = None,
        common_stride: int = 4,
        num_feature_levels: int = 3,
        total_num_feature_levels: int = 4,
        feature_order: str = 'low2high',
    ):
        # Provide defaults matching MaskDINOEncoder expectations
        in_channels = in_channels or [256, 512, 1024, 2048]
        in_strides = in_strides or [4, 8, 16, 32]
        transformer_in_features = transformer_in_features or ['res3', 'res4', 'res5']

        # Call base __init__ to build FPN + original transformer
        super().__init__(
            in_channels=in_channels,
            in_strides=in_strides,
            transformer_dropout=transformer_dropout,
            transformer_nheads=transformer_nheads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_enc_layers=transformer_enc_layers,
            conv_dim=conv_dim,
            mask_dim=mask_dim,
            norm_cfg=norm_cfg,
            transformer_in_features=transformer_in_features,
            common_stride=common_stride,
            num_feature_levels=num_feature_levels,
            total_num_feature_levels=total_num_feature_levels,
            feature_order=feature_order,
        )

        # ------------------------------------------------------------------
        # Replace the Transformer encoder with ViT-5 version
        # ------------------------------------------------------------------
        self.transformer = ViT5MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            nhead=transformer_nheads,
            num_encoder_layers=transformer_enc_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            num_feature_levels=total_num_feature_levels,
            enc_n_points=4,
            init_values=init_values,
            swiglu_ffn_ratio=swiglu_ffn_ratio,
            # act_cfg is intentionally not forwarded: SwiGLU uses fixed SiLU.
            # The parameter is accepted by ViT5MSDeformAttnTransformerEncoderOnly
            # for API compatibility but has no effect.
        )

        print(
            f"[ViT5MaskDINOEncoder] Upgraded transformer encoder:\n"
            f"  Norm       : LayerNorm → RMSNorm\n"
            f"  Norm order : Post-Norm → Pre-Norm\n"
            f"  FFN        : ReLU-FFN  → SwiGLU "
            f"(hidden={int(transformer_dim_feedforward * swiglu_ffn_ratio)})\n"
            f"  Residual   : bare      → LayerScale (init={init_values:.0e})\n"
            f"  Layers     : {transformer_enc_layers}  |  "
            f"Heads: {transformer_nheads}  |  d_model: {conv_dim}"
        )
