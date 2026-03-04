# Copyright (c) OpenMMLab. All rights reserved.
# ViT-5 Transformer Optimization for MaskDINO Decoder
# Stage 1+2: RMSNorm + Pre-Norm + SwiGLU FFN + LayerScale
#
# Changes vs original maskdino_decoder_layers.py:
#   [DeformableTransformerDecoderLayer → ViT5DeformableTransformerDecoderLayer]
#     1. Post-Norm → Pre-Norm  (norm applied before sublayer, not after)
#     2. nn.LayerNorm → RMSNorm  (all three norms: SA, CA, FFN)
#     3. ReLU FFN → SwiGLU FFN  (gated activation, stronger expressivity)
#     4. Bare residual → LayerScale residual  (gamma init=1e-4 per ViT-5)
#
#   [MaskDINODecoder → ViT5MaskDINODecoder]
#     5. enc_output_norm: nn.LayerNorm → RMSNorm  (two-stage query selection)
#     6. decoder_norm   : nn.LayerNorm → RMSNorm  (final output norm)
#     7. decoder_layer  : replaced with ViT5DeformableTransformerDecoderLayer
#
# Everything else (DN, Two-Stage, TransformerDecoder loop, mask head) is unchanged.

import copy
from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import caffe2_xavier_init
from torch import Tensor
from torch.cuda.amp import autocast
from torch.nn import functional as F

from mmdet.models.layers import MLP, coordinate_to_encoding, inverse_sigmoid
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

# ViT-5 primitives
from .vit5_transformer_utils import RMSNorm, SwiGLU, LayerScale

# Re-export helpers that MaskDINODecoder needs (unchanged from original)
from .maskdino_decoder_layers import (
    setup_seed,
    masks_to_boxes,
    MaskDINODecoder,          # base class – we subclass, not rewrite
    TransformerDecoder,       # loop logic unchanged
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
)


# =============================================================================
# ViT5DeformableTransformerDecoderLayer
# Drop-in replacement for DeformableTransformerDecoderLayer
# =============================================================================

class ViT5DeformableTransformerDecoderLayer(nn.Module):
    """Decoder layer with ViT-5-style Transformer improvements.

    Architecture per sub-layer (Pre-Norm + LayerScale):
        x = x + LayerScale_sa( SA( RMSNorm(x) ) )
        x = x + LayerScale_ca( CA( RMSNorm(x) ) )
        x = x + LayerScale_ffn( SwiGLU( RMSNorm(x) ) )

    Args:
        d_model (int): Hidden dimension. Default: 256.
        d_ffn (int): FFN hidden dimension (gate branch width).
            Tip: pass int(d_model * 4 * 2/3) ≈ 682 to match standard FFN
            parameter count, or keep 1024 for a larger capacity upgrade.
            Default: 1024.
        dropout (float): Dropout in attention and FFN. Default: 0.1.
        n_levels (int): Number of feature levels for deformable cross-attn. Default: 4.
        n_heads (int): Number of attention heads. Default: 8.
        n_points (int): Deformable sampling points per head per level. Default: 4.
        init_values (float): LayerScale initial value. Default: 1e-4 (ViT-5).
        use_deformable_box_attn (bool): Not implemented, must be False. Default: False.
        key_aware_type (str | None): Key-aware cross-attention type. Default: None.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        init_values: float = 1e-4,
        use_deformable_box_attn: bool = False,
        key_aware_type=None,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Cross-Attention (deformable, unchanged from original)
        # ------------------------------------------------------------------
        if use_deformable_box_attn:
            raise NotImplementedError(
                "use_deformable_box_attn is not supported in ViT5 decoder layer."
            )
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            dropout=dropout,
        )
        # Stage 1: RMSNorm (replaces LayerNorm for cross-attn)
        self.norm_ca = RMSNorm(d_model)
        # Stage 2: LayerScale for cross-attn residual
        self.ls_ca = LayerScale(d_model, init_values=init_values)

        # ------------------------------------------------------------------
        # Self-Attention (standard multi-head, unchanged structure)
        # ------------------------------------------------------------------
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_sa = nn.Dropout(dropout)
        # Stage 1: RMSNorm (replaces LayerNorm for self-attn)
        self.norm_sa = RMSNorm(d_model)
        # Stage 2: LayerScale for self-attn residual
        self.ls_sa = LayerScale(d_model, init_values=init_values)

        # ------------------------------------------------------------------
        # FFN: SwiGLU replaces Linear→ReLU→Linear
        # ------------------------------------------------------------------
        # Stage 2: SwiGLU FFN
        self.ffn = SwiGLU(
            in_features=d_model,
            hidden_features=d_ffn,
            out_features=d_model,
            dropout=dropout,
        )
        # Stage 1: RMSNorm (replaces LayerNorm for FFN)
        self.norm_ffn = RMSNorm(d_model)
        # Stage 2: LayerScale for FFN residual
        self.ls_ffn = LayerScale(d_model, init_values=init_values)

        # Key-aware cross-attention (optional, kept for compatibility)
        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        if key_aware_type == 'proj_mean':
            self.key_aware_proj = nn.Linear(d_model, d_model)

    def rm_self_attn_modules(self):
        """Remove self-attention modules (used by MaskDINO for certain configs)."""
        self.self_attn = None
        self.dropout_sa = None
        self.norm_sa = None
        self.ls_sa = None

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    @autocast(enabled=False)
    def forward(
        self,
        # query
        tgt: Optional[Tensor],                  # [nq, bs, d_model]
        tgt_query_pos: Optional[Tensor] = None, # [nq, bs, d_model]  MLP(Sine(ref))
        tgt_query_sine_embed: Optional[Tensor] = None,  # unused in deformable
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None, # [nq, bs, n_levels, 4]
        # memory (encoder output)
        memory: Optional[Tensor] = None,                # [hw, bs, d_model]
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,
        memory_spatial_shapes: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        # masks
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pre-Norm + LayerScale forward pass.

        Stage 1 change: norm is applied BEFORE each sublayer (Pre-Norm).
        Stage 2 change: LayerScale modulates each residual branch.
        """

        # ==================================================================
        # 1. Self-Attention  (Pre-Norm)
        # ==================================================================
        if self.self_attn is not None:
            # Pre-Norm: normalize tgt before computing q, k, v
            tgt_normed = self.norm_sa(tgt)
            q = k = self.with_pos_embed(tgt_normed, tgt_query_pos)
            sa_out, _ = self.self_attn(
                query=q,
                key=k,
                value=tgt_normed,
                attn_mask=self_attn_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
            # LayerScale residual (Stage 2)
            tgt = tgt + self.ls_sa(self.dropout_sa(sa_out))

        # ==================================================================
        # 2. Cross-Attention  (Pre-Norm, deformable)
        # ==================================================================
        # Optional key-aware conditioning (unchanged logic)
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError(
                    f"Unknown key_aware_type: {self.key_aware_type}"
                )

        # Pre-Norm: normalize tgt before cross-attention
        tgt_normed = self.norm_ca(tgt)

        # MultiScaleDeformableAttention:
        #   - query=tgt_normed, query_pos=tgt_query_pos  follows the mmdet convention
        #     (mmcv internally computes tgt_normed + tgt_query_pos for sampling offsets,
        #     which is the correct Pre-Norm behavior: norm features first, then add pos).
        #   - identity=zeros_like(tgt_normed) disables the internal residual addition
        #     so that ca_out is the pure attention output; we apply LayerScale ourselves.
        ca_out = self.cross_attn(
            query=tgt_normed,
            query_pos=tgt_query_pos,
            value=memory,
            identity=torch.zeros_like(tgt_normed),   # disable internal residual
            key_padding_mask=memory_key_padding_mask,
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
        )
        # LayerScale residual (Stage 2)
        tgt = tgt + self.ls_ca(ca_out)

        # ==================================================================
        # 3. FFN  (Pre-Norm, SwiGLU)
        # ==================================================================
        # Pre-Norm: normalize before FFN
        tgt_normed = self.norm_ffn(tgt)
        # SwiGLU forward (Stage 2 FFN replacement)
        ffn_out = self.ffn(tgt_normed)
        # LayerScale residual (Stage 2)
        tgt = tgt + self.ls_ffn(ffn_out)

        return tgt


# =============================================================================
# ViT5MaskDINODecoder
# Subclass of MaskDINODecoder, overrides only the transformer components.
# =============================================================================

class ViT5MaskDINODecoder(MaskDINODecoder):
    """MaskDINO Decoder with ViT-5 Transformer improvements (Stage 1 + 2).

    Overrides three components compared to the base MaskDINODecoder:
        1. enc_output_norm  : nn.LayerNorm → RMSNorm
        2. decoder_norm     : nn.LayerNorm → RMSNorm  (final output norm)
        3. decoder_layer    : DeformableTransformerDecoderLayer
                            → ViT5DeformableTransformerDecoderLayer

    All DN, Two-Stage, query initialization, and prediction head logic
    is inherited unchanged from MaskDINODecoder.

    Args:
        init_values (float): LayerScale init value for decoder layers. Default: 1e-4.
        swiglu_ffn_ratio (float): Multiplier on dim_feedforward for SwiGLU hidden
            dimension. Use 2/3 to match standard FFN param count, 1.0 to keep
            dim_feedforward as-is (larger capacity). Default: 1.0.
        All other args: same as MaskDINODecoder.
    """

    def __init__(
        self,
        # ViT-5 specific kwargs
        init_values: float = 1e-4,
        swiglu_ffn_ratio: float = 1.0,
        # All original MaskDINODecoder kwargs follow (no defaults – must be explicit)
        in_channels: int = 256,
        num_classes: int = 133,
        hidden_dim: int = 256,
        num_queries: int = 300,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dec_layers: int = 9,
        mask_dim: int = 256,
        enforce_input_project: bool = False,
        two_stage: bool = True,
        dn: str = 'standard',
        noise_scale: float = 0.4,
        dn_num: int = 100,
        initialize_box_type: bool = False,
        initial_pred: bool = True,
        learn_tgt: bool = False,
        total_num_feature_levels: int = 4,
        dropout: float = 0.0,
        activation: str = 'relu',  # kept for API compat, ignored (SwiGLU used)
        nhead: int = 8,
        dec_n_points: int = 4,
        mask_classification: bool = True,
        return_intermediate_dec: bool = True,
        query_dim: int = 4,
        dec_layer_share: bool = False,
        semantic_ce_loss: bool = False,
    ):
        # Call base __init__ first – this builds the original decoder
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            two_stage=two_stage,
            dn=dn,
            noise_scale=noise_scale,
            dn_num=dn_num,
            initialize_box_type=initialize_box_type,
            initial_pred=initial_pred,
            learn_tgt=learn_tgt,
            total_num_feature_levels=total_num_feature_levels,
            dropout=dropout,
            activation=activation,
            nhead=nhead,
            dec_n_points=dec_n_points,
            mask_classification=mask_classification,
            return_intermediate_dec=return_intermediate_dec,
            query_dim=query_dim,
            dec_layer_share=dec_layer_share,
            semantic_ce_loss=semantic_ce_loss,
        )
        # ------------------------------------------------------------------
        # Stage 1: Replace LayerNorm → RMSNorm
        # ------------------------------------------------------------------
        # (a) Two-stage encoder output normalization
        if two_stage and hasattr(self, 'enc_output_norm'):
            self.enc_output_norm = RMSNorm(hidden_dim)

        # (b) Final decoder output norm (lives inside TransformerDecoder)
        self.decoder.norm = RMSNorm(hidden_dim)

        # ------------------------------------------------------------------
        # Stage 1+2: Replace decoder layers with ViT5 versions
        # ------------------------------------------------------------------
        # Compute SwiGLU hidden dim
        swiglu_hidden = int(dim_feedforward * swiglu_ffn_ratio)

        # Build one prototype layer, then clone for all dec_layers
        vit5_layer = ViT5DeformableTransformerDecoderLayer(
            d_model=hidden_dim,
            d_ffn=swiglu_hidden,
            dropout=dropout,
            n_levels=total_num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
            init_values=init_values,
        )
        self.decoder.layers = _get_clones(
            vit5_layer,
            dec_layers,              # positional: _get_clones(module, N, layer_share)
            layer_share=dec_layer_share,
        )

        # Log the upgrade for visibility
        print(
            f"[ViT5MaskDINODecoder] Upgraded decoder:\n"
            f"  Norm       : LayerNorm → RMSNorm\n"
            f"  Norm order : Post-Norm → Pre-Norm\n"
            f"  FFN        : ReLU-FFN  → SwiGLU (hidden={swiglu_hidden})\n"
            f"  Residual   : bare      → LayerScale (init={init_values:.0e})\n"
            f"  Layers     : {dec_layers}  |  Heads: {nhead}  |  d_model: {hidden_dim}"
        )