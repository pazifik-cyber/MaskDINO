# Copyright (c) 2026. ViT-5 Transformer Utilities for MaskDINO
# Ported from ViT-5 (https://arxiv.org/abs/2602.08071) into the MaskDINO framework.
#
# This module provides three drop-in replacements for standard PyTorch/mmdet components:
#   - RMSNorm       : replaces nn.LayerNorm  (faster, no mean centering, per LLaMA practice)
#   - SwiGLU        : replaces ReLU-FFN      (gated activation, stronger expressivity)
#   - LayerScale    : replaces bare residual  (trainable per-channel scale, init=1e-4)
#
# Usage:
#   from vit5_transformer_utils import RMSNorm, SwiGLU, LayerScale

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. RMSNorm
#    Equivalent to LayerNorm without the mean-centering step.
#    y = x / RMS(x) * weight
#    where RMS(x) = sqrt(mean(x^2) + eps)
#
#    Source: ViT-5 models_vit5.py → class RMSNorm
#    Advantage over LayerNorm: ~15% faster, trains equally stable on vision tasks.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean shift).

    Args:
        hidden_size (int): Feature dimension to normalize.
        eps (float): Numerical stability epsilon. Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        # Upcast to float32 for numerics, same as ViT-5
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"


# ---------------------------------------------------------------------------
# 2. SwiGLU FFN
#    Replaces the standard: Linear → ReLU → Dropout → Linear
#    With the gated variant:  (W1·x) ⊙ SiLU(W2·x) → W3
#
#    Source: ViT-5 models_vit5.py → class SwiGLU
#    Param count note: With hidden_features = int(ffn_ratio * d_model * 2/3),
#    the total params match a standard FFN. For a drop-in replacement that
#    keeps the same hidden_features arg, param count increases by ~50%
#    (three matrices instead of two). Adjust hidden_features * 2/3 if needed.
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    Gated activation: output = W3( SiLU(W1(x)) * W2(x) )

    Args:
        in_features  (int): Input feature dimension.
        hidden_features (int): Hidden (gate) feature dimension.
            Tip: use int(d_model * ffn_ratio * 2 / 3) to match standard FFN param count.
        out_features (int): Output feature dimension. Defaults to in_features.
        dropout (float): Dropout probability applied after gating. Default: 0.0.
        bias (bool): Whether to use bias in linear layers. Default: True.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Gate branch: SiLU(W1 · x)
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        # Value branch: W2 · x
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        # Output projection: W3
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Gated activation: element-wise product of gate and value
        gate = self.act(self.w1(x))
        value = self.w2(x)
        x = self.drop(gate * value)
        return self.w3(x)

    def extra_repr(self) -> str:
        return (
            f"in={self.w1.in_features}, "
            f"hidden={self.w1.out_features}, "
            f"out={self.w3.out_features}"
        )


# ---------------------------------------------------------------------------
# 3. LayerScale
#    A learnable per-channel scalar that modulates the residual branch output.
#    residual_output = gamma ⊙ sublayer_output   (gamma init = init_values)
#
#    Source: ViT-5 Block → self.gamma_1 / self.gamma_2
#    Why: With init_values=1e-4, the residual branch contributes almost nothing
#    at the start of training, letting the main path stabilize. Especially
#    important for deep models (>12 layers) and fp16/bf16 mixed precision.
# ---------------------------------------------------------------------------
class LayerScale(nn.Module):
    """Per-channel learnable scalar for residual branch scaling.

    Args:
        dim (int): Feature dimension (one scalar per channel).
        init_values (float): Initial scale value. Default: 1e-4 (ViT-5 default).
    """

    def __init__(self, dim: int, init_values: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.gamma * x

    def extra_repr(self) -> str:
        return f"dim={self.gamma.shape[0]}, init={self.gamma[0].item():.1e}"


# ---------------------------------------------------------------------------
# Convenience factory: build a norm layer by name
# ---------------------------------------------------------------------------
def build_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """Factory for norm layers.

    Args:
        norm_type: 'rms' or 'ln' (LayerNorm).
        dim: Feature dimension.
        eps: Epsilon for numerical stability.
    """
    if norm_type == 'rms':
        return RMSNorm(dim, eps=eps)
    elif norm_type == 'ln':
        return nn.LayerNorm(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Choose 'rms' or 'ln'.")