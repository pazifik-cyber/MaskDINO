# https://github.com/NVlabs/RADIO
# Modified for C-RADIOv4 Model Family - HuggingFace version

import math
from typing import List, Optional, Sequence, OrderedDict
import warnings
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# einops uses set.symmetric_difference internally, which TorchDynamo cannot
# trace.  This one call registers einops functions as "safe" so torch.compile
# includes them in the compiled graph instead of breaking on them.
# Requires einops >= 0.6.1.  torch >= 2.4 does this automatically.
try:
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()
except ImportError:
    pass  # einops < 0.6.1 or not installed; torch >= 2.4 handles this automatically

from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import BaseModule
from mmdet.registry import MODELS

try:
    from transformers import AutoModel, CLIPImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..utils.FADE_L2H import FADE

_logger = logging.getLogger(__name__)


@MODELS.register_module()
class RADIO(BaseModule):
    """RADIO backbone for C-RADIOv4 Model Family (HuggingFace version).

    Uses HuggingFace transformers to load models from nvidia/C-RADIOv4-*.
    Supports model versions: 'C-RADIOv4-H', 'C-RADIOv4-SO400M', etc.

    Args:
        model_version (str): Model version to load, e.g., 'nvidia/C-RADIOv4-H'.
        init_cfg (dict, optional): Initialization config. Default: None.
        align_resolution (bool): Whether to align input resolution to model's
            supported resolutions. Default: True.
        patch_size (int): Patch size for spatial feature reconstruction. Default: 16.
    """

    # HuggingFace model repositories
    HF_REPOS = {
        'c-radio_v4-h': 'nvidia/C-RADIOv4-H',
        'c-radio_v4-so400m': 'nvidia/C-RADIOv4-SO400M',
        'C-RADIOv4-H': 'nvidia/C-RADIOv4-H',
        'C-RADIOv4-SO400M': 'nvidia/C-RADIOv4-SO400M',
    }

    # Default intermediate layer indices for different model versions
    DEFAULT_INDICES = {
        'c-radio_v4-h': [7, 15, 23, 31],  # ViT-H/16
        'c-radio_v4-so400m': [5, 11, 17, 23],  # ViT-SO400M/16
        'C-RADIOv4-H': [7, 15, 23, 31],
        'C-RADIOv4-SO400M': [5, 11, 17, 23],
    }

    def __init__(
        self,
        model_version: str,
        init_cfg=None,
        align_resolution: bool = True,
        patch_size: int = 16,
    ):
        super().__init__(init_cfg=init_cfg)

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required for HuggingFace RADIO. "
                "Please install it with: pip install transformers"
            )

        self.model_version = model_version
        self.align_resolution = align_resolution
        self.patch_size = patch_size

        # Map to HuggingFace repo
        if model_version in self.HF_REPOS:
            self.hf_repo = self.HF_REPOS[model_version]
        elif model_version.startswith('nvidia/'):
            self.hf_repo = model_version
        else:
            self.hf_repo = f'nvidia/C-RADIOv4-{model_version.upper()}'
            _logger.warning(f"Unknown model version {model_version}, trying {self.hf_repo}")

        # Load model via HuggingFace
        _logger.info(f"Loading RADIO model from HuggingFace: {self.hf_repo}")
        self.image_processor = CLIPImageProcessor.from_pretrained(self.hf_repo)
        self.base_model = AutoModel.from_pretrained(
            self.hf_repo,
            trust_remote_code=True
        )
        self.base_model.eval()
        self.base_model.cuda()

        # Debug: print model structure
        _logger.info(f"Base model type: {type(self.base_model)}")
        _logger.info(f"Base model attributes: {dir(self.base_model)}")
        if hasattr(self.base_model, 'config'):
            _logger.info(f"Model config: {self.base_model.config}")
            _logger.info(f"Config attributes: {dir(self.base_model.config)}")
        if hasattr(self.base_model, 'model'):
            _logger.info(f"Inner model type: {type(self.base_model.model)}")
            _logger.info(f"Inner model attributes: {dir(self.base_model.model)}")

        # Get embed_dim from model config or attributes
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'hidden_size'):
            self.embed_dim = self.base_model.config.hidden_size
            _logger.info(f"Got embed_dim from config.hidden_size: {self.embed_dim}")
        elif hasattr(self.base_model, 'embed_dim'):
            self.embed_dim = self.base_model.embed_dim
            _logger.info(f"Got embed_dim from base_model.embed_dim: {self.embed_dim}")
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_dim'):
            self.embed_dim = self.base_model.model.embed_dim
            _logger.info(f"Got embed_dim from base_model.model.embed_dim: {self.embed_dim}")
        else:
            # Try to infer from patch embedding if available
            if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'patch_embed'):
                self.embed_dim = self.base_model.model.patch_embed.proj.out_channels
                _logger.info(f"Got embed_dim from base_model.model.patch_embed: {self.embed_dim}")
            elif hasattr(self.base_model, 'patch_embed'):
                self.embed_dim = self.base_model.patch_embed.proj.out_channels
                _logger.info(f"Got embed_dim from base_model.patch_embed: {self.embed_dim}")
            else:
                # Default to common ViT-H dimension for C-RADIOv4-H
                self.embed_dim = 1280
                _logger.warning(f"Could not determine embed_dim from model, defaulting to {self.embed_dim}")

        _logger.info(f"RADIO model loaded. embed_dim: {self.embed_dim}")

        # Determine intermediate indices
        if model_version in self.DEFAULT_INDICES:
            self.intermediate_indices = self.DEFAULT_INDICES[model_version]
        else:
            # Default to c-radio_v4-h indices
            self.intermediate_indices = self.DEFAULT_INDICES['c-radio_v4-h']
            _logger.warning(f"Unknown model version {model_version}, using default indices")

        # Freeze base model parameters
        self._freeze_base_model()

    def _freeze_base_model(self):
        """Freeze all parameters in the base model."""
        frozen_count = 0
        for param in self.base_model.parameters():
            param.requires_grad = False
            frozen_count += 1
        _logger.info(f"Frozen {frozen_count} parameters in base_model")

    def train(self, mode=True):
        """Set training mode. RADIO backbone always stays in eval mode."""
        super().train(mode)
        if mode:
            warnings.warn("RADIO backbone is always in eval mode (frozen).")
        # Ensure base_model stays in eval mode
        self.base_model.eval()
        return self

    def init_weights(self):
        """No-op: weights are loaded during initialization."""
        pass

    def _preprocess_images(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for RADIO model.

        Args:
            x: Input tensor (B, C, H, W) with values in [0, 1].

        Returns:
            Preprocessed pixel values tensor.
        """
        # RADIO expects values in [0, 1], image_processor will normalize
        # CLIPImageProcessor expects PIL images or numpy arrays
        # But we can use the processor's normalization directly

        # The image_processor normalizes to mean=0, std=1 internally
        # Input x is already in [0, 1], so we just need to resize
        B, C, H, W = x.shape

        # Resize if needed (align to patch size)
        if self.align_resolution:
            # Round to nearest multiple of patch_size
            target_H = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
            target_W = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size

            if (H, W) != (target_H, target_W):
                x = F.interpolate(
                    x, size=(target_H, target_W),
                    mode='bilinear', align_corners=False
                )

        return x

    def _reconstruct_spatial_features(self, features: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reconstruct spatial features from flattened tokens.

        Args:
            features: Flattened features (B, T, D) where T = H//patch * W//patch
            H: Original height
            W: Original width

        Returns:
            Spatial features (B, D, H//patch, W//patch)
        """
        B, T, D = features.shape
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size

        assert T == H_patches * W_patches, f"Token count {T} doesn't match {H_patches}x{W_patches}"

        # Reshape to (B, D, H_patches, W_patches)
        features = features.transpose(1, 2).view(B, D, H_patches, W_patches)
        return features

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning intermediate features.

        Args:
            x: Input tensor (B, C, H, W). Values should be in [0, 1].
                RADIO will normalize internally to mean=0, std=1.

        Returns:
            List of 4 intermediate features in NCHW format.
        """
        B, C, H_in, W_in = x.shape

        # Preprocess images
        x = self._preprocess_images(x)
        _, _, H, W = x.shape

        # Forward through RADIO model
        # Returns: summary (B, C_summary) and features (B, T, D)
        summary, spatial_features = self.base_model(x)

        # Reconstruct spatial features
        spatial_features = self._reconstruct_spatial_features(spatial_features, H, W)
        # spatial_features shape: (B, embed_dim, H//16, W//16)

        # For C-RADIOv4, we need to extract intermediate features
        # Since the HF model returns final features, we need to modify to get intermediates
        # Option 1: Use forward_intermediates if available
        # Option 2: Extract from specific layers using hooks

        # Try to get intermediate features if the model supports it
        if hasattr(self.base_model, 'forward_intermediates'):
            outputs = self.base_model.forward_intermediates(
                x,
                indices=self.intermediate_indices,
                return_prefix_tokens=False,
                norm=True,
                stop_early=False,
                output_fmt='NCHW',
                intermediates_only=True,
                aggregation='sparse'
            )

            # Extract features from outputs
            if hasattr(outputs, 'features'):
                features = [o.features for o in outputs]
            else:
                features = list(outputs)
        else:
            # Fallback: create multi-scale features from single output
            # Using spatial_features as the base feature at H/16
            features = self._create_multiscale_features(spatial_features, H_in, W_in)

        # Ensure we have 4 features
        if len(features) < 4:
            _logger.error(f"Expected 4 features, got {len(features)}")
            raise ValueError(f"Insufficient features from RADIO: got {len(features)}, need 4")

        return features[:4]

    def _create_multiscale_features(self, base_feature: torch.Tensor, H: int, W: int) -> List[torch.Tensor]:
        """Create multi-scale features from base feature using interpolation.

        Args:
            base_feature: Base feature (B, C, H//16, W//16)
            H: Original height
            W: Original width

        Returns:
            List of 4 features at H/16 resolution (will be processed by adapter)
        """
        B, C, H_feat, W_feat = base_feature.shape

        # Create 4 identical features at H/16 resolution
        # The RADIOAdapter will handle upsampling/downsampling
        f1 = base_feature  # Will be upsampled to H/4
        f2 = base_feature  # Will be upsampled to H/8
        f3 = base_feature  # H/16
        f4 = base_feature  # Will be downsampled to H/32

        return [f1, f2, f3, f4]


class ConvFFN(nn.Module):
    """Convolutional FFN with depth-wise convolution."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """Depth-wise convolution for ConvFFN."""

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        # x shape: (B, N, C), N = H * W
        B, N, C = x.shape
        assert N == H * W, f"Input sequence length {N} does not match H*W {H*W}"
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Extractor(nn.Module):
    """Multi-scale deformable attention extractor."""

    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(
            embed_dims=dim,
            num_heads=num_heads,
            num_levels=n_levels,
            num_points=n_points,
            dropout=drop,
            batch_first=True,
            value_proj_ratio=deform_ratio
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=dim,
                hidden_features=int(dim * cffn_ratio),
                drop=drop
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            attn = self.attn(
                query=self.query_norm(query),
                value=self.feat_norm(feat),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


def get_reference_points(spatial_shapes, device):
    """Generate reference points for deformable attention."""
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


@MODELS.register_module()
class RADIOAdapter(RADIO):
    """RADIO Adapter with SPM and FADE fusion.

    Outputs 4-scale features:
    - f1: H/4 resolution, from SPM + FADE fusion
    - f2: H/8 resolution, from SPM + FADE fusion
    - f3: H/16 resolution, raw RADIO feature
    - f4: H/32 resolution, downsampled from RADIO feature

    Args:
        model_version (str): RADIO model version, e.g., 'C-RADIOv4-H'.
        spm_stem_channels (List[int]): Channels for SPM features [c1, c2].
            Default: [embed_dim // 4, embed_dim // 4].
        fade_up_kernel_size (int): Upsample kernel size for FADE. Default: 5.
        align_resolution (bool): Whether to align input resolution. Default: True.
    """

    def __init__(
        self,
        model_version: str,
        spm_stem_channels: Optional[List[int]] = None,
        fade_up_kernel_size: int = 5,
        align_resolution: bool = True,
        **kwargs
    ):
        super().__init__(
            model_version=model_version,
            align_resolution=align_resolution,
            **kwargs
        )

        # Default SPM channels
        if spm_stem_channels is None:
            spm_stem_channels = [self.embed_dim // 4] * 2
        self.spm_stem_channels = spm_stem_channels

        # SPM layers
        self.spm_stem = nn.Sequential(
            nn.Conv2d(3, spm_stem_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(spm_stem_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.spm_conv2 = nn.Sequential(
            nn.Conv2d(spm_stem_channels[0], spm_stem_channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(spm_stem_channels[1]),
            nn.ReLU(inplace=True)
        )

        # Projection layers for RADIO features
        self.proj_radio_r1 = nn.Conv2d(self.embed_dim, spm_stem_channels[0], kernel_size=1)
        self.proj_radio_r2 = nn.Conv2d(self.embed_dim, spm_stem_channels[1], kernel_size=1)

        # FADE fusion modules
        self.fade_c1 = FADE(
            in_channels_en=spm_stem_channels[0],
            in_channels_de=spm_stem_channels[0],
            scale=4,
            up_kernel_size=fade_up_kernel_size
        )
        self.fade_c2 = FADE(
            in_channels_en=spm_stem_channels[1],
            in_channels_de=spm_stem_channels[1],
            scale=2,
            up_kernel_size=fade_up_kernel_size
        )

        # Normalization layers
        self.norm1 = nn.BatchNorm2d(spm_stem_channels[0])
        self.norm2 = nn.BatchNorm2d(spm_stem_channels[1])
        self.norm4 = nn.BatchNorm2d(self.embed_dim)

        # Initialize adapter weights
        self._init_adapter_weights()

    def _init_weights(self, m):
        """Initialize weights for adapter modules."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_adapter_weights(self):
        """Initialize all adapter-specific modules."""
        _logger.info("Initializing RADIOAdapter-specific modules...")
        self.spm_stem.apply(self._init_weights)
        self.spm_conv2.apply(self._init_weights)
        self.proj_radio_r1.apply(self._init_weights)
        self.proj_radio_r2.apply(self._init_weights)
        self.norm1.apply(self._init_weights)
        self.norm2.apply(self._init_weights)
        self.norm4.apply(self._init_weights)
        _logger.info("Adapter modules initialized")

    def init_weights(self):
        """Initialize adapter weights. Base model is frozen."""
        self._init_adapter_weights()

    @torch.no_grad()
    def get_radio_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate features from frozen RADIO model."""
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W), values in [0, 1].

        Returns:
            List of 4 features [f1, f2, f3, f4] at H/4, H/8, H/16, H/32.
        """
        B, _, H_in, W_in = x.shape
        radio_features = self.get_radio_features(x)
        r1, r2, r3, r4 = radio_features
        
        print(f"=== RADIOAdapter Debug ===")
        print(f"Input: {x.shape}")
        print(f"r1: {r1.shape}, r2: {r2.shape}, r3: {r3.shape}, r4: {r4.shape}")
        
        spm_c1 = self.spm_stem(x)
        spm_c2 = self.spm_conv2(spm_c1)
        print(f"spm_c1: {spm_c1.shape}, spm_c2: {spm_c2.shape}")
        
        r1_proj = self.proj_radio_r1(r1)
        r2_proj = self.proj_radio_r2(r2)
        print(f"r1_proj: {r1_proj.shape}, r2_proj: {r2_proj.shape}")
        
        f1 = self.fade_c1(en=spm_c1, de=r1_proj)
        f2 = self.fade_c2(en=spm_c2, de=r2_proj)
        f3 = r3
        target_H32, target_W32 = H_in // 32, W_in // 32
        f4 = F.interpolate(r4, size=(target_H32, target_W32), mode='bilinear', align_corners=False)
        
        print(f"f1: {f1.shape}, f2: {f2.shape}, f3: {f3.shape}, f4: {f4.shape}")
        print(f"Expected — f1:(B,256,{H_in//4},{W_in//4}), f2:(B,512,{H_in//8},{W_in//8}), "
            f"f3:(B,1152,{H_in//16},{W_in//16}), f4:(B,1152,{H_in//32},{W_in//32})")
        # 7. Normalize
        f1 = self.norm1(f1)
        f2 = self.norm2(f2)
        f4 = self.norm4(f4)

        return [f1, f2, f3, f4]


def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    """Basic conv-bn-relu block."""
    if pad is None:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class Upsample(nn.Module):
    """Upsample module with CARAFE."""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        try:
            from mmcv.ops import CARAFEPack
            self.upsample = nn.Sequential(
                BasicConv(in_channels, out_channels, 1),
                CARAFEPack(out_channels, scale_factor=scale_factor, compressed_channels=64)
            )
        except ImportError:
            _logger.warning("CARAFE not available, using bilinear upsampling")
            self.upsample = nn.Sequential(
                BasicConv(in_channels, out_channels, 1),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
            )

    def forward(self, x):
        return self.upsample(x)


@MODELS.register_module()
class RADIOFPN(BaseModule):
    """RADIO backbone with FPN layers.

    Uses frozen RADIO model to extract features and applies FPN
    to generate multi-scale features [P2, P3, P4, P5].

    Args:
        model_version (str): RADIO model version.
        fpn_out_channels (List[int]): Output channels for each FPN level [P2, P3, P4, P5].
            Default: [embed_dim] * 4.
        align_resolution (bool): Whether to align input resolution. Default: True.
    """

    def __init__(
        self,
        model_version: str,
        fpn_out_channels: Optional[List[int]] = None,
        align_resolution: bool = True,
        init_cfg=None,
        **kwargs
    ):
        super().__init__(init_cfg=init_cfg)

        # Load RADIO backbone
        self.backbone = RADIO(
            model_version=model_version,
            align_resolution=align_resolution,
            **kwargs
        )

        self.embed_dim = self.backbone.embed_dim

        # FPN output channels
        if fpn_out_channels is None:
            self.fpn_out_channels = [self.embed_dim] * 4
        else:
            assert len(fpn_out_channels) == 4, "fpn_out_channels must have 4 values"
            self.fpn_out_channels = fpn_out_channels

        # FPN layers
        # P2: H/16 -> H/4 (upsample 4x)
        self.fpn1 = Upsample(self.embed_dim, self.fpn_out_channels[0], scale_factor=4)
        # P3: H/16 -> H/8 (upsample 2x)
        self.fpn2 = Upsample(self.embed_dim, self.fpn_out_channels[1], scale_factor=2)
        # P4: H/16 -> H/16
        if self.fpn_out_channels[2] == self.embed_dim:
            self.fpn3 = nn.Identity()
        else:
            self.fpn3 = nn.Conv2d(self.embed_dim, self.fpn_out_channels[2], kernel_size=1)
        # P5: H/16 -> H/32 (downsample 2x)
        if self.fpn_out_channels[3] == self.embed_dim:
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(self.embed_dim, self.fpn_out_channels[3], kernel_size=1)
            )

        self._init_fpn_weights()

    def _init_weights(self, m):
        """Initialize weights helper."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_fpn_weights(self):
        """Initialize FPN layer weights."""
        _logger.info("Initializing FPN weights...")
        for m in self.modules():
            self._init_weights(m)

    def init_weights(self):
        """Initialize FPN weights. Backbone is frozen."""
        self._init_fpn_weights()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W), values in [0, 1].

        Returns:
            List of FPN features [P2, P3, P4, P5].
        """
        # Get backbone features (4 intermediate features)
        features = self.backbone(x)

        # Apply FPN layers
        outputs = [
            self.fpn1(features[0]),  # P2
            self.fpn2(features[1]),  # P3
            self.fpn3(features[2]),  # P4
            self.fpn4(features[3]),  # P5
        ]

        return outputs

    def train(self, mode=True):
        """Set training mode. Backbone always stays frozen."""
        super().train(mode)
        # Ensure backbone stays in eval mode
        self.backbone.eval()
        return self