import copy
import os
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_

try:
    from mmdet.models.builder import BACKBONES as MMDET2_BACKBONES
except Exception:
    MMDET2_BACKBONES = None

try:
    from mmdet.registry import MODELS as MMDET3_MODELS
except Exception:
    MMDET3_MODELS = None

try:
    from mmdet.utils import get_root_logger as mmdet2_get_root_logger
except Exception:
    mmdet2_get_root_logger = None

try:
    from mmengine.logging import MMLogger
except Exception:
    MMLogger = None

try:
    from mmcv.runner import _load_checkpoint as mmcv_load_checkpoint
except Exception:
    mmcv_load_checkpoint = None

try:
    from mmengine.runner.checkpoint import _load_checkpoint as mmengine_load_checkpoint
except Exception:
    mmengine_load_checkpoint = None

from model.tinyvim import Embedding, LocalBlock, TinyViM_depth, TinyViM_width, stem
from model.tvimblock import Conv2d_BN, FFN, TViMBlock


def get_root_logger():
    if mmdet2_get_root_logger is not None:
        return mmdet2_get_root_logger()
    if MMLogger is not None:
        return MMLogger.get_current_instance()
    raise RuntimeError("No compatible logger backend found for HybridMamba.")


def load_checkpoint_compat(path, logger):
    if mmcv_load_checkpoint is not None:
        return mmcv_load_checkpoint(path, logger=logger, map_location='cpu')
    if mmengine_load_checkpoint is not None:
        return torch.load(path, map_location='cpu', weights_only=False)
    raise RuntimeError("No compatible checkpoint loader found for HybridMamba.")


def register_backbone(factory):
    if MMDET2_BACKBONES is not None:
        MMDET2_BACKBONES.register_module()(factory)
    if MMDET3_MODELS is not None:
        MMDET3_MODELS.register_module()(factory)
    return factory


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


class FrequencySplitter(nn.Module):
    def __init__(self, dim: int, mode: str = "avg3") -> None:
        super().__init__()
        kernel_size = 5 if mode == "avg5" else 3
        self.mode = mode
        self.low_pass = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.detail_norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "identity":
            return x, torch.zeros_like(x)
        low = self.low_pass(x)
        high = self.detail_norm(x - low)
        return low, high


class EdgeAwareGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.edge = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.edge.weight.requires_grad = False
        self.gate = nn.Sequential(
            Conv2d_BN(dim * 2, dim, 1),
            nn.Sigmoid(),
        )
        self._init_edge_kernel()

    def _init_edge_kernel(self) -> None:
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        weight = kernel.view(1, 1, 3, 3).repeat(self.edge.weight.shape[0], 1, 1, 1)
        with torch.no_grad():
            self.edge.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = torch.abs(self.edge(x))
        return self.gate(torch.cat([x, edge], dim=1))


class DetailBranch(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw3 = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)
        self.dw5 = Conv2d_BN(dim, dim, 5, 1, 2, groups=dim)
        self.mix = Conv2d_BN(dim * 2, dim, 1)
        self.gate = EdgeAwareGate(dim)
        self.out = Conv2d_BN(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = torch.cat([self.dw3(x), self.dw5(x)], dim=1)
        detail = self.mix(detail)
        return self.out(self.gate(detail) * detail)


class HybridMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        index: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        freq_split: str = "avg3",
        detail_branch: str = "dw_gate",
        fusion_alpha: float = 0.75,
        enable_detail: bool = True,
        ssm_d_state: int = 8,
        ssm_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.enable_detail = enable_detail and detail_branch != "none"
        self.fusion_alpha = fusion_alpha if self.enable_detail else 0.0
        self.splitter = FrequencySplitter(dim=dim, mode=freq_split)
        self.mamba = TViMBlock(
            hidden_dim=dim,
            drop_path=0.0,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_conv_bias=False,
            mlp_ratio=0.0,
            index=index,
        )
        self.detail = DetailBranch(dim) if self.enable_detail else nn.Identity()
        self.mlp = FFN(dim, int(mlp_ratio * dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, high = self.splitter(x)
        low_delta = self.mamba(low) - low
        detail_delta = self.detail(high) if self.enable_detail else 0.0
        x = x + self.drop_path(low_delta + self.fusion_alpha * detail_delta)
        x = x + self.drop_path(self.mlp(x))
        return x


def HybridStage(
    dim: int,
    index: int,
    layers: Sequence[int],
    mlp_ratio: float = 4.0,
    ssm_d_state: int = 8,
    ssm_ratio: float = 1.0,
    ssm_num: int = 1,
    detail_stages: Iterable[int] = (0, 1, 2),
    freq_split: str = "avg3",
    detail_branch: str = "dw_gate",
    fusion_alpha: float = 0.75,
) -> nn.Sequential:
    blocks = []
    detail_stage_set = set(detail_stages)
    for block_idx in range(layers[index]):
        use_mamba = layers[index] - block_idx <= ssm_num or (index == 2 and block_idx == layers[index] // 2)
        if use_mamba:
            blocks.append(
                HybridMambaBlock(
                    dim=dim,
                    index=index,
                    mlp_ratio=mlp_ratio,
                    freq_split=freq_split,
                    detail_branch=detail_branch,
                    fusion_alpha=fusion_alpha,
                    enable_detail=index in detail_stage_set,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                )
            )
        else:
            blocks.append(LocalBlock(dim=dim, hidden_dim=int(mlp_ratio * dim)))
    return nn.Sequential(*blocks)


class HybridMambaDet(nn.Module):
    def __init__(
        self,
        layers,
        embed_dims=None,
        mlp_ratios=4,
        downsamples=None,
        num_classes=1000,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        fork_feat=False,
        init_cfg=None,
        pretrained=None,
        ssm_num=1,
        distillation=True,
        freq_split="avg3",
        detail_branch="dw_gate",
        detail_stages=(0, 1, 2),
        fusion_alpha=0.75,
        **kwargs,
    ):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = HybridStage(
                embed_dims[i],
                i,
                layers,
                mlp_ratio=mlp_ratios,
                ssm_num=ssm_num,
                detail_stages=detail_stages,
                freq_split=freq_split,
                detail_branch=detail_branch,
                fusion_alpha=fusion_alpha,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    Embedding(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )

        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                layer = nn.Identity() if i_emb == 0 and os.environ.get("FORK_LAST3", None) else nn.BatchNorm2d(embed_dims[i_emb])
                self.add_module(f"norm{i_layer}", layer)
        else:
            self.norm = nn.BatchNorm2d(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warning("No pre-trained weights for %s, training starts from scratch", self.__class__.__name__)
            return
        assert "checkpoint" in self.init_cfg, "Only init_cfg checkpoint loading is supported."
        ckpt_path = self.init_cfg["checkpoint"] if self.init_cfg is not None else pretrained
        ckpt = load_checkpoint_compat(ckpt_path, logger=logger)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        self.load_state_dict(state_dict, strict=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                outs.append(norm_layer(x))
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        return cls_out


@register_backbone
def HybridMamba_Base_B(pretrained=False, **kwargs):
    model = HybridMambaDet(
        layers=TinyViM_depth["B"],
        embed_dims=TinyViM_width["B"],
        downsamples=[True, True, True, True],
        freq_split=kwargs.pop("freq_split", "avg3"),
        detail_branch="none",
        fusion_alpha=0.0,
        **kwargs,
    )
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_backbone
def HybridMambaDet_B(pretrained=False, **kwargs):
    model = HybridMambaDet(
        layers=TinyViM_depth["B"],
        embed_dims=TinyViM_width["B"],
        downsamples=[True, True, True, True],
        **kwargs,
    )
    model.default_cfg = _cfg(crop_pct=0.9)
    return model
