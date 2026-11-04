"""
models - Backbone transfer learning + extension Squeeze-and-Excitation (T4, E1).

Satu kelas Detector untuk keenam konfigurasi ablation:
    backbone.forward_features -> [SEBlock jika use_se] -> backbone.forward_head (2 logit)

Lokasi SE: satu blok pada peta fitur terakhir backbone, sebelum global pooling,
reduction ratio dari configs/train.yaml (default 16). Lokasi ini sama untuk
ketiga arsitektur sehingga ablation membandingkan efek extension yang sama
(lihat docs/decisions.md D-02; EfficientNetV2-B0 dan MobileNetV3 sudah
memiliki SE internal di blok MBConv-nya dan tidak diubah).
"""

import timm
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu dkk., CVPR 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class Detector(nn.Module):
    def __init__(self, arch: str, timm_name: str, use_se: bool, se_reduction: int = 16,
                 pretrained: bool = True, input_size: int = 224, drop_rate: float = 0.3):
        super().__init__()
        self.arch, self.timm_name, self.use_se = arch, timm_name, use_se
        self.backbone = timm.create_model(timm_name, pretrained=pretrained,
                                          num_classes=2, drop_rate=drop_rate)
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()
            channels = self.backbone.forward_features(
                torch.zeros(1, 3, input_size, input_size)).shape[1]
            self.backbone.train(was_training)
        self.se = SEBlock(channels, se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        return self.backbone.forward_head(self.se(self.backbone.forward_features(x)))

    def param_groups(self, lr_backbone: float, lr_head: float):
        """LR rendah untuk backbone pretrained, LR tinggi untuk head klasifikasi + SE."""
        head_ids = {id(p) for p in self.backbone.get_classifier().parameters()}
        head_ids |= {id(p) for p in self.se.parameters()}
        backbone = [p for p in self.parameters() if id(p) not in head_ids]
        head = [p for p in self.parameters() if id(p) in head_ids]
        return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]

    def target_layer(self):
        """Layer untuk Grad-CAM: blok SE bila ada, selain itu stage terakhir backbone."""
        if self.use_se:
            return self.se
        for attr in ("layer4", "blocks", "stages"):
            if hasattr(self.backbone, attr):
                return getattr(self.backbone, attr)[-1]
        raise ValueError(f"Target layer Grad-CAM tidak diketahui untuk {self.timm_name}")


def config_name(arch: str, use_se: bool) -> str:
    return f"{arch}_{'se' if use_se else 'base'}"


def build_model(train_cfg: dict, arch: str, use_se: bool, pretrained=None) -> Detector:
    return Detector(
        arch=arch,
        timm_name=train_cfg["architectures"][arch],
        use_se=use_se,
        se_reduction=train_cfg["se"]["reduction"],
        pretrained=train_cfg["pretrained"] if pretrained is None else pretrained,
        input_size=train_cfg["input_size"],
    )


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
