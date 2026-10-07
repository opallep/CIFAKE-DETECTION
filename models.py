"""
models.py - Definisi Arsitektur Model CNN untuk CIFAKE Detection
================================================================
Tiga arsitektur model:
1. LightweightCNN     - CNN custom 8 conv layer, from scratch
2. ResNet50Transfer   - Transfer learning ResNet-50 (ImageNet)
3. EfficientNetV2B0   - Transfer learning EfficientNetV2-B0 (ImageNet)

Setiap model menerima citra RGB dan menghasilkan output 2 kelas
(FAKE vs REAL) sebagai logits.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm tidak terinstall. EfficientNetV2-B0 tidak tersedia.")
    print("Install: pip install timm")

import config


def freeze_bn_stats(module: nn.Module):
    """
    Set BatchNorm pada layer beku (parameter tidak dilatih) ke mode eval.

    Tanpa ini, model.train() tetap memperbarui running_mean/running_var
    layer beku dengan statistik batch CIFAKE, sehingga fitur pretrained
    ImageNet ikut bergeser walaupun bobotnya "dibekukan".
    """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


# ============================================================
# 1. LIGHTWEIGHT CNN (8 Convolutional Layers, From Scratch)
# ============================================================

class LightweightCNN(nn.Module):
    """
    CNN ringan custom dengan 8 layer konvolusi.
    
    Arsitektur:
    - Block 1: 2× Conv2d(32) + BatchNorm + ReLU → MaxPool → Dropout
    - Block 2: 2× Conv2d(64) + BatchNorm + ReLU → MaxPool → Dropout
    - Block 3: 2× Conv2d(128) + BatchNorm + ReLU → MaxPool → Dropout
    - Block 4: 2× Conv2d(128) + BatchNorm + ReLU → Global Avg Pool → Dropout
    - Classifier: Linear(128 → num_classes)
    
    Total: 8 convolutional layers
    Filter progression: 32 → 32 → 64 → 64 → 128 → 128 → 128 → 128
    
    Cocok untuk input 32×32 (ukuran asli CIFAKE).
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        
        # Block 1: 32 filters (2 conv layers)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 32→16
            nn.Dropout2d(0.25),
        )
        
        # Block 2: 64 filters (2 conv layers)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 16→8
            nn.Dropout2d(0.25),
        )
        
        # Block 3: 128 filters (2 conv layers)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 8→4
            nn.Dropout2d(0.25),
        )
        
        # Block 4: 128 filters (2 conv layers) + Global Average Pooling
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)
        
        # Inisialisasi bobot (Kaiming/He initialization)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization untuk konvergensi lebih cepat."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)   # Flatten: (B, 128, 1, 1) → (B, 128)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ============================================================
# 2. RESNET-50 (Transfer Learning)
# ============================================================

class ResNet50Transfer(nn.Module):
    """
    ResNet-50 dengan transfer learning dari ImageNet.
    
    Strategi fine-tuning:
    - Freeze: conv1, bn1, layer1, layer2, layer3
    - Unfreeze: layer4 (fine-tune fitur high-level)
    - Replace: fc layer → Dropout + Linear(2048 → num_classes)
    - BatchNorm di layer beku tetap mode eval (running stats ImageNet dipertahankan)

    Parameter dilatih: ±15,0 juta dari 23,5 juta (63,7%).
    Input: 224×224 RGB (resize dari 32×32 CIFAKE)
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        
        # Load pretrained ResNet-50
        self.model = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V2
        )
        
        # Freeze semua parameter terlebih dahulu
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 untuk fine-tuning fitur high-level
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        # Replace fully-connected layer
        in_features = self.model.fc.in_features  # 2048
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            freeze_bn_stats(self.model)
        return self

    def forward(self, x):
        return self.model(x)
    
    def get_param_groups(self):
        """
        Mengembalikan parameter groups dengan learning rate berbeda:
        - Pretrained layers (layer4): LR rendah
        - Classification head (fc): LR tinggi
        """
        pretrained_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "fc" in name:
                    head_params.append(param)
                else:
                    pretrained_params.append(param)
        
        return [
            {"params": pretrained_params, "lr": config.LEARNING_RATE_PRETRAINED},
            {"params": head_params, "lr": config.LEARNING_RATE_HEAD},
        ]


# ============================================================
# 3. EFFICIENTNETV2-B0 (Transfer Learning)
# ============================================================

class EfficientNetV2B0Transfer(nn.Module):
    """
    EfficientNetV2-B0 dengan transfer learning dari ImageNet.
    Menggunakan library timm (PyTorch Image Models) untuk akses
    arsitektur EfficientNetV2-B0 yang tidak tersedia di torchvision.
    
    Strategi fine-tuning:
    - Freeze: conv_stem, bn1, dan stage 1-4 dari 6 stage backbone
    - Unfreeze: 2 stage terakhir (stage 5-6) + conv_head + bn2 + classifier
    - Replace: classifier → Linear(1280 → num_classes)
    - BatchNorm di layer beku tetap mode eval (running stats ImageNet dipertahankan)

    Catatan: stage 5-6 berisi sebagian besar parameter, sehingga yang dilatih
    ±5,4 juta dari 5,9 juta (92,7%), hampir setara full fine-tuning.
    Bandingkan ResNet-50: 63,7%.

    Input: 224×224 RGB
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError(
                "Library 'timm' diperlukan untuk EfficientNetV2-B0. "
                "Install: pip install timm"
            )
        
        # Load pretrained EfficientNetV2-B0 via timm
        self.model = timm.create_model(
            "tf_efficientnetv2_b0",
            pretrained=True,
            num_classes=num_classes,
        )
        
        # Freeze semua parameter terlebih dahulu
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier (head baru)
        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        
        # Unfreeze conv_head dan bn2 jika ada
        if hasattr(self.model, "conv_head"):
            for param in self.model.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.model, "bn2"):
            for param in self.model.bn2.parameters():
                param.requires_grad = True
        
        # Unfreeze 2 stage terakhir dari backbone (blocks = 6 stage)
        if hasattr(self.model, "blocks"):
            num_blocks = len(self.model.blocks)
            for param in self.model.blocks[max(0, num_blocks - 2):].parameters():
                param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            freeze_bn_stats(self.model)
        return self

    def forward(self, x):
        return self.model(x)
    
    def get_param_groups(self):
        """
        Mengembalikan parameter groups dengan learning rate berbeda.
        """
        pretrained_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "classifier" in name:
                    head_params.append(param)
                else:
                    pretrained_params.append(param)
        
        return [
            {"params": pretrained_params, "lr": config.LEARNING_RATE_PRETRAINED},
            {"params": head_params, "lr": config.LEARNING_RATE_HEAD},
        ]


# ============================================================
# FACTORY FUNCTION
# ============================================================

def get_model(model_name: str) -> nn.Module:
    """
    Factory function untuk membuat model berdasarkan nama.
    
    Args:
        model_name: Salah satu dari config.ALL_MODELS
    
    Returns:
        nn.Module model yang sudah siap training
    """
    models_map = {
        config.MODEL_LIGHTWEIGHT: LightweightCNN,
        config.MODEL_RESNET50: ResNet50Transfer,
        config.MODEL_EFFICIENTNET: EfficientNetV2B0Transfer,
    }
    
    if model_name not in models_map:
        raise ValueError(
            f"Model '{model_name}' tidak dikenal. "
            f"Pilihan: {list(models_map.keys())}"
        )
    
    model = models_map[model_name]()
    model = model.to(config.DEVICE)
    
    return model


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def count_parameters(model: nn.Module) -> dict:
    """
    Menghitung jumlah parameter model.
    
    Returns:
        dict dengan total, trainable, dan frozen parameter count
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def get_target_layer(model: nn.Module, model_name: str):
    """
    Mengembalikan target layer untuk Grad-CAM visualization.
    
    Args:
        model: Model instance
        model_name: Nama model
    
    Returns:
        Target layer (nn.Module) untuk Grad-CAM
    """
    if model_name == config.MODEL_LIGHTWEIGHT:
        # Layer conv terakhir di block4 (Conv2d ke-2)
        return model.block4[3]  # Conv2d(128, 128)
    
    elif model_name == config.MODEL_RESNET50:
        # Layer terakhir di layer4
        return model.model.layer4[-1]
    
    elif model_name == config.MODEL_EFFICIENTNET:
        # Block terakhir dari backbone
        if hasattr(model.model, "blocks"):
            return model.model.blocks[-1]
        elif hasattr(model.model, "conv_head"):
            return model.model.conv_head
    
    raise ValueError(f"Target layer tidak ditemukan untuk model: {model_name}")


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing models.py...")
    print(f"Device: {config.DEVICE}")
    print()
    
    for model_name in config.ALL_MODELS:
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        try:
            model = get_model(model_name)
            params = count_parameters(model)
            
            print(f"  Total Parameters    : {params['total']:>12,}")
            print(f"  Trainable Parameters: {params['trainable']:>12,}")
            print(f"  Frozen Parameters   : {params['frozen']:>12,}")
            
            # Test forward pass
            if model_name == config.MODEL_LIGHTWEIGHT:
                dummy = torch.randn(2, 3, 32, 32).to(config.DEVICE)
            else:
                dummy = torch.randn(2, 3, 224, 224).to(config.DEVICE)
            
            with torch.no_grad():
                output = model(dummy)
            
            print(f"  Input Shape         : {dummy.shape}")
            print(f"  Output Shape        : {output.shape}")
            print(f"  Output Sample       : {output[0].cpu().numpy()}")
            print()
        
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
