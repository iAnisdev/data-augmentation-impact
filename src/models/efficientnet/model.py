import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.global_pool(x)
        se = F.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_rate=0.0):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise conv
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, 
                     padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )
        
        # Squeeze and excitation
        if se_ratio > 0:
            self.se = SEBlock(expanded_channels, int(1/se_ratio))
        else:
            self.se = nn.Identity()
        
        # Output phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.use_skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        
        # SE
        x = self.se(x)
        
        # Project
        x = self.project_conv(x)
        
        # Skip connection
        if self.use_skip:
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = x + identity
        
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        
        # Define block configurations [expand_ratio, channels, num_blocks, stride, kernel_size]
        block_configs = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        # Stem
        stem_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            Swish()
        )
        
        # Build blocks
        self.blocks = nn.ModuleList()
        in_ch = stem_channels
        
        for expand_ratio, channels, num_blocks, stride, kernel_size in block_configs:
            out_ch = int(channels * width_mult)
            num_blocks = int(math.ceil(num_blocks * depth_mult))
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(
                        in_ch, out_ch, kernel_size, block_stride, 
                        expand_ratio, se_ratio=0.25, drop_rate=0.1
                    )
                )
                in_ch = out_ch
        
        # Head
        head_channels = int(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_channels, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


def EfficientNetB0(num_classes=10, in_channels=3):
    """EfficientNet-B0 model"""
    return EfficientNet(
        num_classes=num_classes,
        in_channels=in_channels,
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.2
    )


# Alternative: Using torchvision pretrained models (more efficient for benchmarking)
def get_pretrained_efficientnet_b0(num_classes=10, pretrained=True):
    """Get pretrained EfficientNet-B0 from torchvision"""
    try:
        import torchvision.models as models
        
        # Try to use the newer efficientnet_b0 if available
        if hasattr(models, 'efficientnet_b0'):
            model = models.efficientnet_b0(pretrained=pretrained)
            # Modify the final layer for the specific number of classes
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            # Fallback to custom implementation
            model = EfficientNetB0(num_classes=num_classes, in_channels=3)
        
        return model
    except ImportError:
        # If torchvision doesn't have EfficientNet, use custom implementation
        return EfficientNetB0(num_classes=num_classes, in_channels=3)
