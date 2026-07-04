"""
DenseNet-121 models for different datasets
Adapted from torchvision.models.densenet121
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DenseNetWrapper(nn.Module):
    """
    包装类：为 torchvision DenseNet 添加 return_hidden 支持
    用于兼容需要 return_hidden 参数的防御方法（如 AC）
    """
    def __init__(self, densenet_model):
        super(DenseNetWrapper, self).__init__()
        self.densenet = densenet_model
        # 关键：把 DenseNet 的核心子模块“提升”到 wrapper 顶层
        # 这样像 GradCAM 这类依赖 arch._modules['features'] 的代码也能直接工作，
        # 而不需要改动 GradCAM 的实现。
        self.features = densenet_model.features
        self.classifier = densenet_model.classifier
    
    def forward(self, x, return_hidden=False):
        """
        Args:
            x: 输入张量
            return_hidden: 是否返回隐藏层特征（用于 AC 等防御方法）
        Returns:
            如果 return_hidden=True: (logits, hidden_features)
            否则: logits
        """
        # DenseNet 的前向传播
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # 全局平均池化
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # 展平
        hidden = out.view(out.size(0), -1)
        # 分类层
        out = self.classifier(hidden)
        
        if return_hidden:
            return out, hidden
        else:
            return out
    
    # for FeatureRE
    def from_input_to_features(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # 展平
        hidden = out.view(out.size(0), -1)
        return hidden

    def from_features_to_output(self, features):
        out = self.classifier(features)
        return out

    def __getattr__(self, name):
        """
        代理其他属性和方法到原始 DenseNet 模型
        这样包装类可以透明地访问原始模型的所有属性
        """
        try:
            return super(DenseNetWrapper, self).__getattr__(name)
        except AttributeError:
            return getattr(self.densenet, name)


def densenet121_cifar10(num_classes=10):
    """
    DenseNet-121 adapted for CIFAR10 (32x32 input)
    
    Modifications:
    - First conv layer: kernel_size 7->3, stride 2->1, padding 3->1
    - Remove first pooling layer (Identity)
    - Classifier: 1000->num_classes
    - Wrapped with DenseNetWrapper to support return_hidden parameter
    """
    model = torchvision.models.densenet121(pretrained=False, num_classes=num_classes)
    
    # Modify first conv layer for 32x32 input
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove first pooling layer (MaxPool) - use Identity instead
    model.features.pool0 = nn.Identity()
    
    # 包装模型以支持 return_hidden 参数
    return DenseNetWrapper(model)


def densenet121_gtsrb(num_classes=43):
    """
    DenseNet-121 adapted for GTSRB (32x32 input)
    
    Modifications:
    - First conv layer: kernel_size 7->3, stride 2->1, padding 3->1
    - Remove first pooling layer (Identity)
    - Classifier: 1000->num_classes
    - Wrapped with DenseNetWrapper to support return_hidden parameter
    """
    model = torchvision.models.densenet121(pretrained=False, num_classes=num_classes)
    
    # Modify first conv layer for 32x32 input
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove first pooling layer (MaxPool) - use Identity instead
    model.features.pool0 = nn.Identity()
    
    # 包装模型以支持 return_hidden 参数
    return DenseNetWrapper(model)


def densenet121_imagenette(num_classes=10):
    """
    DenseNet-121 for ImageNette (224x224 input)
    Uses standard torchvision DenseNet-121 without modifications
    Wrapped with DenseNetWrapper to support return_hidden parameter
    """
    model = torchvision.models.densenet121(pretrained=False, num_classes=num_classes)
    return DenseNetWrapper(model)


def densenet121_tiny_imagenet(num_classes=200):
    """
    DenseNet-121 adapted for Tiny ImageNet (64x64 input, 200 classes).

    The current Tiny-ImageNet pipeline keeps the original 64x64 image size.
    A 3x3/stride-1 stem without the initial max-pool avoids excessive early
    downsampling while remaining compatible with the existing 64x64 transforms.
    """
    model = torchvision.models.densenet121(pretrained=False, num_classes=num_classes)

    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features.pool0 = nn.Identity()

    # 包装模型以支持 return_hidden 参数
    return DenseNetWrapper(model)


def densenet121_imagenet(num_classes=1000):
    """
    DenseNet-121 for ImageNet (224x224 input)
    Uses standard torchvision DenseNet-121 without modifications
    Wrapped with DenseNetWrapper to support return_hidden parameter
    """
    model = torchvision.models.densenet121(pretrained=False, num_classes=num_classes)
    return DenseNetWrapper(model)



def densenet121_mnistm(num_classes=10):
    """
    DenseNet-121 adapted for MNIST-M (28x28 input)
    
    Note: Uses the same architecture as CIFAR-10 since MNIST-M is similar in scale.
    """
    model = torchvision.models.densenet121(pretrained=False, num_classes=num_classes)
    
    # Modify first conv layer for 32x32/28x28 input
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove first pooling layer (MaxPool) - use Identity instead
    model.features.pool0 = nn.Identity()
    
    # 包装模型以支持 return_hidden 参数
    return DenseNetWrapper(model)
