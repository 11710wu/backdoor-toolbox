'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg19_bn_cifar10', 'vgg19_bn_tiny_imagenet', 'vgg19_bn_mnist', 'vgg19_bn_mnistm',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10, feature_size=None):
        super(VGG, self).__init__()
        self.features = features

        # If feature_size is not provided, use default 512 (for CIFAR-10 like datasets)
        if feature_size is None:
            feature_size = 512  # Default for standard VGG configurations

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(feature_size, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, return_hidden=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if return_hidden:
            hidden = x
        x = self.classifier(x)
        if return_hidden:
            return x, hidden
        else:
            return x

    # for FeatureRE
    def from_input_to_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def from_features_to_output(self, features):
        x = self.classifier(features)
        return x

    def freeze_feature(self):
        for name, para in self.named_parameters():
            if name.count('classifier') == 0: # non-linear layer
                para.requires_grad = False
    
    def unfreeze_feature(self):
        for name, para in self.named_parameters():
            if name.count('classifier') == 0: # non-linear layer
                para.requires_grad = True

    def freeze_fc(self):
        for name, para in self.named_parameters():
            if name.count('classifier') > 0: # non-linear layer
                para.requires_grad = False

    def unfreeze_fc(self):
        for name, para in self.named_parameters():
            if name.count('classifier') > 0: # non-linear layer
                para.requires_grad = True

    def partial_forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        partial_classifier = self.classifier[:-1]
        x = partial_classifier(x)
        return x


class VGG_low_dim(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGG_low_dim, self).__init__()

        self.features = features

        self.reducer = nn.Linear(512, 8)

        self.low_dim_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.ReLU(True),
            nn.Linear(8, num_classes)
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, return_hidden=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.reducer(x)

        if return_hidden:
            hidden = x
        x = self.low_dim_classifier(x)
        if return_hidden:
            return x, hidden
        else:
            return x

    def partial_forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        partial_classifier = self.classifier[:-1]
        x = partial_classifier(x)
        return x




def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn(num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)

def vgg16_low_dim_bn(num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG_low_dim(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)


def vgg19(num_classes=10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes=num_classes)


def vgg19_bn(num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    print(f"[MODEL] Creating VGG19_BN with {num_classes} classes")
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)


def vgg19_bn_cifar10(num_classes=10):
    """
    VGG 19-layer model (configuration 'E') with batch normalization for CIFAR-10 (32x32 input, 10 classes)
    
    Note: Same architecture as standard vgg19_bn, explicitly named for consistency with other datasets.
    """
    print(f"[MODEL] Creating VGG19_BN_CIFAR10 with {num_classes} classes")
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)


def vgg19_bn_tiny_imagenet(num_classes=200):
    """
    VGG 19-layer model (configuration 'E') with batch normalization for Tiny ImageNet (64x64 input, 200 classes)

    Note: Tiny ImageNet uses original 64×64 resolution, feature maps are 2×2×512 = 2048 dimensions.
    """
    print(f"[MODEL] Creating VGG19_BN_TinyImageNet with {num_classes} classes")
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes, feature_size=512*2*2)


def vgg19_bn_mnist(num_classes=10):
    """
    VGG 19-layer model (configuration 'E') with batch normalization for MNIST (28x28 input, 10 classes)

    Note: Modified configuration to reduce pooling layers to prevent feature maps from becoming too small.
    """
    # Modified config for MNIST: remove last two 'M's to keep feature maps at 3x3
    mnist_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
                 512, 512, 512, 512]  # Remove last two 'M's to keep feature maps at 3x3
    print(f"[MODEL] Creating VGG19_BN_MNIST with {num_classes} classes")
    return VGG(make_layers(mnist_cfg, batch_norm=True), num_classes=num_classes, feature_size=512*3*3)


def vgg19_bn_mnistm(num_classes=10):
    """
    VGG 19-layer model (configuration 'E') with batch normalization for MNIST-M (28x28 input, 10 classes)
    
    Note: Uses the same architecture as MNIST since MNIST-M has the same image size and number of classes.
    """
    # Modified config for MNIST-M: same as MNIST
    mnist_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
                 512, 512, 512, 512]  # Remove last two 'M's to keep feature maps at 3x3
    print(f"[MODEL] Creating VGG19_BN_MNISTM with {num_classes} classes")
    return VGG(make_layers(mnist_cfg, batch_norm=True), num_classes=num_classes, feature_size=512*3*3)
