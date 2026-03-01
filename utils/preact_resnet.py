"""Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ind = None

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") and len(self.shortcut) > 0 else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.ind is not None:
            out += shortcut[:, self.ind, :, :]
        else:
            out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") and len(self.shortcut) > 0 else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_hidden=False, return_activation=False):
        out = self.conv1(x)
        out = self.layer1(out)
        activation1 = out
        out = self.layer2(out)
        activation2 = out
        out = self.layer3(out)
        activation3 = out
        out = self.layer4(out)
        out = self.avgpool(out)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)

        if return_hidden:
            return out, hidden
        elif return_activation:  # for NAD
            return out, activation1, activation2, activation3
        else:
            return out

    # for FeatureRE
    def from_input_to_features(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def from_features_to_output(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_layer(self, x, layer_output):
        out = self.conv1(x)
        out = self.layer1(out)
        activation1 = out
        out = self.layer2(out)
        activation2 = out
        out = self.layer3(out)
        activation3 = out
        out = self.layer4(out)
        out = self.avgpool(out)
        if layer_output == 'avgpool':
            return out
        else:
            raise NotImplementedError("`layer_output` must be 'avgpool'!")
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)

    def freeze_feature(self):
        for name, para in self.named_parameters():
            if name.count('linear') == 0:  # non-linear layer
                para.requires_grad = False

    def unfreeze_feature(self):
        for name, para in self.named_parameters():
            para.requires_grad = True

    def freeze_fc(self):
        for name, para in self.named_parameters():
            if name.count('linear') > 0:  # linear layer
                para.requires_grad = False

    def unfreeze_fc(self):
        for name, para in self.named_parameters():
            if name.count('linear') > 0:  # linear layer
                para.requires_grad = True

    def freeze_before_last_block(self):
        for name, para in self.named_parameters():
            para.requires_grad = False

        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

        last_block = self.layer3[-1]
        for name, para in last_block.named_parameters():
            para.requires_grad = True

    def unfreeze(self):
        for name, para in self.named_parameters():
            para.requires_grad = True


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


def PreActResNet34(num_classes=10):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)


def PreActResNet50(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes)


def PreActResNet101(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes)


def PreActResNet152(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes=num_classes)


def PreActResNet18_tiny_imagenet(num_classes=200):
    """
    PreActResNet-18 for Tiny ImageNet (32x32 input, 200 classes)
    
    Note: Tiny ImageNet images are resized to 32×32 before training/testing,
    so this model uses the same architecture as CIFAR-10 version.
    """
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


def test():
    net = PreActResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
