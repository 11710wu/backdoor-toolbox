import torch.nn as nn


class SmallCNN(nn.Module):
    """Lightweight CNN baseline for CIFAR-sized RGB images."""

    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def _forward_features(self, x):
        activation1 = self.block1(x)
        activation2 = self.block2(activation1)
        activation3 = self.block3(activation2)
        return activation1, activation2, activation3

    def forward(self, x, return_hidden=False, return_activation=False):
        activation1, activation2, activation3 = self._forward_features(x)
        hidden = self.avgpool(activation3).view(x.size(0), -1)
        logits = self.classifier(hidden)

        if return_hidden:
            return logits, hidden
        if return_activation:
            return logits, activation1, activation2, activation3
        return logits

    def from_input_to_features(self, x):
        _, _, activation3 = self._forward_features(x)
        return activation3

    def from_features_to_output(self, features):
        hidden = self.avgpool(features).view(features.size(0), -1)
        return self.classifier(hidden)

    def freeze_feature(self):
        for name, param in self.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False

    def unfreeze_feature(self):
        for name, param in self.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = True

    def freeze_fc(self):
        for name, param in self.named_parameters():
            if name.startswith('classifier'):
                param.requires_grad = False

    def unfreeze_fc(self):
        for name, param in self.named_parameters():
            if name.startswith('classifier'):
                param.requires_grad = True


def SmallCNN_cifar10(num_classes=10):
    print(f"[MODEL] Creating SmallCNN_CIFAR10 with {num_classes} classes")
    return SmallCNN(num_classes=num_classes)
