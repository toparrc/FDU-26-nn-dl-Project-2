import torch
from torch import nn


def make_activation(name):
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
        "silu": lambda: nn.SiLU(inplace=True),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(module.bias)


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.0),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(0.0),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(0.1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        initialize_weights(self)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y = self.classifier(x)
        return y


class ConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu", dropout=0.0):
        super().__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.stage(x)


class CIFAR10PlainCNN(nn.Module):
    def __init__(self, channels, activation="relu", num_classes=10):
        super().__init__()
        dropouts = [0.0, 0.0, 0.1, 0.1]
        in_channels = 3
        layers = []
        for idx, out_channels in enumerate(channels):
            layers.append(
                ConvStage(
                    in_channels,
                    out_channels,
                    activation=activation,
                    dropout=dropouts[idx] if idx < len(dropouts) else 0.1,
                )
            )
            in_channels = out_channels

        spatial_size = 32 // (2 ** len(channels))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * spatial_size * spatial_size, 512),
            make_activation(activation),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        y = self.classifier(x)
        return y


class CIFAR10CNN4(CIFAR10PlainCNN):
    def __init__(self, activation="relu", num_classes=10):
        super().__init__(
            channels=(64, 128, 256, 512),
            activation=activation,
            num_classes=num_classes,
        )


class CIFAR10CNN3(CIFAR10PlainCNN):
    def __init__(self, activation="relu", num_classes=10):
        super().__init__(
            channels=(64, 128, 256),
            activation=activation,
            num_classes=num_classes,
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, activation="relu"):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = make_activation(activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        y = self.main(x)
        y = self.activation(y + self.shortcut(x))
        y = self.pool(y)
        y = self.dropout(y)
        return y


class CIFAR10ResidualCNN(nn.Module):
    def __init__(self, activation="relu", num_classes=10):
        super().__init__()
        self.layer1 = ResidualBlock(3, 64, dropout=0.0, activation=activation)
        self.layer2 = ResidualBlock(64, 128, dropout=0.0, activation=activation)
        self.layer3 = ResidualBlock(128, 256, dropout=0.1, activation=activation)
        self.layer4 = ResidualBlock(256, 512, dropout=0.1, activation=activation)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            make_activation(activation),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y = self.classifier(x)
        return y
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    for model_cls in (CIFAR10CNN, CIFAR10ResidualCNN):
        model = model_cls()
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        print(model_cls.__name__, y.shape, count_parameters(model))
