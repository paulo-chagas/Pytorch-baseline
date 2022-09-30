import sys
import torch
import torchvision


def get_model(name, num_classes, freeze_backbone, dp_rate):
    if name == 'densenet':
        return Densenet(num_classes, freeze_backbone)
    elif name == 'wideresnet':
        return WideResNet(num_classes, freeze_backbone)
    elif name == 'wideresnet_dp':
        return WideResNet_Dp(num_classes, freeze_backbone, dp_rate)
    else:
        print('\nNot implemmented model')
        sys.exit(1)


class Identity(torch.nn.Module):
    """
    Identity operation; it passes the data foward
    Identity(x) = x
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Densenet(torch.nn.Module):
    """
    Normal DenseNet
    """

    def __init__(self, num_classes, freeze_backbone):
        super().__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.classifier = Identity()
        self.fc = torch.nn.Linear(in_features=1024, out_features=num_classes)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, dropout=False, p=0.5):
        x = self.model(x)
        y = self.fc(x)
        return y


class WideResNet(torch.nn.Module):
    """
    Normal WideResNet
    """

    def __init__(self, num_classes, freeze_backbone):
        super().__init__()
        self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.fc = Identity()
        self.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, dropout=False, p=0.5):
        x = self.model(x)
        y = self.fc(x)
        return y


class WideResNet_Dp(torch.nn.Module):
    """
    WideResNet with dropout before the last fc layer
    """

    def __init__(self, num_classes, freeze_backbone, dp_rate):
        super().__init__()
        self.dp_rate = dp_rate
        self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.fc = Identity()
        self.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, dropout=True):
        x = self.model(x)
        if dropout:
            x = torch.nn.functional.dropout(x, p=self.dp_rate,
                                            training=True,
                                            inplace=False)
        y = self.fc(x)
        return y
