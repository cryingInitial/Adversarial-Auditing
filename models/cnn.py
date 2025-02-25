"""Convolutional Neural Networks from https://github.com/OsvaldFrisk/dp-not-all-noise-is-equal/blob/master/src/networks.py
Paper: https://arxiv.org/pdf/2110.06255"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_shape=None, out_dim=10, dropout_rate=0):
        super().__init__()
        if in_shape[1] == 1:
            # MNIST
            self.net = SmallNetwork(out_dim=out_dim, dropout_rate=dropout_rate)
        elif in_shape[1] == 3:
            # CIFAR-10
            self.net = BigNetwork(out_dim=out_dim, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.net(x)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        # group normalization
        self.bn1 = nn.GroupNorm(16, in_planes)
        # self.relu1 = nn.ReLU(inplace=False)
        self.relu1 = nn.Tanh()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=False)
        self.bn2 = nn.GroupNorm(16, out_planes)
        # self.relu2 = nn.ReLU(inplace=False)
        self.relu2 = nn.Tanh()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropout_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            # x = self.relu1(x)
        else:
            out = self.relu1(self.bn1(x))
            # out = self.relu1(x)
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.relu2(self.conv1(out if self.equalInOut else x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class BigNetwork(nn.Module):
    """Network used in the experiments on CIFAR-10"""

    def __init__(self, act_func=nn.Tanh, input_channels: int = 3, out_dim = 10, dropout_rate=0):
        super(BigNetwork, self).__init__()
        self.in_channels: int = input_channels

        # Variables to keep track of taken steps and samples in the model
        self.n_samples: int = 0
        self.n_steps: int = 0

        # Feature Layers
        feature_layer_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
        feature_layers = []

        c = self.in_channels
        for v in feature_layer_config:
            if v == 'M':
                feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                feature_layers += [conv2d, act_func()]
                c = v
        self.features = nn.Sequential(*feature_layers)

        # Classifier Layers
        num_hidden: int = 128
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(c * 4 * 4, num_hidden), act_func(), nn.Linear(num_hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(self.dropout(x))
        return x


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=16, out_dim=10, widen_factor=4, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        # self.bn1 = nn.BatchNorm2d(nChannels[3], track_running_stats=False)
        self.bn1 = nn.GroupNorm(16, nChannels[3])
        # self.relu = nn.ReLU(inplace=False)
        self.relu = nn.Tanh()
        self.fc = nn.Linear(nChannels[3], out_dim)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class SmallNetwork(nn.Module):
    """Network used in the experiments on MNIST"""

    def __init__(self, act_func=torch.tanh, out_dim=10, dropout_rate=0.0) -> None:
        super(SmallNetwork, self).__init__()

        # Variables to keep track of taken steps and samples in the model
        self.n_samples: int = 0
        self.n_steps: int = 0

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, out_dim)

        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_func(F.max_pool2d(self.conv1(x), (2, 2)))
        x = self.act_func(F.max_pool2d(self.conv2(x), (2, 2)))
        x = x.view(-1, 512)
        x = self.act_func(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x