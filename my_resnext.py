import torch
import torch.nn as nn
import math


# TypeA is not tested yet.
class BottleneckTypeA(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_paths, stride):
        """
        Args:
            in_channels: number of input channels, the same for each path and for the bottleneck
            mid_channels: number of middle channels for each path
            out_channels: number of output channels for each path, the same for each path and for the bottleneck
            num_paths: num of paths
            stride: conv stride, for the middle layer
        """
        super(BottleneckTypeA, self).__init__()

        def make_path():
            return nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )

        self.paths = nn.ModuleList([make_path() for _ in range(num_paths)])

        if (in_channels != out_channels) or (stride != 1):
            self.shortcut_transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut_transform = None

        # BN layer applied to the summation of paths
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU applied to the summation of main branch and shortcut_transform
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut_out = x if self.shortcut_transform is None else self.shortcut_transform(x)
        paths_out = [path(x) for path in self.paths]
        main_branch_out = self.bn(torch.sum(torch.stack(paths_out), 0))
        out = main_branch_out + shortcut_out
        out = self.relu(out)
        return out


class BottleneckTypeC(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_paths, stride):
        """
        Args:
            in_channels: number of input channels, the same for each path and for the bottleneck
            mid_channels: number of middle channels for each path
            out_channels: number of output channels for each path, the same for each path and for the bottleneck
            num_paths: num of paths
            stride: conv stride, for the middle layer
        """
        super(BottleneckTypeC, self).__init__()

        total_mid_channels = mid_channels * num_paths
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, total_mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(total_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_mid_channels, total_mid_channels, kernel_size=3, stride=stride, padding=1, groups=num_paths, bias=False),
            nn.BatchNorm2d(total_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if (in_channels != out_channels) or (stride != 1):
            self.shortcut_transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut_transform = None

        # ReLU applied to the summation of main branch and shortcut_transform
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut_out = x if self.shortcut_transform is None else self.shortcut_transform(x)
        main_branch_out = self.main_branch(x)
        out = main_branch_out + shortcut_out
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, config, type='C', num_classes=1000):
        super(ResNeXt, self).__init__()
        self.block = BottleneckTypeA if type == 'A' else BottleneckTypeC
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self.make_stage(config['stage1'])
        self.stage2 = self.make_stage(config['stage2'])
        self.stage3 = self.make_stage(config['stage3'])
        self.stage4 = self.make_stage(config['stage4'])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_stage(self, config):
        layers = []
        for i in range(config['num_blocks']):
            # If a stage has stride=2, it is applied in the first block of the stage.
            stride = config['stride'] if i == 0 else 1
            layers.append(self.block(self.in_channels, config['mid_channels'], config['out_channels'], config['num_paths'], stride))
            self.in_channels = config['out_channels']
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Since the input channels of the blocks inside a stage may be different,
# we do not specify it here. Instead, it is determined by the output
# channels of the preceding layer.
resnext50_32x4d_config = dict(
    stage1=dict(num_blocks=3, num_paths=32, mid_channels=4, out_channels=256, stride=1),
    stage2=dict(num_blocks=4, num_paths=32, mid_channels=8, out_channels=512, stride=2),
    stage3=dict(num_blocks=6, num_paths=32, mid_channels=16, out_channels=1024, stride=2),
    stage4=dict(num_blocks=3, num_paths=32, mid_channels=32, out_channels=2048, stride=2),
)


def create_resnext50_32x4d(**kwargs):
    model = ResNeXt(resnext50_32x4d_config, **kwargs)
    return model
