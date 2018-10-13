import torch
import torch.nn as nn

side_length = 256
epoch = 50
batch_size = 10
learning_rate = 1e-3
momentum_parameter = 0


class UNet(nn.Module):

    # UNet's Downward Convolutional Block.
    # i: number of input channels
    # j: number of output channels
    @classmethod
    def step_down_a(cls, i, j):
        maps = nn.Sequential(
            nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(j),
            nn.Conv2d(in_channels=j, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(j)
        )
        return maps

    # UNet's downsampling block.
    # j: number of input channels for batch normalization.
    @classmethod
    def step_down_b(cls, j):
        maps = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(j)
        )
        return maps

    # UNet's "bottom_out layers".
    # i: number of input channels.
    @classmethod
    def bottom_out(cls, i, j):
        maps: torch.nn.modules.container.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(j),
            nn.Conv2d(in_channels=j, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(j),
            nn.ConvTranspose2d(in_channels=j, out_channels=i, kernel_size=2, stride=2, bias=True, padding=0)
        )
        return maps

    # UNet's Upward Convolutional Block.
    # i: number of input channels.
    # j: number of output channels following the first convolution.
    # k: number of output channels following transpose convolution.
    @classmethod
    def step_up(cls, i, j, k):
        maps = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(i),
            nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(j),
            nn.Conv2d(in_channels=j, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(j),
            nn.ConvTranspose2d(in_channels=j, out_channels=k, kernel_size=2, stride=2, bias=True, padding=0)
        )
        return maps

    # UNet's Output Segmentation Block.
    # i: number of input channels.
    # j: number of output channels following the first convolution.
    # k: number of mask channels.
    @classmethod
    def segment_output(cls, i, j, k):
        maps = nn.Sequential(
            nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(j),
            nn.Conv2d(in_channels=j, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(j),
            nn.Conv2d(in_channels=j, out_channels=k, kernel_size=1, stride=1, bias=True, padding=0)
        )
        return maps

    def __init__(self):

        super().__init__()

        self.normalize = nn.BatchNorm2d(1)

        self.encode1_a = UNet.step_down_a(1, 64)
        self.encode1_b = UNet.step_down_b(64)
        self.encode2_a = UNet.step_down_a(64, 128)
        self.encode2_b = UNet.step_down_b(128)
        self.encode3_a = UNet.step_down_a(128, 256)
        self.encode3_b = UNet.step_down_b(256)
        self.encode4_a = UNet.step_down_a(256, 512)
        self.encode4_b = UNet.step_down_b(512)

        self.bottom = UNet.bottom_out(512, 1024)

        self.decode4 = UNet.step_up(1024, 512, 256)
        self.decode3 = UNet.step_up(512, 256, 128)
        self.decode2 = UNet.step_up(256, 128, 64)
        self.segment = UNet.segment_output(128, 64, 1)

        self.activate = nn.Softmax(dim=2)

        # Produce weights with He initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.normalize(x)

        # Encoding blocks are partitioned into two steps, so that
        # the a convolved image can be stored for a skip connection.
        x = self.encode1_a(x)
        x1 = x
        x = self.encode1_b(x)

        x = self.encode2_a(x)
        x2 = x
        x = self.encode2_b(x)

        x = self.encode3_a(x)
        x3 = x
        x = self.encode3_b(x)

        x = self.encode4_a(x)
        x4 = x
        x = self.encode4_b(x)

        x = self.bottom(x)

        # Decoding blocks are preceded by a skip connection. The
        # connection is a concatenation rather than a sum.
        x = torch.cat((x4, x), 1)
        x = self.decode4(x)

        x = torch.cat((x3, x), 1)
        x = self.decode3(x)

        x = torch.cat((x2, x), 1)
        x = self.decode2(x)

        x = torch.cat((x1, x), 1)
        x = self.segment(x)

        # nn.Softmax() normalizes across only one dimension.
        # Hence the following:
        s1 = x.size(0), x.size(1), x.size(2) * x.size(3)  # Shape parameters for return
        s2 = x.shape  # Shape parameters for Softmax activation

        x = x.view(s1)  # Reshape for Softmax activation
        x = self.activate(x)
        x = x.view(s2)  # Reshape for return

        return x
