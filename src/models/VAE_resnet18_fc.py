import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class ResizeArbitrary(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_size, mode='nearest'):
        super().__init__()
        self.out_size = out_size
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        x = F.interpolate(x, size=self.out_size, mode=self.mode)
        x = torch.relu(self.conv(x))
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()
        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, 
        # but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3, linear_downsample_factor = 8):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=2)
        self.fc_layer_len = 512 * 2 * 2
        self.linear_block = nn.Sequential(
            nn.Linear(
                self.fc_layer_len,
                self.fc_layer_len
            ), 
            nn.ReLU(),
            nn.Linear(
                self.fc_layer_len,
                z_dim * 2
            )
        )
    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(-1,self.fc_layer_len)
        x = self.linear_block(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, spatial_dim_bottle, num_Blocks=[2,2,2,2], z_dim=10, nc=3, linear_downsample_factor =8):
        super().__init__()
        self.in_planes = 512
        self.nc = nc
        self.shape_first_img = (512,spatial_dim_bottle[0],spatial_dim_bottle[1])
        self.fc_layer_len = 512 * 2 * 2

        self.linear_block = nn.Sequential(
            nn.Linear(
                z_dim,
                self.fc_layer_len,
            ),
            nn.ReLU(),
            nn.Linear(
                self.fc_layer_len,
                self.fc_layer_len,
            ), 
        )
        self.upscale = ResizeArbitrary(512,512,3,spatial_dim_bottle,mode='bicubic')
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear_block(z)
        x = x.view(-1, 512, 2, 2)
        x = self.upscale(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        return x


class VAEResNet18_Linear(nn.Module):
    def __init__(self, nc, z_dim, input_spatial_dim):
        super().__init__()
        self.in_spatial_shape = input_spatial_dim
        self.spat_shape_bottle = self.compute_spatial_shape(4)
        self.spat_shape_bottle = (self.spat_shape_bottle[0],self.spat_shape_bottle[1])
        self.encoder = ResNet18Enc(nc=nc, z_dim=z_dim)
        self.decoder = ResNet18Dec(nc=nc, z_dim=z_dim, spatial_dim_bottle=self.spat_shape_bottle)
        self.enc_linear = nn.Sequential(

        )

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, z, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def compute_spatial_shape(self, level: int) -> tuple[int, int]:
        # TODO Add warning when shape is odd before maxpool
        spatial_shape = np.array(self.in_spatial_shape)
        if level == 0:
            return spatial_shape
        spatial_shape = np.array(self.compute_spatial_shape(level-1)) // 2 
        if any([s%2 != 0 for s in spatial_shape]):
            raise ValueError("Can't Decode Because Input Dimension is Lost during Downsampling")
        return spatial_shape
