from torch import nn
import torch
import torch.nn.functional as F
from UNetVit import VIT


class Tran_conv(nn.Module):
    def __init__(self,in_channels):
        super(Tran_conv, self).__init__()
        self.convX1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels + in_channels // 2, 1, 1),
            nn.BatchNorm2d(in_channels + in_channels // 2)
        )
        self.convX2 = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 2, in_channels*2, 1, 2),
            nn.BatchNorm2d(in_channels*2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels+in_channels//2, 1, 1),
            nn.BatchNorm2d(in_channels+in_channels//2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels+in_channels//2, in_channels+in_channels//2, 1, 1),
            nn.BatchNorm2d(in_channels+in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels+in_channels//2, in_channels+in_channels//2, 3, 2, 1),
            nn.BatchNorm2d(in_channels+in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels+in_channels//2, in_channels*2, 1, 1),
            nn.BatchNorm2d(in_channels*2)
        )
        self.action = nn.ReLU(inplace=True)

        self.vit = VIT.VisionTransformer(depth=6, n_heads=12, img_size=16, dim=768, patch_size=2, pos_1d=True,
                                           hybr=False, n_classes=1, n_chan=512)
        self.con_change = nn.Sequential(
            nn.Conv2d(768, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
    def forward(self,x1,x2):
        x2 = self.vit(x2)
        x2 = self.con_change(x2)

        input1 = self.convX1(x1)
        branch1 = self.conv1(x1)
        branch2 = self.conv2(branch1)
        branch3 = self.conv3(branch2)
        branch3 = self.action(input1+branch3)

        input2 = self.convX2(branch3)
        branch4 = self.conv4(branch3)
        branch5 = self.conv5(branch4)
        branch6 = self.conv6(branch5)
        branch6 = self.action(input2 + branch6)

        return branch6 + x2

if __name__ == '__main__':
    unet = Tran_conv(512)
    # unet.eval()
    Ae = torch.randn([1, 512, 16, 16])
    Ae2 = torch.randn([1, 512, 16, 16])
    out1 = unet(Ae,Ae2).size()
    print(out1)

