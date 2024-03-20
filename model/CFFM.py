from torch import nn
import torch
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x


class CFFM(nn.Module):
    def __init__(self,inchannel,dim):
        super(CFFM, self).__init__()
        self.global_h1 = nn.AdaptiveMaxPool2d((None, 1))
        self.global_w1 = nn.AdaptiveMaxPool2d((1, None))

        self.global_h2 = nn.AdaptiveMaxPool2d((None, 1))
        self.global_w2 = nn.AdaptiveMaxPool2d((1, None))

        self.conv1 = SeparableConv2d(inchannel, dim, 3, 1, padding=2,dilation=2)
        self.conv2 = SeparableConv2d(inchannel, dim, 3, 1, padding=2,dilation=2)

        self.conv3 = nn.Conv2d(dim*2+4, inchannel, 3, 1, padding=2, dilation=2)

        self.bn = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(inchannel)
        self.action = nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        change1 = self.action(self.bn(self.conv1(x1)))
        h1 = self.global_h1(change1)
        w1 = self.global_w1(change1)
        h1 = torch.transpose(h1,1,3)
        w1 = torch.transpose(w1,1,2)

        change2 = self.conv2(x2)
        h2 = self.global_h2(change2)
        w2 = self.global_w2(change2)
        h2 = torch.transpose(h2, 1, 3)
        w2 = torch.transpose(w2, 1, 2)

        all = torch.cat([change1,h1,w1,change2,h2,w2],dim=1)
        all = self.action(self.bn2(self.conv3(all)))
        return all








