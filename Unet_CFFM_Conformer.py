from thop import profile
from torch import nn
import torch
import torch.nn.functional as F
import GRU
from my_modoule import FCM
from my_modoule import CFC
from my_modoule import Conformer

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


class ASPP(nn.Module):
    def __init__(self,in_channel,output):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block6 = SeparableConv2d(in_channel, in_channel, 3, 1, padding=5, dilation=5)
        self.atrous_block12 = SeparableConv2d(in_channel, in_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block18 = SeparableConv2d(in_channel, in_channel, 3, 1, padding=7, dilation=7)
        self.conv_1x1_output = nn.Conv2d(in_channel * 4, output, 1, 1)
        self.batchnorm = nn.BatchNorm2d(output)
        self.action = nn.ReLU(inplace=True)

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        cat = torch.cat([atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        net = self.batchnorm(net)
        net = self.action(net)

        return net


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        # self.pool4 = nn.MaxPool2d(2, stride=2)
        # self.conv5 = DoubleConv(512, 1024)
        self.Conformer = Conformer.Conformer(512)
        self.up6 = nn.ConvTranspose2d(1024, 512,2,2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256,2,2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128,2,2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64,2,2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.dropout = nn.Dropout2d(p=0.5)

        self.aspp1 = ASPP(1, 64)
        self.aspp2 = ASPP(64, 128)
        self.aspp3 = ASPP(128, 256)
        self.aspp4 = ASPP(256, 512)

        self.poola = nn.AvgPool2d(2, stride=2)

        self.ram1 = CFC.CFC(64, 128)
        self.ram2 = CFC.CFC(128, 64)
        self.ram3 = CFC.CFC(256, 32)
        self.ram4 = CFC.CFC(512, 16)



    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res1 = self.aspp1(x)
        c1 = self.conv1(x)
        # add1 = res1+c1
        add1 = self.ram1(c1, res1)
        p1 = self.pool1(add1)

        res2 = self.aspp2(self.poola(res1))
        c2 = self.conv2(p1)
        # add2 = res2 + c2
        add2 = self.ram2(c2, res2)
        p2 = self.pool2(add2)

        res3 = self.aspp3(self.poola(res2))
        c3 = self.conv3(p2)
        # add3 = res3 + c3
        add3 = self.ram3(c3,res3)
        p3 = self.pool3(add3)

        res4 = self.aspp4(self.poola(res3))
        c4 = self.conv4(p3)
        # add4 = res4 + c4
        add4 = self.ram4(c4,res4)
        mid1 = self.dropout(add4)
        # p4 = self.pool4(mid1)

        # c5 = self.conv5(p4)
        c5=self.Conformer(mid1,res4)
        mid2 = self.dropout(c5)
        up_6 = self.up6(mid2)
        merge6 = torch.cat([up_6, add4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, add3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, add2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, add1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10

#==================================================================================
if __name__ == '__main__':
    unet = Unet(1, 1)
    unet.eval()
    rgb = torch.randn([1, 1, 128, 128])
    out1 = unet(rgb).size()

    flops, params = profile(unet, inputs=(rgb,))
    flop_g = flops / (10 ** 9)
    param_mb = params / (1024 * 1024)  # 转换为MB

    print(f"模型的FLOP数量：{flop_g}G")
    print(f"参数数量: {param_mb} MB")
    print(out1)





















