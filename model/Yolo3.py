import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.conv_block(x)

class HeadBlock(nn.Module):
    '''
    out_h = in_h // 2, out_w = in_h // 2
    '''
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 2 == 0
        self.head = nn.Sequential(
            ConvBlock(in_channel=in_channel, out_channel=out_channel//2, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=out_channel//2, out_channel=out_channel, kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x):
        return self.head(x)


class ResBlock(nn.Module):
    '''
    out_channel = in_channel * 2 \\
    out_h = in_h // 2, out_w = in_h // 2
    '''
    def __init__(self, in_channel, res_num):
        super().__init__()
        assert in_channel % 2 == 0
        self.res =  nn.ModuleList()
        for i in range(res_num):
            self.res.add_module(name=f"res_block_{i}", module=self.make_res_block(in_channel))
        self.down_sample = ConvBlock(in_channel=in_channel, out_channel=in_channel*2, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        for res_block in self.res:
            x = x + res_block(x)
        x = self.down_sample(x)
        return x
    
    def make_res_block(self, in_channel):
        block = nn.Sequential(
            ConvBlock(in_channel=in_channel, out_channel=in_channel//2, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channel=in_channel//2, out_channel=in_channel, kernel_size=3, stride=1, padding=1)
        )
        return block


class Darknet53(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.head = ConvBlock(in_channel=in_channel, out_channel=32, kernel_size=3, stride=1, padding=1)
        self.res_0 = ResBlock(in_channel=32, res_num=1)
        self.res_1 = ResBlock(in_channel=64, res_num=2)
        self.res_2 = ResBlock(in_channel=128, res_num=8)
        self.res_3 = ResBlock(in_channel=256, res_num=8)
        self.res_4 = ResBlock(in_channel=512, res_num=4)
    
    def forward(self, x):
        x = self.head(x)
        x = self.res_0(x)
        x = self.res_1(x)
        y_256 = self.res_2(x)
        y_512 = self.res_3(y_256)
        y_1024 = self.res_4(y_512)

        y = (y_256, y_512, y_1024)
        return y

class YoloBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.f_1024_connect_darknet_block = self.make_connect_darknet_block(in_channel=1024, out_channel=1024//2)
        self.f_1024_feature_map_block = self.make_feature_map_block(512)
        self.f_1024_upsample_block = self.make_upsample_block(512)

        self.f_512_connect_darknet_block = self.make_connect_darknet_block(in_channel=(512+256), out_channel=(512+256)//3)
        self.f_512_feature_map_block = self.make_feature_map_block(256)
        self.f_512_upsample_block = self.make_upsample_block(256)

        self.f_256_connect_darknet_block = self.make_connect_darknet_block(in_channel=(256+128), out_channel=(256+128)//3)
        self.f_256_feature_map_block = self.make_feature_map_block(128)
    
    def forward(self, x_256, x_512, x_1024):
        x_1024 = self.f_1024_connect_darknet_block(x_1024)
        feature_map_1024 = self.f_1024_feature_map_block(x_1024)

        x_1024 = self.f_1024_upsample_block(x_1024)
        x_512 = torch.cat((x_1024, x_512), dim=1)
        x_512 = self.f_512_connect_darknet_block(x_512)
        feature_map_512 = self.f_512_feature_map_block(x_512)

        x_512 = self.f_512_upsample_block(x_512)
        x_256 = torch.cat((x_512, x_256), dim=1)
        x_256 = self.f_256_connect_darknet_block(x_256)
        feature_map_256 = self.f_256_feature_map_block(x_256)

        return (feature_map_256, feature_map_512, feature_map_1024)

    def make_connect_darknet_block(self, in_channel, out_channel):
        block = nn.Sequential(
            ConvBlock(in_channel=in_channel, out_channel=out_channel, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channel=out_channel, out_channel=out_channel, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channel=out_channel, out_channel=out_channel*2, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=out_channel*2, out_channel=out_channel, kernel_size=1, stride=1, padding=0),
        )
        return block
    
    def make_upsample_block(self, in_channel):
        assert in_channel % 2 == 0
        block = nn.Sequential(
            ConvBlock(in_channel=in_channel, out_channel=in_channel//2, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        return block
    
    def make_feature_map_block(self, in_channel):
        block = nn.Sequential(
            ConvBlock(in_channel=in_channel, out_channel=in_channel*2, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channel=in_channel*2, out_channel=255, kernel_size=1, stride=1, padding=0)
        )
        return block

class Yolo3(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.darknet53 = Darknet53(in_channel)
        self.yolo = YoloBlock()
    
    def forward(self, x):
        (x_256, x_512, x_1024) = self.darknet53(x)
        y = (y_256, y_512, y_1024) = self.yolo(x_256, x_512, x_1024)
        return y
