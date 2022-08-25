import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MSRFNet"]

class Dconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Dconv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, d2=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        
        self.dd2conv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d2, 0), dilation=(d2, 1), groups=nIn // 2, bn_acti=True)
        self.dd2conv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d2), dilation=(1, d2), groups=nIn // 2, bn_acti=True)

        

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.d = d
        self.d2 = 5-d

    def forward(self, input):
        w,h = input.size(2),input.size(3)
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)


        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        
        ds1 = self.ddconv3x1(output+br1) 
        ds1 = self.ddconv1x3(ds1)

        ds2 = self.dd2conv3x1(output+br1+ds1)
        ds2 = self.dd2conv1x3(ds2)
        

        output = br1 + ds1 + ds2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class MRFDCNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3)

        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block1_1 = DABModule(64, d=2, d2 = 4)
        self.DAB_Block1_2 = DABModule(64, d=4, d2 = 2)
        self.bn_prelu_2 = BNPReLU(128 + 3)

        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.DAB_Block2_1 = DABModule(128, d=2, d2 = 16)
        self.DAB_Block2_2 = DABModule(128, d=4, d2 = 8)
        self.DAB_Block2_3 = DABModule(128, d=8, d2 = 4)
        self.DAB_Block2_4 = DABModule(128, d=16, d2 = 2)
       
        self.DAB_Block3_1 = DABModule(128, d=2, d2 = 16)
        self.DAB_Block3_2 = DABModule(128, d=4, d2 = 8)
        self.DAB_Block3_3 = DABModule(128, d=8, d2 = 4)
        self.DAB_Block3_4 = DABModule(128, d=16, d2 = 2)

        self.bn_prelu_3 = BNPReLU(256)

        self.BGA = GAM()

    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))


        output1_0 = self.downsample_1(output0_cat)
        DABb1_1 = self.DAB_Block1_1(output1_0)
        DABb1_2 = self.DAB_Block1_2(DABb1_1)
        output1_cat = self.bn_prelu_2(torch.cat([DABb1_2, output1_0, down_2], 1))


        output2_0 = self.downsample_2(output1_cat)
        DABb2_1 = self.DAB_Block2_1(output2_0)
        DABb2_2 = self.DAB_Block2_2(DABb2_1)
        DABb2_3 = self.DAB_Block2_3(DABb2_2)
        DABb2_4 = self.DAB_Block2_4(DABb2_3)

        DABb3_1 = self.DAB_Block3_1(DABb2_4)
        DABb3_2 = self.DAB_Block3_2(DABb3_1)
        DABb3_3 = self.DAB_Block3_3(DABb3_2)
        DABb3_4 = self.DAB_Block3_4(DABb3_3)
       
        output2_cat = self.bn_prelu_3(torch.cat([DABb3_4, output2_0], 1))

        BGA = self.BGA(output1_cat, output2_cat, down_3)

        out = F.interpolate(BGA, input.size()[2:], mode='bilinear', align_corners=False)

        return out

class GAM(nn.Module):

    def __init__(self):
        super(GAM, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                131, 256, kernel_size=1, stride=1,
                padding=0, bias=False),
                nn.BatchNorm2d(256),
            nn.Conv2d(
                256, 256, kernel_size=(3, 1), stride=1,
                padding=(1, 0), groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(256),
            nn.Conv2d(
                256, 256, kernel_size=(1, 3), stride=1,
                padding=(0, 1), groups=256, bias=False),
            nn.BatchNorm2d(256)
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                131, 131, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(131)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(256),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                259, 259, kernel_size=3, stride=1,
                padding=1, groups=259, bias=False),
            nn.BatchNorm2d(259),
            nn.Conv2d(
                259, 131, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                387, 19, kernel_size=1, stride=1,
                padding=0, bias=False)
        )

        self.bn_prelu = BNPReLU(259)

    def forward(self, x_d, x_s, down_3):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        
        x_s_c = self.bn_prelu(torch.cat([x_s, down_3], 1))
        
        right2 = self.right2(x_s_c)
        right2 = F.interpolate(
            right2, size=dsize, mode='bilinear', align_corners=True)
        
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)      
        
        out = self.conv(torch.cat([left, right], 1))
        
        return out
