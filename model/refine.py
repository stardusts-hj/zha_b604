import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from model.BackBone import Head
import torch.nn.functional as F
from model.RRRB import RRRB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, dw = False, bias = True):
    if dw:
        return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, groups=in_planes,
                  padding=padding, dilation=dilation, bias=bias),
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.PReLU(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



################## EMA-VFI refine ###########################


class EMA_Unet(nn.Module):
    def __init__(self, c, out=3):
        super(EMA_Unet, self).__init__()
        self.down0 = Conv2(17+c, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, out, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow,c0[0], c1[0]), 1))
        s1 = self.down1(torch.cat((s0, c0[1], c1[1]), 1))
        s2 = self.down2(torch.cat((s1, c0[2], c1[2]), 1))
        s3 = self.down3(torch.cat((s2, c0[3], c1[3]), 1))
        x = self.up0(torch.cat((s3, c0[4], c1[4]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        return torch.sigmoid(x)


############################## NAFNet Unet Refine #################################

class NAF_Unet(nn.Module):
    def __init__(self, in_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], dw = True):
        super().__init__()
        self.intro = conv(in_channel,width,3,1,1, dw=dw)
        if dw:
            self.ending = nn.Sequential(
                nn.Conv2d(width, width, 3 , 1 ,1, groups=width),
                nn.Conv2d(width, 3, 1, 1)
                )
        else:
            self.ending = nn.Conv2d(width,3,3,1,1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[conv(chan,chan,3,1,1, dw=dw) for _ in range(num)]
                )
            )
            self.downs.append(
                conv(chan, 2*chan, 2, 2, padding=0, dw=dw)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[conv(chan,chan,3,1,1, dw=dw) for _ in range(middle_blk_num)]
            )
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    conv(chan, chan * 2, 1, bias=False, padding=0, dw=dw),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[conv(chan,chan,3,1,1, dw=dw) for _ in range(num)]
                )
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, inp, time=None):

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        ## Note: must use sigmoid as act
        x = self.ending(x)
        x = torch.sigmoid(x)

        return x

############################## NAFNet Unet_v2 Refine #################################


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, dw=False, rep= False):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1, dw=dw),
            conv(c//2, c, 3, 2, 1, dw=dw),
            )
        if rep:
            self.convblock = nn.Sequential(
                RRRB(c),
                RRRB(c),
                RRRB(c),
                RRRB(c),
                RRRB(c),
                RRRB(c),
                RRRB(c),
                RRRB(c),
            )
        else :
            self.convblock = nn.Sequential(
                conv(c, c, dw=dw),
                conv(c, c, dw=dw),
                conv(c, c, dw=dw),
                conv(c, c, dw=dw),
                conv(c, c, dw=dw),
                conv(c, c, dw=dw),
                conv(c, c, dw=dw),
                conv(c, c, dw=dw),
            )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow=None, scale=2):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask



class Stage_Refine(nn.Module):
    def __init__(self, in_channel=3, width=16, dw = True, rep=False):
        super().__init__()
        self.block1 = IFBlock(in_planes=in_channel, c=width, dw=dw, rep=rep)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, inp, time=None):

        B, C, H, W = inp.shape
        x = self.block1(torch.cat([inp, time.repeat(1,1,H,W), (1 - time).repeat(1,1,H,W)], 1), None)

        return x