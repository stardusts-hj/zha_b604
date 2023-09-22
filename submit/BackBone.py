import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
from warplayer import warp
from feature_extractor import MotionFormer



def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, dw = False):
    if dw:
        return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, groups=in_planes,
                  padding=padding, dilation=dilation, bias=True),
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.PReLU(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True),
            nn.PReLU(out_planes)
        )


############################ BFAT_backbone ############################

class IFBlock(nn.Module):
    def __init__(self, in_planes, out_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = conv(c, out_planes)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = F.interpolate(x, scale_factor = scale * 4, mode="bilinear", align_corners=False)
        tmp = self.lastconv(tmp)
        flow = tmp[:, :4].contiguous() * scale * 2
        feature = tmp[:, 4:].contiguous()
        return flow, feature
    

    
class BFAT_backbone(nn.Module):
    def __init__(self, **kargs):
        super(BFAT_backbone, self).__init__()
        ft = kargs['ft']
        self.block0 = IFBlock(6, ft, c=64) # 6  = 2 x (image channel)
        self.block1 = IFBlock(12+ft, ft, c=64) # 12 + ft = 4 x (image channel) + ft
        self.block2 = IFBlock(12+ft, ft, c=32)
        # self.block_tea = IFBlock(15+4, 32, c=64)

    def forward(self, x, scale=[4,2,1]):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        flow_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, feature_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, feature), 1), flow, scale=scale[i])
                flow = flow + flow_d
                feature = feature + feature_d
            else:
                flow, feature = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2].contiguous())
            warped_img1 = warp(img1, flow[:, 2:4].contiguous())
        
        return flow, feature
    

############################ EMA-VFI backbone ############################

class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17,fc=32):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else, c),
                                  conv(c, c),
                                  conv(c, 4+fc),
                                  )

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature = self.upsample(motion_feature) #/4 /2 /1
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
            flow = x[:, :4].contiguous() * (self.scale // 4)
        else:
            flow = x[:, :4].contiguous()
        feature = x[:, 4:].contiguous()
        return flow, feature


class EMA_Backbone(nn.Module):
    def __init__(self, **kargs):
        super(EMA_Backbone, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.backbone = MotionFormer(**kargs)
        self.block = nn.ModuleList([Head(kargs['motion_dims'][-1-i] * kargs['depths'][-1-i] + kargs['embed_dims'][-1-i],
                            kargs['scales'][-1-i], 
                            kargs['hidden_dims'][-1-i],
                            6 if i==0 else 3*4+4+kargs['fc'],
                            kargs['fc']) 
                            for i in range(self.flow_num_stage)])
        
    def forward(self, x):
        img0 = x[:, :3].contiguous()
        img1 = x[:, 3:6].contiguous()
        B = x.size(0)
        # feature_list = []
        # flow_list = []
        af, mf = self.backbone(img0, img1)
        flow = None

        for i in range(self.flow_num_stage):
            # t = torch.full(mf[-1-i][:B].shape, timestamp, dtype=torch.float).cuda()
            if flow != None:
                flow_d, feature_d = self.block[i]( torch.cat([mf[-1-i][:B], mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1, warped_img0, warped_img1, feature), 1), flow)
                flow = flow + flow_d
                feature = feature + feature_d
            else:
                flow, feature = self.block[i]( torch.cat([mf[-1-i][:B], mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1), 
                                            torch.cat((img0, img1), 1), None)
            # feature_list.append(feature)
            # flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2].contiguous())
            warped_img1 = warp(img1, flow[:, 2:4].contiguous())
            
        


        return flow, feature
    

