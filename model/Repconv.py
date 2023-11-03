from model.RRRB import RRRB
from model.dbb_block import DiverseBranchBlock
import torch.nn as nn

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, dw = False, bias = True, rep='none'):
    if rep == 'none':
        if dw:
            return nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, groups=in_planes,
                    padding=padding, dilation=dilation, bias=bias),
            nn.PReLU(in_planes),
            nn.Conv2d(in_planes, out_planes, 1, 1),
            nn.PReLU(out_planes)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
                nn.PReLU(out_planes)
            )
    elif rep == 'RRRB':
        assert in_planes==out_planes, "RRRB only support c_in = c_out"
        assert dw==False, "RRRB not support depth-wise conv"
        return RRRB(in_planes, act=True)
    elif rep == 'RRRB_no_act':
        assert in_planes==out_planes, "RRRB only support c_in = c_out"
        assert dw==False, "RRRB not support depth-wise conv"
        return RRRB(in_planes, act=False)
    elif rep == 'DBB':
        if dw:
            return nn.Sequential(
                DiverseBranchBlock(in_planes, out_channels=in_planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=in_planes),
                nn.PReLU(in_planes),
                DiverseBranchBlock(in_planes, out_planes, 1, 1),
                nn.PReLU(out_planes),
            )
        else:
            return nn.Sequential(
                DiverseBranchBlock(in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation),
                nn.PReLU(out_planes),
            )