import torch
import torch.nn as nn
import torch.nn.functional as F
import basicsr.models.archs.dbb_block as dbbblock

def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t


def get_bn_bias(bn_layer):
    gamma, beta, mean, var, eps = bn_layer.weight, bn_layer.bias, bn_layer.running_mean, bn_layer.running_var, bn_layer.eps
    std = (var + eps).sqrt()
    bn_bias = beta - mean * gamma / std

    return bn_bias


class RRRB(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|


    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2, rep='plain'):
        super(RRRB, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, ratio*n_feats, 1, 1, 0)
        if rep == 'plain' :
            self.fea_conv = nn.Conv2d(ratio*n_feats, ratio*n_feats, 3, 1, 0)
        elif rep == 'DBB':
            self.fea_conv = dbbblock.DiverseBranchBlock(
                ratio*n_feats,
                ratio*n_feats,
                3,1,0
                )
        # self.fea_conv = nn.Conv2d(ratio*n_feats, ratio*n_feats, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(ratio*n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        if hasattr(self, 'rrrb_reparam'):
            return self.nonlinear(self.rrrb_reparam(x))
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out

    def switch_to_deploy(self):
        if hasattr(self, 'rrrb_reparam'):
            return
        k0 = self.expand_conv.weight.data
        k1 = self.fea_conv.weight.data
        k2 = self.reduce_conv.weight.data
        
        b0 = self.expand_conv.bias.data
        b1 = self.fea_conv.bias.data
        b2 = self.reduce_conv.bias.data
        
        mid_feats, n_feats = k0.shape[:2]
        # first step: remove the middle identity
        for i in range(mid_feats):
            k1[i, i, 1, 1] += 1.0
            
        # second step: merge the first 1x1 convolution and the next 3x3 convolution
        merge_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
        merge_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).cuda()
        merge_b0b1 = F.conv2d(input=merge_b0b1, weight=k1, bias=b1)
        
        # third step: merge the remain 1x1 convolution
        merge_k0k1k2 = F.conv2d(input=merge_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
        merge_b0b1b2 = F.conv2d(input=merge_b0b1, weight=k2, bias=b2).view(-1)
    
        for i in range(n_feats):
            merge_k0k1k2[i, i, 1, 1] += 1.0
            
        self.rrrb_reparam = nn.Conv2d(in_channels=n_feats, out_channels=n_feats,
                                     kernel_size=3, stride=1,
                                     padding=1)
        self.rrrb_reparam.weight.data = merge_k0k1k2
        self.rrrb_reparam.bias.data = merge_b0b1b2
        for para in self.parameters():
            para.detach_()
        self.__delattr__('expand_conv')
        self.__delattr__('fea_conv')
        self.__delattr__('reduce_conv')

