from functools import partial
import torch.nn as nn
from model.loss import LapLoss

'''==========Model config=========='''
def init_backbone_model_config(F=32, W=9, depth=[2, 2, 4, 4, 4]):
    '''This function should not be modified'''
    return { 
        'type': 'EMA_Backbone',
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*F//32, 16*F//32],
        'window_sizes':[W, W],
        'scales':[4, 8, 16],
        'hidden_dims':[4*F, 4*F],
        'c':F,
        'mlp_ratios':[4, 4],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
        'fc': 32,
        'dw': False
    }


def init_refine_model_config():
    '''This function should not be modified'''
    return {
        'type': 'NAF_Unet',
        'in_channel': 32+15+4,
        'width': 16,
        'enc_blk_nums': [1] * 3,
        'dec_blk_nums': [1] * 3,
        'middle_blk_num': 1,
        'dw':False
    }

MODEL_CONFIG = {
    'LOGNAME': 'debug',
    'backbone': init_backbone_model_config(F = 16,
        depth = [2, 2, 2, 2, 2]),
    'refine':   init_refine_model_config()
}

class Total_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = [LapLoss()]
        self.weight = [1.]
    
    def forward(self,img0, img1):
        loss = 0
        for l, w in zip(self.loss, self.weight):
            loss = loss + l(img0, img1) * w
            
        return loss