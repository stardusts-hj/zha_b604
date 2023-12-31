from functools import partial
import torch.nn as nn
from model.loss import LapLoss, Charbonnier_Loss, Smoothloss

'''==========Model config=========='''
def init_backbone_model_config(F=16, W=9, depth=[2, 2, 2, 4, 4]):
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
        'dw': True
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
        'dw': True
    }

MODEL_CONFIG = {
    'LOGNAME': 'debug',
    'pad':True,
    'backbone': init_backbone_model_config(),
    'refine':   init_refine_model_config(),
    'linear_blend': True
}

class Total_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = [Charbonnier_Loss(), LapLoss()]
        self.weight = [1., 0.5]
        self.char_loss = Charbonnier_Loss()
        self.smooth_loss = Smoothloss()
    
    def forward(self,pred, gt, extra_info=None, imgs = None):
        loss = 0
        loss_dict = {}
        for l, w in zip(self.loss, self.weight):
            loss = loss + l(pred, gt) * w
        loss_dict['init_pred_loss'] = loss.detach().cpu().item()
        warped_r_to_l_loss = self.char_loss(extra_info['warped_r_to_l'], imgs[:,:3]) * 0.2
        loss_dict['warped_r_to_l_loss'] = warped_r_to_l_loss.detach().cpu().item()
        warped_l_to_r_loss = self.char_loss(extra_info['warped_l_to_r'], imgs[:,3:6]) * 0.2
        loss_dict['warped_l_to_r_loss'] = warped_l_to_r_loss.detach().cpu().item()
        warped_mid_loss = self.char_loss(extra_info['warped_img0'], extra_info['warped_img1']) * 0.2
        loss_dict['warped_mid_loss'] = warped_mid_loss.detach().cpu().item()
        reg_loss = self.smooth_loss(extra_info['flow']) * 0.01
        loss_dict['reg_loss'] = reg_loss.detach().cpu().item()
        
        loss = loss + warped_r_to_l_loss + warped_l_to_r_loss + warped_mid_loss + reg_loss
            
        return loss, loss_dict