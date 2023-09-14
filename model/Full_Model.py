import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from model.warplayer import warp
import model.BackBone as BackBone

import model.Refine as Refine

class FullModel(nn.Module):
    def __init__(self, **kwargs):
        super(FullModel, self).__init__()
        cls_tmp = getattr(BackBone, kwargs['backbone'].pop('type', None), None)
        self.backone = cls_tmp(**kwargs['backbone'])

        cls_tmp = getattr(Refine, kwargs['refine'].pop('type', None), None)
        self.refine = cls_tmp(**kwargs['refine'])

    def forward(self, x, timestamp = 0.5):
        extra_info = {}
        flow, feature = self.backone(x)
        img0 = x[:, :3].contiguous()
        img1 = x[:, 3:6].contiguous()
        B = timestamp.shape[0]
        timestamp = timestamp.reshape(B,1,1,1)
        warped_img0 = warp(img0, flow[:, :2] * timestamp)
        warped_img1 = warp(img1, flow[:, 2:4] * (1 - timestamp))

        init_pred = warped_img0 * (1 - timestamp) + warped_img1 * (timestamp)

        res_out = self.refine(torch.cat([flow, feature, img0, img1, warped_img0, warped_img1, init_pred], dim=1))
        
        ### follow the implementation of EMA-VFI and RIFE
        res = res_out[:, :3] * 2 - 1
        interp_img = torch.clamp(res + init_pred, 0, 1)

        extra_info['warped_img0'] = warped_img0
        extra_info['warped_img1'] = warped_img1
        extra_info['init_pred'] = init_pred
        extra_info['res'] = res



        return interp_img, extra_info

    def calculate_flow(self, x, timestamp = 0.5, flow=None, feature=None):
        assert flow is not None, \
        '''
        multiframe inference use the same backbone,
        please inference the backbone first
        '''
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        warped_img0 = warp(img0, flow[:, :2] * timestamp)
        warped_img1 = warp(img1, flow[:, 2:4] * (1 - timestamp))

        init_pred = warped_img0 * (1 - timestamp) + warped_img1 * (timestamp)

        res = self.refine(torch.cat([flow, feature, img0, img1, warped_img0, warped_img1, init_pred], dim=1))

        interp_img = res + init_pred

        return interp_img


    def multi_inference(self, x, time_list = []):
        assert len(time_list) > 0, 'Time_list should not be empty!'

        pred_list = []
        flow, feature = self.backone(x)
        for timestep in time_list:
            pred = self.calculate_flow(x, timestamp=timestep, flow=flow, feature=feature)
            pred_list.append(pred)

        return pred_list
