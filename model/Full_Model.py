import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from model.warplayer import warp, warp_pad_zero
import model.BackBone as BackBone

import model.Refine as Refine

class FullModel(nn.Module):
    def __init__(self, **kwargs):
        super(FullModel, self).__init__()
        cls_tmp = getattr(BackBone, kwargs['backbone'].pop('type', None), None)
        self.backone = cls_tmp(**kwargs['backbone'])

        cls_tmp = getattr(Refine, kwargs['refine'].pop('type', None), None)
        self.refine = cls_tmp(**kwargs['refine'])
        self.linear_blend = kwargs.pop('linear_blend', False)
        self.refine_5 = kwargs.pop('refine_5', False)
        self.refine_twice = kwargs.pop('refine_twice', False)
        self.pad = kwargs.pop('pad', False)
        self.pad_grad = kwargs.pop('pad_grad', False)

    def forward(self, x, timestamp = 0.5):
        extra_info = {}
        flow, feature = self.backone(x)
        img0 = x[:, :3].contiguous()
        img1 = x[:, 3:6].contiguous()
        if not isinstance(timestamp,torch.Tensor):
            timestamp = torch.Tensor([timestamp]).to(x)
        B = timestamp.shape[0]
        timestamp = timestamp.reshape(B,1,1,1).to(x)
        
        if self.linear_blend:
            flow_forward = flow[:, 2:4]*(1 - timestamp)**2 - flow[:,:2]*timestamp*(1-timestamp)
            flow_backward = flow[:, :2]*timestamp**2 - flow[:,2:4]*timestamp*(1-timestamp)
        else:
            flow_forward = flow[:, 2:4] * (1 - timestamp)
            flow_backward = flow[:, :2] * timestamp

        warped_img0 = warp(img0, flow_backward)
        warped_img1 = warp(img1, flow_forward)


        warped_l_to_r = warp(img0, flow[:, :2])
        warped_r_to_l = warp(img1, flow[:, 2:4])

        if self.pad or self.pad_grad:
            if self.pad_grad:
                zero_warped_img0 = warp_pad_zero(img0 + 1, flow_backward) > 0
                zero_warped_img1 = warp_pad_zero(img1 + 1, flow_forward) > 0
            else:
                zero_warped_img0 = warp_pad_zero(img0 + 1, flow_backward).detach() > 0
                zero_warped_img1 = warp_pad_zero(img1 + 1, flow_forward).detach() > 0
            warped_img0 = warped_img1 * ~zero_warped_img0 + warped_img0 * zero_warped_img0
            warped_img1 = warped_img0 * ~zero_warped_img1 + warped_img1 * zero_warped_img1
            extra_info['zero_warped_img0'] = zero_warped_img0
            extra_info['zero_warped_img1'] = zero_warped_img1
        
        init_pred = warped_img0 * (1 - timestamp) + warped_img1 * (timestamp)
        
        
        if self.refine_twice:
            out, mask, final_fuse_mask = self.refine(torch.cat([flow, feature, img0, img1, warped_img0, warped_img1, init_pred], dim=1), time = timestamp)
            out_0 = warp(img0, out[:, :2] + flow_backward)
            out_1 = warp(img1, out[:, 2:4] + flow_forward)
            mask = torch.sigmoid(mask)
            extra_info['mask'] = mask
            interp_img = out_0 * mask + out_1 * (1 - mask)
            final_fuse_mask = torch.sigmoid(final_fuse_mask)
            interp_img = interp_img * final_fuse_mask + init_pred * (1 - final_fuse_mask)
            
            interp_img = torch.clamp(interp_img, 0, 1)
            extra_info['res_flow'] = out[:, :4]
            extra_info['t_flow'] = torch.cat([out[:, :2] + flow_backward,
                                              out[:, 2:4] + flow_forward],1)
        elif not self.refine_5:
    
            res_out = self.refine(torch.cat([flow, feature, img0, img1, warped_img0, warped_img1, init_pred], dim=1), time = timestamp)
            ### follow the implementation of EMA-VFI and RIFE
            res = res_out[:, :3] * 2 - 1
            interp_img = torch.clamp(res + init_pred, 0, 1)
            extra_info['res'] = res
        else:
            out, mask = self.refine(torch.cat([flow, feature, img0, img1, warped_img0, warped_img1, init_pred], dim=1), time = timestamp)
            out_0 = warp(img0, out[:, :2] + flow_backward)
            out_1 = warp(img1, out[:, 2:4] + flow_forward)
            mask = torch.sigmoid(mask)
            extra_info['mask'] = mask
            interp_img = out_0 * mask + out_1 * (1 - mask)
            interp_img = torch.clamp(interp_img, 0, 1)
            extra_info['res_flow'] = out[:, :4]
            extra_info['t_flow'] = torch.cat([out[:, :2] + flow_backward,
                                              out[:, 2:4] + flow_forward],1)
        extra_info['warped_img0'] = warped_img0
        extra_info['warped_img1'] = warped_img1
        extra_info['init_pred'] = init_pred
        extra_info['pred'] = interp_img
        extra_info['warped_l_to_r'] = warped_l_to_r
        extra_info['warped_r_to_l'] = warped_r_to_l
        extra_info['flow'] = flow



        return interp_img, extra_info

    def calculate_flow(self, x, timestamp = 0.5, flow=None, feature=None):
        assert flow is not None, \
        '''
        multiframe inference use the same backbone,
        please inference the backbone first
        '''
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        if not isinstance(timestamp,torch.Tensor):
            timestamp = torch.Tensor([timestamp]).to(x)
        B = timestamp.shape[0]
        timestamp = timestamp.reshape(B,1,1,1).to(x)
        if self.linear_blend:
            flow_forward = flow[:, 2:4]*(1 - timestamp)**2 - flow[:,:2]*timestamp*(1-timestamp)
            flow_backward = flow[:, :2]*timestamp**2 - flow[:,2:4]*timestamp*(1-timestamp)
        else:
            flow_forward = flow[:, 2:4] * (1 - timestamp)
            flow_backward = flow[:, :2] * timestamp
        warped_img0 = warp(img0, flow_backward)
        warped_img1 = warp(img1, flow_forward)


        if self.pad or self.pad_grad:
            if self.pad_grad:
                zero_warped_img0 = warp_pad_zero(img0 + 1, flow_backward) > 0
                zero_warped_img1 = warp_pad_zero(img1 + 1, flow_forward) > 0
            else:
                zero_warped_img0 = warp_pad_zero(img0 + 1, flow_backward).detach() > 0
                zero_warped_img1 = warp_pad_zero(img1 + 1, flow_forward).detach() > 0
            warped_img0 = warped_img1 * ~zero_warped_img0 + warped_img0 * zero_warped_img0
            warped_img1 = warped_img0 * ~zero_warped_img1 + warped_img1 * zero_warped_img1

        init_pred = warped_img0 * (1 - timestamp) + warped_img1 * (timestamp)

        if self.refine_twice:
            out, mask, final_fuse_mask = self.refine(torch.cat([flow, feature, img0, img1, warped_img0, warped_img1, init_pred], dim=1), time = timestamp)
            out_0 = warp(img0, out[:, :2] + flow_backward)
            out_1 = warp(img1, out[:, 2:4] + flow_forward)
            mask = torch.sigmoid(mask)
            interp_img = out_0 * mask + out_1 * (1 - mask)
            final_fuse_mask = torch.sigmoid(final_fuse_mask)
            interp_img = interp_img * final_fuse_mask + init_pred * (1 - final_fuse_mask)
            
            interp_img = torch.clamp(interp_img, 0, 1)
        elif not self.refine_5:
    
            res_out = self.refine(torch.cat([flow, feature, img0, img1, warped_img0, warped_img1, init_pred], dim=1), time = timestamp)
            ### follow the implementation of EMA-VFI and RIFE
            res = res_out[:, :3] * 2 - 1
            interp_img = torch.clamp(res + init_pred, 0, 1)
        else:
            out, mask = self.refine(torch.cat([flow, feature, img0, img1, warped_img0, warped_img1, init_pred], dim=1), time = timestamp)
            out_0 = warp(img0, out[:, :2] + flow_backward)
            out_1 = warp(img1, out[:, 2:4] + flow_forward)
            mask = torch.sigmoid(mask)
            interp_img = out_0 * mask + out_1 * (1 - mask)
            interp_img = torch.clamp(interp_img, 0, 1)
        return interp_img, init_pred


    def multi_inference(self, x, time_list = []):
        assert len(time_list) > 0, 'Time_list should not be empty!'

        pred_list = []
        flow, feature = self.backone(x)
        for timestep in time_list:
            pred = self.calculate_flow(x, timestamp=timestep, flow=flow, feature=feature)
            pred_list.append(pred)

        return pred_list
