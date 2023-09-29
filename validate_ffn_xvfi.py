import os, math
import torch
import pathlib
import logging
import argparse
from tqdm import tqdm
from collections import OrderedDict
from ptflops import get_model_complexity_info
from utils import util_logger
from utils.model_summary import get_model_flops
import sys
from Trainer import Model
from dataset_arbi import VimeoDatasetArbi
from torch.utils.data import DataLoader
import numpy as np
from model.warplayer import warp
import cv2 as cv
import torch.nn.functional as F
from imageio import mimsave
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="baseline_lap_char")
parser.add_argument('--test_data_path', type=str, default='test')
parser.add_argument('--dataset', type=str, default='X4K1000FPS')
parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####   FFN   ####
from ffn.fastflownet import FastFlowNet, centralize
# from torchvision.models.optical_flow import Raft_Small_Weights


model = FastFlowNet().cuda()
model = model.eval()
model.load_state_dict(torch.load('./fastflownet_ft_sintel.pth'))

# dataset_val = VimeoDatasetArbi('test', '/data1/dataset/NeurIPS_CellSegData/vimeo90k/')
# val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=4)

from xvfi_utils import X_Test
test_set = X_Test(args, multiple=8, validation=False)
val_data = DataLoader(test_set, batch_size=2, pin_memory=True, num_workers=0, drop_last=True)

def process(imgs):
    result = []
    for im in imgs:
        im = im.detach().cpu().numpy().transpose(1,2,0)*255.
        result.append(im.astype(np.uint8))
    
    return result
idx = 0
def evaluate(model, val_data):

    psnr = []
    global idx
    for i, (frames, t_value, scene_name, frameRange) in enumerate(val_data):
        # imgs = F.interpolate(imgs, (1088, 1920))
        b, _, _, _ = frames.shape
        imgs = frames
        emb_t = t_value.reshape(b,1,1,1)
        emb_t = emb_t.to(device)
        imgs = imgs.to(device, non_blocking=True)
        img1, img2, gt = imgs[:, :3], imgs[:, 3:6],  imgs[:, 6:]
        img1, img2, rgb_mean = centralize(img1, img2)
        h, w = img1.shape[2:]
        with torch.no_grad():
            flow_forward = model(torch.cat([img1, img2], 1)).data
            flow_forward = 20.0 * F.interpolate(flow_forward, size=(h, w), mode='bilinear', align_corners=False)
            flow_backward = model(torch.cat([img2, img1], 1)).data
            flow_backward = 20.0 * F.interpolate(flow_backward, size=(h, w), mode='bilinear', align_corners=False)
            # warp0 = warp(img1, flow_backward*emb_t)
            # warp1 = warp(img2, flow_forward*(1 - emb_t))
            warp0 = warp(img1, flow_backward*emb_t**2 - flow_forward*emb_t*(1 - emb_t))
            warp1 = warp(img2, flow_forward*(1 - emb_t)**2 - flow_backward*emb_t*(1 - emb_t))
            pred = warp0 * ( 1 - emb_t ) + warp1 * emb_t
            # pred, _ = model.update(imgs, gt, training=False, emb_t = emb_t)
        for j in range(gt.shape[0]):
            pred[j] = pred[j] + rgb_mean[j] - pred[j].mean(dim=[1,2]).reshape(3,1,1)
            pred[j] = torch.clamp(pred[j], 0, 1)
            cri = -10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item())
            if cri < 100:
                i1, i2, gtt, predd= process([imgs[j, 0:3], imgs[j,3:6], gt[j], pred[j]])
                overlap = (gtt.astype(np.float32) + predd.astype(np.float32)) / 2.
                t = float(emb_t[j].detach().cpu())
                base_path = os.path.join('bad_validation_8', f'{i}_{cri}')
                os.makedirs(base_path)
                cv.imwrite(os.path.join(base_path, 'left.png'), i1)
                cv.imwrite(os.path.join(base_path, 'right.png'), i2)
                cv.imwrite(os.path.join(base_path, 'gt.png'), gtt)
                cv.imwrite(os.path.join(base_path, 'pred.png'), predd)
                cv.imwrite(os.path.join(base_path, 'overlap.png'), overlap.astype(np.uint8))
                tmps = [i1[:,:,::-1], predd[:,:,::-1], i2[:,:,::-1]]
                mimsave(os.path.join(base_path, f'out_{t:.4f}.gif'), tmps, duration=int(1/3*1000))
                tmps = [gtt[:,:,::-1], predd[:,:,::-1]]
                mimsave(os.path.join(base_path, f'overlap.gif'), tmps, duration=int(1/2*1000))
                
                # min_psnr = cri
                idx += 1
            psnr.append(cri)
            tmp = pred[j].detach().cpu().numpy().transpose(1,2,0)*255.
            cv.imwrite('tmp.png', tmp.astype(np.uint8))

    # print(psnr)
    # print(len(psnr))
    psnr = np.array(psnr).mean()
    print(f"PSNR: {psnr}")
    print('test/psnr', psnr)

evaluate(model, val_data)
