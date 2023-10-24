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
from torchvision.utils import flow_to_image
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="baseline_lap_char")
parser.add_argument('--test_data_path', type=str, default='test')
parser.add_argument('--dataset', type=str, default='X4K1000FPS')
parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####   RAFT   ####
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from imageio import mimsave

weights = Raft_Large_Weights.DEFAULT
model = raft_large(weights=weights, progress=False).to(device)
model = model.eval()

# dataset_val = VimeoDatasetArbi('test', r'F:\vfi\train\sequence')
# val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=0)

from xvfi_utils import X_Test
test_set = X_Test(args, multiple=8, validation=False)
val_data = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=0, drop_last=True)

def process(imgs):
    result = []
    for im in imgs:
        im = im.detach().cpu().numpy().transpose(1,2,0)*255.
        result.append(im.astype(np.uint8))
    
    return result


padder = InputPadder((1080, 1920), divisor=32)
def evaluate(model, val_data):

    psnr = []
    for i, (frames, t_value, scene_name, frameRange) in enumerate(val_data):
        b, _, _, _ = frames.shape
        imgs = frames
        emb_t = t_value.reshape(b,1,1,1)
        emb_t = emb_t.to(device)
        imgs = imgs.to(device, non_blocking=True)
        img1, img2, gt = imgs[:, :3], imgs[:, 3:6],  imgs[:, 6:]
        rgb_mean = (img1.mean(dim=[-1,-2]) + img1.mean(dim=[-1,-2])) / 2
        rgb_mean = rgb_mean.reshape(b, 3,1,1)
        with torch.no_grad():
            flow_forward = model(img1, img2)[-1]
            flow_backward = model(img2, img1)[-1]
            warp0 = warp(img1, flow_backward*emb_t)
            warp1 = warp(img2, flow_forward*(1 - emb_t))
            pred = warp0 * ( 1 - emb_t ) + warp1 * emb_t
            warped_l_to_r = warp(img1, flow_backward)
            warped_r_to_l = warp(img2, flow_forward)
            warp0 = warp(img1, flow_backward*emb_t**2 - flow_forward*emb_t*(1 - emb_t))
            warp1 = warp(img2, flow_forward*(1 - emb_t)**2 - flow_backward*emb_t*(1 - emb_t))
            pred = warp0 * ( 1 - emb_t ) + warp1 * emb_t
            # pred, _ = model.update(imgs, gt, training=False, emb_t = emb_t)
        for j in range(gt.shape[0]):
            pred[j] = pred[j] + rgb_mean[j] - pred[j].mean(dim=[1,2]).reshape(3,1,1)
            pred[j] = torch.clamp(pred[j], 0, 1)
            warped_l_to_r[j] = warped_l_to_r[j] + rgb_mean[j] - warped_l_to_r[j].mean(dim=[1,2]).reshape(3,1,1)
            warped_l_to_r[j] = torch.clamp(warped_l_to_r[j], 0, 1)
            warped_r_to_l[j] = warped_r_to_l[j] + rgb_mean[j] - warped_r_to_l[j].mean(dim=[1,2]).reshape(3,1,1)
            warped_r_to_l[j] = torch.clamp(warped_r_to_l[j], 0, 1)
            cri = -10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item())
            i1, i2, gtt, predd, w0, w1= process([imgs[j, 0:3], imgs[j,3:6], gt[j], pred[j], warped_l_to_r[j], warped_r_to_l[j]])
            # overlap = (gtt.astype(np.float32) + predd.astype(np.float32)) / 2.
            t = float(emb_t[j].detach().cpu())
            base_path = os.path.join('validation_raft', f'{i}_{cri:.4f}')
            if not os.path.exists(base_path):
               os.makedirs(base_path)
            cv.imwrite(os.path.join(base_path, 'left.png'), i1)
            cv.imwrite(os.path.join(base_path, 'right.png'), i2)
            cv.imwrite(os.path.join(base_path, 'gt.png'), gtt)
            cv.imwrite(os.path.join(base_path, 'pred.png'), predd)
            cv.imwrite(os.path.join(base_path, 'warped_l_to_r.png'), w0)
            cv.imwrite(os.path.join(base_path, 'warped_r_to_l.png'), w1)
            # cv.imwrite(os.path.join(base_path, 'overlap.png'), overlap.astype(np.uint8))
            tmps = [i1[:,:,::-1], predd[:,:,::-1], i2[:,:,::-1]]
            mimsave(os.path.join(base_path, f'out_.gif'), tmps, duration=int(1/3*1000))
            tmps = [gtt[:,:,::-1], predd[:,:,::-1]]
            mimsave(os.path.join(base_path, f'overlap.gif'), tmps, duration=int(1/2*1000))
            f1 = flow_forward
            f2 = flow_backward
            f1 = flow_to_image(f1)
            im = padder.unpad(f1)[0].detach().cpu().numpy().transpose(1, 2, 0)
            cv.imwrite(os.path.join(base_path, f'flow_forward.png'), im.astype(np.uint8))
            f2 = flow_to_image(f2)
            im = padder.unpad(f2)[0].detach().cpu().numpy().transpose(1, 2, 0)
            cv.imwrite(os.path.join(base_path, f'flopw_backward.png'), im.astype(np.uint8))

    # print(psnr)
    # print(len(psnr))
    psnr = np.array(psnr).mean()
    print(f"PSNR: {psnr}")
    print('test/psnr', psnr)

evaluate(model, val_data)
