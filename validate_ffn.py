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
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="baseline_lap_char")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####   FFN   ####
from ffn.fastflownet import FastFlowNet, centralize
# from torchvision.models.optical_flow import Raft_Small_Weights


model = FastFlowNet().cuda()
model = model.eval()
model.load_state_dict(torch.load('./fastflownet_ft_sintel.pth'))

dataset_val = VimeoDatasetArbi('test', r'K:\data')
val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=0)

def evaluate(model, val_data):

    psnr = []
    for i, (imgs, emb_t) in enumerate(val_data):
        emb_t = emb_t.to(device)
        imgs = imgs.to(device, non_blocking=True) / 255.
        img1, img2, gt = imgs[:, :3], imgs[:, 3:6],  imgs[:, 6:]
        img1, img2, rgb_mean = centralize(img1, img2)
        h, w = img1.shape[2:]
        with torch.no_grad():
            flow_forward = model(torch.cat([img1, img2], 1)).data
            flow_forward = 20.0 * F.interpolate(flow_forward, size=(h, w), mode='bilinear', align_corners=False)
            flow_backward = model(torch.cat([img2, img1], 1)).data
            flow_backward = 20.0 * F.interpolate(flow_backward, size=(h, w), mode='bilinear', align_corners=False)
            warp0 = warp(img1, flow_backward*emb_t**2 - flow_forward*emb_t*(1 - emb_t))
            warp1 = warp(img2, flow_forward*(1 - emb_t)**2 - flow_backward*emb_t*(1 - emb_t))
            pred = warp0 * ( 1 - emb_t ) + warp1 * emb_t
            # pred, _ = model.update(imgs, gt, training=False, emb_t = emb_t)
        for j in range(gt.shape[0]):
            # pred[j] = (pred[j] - pred[j].min()) / (pred[j].max() - pred[j].min())
            pred[j] = pred[j] + rgb_mean[j] - pred[j].mean(dim=[1,2]).reshape(3,1,1)
            pred[j] = torch.clamp(pred[j], 0, 1)
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
            # pred[j] = (pred[j] - pred[j].min()) / (pred[j].max() - pred[j].min())
            tmp = pred[j].detach().cpu().numpy().transpose(1,2,0)*255.
            cv.imwrite('tmp.png', tmp.astype(np.uint8))

    # print(psnr)
    # print(len(psnr))
    psnr = np.array(psnr).mean()
    print(f"PSNR: {psnr}")
    print('test/psnr', psnr)

evaluate(model, val_data)
