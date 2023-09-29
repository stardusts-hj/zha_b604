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

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="baseline_lap_char")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####   RAFT   ####
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights


weights = Raft_Large_Weights.DEFAULT
model = raft_large(weights=weights, progress=False).to(device)
model = model.eval()

dataset_val = VimeoDatasetArbi('test', r'F:\vfi\train\sequence')
val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=0)

def evaluate(model, val_data):

    psnr = []
    for i, (imgs, emb_t) in enumerate(val_data):
        emb_t = emb_t.to(device)
        imgs = imgs.to(device, non_blocking=True) / 255.
        img1, img2, gt = imgs[:, :3], imgs[:, 3:6],  imgs[:, 6:]
        rgb_mean = (img1.mean(dim=[-1,-2]) + img1.mean(dim=[-1,-2])) / 2
        rgb_mean = rgb_mean.reshape(8, 3,1,1)
        with torch.no_grad():
            flow_forward = model(img1, img2)[-1]
            flow_backward = model(img2, img1)[-1]
            warp0 = warp(img1, flow_backward*emb_t)
            warp1 = warp(img2, flow_forward*(1 - emb_t))
            pred = warp0 * ( 1 - emb_t ) + warp1 * emb_t
            # pred, _ = model.update(imgs, gt, training=False, emb_t = emb_t)
        for j in range(gt.shape[0]):
            # pred[j] = (pred[j] - pred[j].min()) / (pred[j].max() - pred[j].min())
            pred[j] = pred[j] + rgb_mean[j] - pred[j].mean(dim=[1,2]).reshape(3,1,1)
            pred[j] = torch.clamp(pred[j], 0, 1)
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
            tmp = pred[j].detach().cpu().numpy().transpose(1,2,0)*255.
            cv.imwrite('tmp.png', tmp.astype(np.uint8))

    # print(psnr)
    # print(len(psnr))
    psnr = np.array(psnr).mean()
    print(f"PSNR: {psnr}")
    print('test/psnr', psnr)

evaluate(model, val_data)
