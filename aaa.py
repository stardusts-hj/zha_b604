import numpy as np
import torch
from model.warplayer import warp_pad_zero
import cv2 as cv
import os
import math
gt_path = r'final_sample\00000_2x'
# im_path = r'final_sample_w1_l2_300.pkl\00000_3x'
im_path = r'final_sample_b32_3090_300.pkl\00000_3x'


gt_files = os.listdir(gt_path)
im_files = os.listdir(im_path)

psnr = []
for i, im in enumerate(im_files):
    gt = cv.imread(os.path.join(gt_path, im)).astype(np.float32) / 255.
    pred = cv.imread(os.path.join(im_path, im)).astype(np.float32) / 255.
    if i % 3 != 0:
        psnr.append(-10 * math.log10(((gt - pred) * (gt - pred)).mean()))

psnr = np.array(psnr).mean()
print(psnr)