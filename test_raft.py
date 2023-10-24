import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
import os
from model.warplayer import warp
'''==========import from our code=========='''
import configs.baseline_lap as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = "tight"


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()

I0 = cv2.imread(r'C:\Users\Administrator\Desktop\project\local_test\test_sample\00001_6x\0000001.png')
I2 = cv2.imread(r'C:\Users\Administrator\Desktop\project\local_test\test_sample\00001_6x\0000000.png')
# I0 = cv2.imread(r'F:\code_base\zha_b604\test\Type1\TEST02_045_f0465\0000.png')
# I2 = cv2.imread(r'F:\code_base\zha_b604\test\Type1\TEST02_045_f0465\0032.png')
# gt = cv2.imread('example/img2.jpg')
I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
# gt = (torch.tensor(gt.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=8)
I0_, I2_ = padder.pad(I0_, I2_)

from torchvision.models.optical_flow import raft_small
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.utils import flow_to_image

weights = Raft_Small_Weights.DEFAULT

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

timestep = torch.Tensor([3 / 6.]).reshape(1,1,1,1).to(I0_)

# flow_imgs = flow_to_image(predicted_flows_forward)
# grid = [[img1, flow_img] for (img1, flow_img) in zip(I0_, flow_imgs)]
# plot(grid)
with torch.no_grad():
    list_of_flows_forward = model(I0_, I2_)
    list_of_flows_backward = model(I2_, I0_)
predicted_flows_backward = list_of_flows_backward[-1]
predicted_flows_forward = list_of_flows_forward[-1]


warp0 = warp(I0_, predicted_flows_backward*timestep)
warp1 = warp(I2_, predicted_flows_forward*(1 - timestep))
init_pred = warp0 * ( 1 - timestep ) + warp1 * timestep
grid = [
    [I0_[0], flow_to_image(predicted_flows_forward)[0]],
    [I2_[0], flow_to_image(predicted_flows_backward)[0]],
    [init_pred[0], init_pred[0]],
]
plot(grid)
init_pred = padder.unpad(init_pred)
init_pred = init_pred[0].detach().cpu().numpy().transpose(1,2,0) * 255.
images = [I0[:, :, ::-1], init_pred[:, :, ::-1].astype(np.uint8), I2[:, :, ::-1]]
mimsave('example/out_2x.gif', images, duration=int(1/3*1000))