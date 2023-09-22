import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
import configs.baseline_lap as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='abseline', type=str)
args = parser.parse_args()


'''==========Model setting=========='''

model = Model(-1, 'baseline_lap')
model.load_model('output_lap_char/300.pkl')
model.eval()
model.device()


print(f'=========================Start Generating=========================')

I0 = cv2.imread('/data1/dataset/NeurIPS_CellSegData/vimeo90k/test_sample/00000_2x/0000001.png')
I2 = cv2.imread('/data1/dataset/NeurIPS_CellSegData/vimeo90k/test_sample/00000_2x/0000006.png')

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)
with torch.no_grad():
    pred = model.net(torch.cat([I0_, I2_], 1),timestamp=torch.Tensor([0.5]).cuda())[0]
im = padder.unpad(pred)[0].detach().cpu().numpy().transpose(1, 2, 0)
im = im * 255.
images = [I0[:, :, ::-1], im[:, :, ::-1].astype(np.uint8), I2[:, :, ::-1]]
mimsave('example/out_2x.gif', images, fps=3)


print(f'=========================Done=========================')