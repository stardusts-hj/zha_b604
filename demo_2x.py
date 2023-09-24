import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
import os

'''==========import from our code=========='''
import configs.baseline_lap as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='abseline', type=str)
# parser.add_argument('--n', default=2, type=int)
args = parser.parse_args()


'''==========Model setting=========='''

model = Model(-1, 'baseline_lap_char')
model.load_model('ckpt.pkl')
model.eval()
model.device()


print(f'=========================Start Generating=========================')

I0 = cv2.imread(r'F:\vfi\train\sequence\00001\0026\im1.png')
I2 = cv2.imread(r'F:\vfi\train\sequence\00001\0026\im7.png')

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)
with torch.no_grad():
    pred = model.net(torch.cat([I0_, I2_], 1),timestamp=torch.Tensor([0.5]).cuda())
im = padder.unpad(pred[0])[0].detach().cpu().numpy().transpose(1, 2, 0)
im = im * 255.
images = [I0[:, :, ::-1], im[:, :, ::-1].astype(np.uint8), I2[:, :, ::-1]]
mimsave('example/out_2x_our.gif', images, duration=int(1/3*1000))


if not os.path.exists('results'):
    os.makedirs('results')

for k, v in pred[1].items():
    if k == 'flow':
        f1 = v[:,:2]
        f1 = torchvision.utils.flow_to_image(f1)
        im = padder.unpad(f1)[0].detach().cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(os.path.join('results', f'{k}_forward.png'), im.astype(np.uint8))
        f2 = v[:,2:]
        f2 = torchvision.utils.flow_to_image(f2)
        im = padder.unpad(f2)[0].detach().cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(os.path.join('results', f'{k}_backward.png'), im.astype(np.uint8))
        continue
    im = padder.unpad(v)[0].detach().cpu().numpy().transpose(1, 2, 0)
    im = im * 255.
    cv2.imwrite(os.path.join('results', f'{k}.png'), im.astype(np.uint8))

print(f'=========================Done=========================')