import shutil
import cv2
import math
import sys
from networkx import out_degree_centrality
import torch
import numpy as np
import argparse
from imageio import mimsave
import os
import re
import subprocess

'''==========import from our code=========='''
import configs.baseline_lap as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
import torchvision
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='abseline', type=str)
# parser.add_argument('--n', default=2, type=int)
args = parser.parse_args()


'''==========Model setting=========='''

model = Model(-1, 'final_stage0')
ckpt = 'b32_3090_300.pkl'
path = 'final_sample'
outdir = f'{path}_{ckpt}'
if not os.path.exists(outdir):
    os.makedirs(outdir)
model.load_model(ckpt, -1)
# model = Model(-1, 'tune')
# model.load_model('output_tune_fulldata/270.pkl', -1)
model.eval()
model.device()


print(f'=========================Start Generating=========================')

im_files = {x:sorted(os.listdir(os.path.join(path,x))) for x in sorted(os.listdir(path))}
padder = None
for i, (inter, im_list) in enumerate(im_files.items()):
    num_frame = len(im_list)
    n = int(re.findall(r'\_(\d+)', inter)[0])
    if not os.path.exists(os.path.join(outdir, inter)):
        os.makedirs(os.path.join(outdir, inter))
    print(f'========= {n}x interpolation ===========')
    for j in tqdm(range(num_frame-1)):
        shutil.copy(os.path.join(path, inter, im_list[j]), os.path.join(outdir, inter, f'{n*j:07d}.png'))
        shutil.copy(os.path.join(path, inter, im_list[j+1]), os.path.join(outdir, inter, f'{n*(j+1):07d}.png'))
        I0 = cv2.imread(os.path.join(path, inter, im_list[j]))
        I2 = cv2.imread(os.path.join(path, inter, im_list[j+1]))
        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
        if padder is None:
            padder = InputPadder(I0_.shape, divisor=32)
        I0_, I2_ = padder.pad(I0_, I2_)
        with torch.no_grad():
            pred = model.net.multi_inference(torch.cat([I0_, I2_], 1),time_list=[(i+1)/n for i in range(n-1)])
        for idx, im in enumerate(pred):
            im = padder.unpad(pred[idx])[0].detach().cpu().numpy().transpose(1, 2, 0)
            im = im * 255.
            cv2.imwrite(os.path.join(outdir, inter, f'{n*j+idx+1:07d}.png'), im.astype(np.uint8))
    print(f'=========================saving {n}x video =========================')
    image_folder = os.path.join(outdir, inter, '%07d.png')
    output_video = os.path.join(outdir, f'{n}x_video.mp4')

    command = ["ffmpeg", "-framerate", "30", "-i", f"{image_folder}", "-c:v", "libx264", "-r", "24", output_video]

    subprocess.run(command)

print(f'=========================Done=========================')