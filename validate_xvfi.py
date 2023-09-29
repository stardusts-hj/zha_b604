import os, math
import torch
import pathlib
import logging
import argparse
from tqdm import tqdm
from collections import OrderedDict
import sys
from Trainer import Model
from dataset_arbi import VimeoDatasetArbi
from torch.utils.data import DataLoader
import numpy as np
import cv2 as cv
from imageio import mimsave
import torch.nn.functional as F
from benchmark.utils.padder import InputPadder
padder = InputPadder((1080, 1920), divisor=32)
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="baseline_lap")
parser.add_argument('--test_data_path', type=str, default='test')
parser.add_argument('--dataset', type=str, default='X4K1000FPS')
parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Model(-1, args.model)
# model.load_model('output_baselinev2/28000.pkl', -1)
model.load_model('ckpt.pkl', -1)
model.eval()
model.device()

# dataset_val = VimeoDatasetArbi('test', './vimeo/vimeo90k/')
# val_data = DataLoader(dataset_val, batch_size=2, pin_memory=True, num_workers=0)

from xvfi_utils import X_Test
test_set = X_Test(args, multiple=4, validation=False)
val_data = DataLoader(test_set, batch_size=2, pin_memory=True, num_workers=0, drop_last=True)


def process(imgs):
    result = []
    for im in imgs:
        im = im.detach().cpu().numpy().transpose(1,2,0)*255.
        result.append(im.astype(np.uint8))
    
    return result

i = 0
min_psnr = 100
def evaluate(model, val_data):

    psnr = []
    global min_psnr
    for i, (frames, t_value, scene_name, frameRange) in enumerate(val_data):
        # imgs = F.interpolate(imgs, (1088, 1920))
        b, _, _, _ = frames.shape
        imgs = frames
        emb_t = t_value.reshape(b,1,1,1).to(device)
        imgs = imgs.to(device, non_blocking=True)
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False, emb_t = emb_t)
        for j in range(gt.shape[0]):
            cri = -10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item())
            if cri < min_psnr:
                i1, i2, gtt, predd= process([imgs[j, 0:3], imgs[j,3:6], gt[j], pred[j]])
                # overlap = (gtt.astype(np.float32) + predd.astype(np.float32)) / 2.
                t = float(emb_t[j].detach().cpu())
                base_path = os.path.join('bad_validation', f'{i}_{cri}')
                os.makedirs(base_path)
                cv.imwrite(os.path.join(base_path, 'left.png'), i1)
                cv.imwrite(os.path.join(base_path, 'right.png'), i2)
                cv.imwrite(os.path.join(base_path, 'gt.png'), gtt)
                cv.imwrite(os.path.join(base_path, 'pred.png'), predd)
                # cv.imwrite(os.path.join(base_path, 'overlap.png'), overlap.astype(np.uint8))
                tmps = [i1[:,:,::-1], predd[:,:,::-1], i2[:,:,::-1]]
                mimsave(os.path.join(base_path, f'out_{t:.4f}.gif'), tmps, fps=3)
                tmps = [gtt[:,:,::-1], predd[:,:,::-1]]
                mimsave(os.path.join(base_path, f'overlap.gif'), tmps, fps=2)
                
                # min_psnr = cri
                i += 1

            psnr.append(cri)

    psnr = np.array(psnr).mean()
    print(f"PSNR: {psnr}")
    print('test/psnr', psnr)

evaluate(model, val_data)

