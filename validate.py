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
import cv2 as cv
from imageio import mimsave
from torchvision.utils import flow_to_image
from benchmark.utils.padder import InputPadder
import torch.nn.functional as F

padder = InputPadder((1080, 1920), divisor=32)
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="final_stage0")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Model(-1, args.model)
model.load_model('output_final_stage0/250.pkl', -1)
model.eval()
model.device()

dataset_val = VimeoDatasetArbi('test', r'K:\data')
val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=0)


def process(imgs):
    result = []
    for im in imgs:
        im = im.detach().cpu().numpy().transpose(1,2,0)*255.
        result.append(im.astype(np.uint8))
    
    return result

i = 0
min_psnr = 26
def evaluate(model, val_data):

    psnr = []
    init_psnr = []
    idx = 0
    diff = []
    global min_psnr
    for i, (imgs, emb_t) in enumerate(val_data):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, extra_info = model.update(imgs, gt, training=False, emb_t = emb_t)
        for j in range(gt.shape[0]):
            cri = -10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item())
            init_cri = -10 * math.log10(((gt[j] - extra_info['init_pred'][j]) * (gt[j] - extra_info['init_pred'][j])).mean().cpu().item())


            # i1, i2, gtt, predd, init_pred= process([imgs[j, 0:3], imgs[j,3:6], gt[j], pred[j], extra_info['init_pred'][j]])
            # # overlap = (gtt.astype(np.float32) + predd.astype(np.float32)) / 2.
            # t = float(emb_t[j].detach().cpu())
            # base_path = os.path.join('vimeo_results', 'validation_vimeo_refine_reproduce', f'{idx}_{t:.4f}_{cri:.4f}')
            # if not os.path.exists(base_path):
            #    os.makedirs(base_path)
            # cv.imwrite(os.path.join(base_path, 'init_pred.png'), init_pred)
            # cv.imwrite(os.path.join(base_path, 'left.png'), i1)
            # cv.imwrite(os.path.join(base_path, 'right.png'), i2)
            # cv.imwrite(os.path.join(base_path, 'gt.png'), gtt)
            # cv.imwrite(os.path.join(base_path, 'pred.png'), predd)
            # # cv.imwrite(os.path.join(base_path, 'overlap.png'), overlap.astype(np.uint8))
            # tmps = [i1[:,:,::-1], predd[:,:,::-1], i2[:,:,::-1]]
            # mimsave(os.path.join(base_path, f'out_.gif'), tmps, duration=int(1/3*1000))
            # tmps = [gtt[:,:,::-1], predd[:,:,::-1]]
            # mimsave(os.path.join(base_path, f'overlap.gif'), tmps, duration=int(1/2*1000))
            # # f1 = extra_info['flow'][:, :2]
            # # f2 = extra_info['flow'][:, 2:4]
            # # f1 = flow_to_image(f1)
            # # im = padder.unpad(f1)[0].detach().cpu().numpy().transpose(1, 2, 0)
            # # cv.imwrite(os.path.join(base_path, f'flow_forward.png'), im.astype(np.uint8))
            # # f2 = flow_to_image(f2)
            # # im = padder.unpad(f2)[0].detach().cpu().numpy().transpose(1, 2, 0)
            # # cv.imwrite(os.path.join(base_path, f'flopw_backward.png'), im.astype(np.uint8))
            # for k, v in extra_info.items():
            #     if 'flow' in k:
            #         f1 = v[j][:2]
            #         f1 = flow_to_image(f1)
            #         im = f1.detach().cpu().numpy().transpose(1, 2, 0)
            #         cv.imwrite(os.path.join(base_path, f'{k}_forward.png'), im.astype(np.uint8))
            #         f2 = v[j][2:]
            #         f2 = flow_to_image(f2)
            #         im = f2.detach().cpu().numpy().transpose(1, 2, 0)
            #         cv.imwrite(os.path.join(base_path, f'{k}_backward.png'), im.astype(np.uint8))
            #         continue
            #     im = v[j].detach().cpu().numpy().transpose(1, 2, 0)
            #     im = im * 255.
            #     cv.imwrite(os.path.join(base_path, f'{k}.png'), im.astype(np.uint8))
            
            warp0 = extra_info['warped_img0'][j:j+1]
            warp1 = extra_info['warped_img1'][j:j+1]
            diff.append([F.mse_loss(warp0,warp1).detach().cpu().item(), cri])
            
            psnr.append(cri)

            init_psnr.append(init_cri)
            idx += 1
        # if idx > 2000:
        #     break
    diff = np.array(diff)
    np.save('diff', diff)
    psnr = np.array(psnr).mean()
    print(f"PSNR: {psnr}")
    init_psnr = np.array(init_psnr).mean()
    print(f"INIT PSNR: {init_psnr}")

evaluate(model, val_data)

