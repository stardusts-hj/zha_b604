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
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
padder = InputPadder((1080, 1920), divisor=32)
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="final_stage0")
parser.add_argument('--test_data_path', type=str, default='test')
parser.add_argument('--dataset', type=str, default='X4K1000FPS')
parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Model(-1, args.model)
# model.load_model('output_baselinev2/28000.pkl', -1)
model.load_model('300.pkl', -1)
model.eval()
model.device()

# dataset_val = VimeoDatasetArbi('test', './vimeo/vimeo90k/')
# val_data = DataLoader(dataset_val, batch_size=2, pin_memory=True, num_workers=0)

from xvfi_utils import X_Test
test_set = X_Test(args, multiple=4, validation=False)
val_data = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=0, drop_last=True)


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
    global min_psnr
    for i, (frames, t_value, scene_name, frameRange) in enumerate(val_data):
        # imgs = F.interpolate(imgs, (1088, 1920))
        b, _, _, _ = frames.shape
        imgs = frames
        emb_t = t_value.reshape(b,1,1,1)
        imgs = imgs.to(device, non_blocking=True)
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, extra_info = model.update(imgs, gt, training=False, emb_t = emb_t)
        for j in range(gt.shape[0]):
            save_dict = {}
            cri = -10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item())
            init_cri = -10 * math.log10(((gt[j] - extra_info['init_pred'][j]) * (gt[j] - extra_info['init_pred'][j])).mean().cpu().item())
            save_dict['pred'] = tensor_to_image(pred[j])[:,:,::-1]
            save_dict['gt'] = tensor_to_image(gt[j])[:,:,::-1]
            save_dict['init_pred'] = tensor_to_image(extra_info['init_pred'][j])[:,:,::-1]
            for k, v in extra_info.items():
                if k in save_dict:
                    continue
                if 'flow' in k :
                    f1 = v[j][:2]
                    f1 = flow_to_image(f1).cpu().numpy().transpose(1,2,0)
                    save_dict[k + '_forward'] = f1[:,:,::-1]
                    f2 = v[j][2:4]
                    f2 = flow_to_image(f2).cpu().numpy().transpose(1,2,0)
                    save_dict[k + '_abckward'] = f2[:,:,::-1]
                else:
                    save_dict[k] = tensor_to_image(v[j])[:,:,::-1]
            save_images(save_dict, 3, 6, os.path.join('results', f'{idx}_{cri:.4f}_{init_cri:.4f}.png'))
            idx += 1

            psnr.append(cri)
            init_psnr.append(init_cri)

    psnr = np.array(psnr).mean()
    print(f"PSNR: {psnr}")
    init_psnr = np.array(init_psnr).mean()
    print(f"INIT PSNR: {init_psnr}")


def tensor_to_image(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    image = np.clip(image, 0 , 1)

    return (image * 255.).astype(np.uint8)


def save_images(save_dict, rows, cols, save_path):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    for i, (k, v) in enumerate(save_dict.items()):
        ax = axes.flat[i]
        ax.imshow(v)
        ax.set_title(k, fontsize=5)
        ax.axis('off')
    axes.flat[-1].set_title('pred', fontsize=5)
    axes.flat[-1].imshow(save_dict['pred'])
    axes.flat[-2].set_title('init_pred', fontsize=5)
    axes.flat[-2].imshow(save_dict['init_pred'])
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.005)
    plt.savefig(save_path, dpi=300)

evaluate(model, val_data)