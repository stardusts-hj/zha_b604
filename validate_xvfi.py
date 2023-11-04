import os, math
import torch
import argparse
from Trainer import Model
from torch.utils.data import DataLoader
import numpy as np
import cv2 as cv
from imageio import mimsave
import torch.nn.functional as F
from benchmark.utils.padder import InputPadder
from torchvision.utils import flow_to_image
import torch.nn.functional as F

padder = InputPadder((1080, 1920), divisor=32)
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="rep_final_stage0")
parser.add_argument('--test_data_path', type=str, default='test')
parser.add_argument('--dataset', type=str, default='X4K1000FPS')
parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Model(-1, args.model)
ckpt = 'rep_dbb_32_300.pkl'
print(ckpt)
model.load_model(ckpt, -1)
# model.load_model(f'output_tune_2_stage_refine/20.pkl', -1)
model.eval()
model.device()

# dataset_val = VimeoDatasetArbi('test', './vimeo/vimeo90k/')
# val_data = DataLoader(dataset_val, batch_size=2, pin_memory=True, num_workers=0)

from xvfi_utils import X_Test
test_set = X_Test(args, multiple=8, validation=False)
val_data = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=0, drop_last=True)


def process(imgs):
    result = []
    for im in imgs:
        im = im.detach().cpu().numpy().transpose(1,2,0)*255.
        result.append(im.astype(np.uint8))
    
    return result
from model.loss import Smoothloss, Charbonnier_Loss, LapLoss
sm_loss = Smoothloss()
char_loss = Charbonnier_Loss()
lap_loss = LapLoss()
min_psnr = 100
def evaluate(model, val_data):
    idx = 0
    psnr = []
    init_psnr = []
    diff = []
    upper_bound = []
    global min_psnr
    for i, (frames, t_value, scene_name, frameRange) in enumerate(val_data):
        # imgs = F.interpolate(imgs, (1088, 1920))
        b, _, _, _ = frames.shape
        imgs = frames
        emb_t = t_value.reshape(b,1,1,1).to(device)
        imgs = imgs.to(device, non_blocking=True)
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, extra_info = model.update(imgs, gt, training=False, emb_t = emb_t)

        # pred = extra_info['init_pred'] * (1 - extra_info['mask']) + pred * extra_info['mask']
        pred = (pred + extra_info['init_pred']) / 2.
        for j in range(gt.shape[0]):
            cri = -10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item())
            init_cri = -10 * math.log10(((gt[j] - extra_info['init_pred'][j]) * (gt[j] - extra_info['init_pred'][j])).mean().cpu().item())
            # i1, i2, gtt, predd, init_pred= process([imgs[j, 0:3], imgs[j,3:6], gt[j], pred[j], extra_info['init_pred'][j]])
            # # overlap = (gtt.astype(np.float32) + predd.astype(np.float32)) / 2.
            # t = float(emb_t[j].detach().cpu())
            # base_path = os.path.join(f'validation_{ckpt}', f'{i}_{cri:.4f}_{init_cri:.4f}')
            # if not os.path.exists(base_path):
            #    os.makedirs(base_path)
            # cv.imwrite(os.path.join(base_path, 'left.png'), i1)
            # cv.imwrite(os.path.join(base_path, 'right.png'), i2)
            # cv.imwrite(os.path.join(base_path, 'gt.png'), gtt)
            # cv.imwrite(os.path.join(base_path, 'pred.png'), predd)
            # cv.imwrite(os.path.join(base_path, 'init_pred.png'), init_pred)
            # # cv.imwrite(os.path.join(base_path, 'overlap.png'), overlap.astype(np.uint8))
            # tmps = [i1[:,:,::-1], predd[:,:,::-1], i2[:,:,::-1]]
            # mimsave(os.path.join(base_path, f'out_.gif'), tmps, duration=int(1/3*1000))
            # tmps = [gtt[:,:,::-1], predd[:,:,::-1]]
            # mimsave(os.path.join(base_path, f'overlap.gif'), tmps, duration=int(1/2*1000))
            # f1 = extra_info['flow'][:, :2]
            # f2 = extra_info['flow'][:, 2:4]
            # f1 = flow_to_image(f1)
            # im = padder.unpad(f1)[0].detach().cpu().numpy().transpose(1, 2, 0)
            # cv.imwrite(os.path.join(base_path, f'flow_forward.png'), im.astype(np.uint8))
            # f2 = flow_to_image(f2)
            # im = padder.unpad(f2)[0].detach().cpu().numpy().transpose(1, 2, 0)
            # cv.imwrite(os.path.join(base_path, f'flopw_backward.png'), im.astype(np.uint8))
            
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

            # warp0 = extra_info['warped_img0'][j:j+1]
            # warp1 = extra_info['warped_img1'][j:j+1]

            psnr.append(cri)
            init_psnr.append(init_cri)
            upper_bound.append(max(cri, init_cri))
            idx += 1

    diff = np.array(diff)
    np.save('diff_xvfi', diff)
    psnr = np.array(psnr).mean()
    print(f"PSNR: {psnr}")
    init_psnr = np.array(init_psnr).mean()
    print(f"INIT PSNR: {init_psnr}")
    upper_bound = np.array(upper_bound).mean()
    print(f"upper bound PSNR: {upper_bound}")

evaluate(model, val_data)

# for idx in ['10', '20', '30', '40', '50', '60']:
#     print('current epoch', idx)
#     model.net.load_state_dict(torch.load(f'output_tune_2_stage_refine/{idx}.pkl'), strict=False)
#     # model.load_model(f'output_tune_2_stage_refine/{idx}.pkl', -1)
#     model.eval()
#     model.device()
#     evaluate(model, val_data)