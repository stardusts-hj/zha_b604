import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
import datetime
import logging
from Trainer import Model
from dataset_arbi import VimeoDatasetArbi
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]
def get_learning_rate(step):
    if step < args.warm_up:
        mul = step / args.warm_up
        return args.lr * mul
    else:
        mul = np.cos((step - args.warm_up) / (args.epoch * args.step_per_epoch - args.warm_up) * math.pi) * 0.5 + 0.5
        return (args.lr - 1e-6) * mul + 1e-6

def train(model, local_rank, batch_size, data_path, log_dir):


    step = 0
    nr_eval = 0
    dataset = VimeoDatasetArbi('train', data_path)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    
    # dataset_val = VimeoDatasetArbi('test', data_path)
    # val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=8)
    from xvfi_utils import X_Test
    test_set = X_Test(args, multiple=8, validation=False)
    val_data = DataLoader(test_set, batch_size=2, pin_memory=True, num_workers=8, drop_last=True)
    
    if local_rank == 0:
        writer = SummaryWriter(log_dir+'/logs', filename_suffix='train')
        
        logger = logging.getLogger('train')
        logger.addHandler(file_handler)
        msg = ['get configs'] + [f'{k} : {v} ' for k, v in args._get_kwargs()] + [f'iter per epoch: {args.step_per_epoch}'] + \
                [f'total iters: {args.step_per_epoch * args.epoch}'] + ['\n']
        logger.info('\n'.join(msg))
    
    if local_rank == 0:
        logger.info(f'training..., total epoch:{args.epoch:d}')
    start_time = time.time()
    ave_loss = 0
    if local_rank == 0:
        evaluate_xvfi(model, val_data, nr_eval, writer, logger)
    dist.barrier()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        # if local_rank == 0:
        #     evaluate_xvfi(model, val_data, nr_eval, writer, logger)
        for i, (imgs,emb_t)  in enumerate(train_data):
            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step)
            _, loss = model.update(imgs, gt, learning_rate, training=True, emb_t = emb_t)
            ave_loss += loss.item()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('train/learning_rate', learning_rate, step)
                writer.add_scalar('train/loss', loss, step)
            if step % 30 == 0 and local_rank == 0:
                total_time = time.time() - start_time
                time_sec_avg = total_time / (step - 0 + 1)
                eta_sec = time_sec_avg * (args.epoch * args.step_per_epoch - step - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                logger.info(f'epoch:{epoch} {i}/{args.step_per_epoch}, loss:{ave_loss / 10:.4e}, lr:{learning_rate:.4e}, ETA:{eta_str} ')
                ave_loss = 0
            step += 1
        nr_eval += 1
        if nr_eval % 3 == 0 and local_rank == 0:
            # evaluate(model, val_data, nr_eval, writer, logger)
            evaluate_xvfi(model, val_data, nr_eval, writer, logger)
        if nr_eval % 10 == 0 and local_rank == 0:
            model.save_model(f'{log_dir}/{nr_eval}.pkl', local_rank)
            
        dist.barrier()

def evaluate(model, val_data, nr_eval, writer, logger):

    psnr = []
    for i, (imgs, emb_t) in enumerate(val_data):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False, emb_t = emb_t)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))

    psnr = np.array(psnr).mean()
    logger.info(f"Test Vimeo epoch: {nr_eval}, PSNR: {psnr}")
    # logging.INFO(str(nr_eval), psnr)
    writer.add_scalar('test_vimeo/psnr', psnr, nr_eval)

def evaluate_xvfi(model, val_data, nr_eval, writer, logger):
    
    psnr = []
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
            psnr.append(cri)

    psnr = np.array(psnr).mean()
    logger.info(f"Test XVFI epoch: {nr_eval}, PSNR: {psnr}")
    writer.add_scalar('test_xvfi/psnr', psnr, nr_eval)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_data_path', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='X4K1000FPS')
    parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
    
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--warm_up', default=2000, type=int)
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--world_size', type=int, default=4, help='world size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--data_path', type=str, default='vimeo/vimeo90k', help='data path of vimeo90k')
    parser.add_argument('--config', type=str, default='tune_2_stage_refine', help='path of configs')
    parser.add_argument('--log_dir', type=str, default=None, help='path of configs')
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(device_id)
    torch.cuda.set_device(device)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if not args.log_dir:
        log_dir = 'output_' + args.config
    else:
        log_dir = args.log_dir
    if not os.path.exists(log_dir) and args.local_rank == 0:
        os.makedirs(log_dir)
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=logging.INFO)
    if device_id == 0:
        logger = logging.getLogger('train')
        file_handler = logging.FileHandler(log_dir+'/log.txt', 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    model = Model(device_id, args.config)
    if args.ckpt :
        model.load_model(args.ckpt, device_id)
        logger = logging.getLogger('train')
        logger.info(f'load ckpt to device: {device}  from {args.ckpt}')
    train(model, device_id, args.batch_size, args.data_path, log_dir = log_dir)
        
