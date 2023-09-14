from model.loss import *
import cv2 as cv
import numpy as np
import torch
import logging
import argparse
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=30000, type=int)
    parser.add_argument('--warm_up', default=200, type=int)
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--world_size', type=int, default=4, help='world size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--data_path', type=str, default='data', help='data path of vimeo90k')
    parser.add_argument('--config', type=str, default='baseline', help='path of configs')
    args = parser.parse_args()
    msg = ['start logging ']+[f'{k} : {v} ' for k, v in args._get_kwargs()]
    print('\n'.join(msg))