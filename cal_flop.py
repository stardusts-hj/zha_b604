import os
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
import importlib
from model.Full_Model import FullModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="baseline_more_loss")
args = parser.parse_args()

torch.cuda.current_device()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cf = importlib.import_module('configs.' + args.model)
net = FullModel(**cf.MODEL_CONFIG).cuda().eval()
size = (3, 1080, 1920)
I0_ = (torch.randn(*size).cuda() / 255.).unsqueeze(0)
I2_ = (torch.randn(*size).cuda() / 255.).unsqueeze(0)
# input_data = torch.randn((1, 3, args.crop_size[0], args.crop_size[1])).to(device)

from benchmark.utils.padder import InputPadder
padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)
print(I0_.shape)

with torch.no_grad():
    flops, params = get_model_complexity_info(net, (6, *I0_.shape[2:]), as_strings=True, print_per_layer_stat=False)
    print("{:>16s} : {:s}".format("FLOPs", flops))
    print("{:>16s} : {:s}".format("#Params", params))
# def input_constructor(input_res):
#     batch = torch.Tensor(1, input_res[0]*2, *input_res[1:]).to(device)
#     return (batch)
# with torch.no_grad():
#     flops = get_model_flops(net, tuple(I0_.shape[1:]), print_per_layer_stat=False, input_constructor=input_constructor)
#     print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops / 1e9))