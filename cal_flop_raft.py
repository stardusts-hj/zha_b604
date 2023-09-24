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

size = (3, 1080, 1920)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
I0_ = (torch.randn(*size).cuda() / 255.).unsqueeze(0)
I2_ = (torch.randn(*size).cuda() / 255.).unsqueeze(0)

from benchmark.utils.padder import InputPadder
padder = InputPadder(I0_.shape, divisor=8)
I0_, I2_ = padder.pad(I0_, I2_)

def input_constructor(input_res):
    batch = torch.Tensor(1, 3, *I0_.shape[2:]).to(device)
    return {'image1':batch,
            'image2':batch}


#------------ load RAFT ------------------#

from torchvision.models.optical_flow import raft_small
from torchvision.models.optical_flow import Raft_Small_Weights

weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights, progress=False).to(device)
model = model.eval()

with torch.no_grad():
    flops, params = get_model_complexity_info(model, tuple(I2_.shape[2:]), input_constructor=input_constructor, as_strings=True, print_per_layer_stat=False)
    print("{:>16s} : {:s}".format("FLOPs", flops))
    print("{:>16s} : {:s}".format("#Params", params))