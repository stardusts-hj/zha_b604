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


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="baseline")

# specify dirs
parser.add_argument("--save-dir", type=str, default="log")

# specify test case
parser.add_argument("--repeat", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--model", type=str, default="baseline_lap_char")


parser.add_argument('--n', default=10, type=int)
args = parser.parse_args()



"""
SETUP DIRS
"""
pathlib.Path(os.path.join(args.save_dir, "results")).mkdir(parents=True, exist_ok=True)

"""
SETUP LOGGER
"""
util_logger.logger_info("IACC-VFI", log_path=os.path.join(args.save_dir, f"Submission.txt"))
logger = logging.getLogger("IACC-VFI")

"""
BASIC SETTINGS
"""
torch.cuda.current_device()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
"""
LOAD MODEL
"""

'''==========import from our code=========='''
sys.path.append('.')
import configs.baseline_lap as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

model = Model(-1, args.model)
model.eval()
model.device()

pipeline = model.net.multi_inference

network = model.net
'''================== end ======================'''
    

number_parameters = sum(map(lambda x: x.numel(), network.parameters()))
logger.info('>>>>>>>>>>>>>>>>>>>>>>>> model name: {} <<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model_name))
logger.info(f"{args.n}X interpolate")

"""
SETUP RUNTIME
"""
test_results = OrderedDict()
test_results["runtime"] = []
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
    
"""
TESTING
"""  
size = (3, 1920, 1080)
I0_ = (torch.randn(*size).cuda() / 255.).unsqueeze(0)
I2_ = (torch.randn(*size).cuda() / 255.).unsqueeze(0)
# input_data = torch.randn((1, 3, args.crop_size[0], args.crop_size[1])).to(device)
padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

def input_constructor(input_res):
    batch = torch.Tensor(1, input_res[0]*2, *input_res[1:]).to(device)
    return (batch)

# GPU warmp up
print("Warm up ...")
with torch.no_grad():
    for _ in range(20): # 50
        out = pipeline(torch.cat([I0_, I2_], 1), time_list=[torch.Tensor([(i+1)*(1./args.n)]).cuda() for i in range(args.n - 1)])
    print('input data shape: {} \n model out shape: {}'.format(I0_.shape, out[0].shape))
print("Start timing ...")
torch.cuda.synchronize()

with torch.no_grad():
    for _ in tqdm(range(args.repeat)):
        start.record()
        _ = pipeline(torch.cat([I0_, I2_], 1), time_list=[torch.Tensor([(i+1)*(1./args.n)]).cuda() for i in range(args.n - 1)])
        end.record()

        torch.cuda.synchronize()

        test_results["runtime"].append(start.elapsed_time(end))  # milliseconds

    ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"])
    logger.info('------> Average runtime of ({}) is : {:.6f} ms'.format(args.model_name, ave_runtime / args.batch_size))
    logger.info('------> FPS of ({}) is : {:.2f} fps'.format(args.model_name, 1 / (ave_runtime / args.batch_size / 1000.)))

    # flops = get_model_flops(network, tuple(I0_.shape[1:]), print_per_layer_stat=False, input_constructor=input_constructor)
    # flops = flops / 1e9
    # logger.info('thops')
    # logger.info("{:>16s} : {:.4f} [G]".format("FLOPs", flops))

    # num_parameters = sum(map(lambda x: x.numel(), network.parameters()))
    # num_parameters = num_parameters / 1e6
    # logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(network, (6, *I0_.shape[2:]), as_strings=True, print_per_layer_stat=False)
    # flops = flops / 1e9
    # params = params / 1e6
    logger.info('ptflops')
    logger.info("{:>16s} : {:s}".format("FLOPs", flops))
    logger.info("{:>16s} : {:s}".format("#Params", params))


