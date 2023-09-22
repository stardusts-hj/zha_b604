import numpy as np
from model.loss import Ternary
import torch
device = "cuda"
loss = Ternary(device)

x1 = torch.randn((1,3,256,256)).to(device)
x2 = torch.randn((1,3,256,256)).to(device)

print(loss(x1, x2))