import numpy as np
import torch
from model.warplayer import warp_pad_zero
import cv2 as cv
flow = torch.ones((1, 2, 400, 400)) * 4
img = torch.ones((1,3, 400, 400))

warpped = warp_pad_zero(img.cuda(), flow.cuda())

warpped = warpped.squeeze().cpu().numpy().transpose(1,2,0)

cv.imshow('1', warpped)
cv.waitKey(0)