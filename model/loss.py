import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

class Ternary(nn.Module):
    def __init__(self, device):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)

## charbonnierloss  
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

## lpis loss
class LPIPS(nn.Module):
    def __init__(self, net='alex', cuda=True):
        super(LPIPS, self).__init__()
        self.net = net
        self.model = lpips.LPIPS(net=net)
        if cuda:
            self.model.cuda()

    def forward(self, im0, im1):
        return self.model.forward(im0, im1)

"""
# cuda
loss = LPIPS()
fn = loss(img1, img2)
"""

## cycle consistency loss
"""
https://openaccess.thecvf.com/content_ICCV_2019/papers/Reda_Unsupervised_Video_Interpolation_Using_Cycle_Consistency_ICCV_2019_paper.pdf
https://github.dev/NVIDIA/unsupervised-video-interpolation
"""
class CCLoss(nn.Module):
    pass

## optical smoothness loss
"""
flow [:, :, :, :]
https://github.dev/avinashpaliwal/Super-SloMo
"""
class Smoothloss(nn.Module):
    def __init__(self):
        super(Smoothloss, self).__init__()
        self.l1 = nn.L1Loss(reduce='mean')
    def forward(self, flow):
        fw = flow[:, :2, :, :]
        bw = flow[:, 2:, :, :]
        smooth_fwd = self.l1(fw[:,:,:,:-1], fw[:,:,:,1:]) + self.l1(fw[:,:,:-1,:], fw[:,:,1:,:])
        smotth_bwd = self.l1(bw[:,:,:,:-1], bw[:,:,:,1:]) + self.l1(bw[:,:,:-1,:], bw[:,:,1:,:])
        return smooth_fwd + smotth_bwd


## warping loss
"""
输入: img1, img3, gt, predf, predb, pred1, pred3
img1 和 img3, t ,输出 predf 预测 中间帧 gt
img3 和 img1, 1-t, 输出 predb 预测中间帧 gt
img1 和 t=1 , 输出 pred3 预测 img3
img3 和 t=1 , 输出 pred1 预测 img1
https://github.dev/avinashpaliwal/Super-SloMo
""" 
class WarpingLoss(nn.Module):
    def __init__(self):
        super(WarpingLoss, self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self, img1, img3, gt, predf, predb, pred1, pred3):
        warploss = self.L1(predf, gt) + self.L1(predb, gt) + self.L1(pred1, img1) + self.L1(pred3, img3)
        return warploss
    



# loss parameters
"""
charbonnierloss   lpipsloss   warpingloss  smoothloss     cc
0.8               0.005         0.4         1 
"""


if __name__ == '__main__':
    device = 'cpu'
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (3, 3, 256, 256))).float().to(device)
    ternary_loss = Ternary(device)
    print(ternary_loss(img0, img1).shape)