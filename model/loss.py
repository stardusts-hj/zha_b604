import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import torchvision.models as models
import torchvision

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
        
    def forward(self, input, target, mask=None):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        if mask is None:
            return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        else:
            return sum((torch.nn.functional.l1_loss(a, b, reduction='none') * mask).mean() for a, b in zip(pyr_input, pyr_target))

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
        return (self.hamming(img0, img1) * self.valid_mask(img0, 1)).mean()

class Charbonnier_Loss(nn.Module):
    def __init__(self):
        super(Charbonnier_Loss, self).__init__()

    def forward(self, img0, img1, mask=None):
        if mask is None:
            loss = (((img0 - img1) ** 2 + 1e-6) ** 0.5).mean()
        else:
            loss = ((((img0 - img1) ** 2 + 1e-6) ** 0.5) * mask).mean()
        return loss
    

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

## perceptual loss:  note X ad Y is not the same
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2,7,16,25,34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """

    def __init__(self, feature_layer=[2,7,16,25,34], weights=[0.1,0.1,1.0,1.0,1.0], lossfn_type='l1', use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

class Smoothloss(nn.Module):
    def __init__(self):
        super(Smoothloss, self).__init__()
        self.l1 = nn.L1Loss(reduction='none')
    def forward(self, flow, mask=None):
        fw = flow[:, :2, :, :]
        bw = flow[:, 2:4, :, :]
        if mask is None:
            smooth_fwd = self.l1(fw[:,:,:,:-1], fw[:,:,:,1:]) + self.l1(fw[:,:,:-1,:], fw[:,:,1:,:])
            smooth_bwd = self.l1(bw[:,:,:,:-1], bw[:,:,:,1:]) + self.l1(bw[:,:,:-1,:], bw[:,:,1:,:])
        else:
            smooth_fwd = (self.l1(fw[:,:,:,:-1], fw[:,:,:,1:]) + self.l1(fw[:,:,:-1,:], fw[:,:,1:,:])) * mask
            smooth_bwd = (self.l1(bw[:,:,:,:-1], bw[:,:,:,1:]) + self.l1(bw[:,:,:-1,:], bw[:,:,1:,:])) * mask
        return smooth_fwd.mean() + smooth_bwd.mean()

class set_smoothness_loss(nn.Module):
    def __init__(self, weight=150.0, edge_aware=True):
        super(set_smoothness_loss, self).__init__()
        self.edge_aware = edge_aware
        self.weight = weight ** 2

    def forward(self, flow, img):
        img_gh = torch.mean(torch.pow((img[:, :, 1:, :] - img[:, :, :-1, :]), 2), dim=1, keepdims=True)
        img_gw = torch.mean(torch.pow((img[:, :, :, 1:] - img[:, :, :, :-1]), 2), dim=1, keepdims=True)

        weight_gh = torch.exp(-self.weight * img_gh)
        weight_gw = torch.exp(-self.weight * img_gw)

        flow_gh = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        flow_gw = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        if self.edge_aware:
            return (torch.mean(weight_gh * flow_gh) + torch.mean(weight_gw * flow_gw)) * 0.5
        else:
            return (torch.mean(flow_gh) + torch.mean(flow_gw)) * 0.5

if __name__== '__main__':
    x = torch.randn(1,3,256,256).cuda()
    y = torch.randn(1,3,256,256).cuda()
    loss = PerceptualLoss().cuda()
    print(loss(x, y))