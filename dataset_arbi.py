import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from random import shuffle
cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class VimeoDatasetArbi(Dataset):
    def __init__(self, dataset_name, path, batch_size=32, model="RIFE"):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = model
        self.h = 256
        self.w = 448
        self.data_root = path
        self.order = list(range(7))
        self.image_root = os.path.join(self.data_root, 'sequence')
        train_fn = os.path.join(self.data_root, 'sub_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sub_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()                                                    
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

#     def getimg(self, index):
#         imgpath = os.path.join(self.image_root, self.meta_data[index])
#         imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
#         img0 = cv2.imread(imgpaths[0])
#         gt = cv2.imread(imgpaths[1])
#         img1 = cv2.imread(imgpaths[2])
#         return img0, gt, img1
    
    def getimg_arbi(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        filename = os.listdir(imgpath)
        filename.sort()
        shuffle(self.order)
        order  = self.order[0:3]
        order.sort()
        imgpaths = [ imgpath + '/' + filename[order[0]], imgpath + '/' + filename[order[1]], imgpath + '/' + filename[order[2]] ]
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = (order[1] - order[0]) * 1.0 / (order[2] - order[0])
        
        return img0,gt,img1,timestep
    
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg_arbi(index)
                
        if 'train' in self.dataset_name:
            img0, gt, img1 = self.aug(img0, gt, img1, 256, 256)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
                timestep = 1 - timestep
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1,1,1)
        return torch.cat((img0, img1, gt), 0), timestep

    
if __name__ == '__main__':
    data_path = r"/data/lbh666/VFI"
    batch_size = 100
    dataset = VimeoDatasetArbi('train', data_path)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True)
    for i, imgs in enumerate(train_data):
        imgs = imgs
        
        img = imgs[0] # [8, 9, 256, 448]      img0, gt, img1
        timestep = imgs[1]  # [8]     timestep
        #
        print(i)