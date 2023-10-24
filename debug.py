import cv2 as cv
import numpy as np
import os
from PIL import Image
# path1 = 'validation_final_stage0'
path1 = 'validation_output_tune_fulldata_xvfi'
path2 = 'validation_output_tune_fulldata_270'
param = dict()

im_list1 = sorted(os.listdir(path1))
im_list2 = sorted(os.listdir(path2))
h, w = 600, 400
for ima, imb in zip(im_list1, im_list2):
    gif1 = Image.open(os.path.join(path1, ima, 'out_.gif'))
    gif2 = Image.open(os.path.join(path2, imb, 'out_.gif'))
    while True:
        print(f'left:{ima}, right:{imb}')
        for frame in range(3):
            gif1.seek(frame)
            gif2.seek(frame)
            image1 = gif1.copy()
            image2 = gif2.copy()
            concatenated_width = image1.width + image2.width
            concatenated_height = image1.height
            concatenated_image = Image.new("RGB", (concatenated_width, concatenated_height))
            concatenated_image.paste(image1, (0, 0))
            concatenated_image.paste(image2, (image1.width, 0))
            res = np.abs(np.array(image1).astype(np.float32) - np.array(image2).astype(np.float32))
            res = (res - res.min()) / (res.max() - res.min()) * 255.
            res = cv.resize(res, (600, 400))
            concatenated_image = np.array(concatenated_image).astype(np.float32)
            concatenated_image = cv.resize(concatenated_image, (1200, 400))
            cv.imshow('1', concatenated_image.astype(np.uint8)[:,:,::-1])
            cv.imshow('res', res.astype(np.uint8))
            cv.waitKey(0)
        key = cv.waitKey(0) & 0xFF
        if key == ord('q'):
            break   
                
