import os
import math
import numpy as np
import cv2
import glob
import os
# import PIL
from PIL import Image
from skimage import data, util,io
# from skimage.metrics import peak_signal_noise_ratio
import skimage
# 读取原图



#这个程序可以用来生成bicubic的图像以及计算psnr和ssim

# im = Image.open('C:/Users/wslgq/Desktop/img/x3/bicubic_img_030.png')   #读取原始图像
# print(im.size)
# # im_resize0 = im.resize((im.width*2, im.height*2), Image.BILINEAR)    #3种上采样方式
# # print(im_resize0.size)
# im_resize1 = im.resize((im.width*3, im.height*3), Image.BICUBIC)
# print(im_resize1.size)
# im_resize2 = im.resize((im.width*2, im.height*2), Image.ANTIALIAS)
# print(im_resize2.size)

# path = r"./001.jpg"
# img = Image.open(path)
# im_resize1.save("C:/Users/wslgq/Desktop/img/x3/bicu_img_030.png")
# im_resize2.save("2.jpg")
# im_resize0.save("0.jpg")








def main():
    im1 = io.imread("bicu_butterfly.png")
    im2 = io.imread("hr_butterfly.png")
    psnr = skimage.measure.compare_psnr(im1, im2, 255)
    ssim = skimage.measure.compare_ssim(im1, im2, data_range=255,win_size = 3, multichannel=True)
    print(psnr,'psnr')
    print(ssim,'ssim')
    pass
if __name__ == '__main__':
    main()


