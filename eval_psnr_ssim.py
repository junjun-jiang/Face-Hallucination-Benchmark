import time
from skimage import measure
import torch
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import math

import cv2

# PSNR & SSIM in DIC
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    #
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    mse = np.mean((img1_np - img2_np)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calc_metrics_common(img1, img2, crop_border=8, test_Y=True):
    #
    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2

    if im1_in.ndim == 3:
        cropped_im1 = im1_in[:, crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[:, crop_border:-crop_border, crop_border:-crop_border]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    print("cropped: ", cropped_im1.shape, cropped_im2.shape)
    psnr = calc_psnr(cropped_im1, cropped_im2)

    ssim = measure.compare_ssim(cropped_im1, cropped_im2, data_range=255)

    return psnr, ssim


if __name__ == "__main__":
    import os
    import glob
    from PIL import Image
    GT_path = 'data/CelebA/test/HR/'
    SR_path = '/comparison_results/x8/urdgn'
    files_GT = sorted(glob.glob(os.path.join(GT_path, "*.png")))
    files_SR = sorted(glob.glob(os.path.join(SR_path, "*.png")))
    psnr_sum = 0
    ssim_sum = 0

    for i in range(len(files_GT)):
        SR = Image.open(files_SR[i])
        HR = Image.open(files_GT[i])
        SR = np.array(SR)
        HR = np.array(HR)
        # print(np.max(SR))
        psnr_current, ssim_current = calc_metrics_common(SR, HR, crop_border=8)
        psnr_sum = psnr_sum + psnr_current
        ssim_sum = ssim_sum + ssim_current

    print(psnr_sum/len(files_GT), ssim_sum/len(files_GT))

