import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

import torch
import numpy as np

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

def calc_metrics_common(img1, img2, crop_border=8, test_Y=True):

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2


    if im1_in.ndim == 3:
        # cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border, :]
        # cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_im1 = im1_in[:, crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[:, crop_border:-crop_border, crop_border:-crop_border]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))


    cropped_im2 = torch.from_numpy(cropped_im2)
    cropped_im1 = torch.from_numpy(cropped_im1)
    cropped_im1 = cropped_im1.type(torch.FloatTensor)
    cropped_im2 = cropped_im2.type(torch.FloatTensor)
    d = loss_fn_alex(cropped_im1, cropped_im2)

    return d
if __name__ == "__main__":
    import os
    import glob
    from PIL import Image
    GT_path = '/data/CelebA/test/HR/'
    SR_path = '/comparison_results/x4/nlsn'
    files_GT = sorted(glob.glob(os.path.join(GT_path, "*.png")))
    files_SR = sorted(glob.glob(os.path.join(SR_path, "*.png")))
    lpips_sum = 0

    for i in range(len(files_GT)):

        basename = os.path.basename(files_GT[i])
        print(basename)
        HR = Image.open(files_GT[i])
        SR = Image.open(os.path.join(SR_path, basename))
        print(os.path.join(SR_path, basename))
        SR = np.array(SR)
        HR = np.array(HR)
        # print(np.max(SR))
        lpips = calc_metrics_common(SR, HR, crop_border=8)

        lpips_sum = lpips_sum + lpips
        print(lpips)

    print(lpips_sum/len(files_GT))
