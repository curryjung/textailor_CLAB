from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize
import numpy as np
import os


Psnr = []
Ssim = []
index = 1

files = os.listdir('/mnt/f06b55a9-977c-474a-bed0-263449158d6a/stargan-v2/calc_img/result/')
for file_name in sorted(files):
    # 원본
    im1 = imread('/mnt/f06b55a9-977c-474a-bed0-263449158d6a/stargan-v2/calc_img/result/' + file_name)
    #im1 = resize(im1, (256, 256))
    # 예측
    im2 = imread('/mnt/f06b55a9-977c-474a-bed0-263449158d6a/stargan-v2/calc_img/style/' + file_name)
    #im2 = resize(im2, (256, 256))

    # 계산
    try:
        Ssim.append(ssim(im1, im2, multichannel=True))
    except:
        im1 = np.stack((im1,)*3, axis=-1)
    Psnr.append(psnr(im1, im2))

    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "PSNR: %.4f" % round(np.mean(Psnr), 4),
            "SSIM: %.4f" % round(np.mean(Ssim), 4),
        )
    index += 1


print("FINAL",
    "PSNR: %.4f" % round(np.mean(Psnr), 4),
    "SSIM: %.4f" % round(np.mean(Ssim), 4)
)