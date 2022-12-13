import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
import os

style_dir = '/home/jaeseok/style-aware-discriminator/testphotos/style/'
content_dir = '/home/jaeseok/style-aware-discriminator/testphotos/content/'

#val_dir = '/home/jaeseok/style-aware-discriminator/shadow_datasets/val/'
#val_save = '/home/jaeseok/style-aware-discriminator/crop_shadow_datasets/val/'

style_save = '/home/jaeseok/style-aware-discriminator/test/style/'
content_save = '/home/jaeseok/style-aware-discriminator/test/content/'

style_img = os.listdir(style_dir)
#content_img = os.listdir(content_dir)
print(style_img)
for k in style_img:
    #content_fin_dir = content_dir + "/" + k[:-4] + "_mask.jpg"
    content_fin_dir = content_dir + "/" + k
    style_fin_dir = style_dir + "/" + k

    content_image = cv2.imread(content_fin_dir)
    style_image = cv2.imread(style_fin_dir)

    image_gray = cv2.imread(content_fin_dir, cv2.IMREAD_GRAYSCALE)

    blur = cv2.GaussianBlur(image_gray, ksize=(3,3), sigmaX=0)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blur, 10, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _  = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_xy = np.array(contours)

    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)

    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)

    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min

    if w*h>0:
        mask_img = content_image[y:y+h, x:x+w]
        mask_dir = content_save + "/" + str(k[:-4]) + "_mask.jpg"
        cv2.imwrite(mask_dir, mask_img)

        color_img = style_image[y:y+h, x:x+w]
        color_dir = style_save + "/" + str(k)
        cv2.imwrite(color_dir, color_img)
    else: # w*h==0일 때
        mask_img = content_image
        mask_dir = content_save + "/" + str(k[:-4]) + "_mask.jpg"
        cv2.imwrite(mask_dir, mask_img)

        color_img = style_image
        color_dir = style_save + "/" + str(k)
        cv2.imwrite(color_dir, color_img)
        print(color_dir)