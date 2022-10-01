#when user inputs index of a grid image, this script will visualize or save the grid image

import numpy as np
from PIL import Image
import os
import sys


def get_image(img_path, row, col, image_size=(64,256), grid_width=1):
    left_point = grid_width + (image_size[1] + grid_width) * col
    right_point = left_point + image_size[1]
    top_point = grid_width + (image_size[0] + grid_width) * row
    bottom_point = top_point + image_size[0]

    img = Image.open(img_path)
    cropped_img = img.crop([left_point, top_point, right_point, bottom_point])
    
    return cropped_img



file_name = "/home/sy/textailor_CLAB/run_js/optimize_w_input/0.png"
output_path = "/home/sy/textailor_CLAB/doc/w_optim_sample.png"

img = get_image(file_name, 0, 0)
img.save(output_path)


