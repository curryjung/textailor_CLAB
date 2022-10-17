import os
import shutil
from PIL import Image

root_path = "/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset/preprocessed"
# root_path = "/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset/cropped/train"
# root_path = "/mnt/f06b55a9-977c-474a-bed0-263449158d6a/textailor_CLAB/style_reshape/0"
#current directory
cur_dir = os.getcwd()
img_dir = os.path.join(cur_dir, 'style_reshape')
img_list = os.listdir(root_path)


total_n = 50000

#make a new directory
dir_0 = os.path.join(img_dir, '0')
dir_1 = os.path.join(img_dir, '1')

print('total number of images: ', total_n)

if not os.path.exists(dir_0):
    os.makedirs(dir_0)
if not os.path.exists(dir_1):
    os.makedirs(dir_1)


for img_name in img_list[:total_n//2]:

    extension = os.path.splitext(img_name)[1]
    if extension == '.png':
        img = Image.open(os.path.join(root_path, img_name))
        img = img.resize((256,64))
        img.save(os.path.join(dir_0, img_name))



for img_name in img_list[total_n//2:total_n]:

    extension = os.path.splitext(img_name)[1]
    if extension == '.png':
        img = Image.open(os.path.join(root_path, img_name))
        img = img.resize((256,64))
        img.save(os.path.join(dir_1, img_name))




# python -m pytorch_fid ./style_reshape/0 ./style_reshape/1
