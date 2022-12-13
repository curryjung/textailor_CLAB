import os

file_path = '/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/4paired_20k/imagesA_mask'
file_names = os.listdir(file_path)

for name in file_names:
    src = os.path.join(file_path, name)
    dst = name[:5] + '_mask.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)