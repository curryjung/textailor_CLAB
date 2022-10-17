import os
import numpy as np
import math
import cv2
from torchvision import transforms

path1 = '/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset/preprocessed/' # real image        

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64,256)), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])