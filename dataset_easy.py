from io import BytesIO
import os
import lmdb
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import copy
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw

class tet_ganA(Dataset):
    def __init__(self, img_folder, label_path=None, gray_text_folder=None,train=True, content_resnet=False, transform=None, generate=False, fake_img_folder=None):
        assert os.path.exists(img_folder), "img_folder does not exist"
        
        self.img_folder = img_folder
        self.label_path = label_path
        self.gray_text_folder = gray_text_folder
        self.img_list = os.listdir(img_folder)

        if gray_text_folder is not None:
           self.gray_list = os.listdir(gray_text_folder)
        if label_path is not None:
            assert os.path.exists(label_path), "label_path does not exist"
            self.label_dic = self.load_label(label_path)
        else:
            self.label_dic = None

        self.generate = generate
        self.fake_img_folder = fake_img_folder
        self.train = train
        self.content_resnet = content_resnet
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64,256)), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

        self.gray_transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64,256)), 
                transforms.Normalize((0.5,), (0.5,))
            ])

    def load_label(self, label_path):
        with open(label_path, 'r') as f:
            data = f.readlines()
        data_split = [x.strip().split() for x in data]
        
        return data_split

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        
        # get style image
        img_name = self.img_list[index]
        img_id = int(img_name.split('.')[0])
        
        style_img = cv2.imread(os.path.join(self.img_folder, img_name), cv2.IMREAD_COLOR)
        style_img = self.transform(style_img)

        if self.fake_img_folder is not None:
            fake_img = cv2.imread(os.path.join(self.fake_img_folder, img_name), cv2.IMREAD_GRAYSCALE)
            fake_img = self.gray_transform(fake_img)

        # get style_gray image : contains content of style image
        # gray_name = self.gray_list[index]
        if self.gray_text_folder is not None:
            gray_text_img_name = os.path.join(self.gray_text_folder, str(img_id)+".png")
            if self.content_resnet==False:
                style_gray_img = cv2.imread(gray_text_img_name, cv2.IMREAD_GRAYSCALE)
                style_gray_img = self.gray_transform(style_gray_img)
            else:
                style_gray_img = cv2.imread(gray_text_img_name)
                style_gray_img = self.transform(style_gray_img)
        else:
            style_gray_img = None

        # get style_label
        style_label= ''.join(self.label_dic[img_id][1])

        # get content_gray image : 
        random_idx = np.random.randint(0, len(self.img_list))
        random_img_name = self.img_list[random_idx]
        random_img_id = int(random_img_name.split('.')[0])
        random_gray_text_img_name = os.path.join(self.gray_text_folder, str(random_img_id)+".png")
        if self.content_resnet==False:
            content_gray_img = cv2.imread(random_gray_text_img_name, cv2.IMREAD_GRAYSCALE)
            content_gray_img = self.gray_transform(content_gray_img)
        else:
            content_gray_img = cv2.imread(random_gray_text_img_name)
            content_gray_img = self.transform(content_gray_img)

        # get content_label
        content_label= self.label_dic[random_img_id][1]

        if self.label_dic is not None:
            if self.generate==True:
                return style_img, style_gray_img, style_label, content_gray_img, content_label, img_id
            elif self.fake_img_folder is not None:
                return style_img, style_gray_img, style_label, content_gray_img, content_label, fake_img
            else:
                label = self.label_dic[img_id]
                return style_img, style_gray_img, style_label, content_gray_img, content_label
        else:
            return style_img, style_gray_img

class tet_ganB(Dataset):
    def __init__(self, img_folder, label_path=None, gray_text_folder=None,train=True, content_resnet=False, transform=None, generate=False, fake_img_folder=None):
        assert os.path.exists(img_folder), "img_folder does not exist"
        
        self.img_folder = img_folder
        self.label_path = label_path
        self.gray_text_folder = gray_text_folder
        self.img_list = os.listdir(img_folder)

        if gray_text_folder is not None:
           self.gray_list = os.listdir(gray_text_folder)
        if label_path is not None:
            assert os.path.exists(label_path), "label_path does not exist"
            self.label_dic = self.load_label(label_path)
        else:
            self.label_dic = None

        self.generate = generate
        self.fake_img_folder = fake_img_folder
        self.train = train
        self.content_resnet = content_resnet
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64,256)), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

        self.gray_transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64,256)), 
                transforms.Normalize((0.5,), (0.5,))
            ])

    def load_label(self, label_path):
        with open(label_path, 'r') as f:
            data = f.readlines()
        data_split = [x.strip().split() for x in data]
        return data_split

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        
        # get style image
        img_name = self.img_list[index]
        img_id = int(img_name.split('.')[0])
        
        style_img = cv2.imread(os.path.join(self.img_folder, img_name), cv2.IMREAD_COLOR)
        style_img = self.transform(style_img)

        if self.fake_img_folder is not None:
            fake_img = cv2.imread(os.path.join(self.fake_img_folder, img_name), cv2.IMREAD_GRAYSCALE)
            fake_img = self.gray_transform(fake_img)

        # get style_gray image : contains content of style image
        # gray_name = self.gray_list[index]
        if self.gray_text_folder is not None:
            gray_text_img_name = os.path.join(self.gray_text_folder, str(img_id)+".png")
            if self.content_resnet==False:
                style_gray_img = cv2.imread(gray_text_img_name, cv2.IMREAD_GRAYSCALE)
                style_gray_img = self.gray_transform(style_gray_img)
            else:
                style_gray_img = cv2.imread(gray_text_img_name)
                style_gray_img = self.transform(style_gray_img)
        else:
            style_gray_img = None

        # get style_label
        style_label= self.label_dic[img_id][2]

        # get content_gray image : 
        random_idx = np.random.randint(0, len(self.img_list))
        random_img_name = self.img_list[random_idx]
        random_img_id = int(random_img_name.split('.')[0])
        random_gray_text_img_name = os.path.join(self.gray_text_folder, str(random_img_id)+".png")
        if self.content_resnet==False:
            content_gray_img = cv2.imread(random_gray_text_img_name, cv2.IMREAD_GRAYSCALE)
            content_gray_img = self.gray_transform(content_gray_img)
        else:
            content_gray_img = cv2.imread(random_gray_text_img_name)
            content_gray_img = self.transform(content_gray_img)

        # get content_label
        content_label= self.label_dic[random_img_id][2]

        if self.label_dic is not None:
            if self.generate==True:
                return style_img, style_gray_img, style_label, content_gray_img, content_label, img_id
            elif self.fake_img_folder is not None:
                return style_img, style_gray_img, style_label, content_gray_img, content_label, fake_img
            else:
                label = self.label_dic[img_id]
                return style_img, style_gray_img, style_label, content_gray_img, content_label
        else:
            return style_img, style_gray_img


def pair_check_visual():

    dst_dir = "./dataset_check/"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    base_dir = "/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/tet_gan_20k"
    img_folder = base_dir + "/imagesA"
    test_label_path = base_dir + "/gt.txt"
    gray_text_folder = base_dir + "/gray_textA"

    batch_size=4
    dataset = tet_ganA(img_folder, test_label_path, gray_text_folder, train=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    pbar = tqdm(dataloader)
    
    import torch
    import torchvision.utils as tvutils
    from PIL import Image, ImageFont, ImageDraw

    for i, (style_img, style_gray_img, style_label, content_gray_img, content_label) in enumerate(pbar):
        import pdb;pdb.set_trace()
        if i == 100:
            break
        save_image(style_img, dst_dir + f'style_{str(i).zfill(6)}.png')
        save_image(style_gray_img, dst_dir + f'style_gray_{str(i).zfill(6)}.png')
        save_image(content_gray_img, dst_dir + f'content_gray_{str(i).zfill(6)}.png')
        
        print(style_label)
        print(content_label)  



if __name__=="__main__":
    # img_folder = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/preprocessed"
    # test_label_path = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/label_dic.json"
    # gray_text_folder = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/gray_text"

    # batch_size=16
    # dataset = IMGUR5K_Handwriting(img_folder, test_label_path, gray_text_folder,train=True)

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # pbar = tqdm(dataloader)
    # for i, (imgs, gray_imgs, labels) in enumerate(pbar):
    #     pbar.set_description(desc=f"imgs.shape: {imgs.shape}, gray_imgs.shape: {gray_imgs.shape}, len(labels): {len(labels)}")
    # draw_text_on_image()
    pair_check_visual()


        

