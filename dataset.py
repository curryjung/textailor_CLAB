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

from OCR.demo import demo

class IMGUR5K_Handwriting(Dataset):
    def __init__(self, img_folder, label_path=None, gray_text_folder=None,train=True, content_resnet=False, transform=None, generate=False, fake_img_folder=None):
        assert os.path.exists(img_folder), "img_folder does not exist"
        
        self.img_folder = img_folder
        self.label_path = label_path
        self.gray_text_folder = gray_text_folder
        self.img_list = os.listdir(img_folder)
        self.generate = generate
        self.fake_img_folder = fake_img_folder
        if gray_text_folder is not None:
           self.gray_list = os.listdir(gray_text_folder)
        if label_path is not None:
            assert os.path.exists(label_path), "label_path does not exist"

            self.label_dic = self.load_label(label_path)
        else:
            self.label_dic = None

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
        with open(label_path) as f:
            label_file = json.load(f)
        return label_file

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        
        # get style image
        img_name = self.img_list[index]
        img_id = img_name.split('.')[0]
        
        style_img = cv2.imread(os.path.join(self.img_folder, img_name), cv2.IMREAD_COLOR)
        style_img = self.transform(style_img)

        if self.fake_img_folder is not None:
            fake_img = cv2.imread(os.path.join(self.fake_img_folder, img_name), cv2.IMREAD_GRAYSCALE)
            fake_img = self.gray_transform(fake_img)

        # get style_gray image : contains content of style image
        # gray_name = self.gray_list[index]
        if self.gray_text_folder is not None:
            gray_text_img_name = os.path.join(self.gray_text_folder, img_id+".png")
            if self.content_resnet==False:
                style_gray_img = cv2.imread(gray_text_img_name, cv2.IMREAD_GRAYSCALE)
                style_gray_img = self.gray_transform(style_gray_img)
            else:
                style_gray_img = cv2.imread(gray_text_img_name)
                style_gray_img = self.transform(style_gray_img)
        else:
            style_gray_img = None

        # get style_label
        style_label= self.label_dic[img_id]

        # get content_gray image : 
        random_idx = np.random.randint(0, len(self.img_list))
        random_img_name = self.img_list[random_idx]
        random_img_id = random_img_name.split('.')[0]
        random_gray_text_img_name = os.path.join(self.gray_text_folder, random_img_id+".png")
        if self.content_resnet==False:
            content_gray_img = cv2.imread(random_gray_text_img_name, cv2.IMREAD_GRAYSCALE)
            content_gray_img = self.gray_transform(content_gray_img)
        else:
            content_gray_img = cv2.imread(random_gray_text_img_name)
            content_gray_img = self.transform(content_gray_img)

        # get content_label
        content_label= self.label_dic[random_img_id]

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

        

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


def draw_text_on_image():
    dst_dir = "run_js/dataset_test"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    base_dir = "/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets"
    img_folder = base_dir + "/IMGUR5K-Handwriting-Dataset/preprocessed"
    test_label_path = base_dir + "/IMGUR5K-Handwriting-Dataset/label_dic.json"
    gray_text_folder = base_dir + "/IMGUR5K-Handwriting-Dataset/gray_text"

    batch_size=1
    dataset = IMGUR5K_Handwriting(img_folder, test_label_path, gray_text_folder,train=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    import pdb;pdb.set_trace()
    pbar = tqdm(dataloader)
    for i, (imgs, gray_imgs, labels) in enumerate(pbar):
        if i == 500:
            break
        
        imgs = (imgs + 1)/2
        #torch to pil
        pil_img = transforms.ToPILImage()(imgs[0])
        canvas = ImageDraw.Draw(pil_img)
        try:
            canvas.text((28,36), labels[0], fill=(0,0,0))
        except:
            continue
        pil_img.save("run_js/dataset_test/{}.png".format(i))



def pair_check_visual():

    dst_dir = "run_js/dataset_test"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    base_dir = "/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets"
    img_folder = base_dir + "/IMGUR5K-Handwriting-Dataset/preprocessed"
    test_label_path = base_dir + "/IMGUR5K-Handwriting-Dataset/label_dic.json"
    gray_text_folder = base_dir + "/IMGUR5K-Handwriting-Dataset/gray_text"

    batch_size=1
    dataset = IMGUR5K_Handwriting(img_folder, test_label_path, gray_text_folder,train=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    pbar = tqdm(dataloader)
    
    import torch
    import torchvision.utils as tvutils
    from PIL import Image, ImageFont, ImageDraw

    for i, (real_img, gray_text_img, label) in enumerate(pbar):
        if i == 500:
            break



        width_cell = real_img.shape[3]
        images = [real_img, gray_text_img]
        images = [tvutils.make_grid(image, nrow=1, normalize=True, range=(-1, 1)) for image in images]
        images = torch.cat(images, axis=2) * 255
        images = images.cpu().numpy().astype('uint8').transpose(1, 2, 0)  # H W C
        H, W, C = images.shape

        canvas_height = 30
        canvas = np.ones((canvas_height, W, C), images.dtype) * 255
        font = ImageFont.truetype('NanumGothicBold.ttf', 20)
        canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas)
        padding = 5
        centering = 50
        words = f'real,label_{label}'.split(',')
        for iw, word in enumerate(words):
            offset = iw * (width_cell + padding) + centering
            draw.text((offset, 0), word, fill='black', font=font, stroke_width=3, stroke_fill='white')
        images = np.concatenate([images, np.array(canvas)], axis=0)
        images = Image.fromarray(images)
        images.save(os.path.join(dst_dir,f'{str(i).zfill(6)}.png'))     



        pbar.set_description(f'{label}')



    

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


        

