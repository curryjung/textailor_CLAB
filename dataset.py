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

class IMGUR5K_Handwriting(Dataset):
    def __init__(self, img_folder, label_path=None, train=True, transform=None):
        assert os.path.exists(img_folder), "img_folder does not exist"
        
        self.img_folder = img_folder
        self.img_list = os.listdir(img_folder)
        if label_path is not None:
            assert os.path.exists(label_path), "label_path does not exist"

            self.label_dic = self.load_label(label_path)
        else:
            self.label_dic = None

        self.train = train
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64,256)), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform



    def load_label(self, label_path):
        with open(label_path) as f:
            label_file = json.load(f)
        return label_file

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_id = img_name.split('.')[0]
        
        img = cv2.imread(os.path.join(self.img_folder, img_name), cv2.IMREAD_COLOR)

        img = self.transform(img)

        if self.label_dic is not None:
            label = self.label_dic[img_id]
            return img, label
        else:
            return img

        

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


if __name__=="__main__":
    img_folder = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/images"
    # test_label_path = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/dataset_info/imgur5k_annotations.json"
    label_path = None
    dataset = IMGUR5K_Handwriting(img_folder, label_path, train=True)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for i, (img) in enumerate(dataloader):
        print(img.shape)
        # print(label)
        if i == 0:
            break

