from dataset import IMGUR5K_Handwriting
from torch.utils import data

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)



# def cal_mse(img1, img2):
#     return ((img1 - img2) ** 2).mean()

import torch
import torch.nn as nn

def cal_mse(dataset, loader, batch_size):
    
    #assert if batch_size is not even or not integer
    assert(batch_size % 2 == 0 or type(batch_size) != int)

    mse = []
    stop_index = 50
    cnt = 0

    loss = nn.MSELoss()

    for i, (img, _,_) in enumerate(loader):
        if img.shape[0] != batch_size or i == stop_index:
            break
        left_half = img[:int(batch_size/2)]
        # left_half = left_half.view(left_half.shape[0], -1)
        right_half = img[int(batch_size/2):]
        # right_half = right_half.view(right_half.shape[0], -1)

        # mse += ((left_half - right_half) ** 2).mean(1).sum()
        mse.append(loss(left_half, right_half).item())
        
    return sum(mse)/len(mse)



    
    



if __name__ == '__main__':

    # img_folder = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/preprocessed"
    img_folder = "/home/sy/textailor_CLAB/result/fake_samples"
    test_label_path = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/label_dic.json"
    gray_text_folder = "/hdd/datasets/IMGUR5K-Handwriting-Dataset/gray_text"
    # gray_text_folder = "/home/sy/textailor_CLAB/result/fake_samples"

    batch_size=64
    dataset = IMGUR5K_Handwriting(img_folder, test_label_path, gray_text_folder,train=True)

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # path = '/hdd/datasets/IMGUR5K-Handwriting-Dataset/preprocessed/'
    # batch_size = int(8)


    # dataset = IMGUR5K_Handwriting(path)

    # loader = data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     sampler=data_sampler(dataset, shuffle=True),
    #     drop_last=True,
    # )

    mse = cal_mse(dataset, dataloader, batch_size)

    print(mse)
