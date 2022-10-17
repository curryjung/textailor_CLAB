import argparse
import os

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

from torchvision.transforms import functional
from dataset import IMGUR5K_Handwriting
from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def generate(args, loader, generator, g_ema, device, mean_latent):
    test_path = './fake_image/'
    style_path = './style_reshape/'
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(style_path):
        os.makedirs(style_path)
    
    loader = sample_data(loader)
    for i in range(0, 50128):
        style_img, style_gray_img, style_label, content_gray_img, content_label, img_id = next(loader)
        style_img = style_img.to(device)
        style_gray_img = style_gray_img.to(device)
        utils.save_image(
                style_img,
                f"./style_reshape/{str(img_id[0])}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        style_img_resize = functional.resize(style_img, (256, 256))

        with torch.no_grad():
            generator.eval()
            fake_image, _ = generator(style_gray_img, style_img_resize)

            #g_ema result
            utils.save_image(
                fake_image,
                f"./fake_image/{str(img_id[0])}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/f06b55a9-977c-474a-bed0-263449158d6a/textailor_CLAB_renew/checkpoint/baseline_with_ocr_modified_ce_from_8000/050000.ckpt",
        help="path to the model checkpoint",
    )
    parser.add_argument('--dir_name', type=str, default='hi', help='저장 이름')
    parser.add_argument("--dataset_dir", type=str, default='/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset', help='datset directory')
    parser.add_argument("--batch_size", type=int, default=1, help="output image size of the generator")

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    generator = Generator().to(device)
    g_ema = Generator().to(device)
    checkpoint = torch.load(args.ckpt)

    generator.load_state_dict(checkpoint["g"])
    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    
    img_folder = args.dataset_dir+'/preprocessed'
    label_path = args.dataset_dir+'/label_dic.json'
    gray_text_folder = args.dataset_dir+'/gray_text'

    dataset = IMGUR5K_Handwriting(img_folder, label_path, gray_text_folder, train=True, content_resnet=True, generate=True)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset, shuffle=False),
        drop_last=True,
    )

    generate(args, loader, generator, g_ema, device, mean_latent)