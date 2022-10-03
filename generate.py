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
    test_path = './test/'+ args.dir_name
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    loader = sample_data(loader)
    style_img, content_img, label = next(loader)

    style_img = style_img[8:].to(device)
    content_match = content_img[:8].to(device)
    content_mismatch = content_img[8:].to(device)
    style_img_resize = functional.resize(style_img, (256, 256))

    utils.save_image(
        style_img,
        f'./test/{args.dir_name}/style_fixed.png',
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )
    utils.save_image(
        content_mismatch,
        f'./test/{args.dir_name}/content_mismatch.png',
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )
    utils.save_image(
        content_match,
        f'./test/{args.dir_name}/content_match.png',
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )

    with torch.no_grad():
        g_ema.eval()
        generator.eval()
        g_ema_fin_mismatch, _ = g_ema(content_mismatch, style_img_resize)
        g_ema_fin_match, _ = g_ema(content_match, style_img_resize)
        generator_fin_mismatch, _ = generator(content_mismatch, style_img_resize)
        generator_fin_match, _ = generator(content_match, style_img_resize)

        #g_ema result
        utils.save_image(
            g_ema_fin_mismatch,
            f"./test/{args.dir_name}/result_mismatch_g_ema.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        utils.save_image(
            g_ema_fin_match,
            f"./test/{args.dir_name}/result_match_g_ema.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        #generator result
        utils.save_image(
            generator_fin_mismatch,
            f"./test/{args.dir_name}/result_mismatch_generator.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        utils.save_image(
            generator_fin_match,
            f"./test/{args.dir_name}/result_match_generator.png",
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
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument('--dir_name', type=str, default='hi', help='저장 이름')
    parser.add_argument("--dataset_dir", type=str, default='/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset', help='datset directory')

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

    # dataset = IMGUR5K_Handwriting(args.img_folder, args.test_label_path, args.gray_text_folder, train=True)
    dataset = IMGUR5K_Handwriting(img_folder, label_path, gray_text_folder, train=True, content_resnet=True)
    loader = data.DataLoader(
        dataset,
        batch_size=16,
        sampler=data_sampler(dataset, shuffle=False),
        drop_last=True,
    )

    generate(args, loader, generator, g_ema, device, mean_latent)