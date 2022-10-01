import argparse
from calendar import c
import math
import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import string
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torchvision.transforms import functional
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None

from dataset import IMGUR5K_Handwriting
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

from PIL import Image
# OCR 
from OCR.demo import demo

from torch.utils.tensorboard import SummaryWriter


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def ocr_pred(c_demo_image, p_t, predict_text=False):
    # [1,1,64,256]으로 model 예측이 안됨 
    # [10,1,64,256]으로는 되기 때문에..
    #c_demo_image = torch.cat([c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image], dim=0)
    if predict_text==False:
        pred = demo(args,c_demo_image,p_t) #content image를 넣었을 때 예측되는 글자
        return pred
    else:
        closs_preds, closs_target, pred = demo(args, c_demo_image, p_t, predict_text=True) #content image를 넣었을 때 예측되는 글자
        return closs_preds, closs_target, pred

def c_loss(preds1, preds2):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss = criterion(preds1.view(-1, preds1.shape[-1]), preds2.contiguous().view(-1))
    return loss

def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    imsave_path = './ce_sample/ce_train_final/'+args.dir_name
    model_path = './ce_checkpoint/ce_train_final/'+args.dir_name
    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if args.tensor:
        writer = SummaryWriter()

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    ex_img, gray_text_img_ex, label = next(loader)
    ex_img = ex_img.to(device)
    ex_img_resize = functional.resize(ex_img, (256,256))
    # c_image = input content image
    image = Image.open('./gray_text/result/EARTH.png')
    tf = transforms.ToTensor()
    c_image = tf(image) #c_image shape = [1,64,256]
    c_demo_image = c_image.unsqueeze(dim=0) # shape [1,3,64,256]
    c_demo_image_gray = functional.rgb_to_grayscale(c_demo_image)
    c_demo_image_gray = c_demo_image_gray.to(device)
    c_demo_image_gray = c_demo_image_gray.repeat(args.batch, 1, 1, 1)

    sample_z = torch.randn(4, args.latent, device=device)


    var_dic = {}
    var_dic['random']=[]
    var_dic['encoder']=[]

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        # gray_text_img shape=[batch,1,64,256]
        # real_img shape=[batch,3,64,256]
        real_img, gray_text_img, label = next(loader)

        real_img = real_img.to(device)
        real_img_resize = functional.resize(real_img, (256,256))

        gray_text_img = gray_text_img.to(device)

        # random noise로 만든 image
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(gray_text_img, noise,random_style=True)
        
        _, encoder_latent = generator(gray_text_img, real_img_resize, return_latents=True)
        _, random_latent = generator(gray_text_img, noise, random_style=True, return_latents=True)

        for i in range(0, args.batch):
            var_dic['encoder'].append(encoder_latent[i][0])
            var_dic['random'].append(random_latent[i][0])
        
        if len(var_dic['encoder']) == 100:
            enc_frame = pd.DataFrame(var_dic['encoder'])
            ran_frame = pd.DataFrame(var_dic['random'])
            enc_var = []
            ran_var = []
            for i in range(0,512):
                enc_var.append(enc_frame[i].var())
                ran_var.append(ran_frame[i].var())
            print(f"style encoder : {torch.mean(torch.tensor(enc_var))}")
            print(f"random : {torch.mean(torch.tensor(ran_var))}")
            print("done!")
            break

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--img_folder", type=str, default = '/mnt/f06b55a9-977c-474a-bed0-263449158d6a/textailor_CLAB/dataset/IMGUR5K-Handwriting-Dataset/preprocessed', help="path to the lmdb dataset")
    parser.add_argument("--test_label_path", type=str, default = "/mnt/f06b55a9-977c-474a-bed0-263449158d6a/textailor_CLAB/dataset/IMGUR5K-Handwriting-Dataset/label_dic.json")
    parser.add_argument("--gray_text_folder", type=str, default = "/mnt/f06b55a9-977c-474a-bed0-263449158d6a/textailor_CLAB/dataset/IMGUR5K-Handwriting-Dataset/gray_text")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument('--dir_name', type=str, default='hi', help='저장 이름')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=8,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1000,
        help="save frequency",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--tensor", action="store_true", help="tensorboard"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--freeze", action="store_true", help="generator/discriminator freeze"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--recon_factor",
        type=float,
        default=10.0,
        help="reconstruction loss facator",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--train_store", action="store_true"
    )

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--ocr_batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    """ Analysis """
    parser.add_argument('--return_var', action='store_true', help='return_var')
    parser.add_argument('--mode_check', action='store_true', help='mode collapse check')

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator, Content_Encoder, Style_Encoder

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator().to(device)
    discriminator = Discriminator(channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator().to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    """
    # mapping network lr
    g_optim.add_param_group(
        {
            'params': generator.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    """
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    dataset = IMGUR5K_Handwriting(args.img_folder, args.test_label_path, args.gray_text_folder, train=True)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
