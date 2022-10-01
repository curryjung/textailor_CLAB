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
    # [1,1,64,256]으로 model 예측이 안됨 github에 비슷한 오류를 가진 사람이 많음. 다들 해결 못함
    # [10,1,64,256]으로는 되기 때문에 우선 임시방편
    c_demo_image = torch.cat([c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image], dim=0)
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

def train(args, loader, generator, discriminator, c_encoder, c_image, g_optim, d_optim, c_optim, g_ema, device):
    loader = sample_data(loader)

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
        c_module = c_encoder

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    c_demo_image = c_image.unsqueeze(dim=0) # shape [1,3,64,256]
    c_demo_image_gray = functional.rgb_to_grayscale(c_demo_image)
    c_pred = ocr_pred(c_demo_image_gray, 'hi')

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        if args.freeze:
            requires_grad(discriminator, False)
        else:
            requires_grad(discriminator, True)
        requires_grad(c_encoder, False)

        c_img = c_demo_image.to(device)
        c_img_gray = functional.rgb_to_grayscale(c_img)
        content = c_encoder(c_img_gray) #encoder에 content image 넣기
        content = content.repeat(args.batch, 1, 1, 1)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(content, noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        # fake img,real img shape [4,3,64,256]
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()
        
        if not args.freeze:
            discriminator.zero_grad()
            d_loss.backward()

        if args.tensor:
            writer.add_scalar("Loss/d_loss", d_loss, idx)

        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            if not args.freeze:
                discriminator.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

                d_optim.step()

        loss_dict["r1"] = r1_loss

        # generator 훈련
        if args.freeze:
            requires_grad(generator, False)
        else:
            requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(c_encoder, False)
        
        content = c_encoder(c_img_gray)
        content = content.repeat(args.batch, 1, 1, 1)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(content, noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        if not args.freeze:
            generator.zero_grad()
            g_loss.backward()
            g_optim.step()

        if args.tensor:
            writer.add_scalar("Loss/g_loss", g_loss, idx)
        

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)

            content = c_encoder(c_img_gray)
            content = content.repeat(args.batch, 1, 1, 1)
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)

            fake_img, latents = generator(content, noise, return_latents=True)
        
            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            if not args.freeze:
                weighted_path_loss.backward()
                g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # content encoder 훈련
        requires_grad(generator, False)
        requires_grad(discriminator, False)
        requires_grad(c_encoder, True)

        content = c_encoder(c_img_gray)
        content = content.repeat(args.batch, 1, 1, 1)
        fake_img, _ = generator(content, noise)
        fake_img_gray = functional.rgb_to_grayscale(fake_img)

        closs_preds, closs_target , pred = ocr_pred(fake_img_gray, c_pred, predict_text=True)
        c_enc_loss = c_loss(closs_preds, closs_target)
        c_enc_loss.requires_grad=True

        if args.tensor:
            writer.add_scalar("Loss/c_enc_loss", c_enc_loss, idx)
        loss_dict["c"] = c_enc_loss

        c_encoder.zero_grad()
        c_enc_loss.backward()
        c_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        c_loss_val = loss_reduced["c"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Content encoder": c_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % args.save_freq == 0:
                with torch.no_grad():
                    g_ema.eval()
                    """
                    # train image 외의 image
                    sample = Image.open('./gray_text/result/WARMING.png')
                    tf = transforms.ToTensor()
                    s_image = tf(sample) #c_image shape = [1,64,256]
                    s_image = s_image.to(device)
                    s_demo_image = s_image.unsqueeze(dim=0)
                    s_demo_image = s_demo_image.to(device)
                    s_demo_image_gray = functional.rgb_to_grayscale(s_demo_image)
                    s_content = c_encoder(s_demo_image_gray)
                    s_fake_img, _ = g_ema(s_content, [sample_z])
                    s_sample = torch.cat([s_image.unsqueeze(0), s_fake_img], dim=0)
                    """
                    content = c_encoder(c_img_gray)
                    content = content.repeat(args.n_sample, 1, 1, 1)
                    t_fake_img, _ = g_ema(content, [sample_z])
                    t_sample = torch.cat([c_img, t_fake_img], dim=0)

                    #final_sample = torch.cat([s_sample.detach(), t_sample.detach()], dim=2)

                    utils.save_image(
                        t_sample,
                        f"freeze_ce/sample/{str(i).zfill(6)}.png",
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "c": c_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "c_optim" : c_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"freeze_ce/checkpoint/{str(i).zfill(6)}.pt",
                )

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--path", type=str, default = '/hdd/datasets/IMGUR5K-Handwriting-Dataset/preprocessed/', help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="batch sizes for each gpus"
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
        from model import Generator, Discriminator, Content_Encoder

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator().to(device)
    discriminator = Discriminator(channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator().to(device)
    c_encoder = Content_Encoder().to(device)
    g_ema.eval()
    #accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    c_optim = optim.Adam(
        c_encoder.parameters(),
        lr=args.lr
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

    dataset = IMGUR5K_Handwriting(args.path)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    # c_image = input content image
    image = Image.open('./gray_text/result/EARTH.png')
    tf = transforms.ToTensor()
    c_image = tf(image) #c_image shape = [1,64,256]

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")
    train(args, loader, generator, discriminator, c_encoder, c_image, g_optim, d_optim, c_optim, g_ema, device)
