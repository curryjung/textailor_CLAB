import argparse
from locale import normalize
import math
import random
import os
from re import L

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils
from tqdm import tqdm
import torchvision.transforms.functional as Function
from torchvision.utils import make_grid, save_image


from model import Encoder, Generator, Discriminator
from dataset import IMGUR5K_Handwriting

from torch.utils.tensorboard import SummaryWriter


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def run_z_optimizer(args, loader, generator):

    curdir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(curdir, args.exp_name)
    dst_dir = os.path.join(exp_dir, args.exp_desc)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    loader = sample_data(loader)

    ## set constant attribute in generator which is tranable and matchs the batch size and latent dimension
    # generator.set_style_input(args.batch, args.latent_dim)
    #latent mean
    mean_latent = torch.randn((args.batch,args.latent,1024))
    mean_latent = mean_latent.mean(axis=-1)

    styles = mean_latent
    styles = nn.Parameter(torch.randn(args.batch, args.latent, device=args.device))
    styles = styles.to(args.device)

    #make optimizer for constant_style
    # optimizer = optim.SGD(getattr(generator,'constant_style'), lr=args.lr)
    optimizer = optim.Adam([styles], lr=args.lr)

    requires_grad(generator, False)
    

    pbar = range(args.iter)
    pbar = tqdm(pbar)

    contents = torch.zeros(args.batch, 1, device=args.device)
    batch = next(loader)
    batch = batch.to(args.device)

    for idx in pbar:
        optimizer.zero_grad()

        #forward pass
        
        output,_ = generator(content = contents, styles = [styles], input_is_latent = False)

        #loss function
        loss = F.mse_loss(output, batch)

        loss.backward()
        optimizer.step()


        pbar.set_description(
            f"loss: {loss.item():.4f}")

        if idx % args.log_interval == 0:
            # print('Iter: [{:4d}] MSE Loss: {:.4f}'.format(iter, loss.item()))
            pass

        if idx % args.visual_step == 0:
            #concat sample and batch
            sample = torch.cat((batch.cpu().detach(), output.cpu().detach()), axis=0)

            #make grid image
            sample = make_grid(sample, nrow=args.batch, normalize=True, scale_each=True)

            #save image
            save_image(sample, os.path.join(dst_dir, '{}.png'.format(idx)), nrow=args.batch,normalize=True, scale_each=True)
            print('Saved image at {}'.format(os.path.join(dst_dir, '{}.png'.format(idx))))


def run_w_optimizer(args, loader, generator):

    #set directory
    curdir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(curdir, args.exp_name)
    dst_dir = os.path.join(exp_dir, args.exp_desc)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


    #get dataloader
    loader = sample_data(loader)
    batch = next(loader)
    batch = batch.to(args.device)


    #compute mean_latent(initial point of optimization)
    random_z = torch.randn((1024,args.latent))
    random_z = random_z.to(args.device)

    style_layer = generator.style
    style_layer = style_layer.to(args.device)

    content = torch.randn((1,1), device=args.device)

    mean_latent = style_layer(random_z)
    mean_latent = mean_latent.mean(axis=0,keepdim=True)
    mean_latent = nn.Parameter(mean_latent)

    #get optimizer
    optimizer = optim.Adam([mean_latent], lr=args.lr)

    #memory efficient
    requires_grad(generator, False)

    pbar = range(args.iter)
    pbar = tqdm(pbar)

    for idx in pbar:
        
        optimizer.zero_grad()
        
        output,_ = generator(content,[mean_latent],input_is_latent=True)

        #loss function
        loss = F.mse_loss(output, batch)

        #backward pass
        loss.backward()

        #optimize
        optimizer.step()


        pbar.set_description(
            f"loss: {loss.item():.4f}")

        if idx % args.log_interval == 0:
            # print('Iter: [{:4d}] MSE Loss: {:.4f}'.format(iter, loss.item()))
            pass

        if idx % args.visual_step == 0:
            #concat sample and batch
            sample = torch.cat((batch.cpu().detach(), output.cpu().detach()), axis=0)

            #make grid image
            sample = make_grid(sample, nrow=args.batch, normalize=True, scale_each=True)

            #save image
            save_image(sample, os.path.join(dst_dir, '{}.png'.format(idx)), nrow=args.batch, normalize=True, scale_each=True)
            print('Saved image at {}'.format(os.path.join(dst_dir, '{}.png'.format(idx))))



    img = make_grid(output, nrow=1, normalize=True, scale_each=True)
    save_image(img, os.path.join(dst_dir, '0.png'), nrow=1, scale_each=True)







if __name__ =="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='/hdd/datasets/IMGUR5K-Handwriting-Dataset/preprocessed/')
    parser.add_argument("--g_ckpt", type=str, default='/home/sy/textailor_CLAB/checkpoint/100000.pt')
    parser.add_argument("--e_ckpt", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="run_js")
    parser.add_argument("--exp_desc", type=str, default="optimize_w_input", help="experiment description")
    
    parser.add_argument("--run_z_optimizer", action='store_true')
    parser.add_argument("--run_w_optimizer", action='store_true')

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--iter", type=int, default=1000000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n_sample", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--visual_step", type=int, default=1000)

    parser.add_argument("--vgg", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--adv", type=float, default=0.05)   
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)

    parser.add_argument("--tensorboard", action="store_true")

    
    args = parser.parse_args()

    device = args.device

    print("load generator:", args.g_ckpt)
    g_ckpt = torch.load(args.g_ckpt, map_location=lambda storage, loc: storage)
    g_args = g_ckpt['args']
    
    args.size = g_args.size
    args.latent = g_args.latent
    args.n_mlp = g_args.n_mlp
    args.channel_multiplier = g_args.channel_multiplier

    generator = Generator().to(device)

    generator.load_state_dict(g_ckpt["g_ema"])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = IMGUR5K_Handwriting(args.path)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=False),
        drop_last=True,
    )


    if args.run_w_optimizer:
        print("run_w_optimizer")
        run_w_optimizer(args, loader, generator)
    elif args.run_z_optimizer:
        print("run_z_optimizer")
        run_z_optimizer(args, loader, generator)

