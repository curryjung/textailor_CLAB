import argparse
import math
import random
import os

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
import shutil



from model import Encoder, Generator, Discriminator, ImageToLatent, Encoder_js
from dataset import IMGUR5K_Handwriting

from torch.utils.tensorboard import SummaryWriter


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

    
class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss 


def train(args, loader, encoder, generator, discriminator, e_optim, d_optim, device):

    sample_dir = os.path.join(args.exp_name,args.exp_disc,'encoder_sample')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)


    loader = sample_data(loader)

    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    e_loss_val = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    loss_dict = {}
    vgg_loss = VGGLoss(device=device)

    accum = 0.5 ** (32 / (10 * 1000))

    requires_grad(generator, False)
    
    truncation = 1
    #trunc = generator.mean_latent(4096).detach()
    #trunc.requires_grad = False
    
    if SummaryWriter and args.tensorboard:
        logger = SummaryWriter(args.exp_disc)    
    
    sample_c = torch.randn(args.n_sample, 16, device=device)

    samples = next(loader)
    samples_enc = Function.resize(samples, (256,256))
    samples = samples.to(device)
    samples_enc = samples_enc.to(device)
    

    real_img_enc_64 = next(loader).to(device)
    real_img_enc = Function.resize(real_img_enc_64, (256,256))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        # # D update
        # requires_grad(encoder, False)
        # requires_grad(discriminator, True)
        
        # real_img_dis = next(loader)#[64,256]
        # real_img_enc = next(loader)
        # real_img_enc = Function.resize(real_img_enc, (256,256))

        # real_img_dis = real_img_dis.to(device)
        # real_img_enc = real_img_enc.to(device)

        # latents = encoder(real_img_enc)
        # content = torch.randn(args.batch, 16, device=device)

        # recon_img, _ = generator(content, [latents])

        # recon_pred = discriminator(recon_img)
        # real_pred = discriminator(real_img_dis)
        # d_loss = d_logistic_loss(real_pred, recon_pred)

        # loss_dict["d"] = d_loss

        # discriminator.zero_grad()
        # d_loss.backward()
        # d_optim.step()

        # d_regularize = i % args.d_reg_every == 0

        # if d_regularize:
        #     real_img_dis.requires_grad = True
        #     real_pred = discriminator(real_img_dis)
        #     r1_loss = d_r1_loss(real_pred, real_img_dis)

        #     discriminator.zero_grad()
        #     (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

        #     d_optim.step()

        # loss_dict["r1"] = r1_loss

        # E update
        

        requires_grad(encoder, True)
        requires_grad(discriminator, False)

        encoder.zero_grad()

        if not args.overfit_encoder:
            real_img_enc_64 = next(loader).to(device)
            real_img_enc = Function.resize(real_img_enc_64, (256,256))


        content = torch.randn(args.batch, 16, device=device)


        real_img_enc = real_img_enc.detach()
        real_img_enc.requires_grad = False

        # real_img_dis = real_img_dis.detach()
        # real_img_dis.requires_grad = False
        
        latents = encoder(real_img_enc)
        # latents = torch.randn(args.batch, args.latent, device=device).to(device)

        # def make_noise(batch, latent_dim, n_noise, device):
        #     if n_noise == 1:
        #         return torch.randn(batch, latent_dim, device=device)

        #     noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

        #     return noises

        # def mixing_noise(batch, latent_dim, prob, device):
        #     if prob > 0 and random.random() < prob:
        #         return make_noise(batch, latent_dim, 2, device)

        # latents = mixing_noise(args.batch, args.latent, args.mixing, device)

        # latents = (latents-torch.mean(latents,dim=1,keepdim=True))/torch.std(latents,dim=1, keepdim=True)
        # latents = (latents +0.4674178) * 0.9402938

        recon_img, _ = generator(content, [latents],input_is_latent=args.use_w)

        recon_vgg_loss = vgg_loss(recon_img, real_img_enc_64)
        loss_dict["vgg"] = recon_vgg_loss * args.vgg

        recon_l2_loss = F.mse_loss(recon_img, real_img_enc_64)
        loss_dict["l2"] = recon_l2_loss * args.l2
        
        recon_pred = discriminator(recon_img)
        adv_loss = g_nonsaturating_loss(recon_pred) * args.adv
        loss_dict["adv"] = adv_loss

        e_loss = recon_vgg_loss + recon_l2_loss + adv_loss 
        loss_dict["e_loss"] = e_loss

        
        
        e_loss.backward()
        e_optim.step()

        e_loss_val = loss_dict["e_loss"].item()
        vgg_loss_val = loss_dict["vgg"].item()
        l2_loss_val = loss_dict["l2"].item()
        adv_loss_val = loss_dict["adv"].item()
        # d_loss_val = loss_dict["d"].item()
        # r1_val = loss_dict["r1"].item()

        pbar.set_description(
            (
                f"e: {e_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; l2: {l2_loss_val:.4f}; adv: {adv_loss_val:.4f};"# d: {d_loss_val:.4f}; r1: {r1_val:.4f}; "
                # f"param: {encoder.convs[0][0].weight[0,0,0,0]:.4f}e: {e_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; l2: {l2_loss_val:.4f}; adv: {adv_loss_val:.4f};"# d: {d_loss_val:.4f}; r1: {r1_val:.4f}; "
            
            )
        )

        if SummaryWriter and args.tensorboard:
            logger.add_scalar('E_loss/total', e_loss_val, i)
            logger.add_scalar('E_loss/vgg', vgg_loss_val, i)
            logger.add_scalar('E_loss/l2', l2_loss_val, i)
            logger.add_scalar('E_loss/adv', adv_loss_val, i)
            logger.add_scalar('D_loss/adv', d_loss_val, i)
            # logger.add_scalar('D_loss/r1', r1_val, i)            
        
        if i % args.visual_every == 0:
            if not args.overfit_encoder:
                with torch.no_grad():
                    real_sample = torch.cat([img for img in samples], dim=1)

                    sample_latents = encoder(samples_enc)
                    sample_latents = (sample_latents-torch.mean(sample_latents,dim=1,keepdim=True))/torch.std(sample_latents,dim=1, keepdim=True)
                    sample_latents = (sample_latents +0.4674178) * 0.9402938
                    # recon_samples, _ = generator(sample_c, [sample_latents])
                    recon_samples, _ = generator(content, [sample_latents],input_is_latent=True)
                    recon_sample = torch.cat([img_gen for img_gen in recon_samples], dim=1)

                    final_sample = torch.cat([real_sample.detach(), recon_sample.detach()], dim=2)
                    utils.save_image(
                        final_sample,
                        f"{sample_dir}/encoder_{str(i).zfill(6)}.png",
                        nrow=4,
                        normalize=True,
                        range=(-1, 1),
                    )
                    print(f"Saved {sample_dir}/encoder_{str(i).zfill(6)}.png")
            
            else :
                real_sample = real_img_enc_64
                recon_sample = recon_img
                final_sample = torch.cat([real_sample.detach(), recon_sample.detach()], dim=2)
                utils.save_image(
                    final_sample,
                    f"{sample_dir}/encoder_{str(i).zfill(6)}.png",
                    nrow=4,
                    normalize=True,
                    range=(-1, 1),
                )
                print(f"Saved {sample_dir}/encoder_{str(i).zfill(6)}.png")

        if i % args.save_every == 0:
            torch.save(
                {
                    "e": encoder.state_dict(),
                    "d": discriminator.state_dict(),
                    "e_optim": e_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                f"/home/sy/textailor_CLAB/checkpoint/encoder_{args.exp_name}_{args.exp_disc}_{str(i).zfill(6)}.pt",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='/hdd/datasets/IMGUR5K-Handwriting-Dataset/preprocessed/')
    parser.add_argument("--g_ckpt", type=str, default='/home/sy/textailor_CLAB/checkpoint/100000.pt')
    parser.add_argument("--e_ckpt", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="run_js")
    parser.add_argument("--exp_disc", type=str, default="overfit_style_encoder_only_batch_8_lr_0.005")

    parser.add_argument("--use_w", type=bool, default=True)
    parser.add_argument("--use_only_one_sample", type=bool, default= True)
    parser.add_argument("--overfit_encoder", type=bool, default=True)

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--iter", type=int, default=1000000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--vgg", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--adv", type=float, default=0.0)   
    parser.add_argument("--r1", type=float, default=0.0)
    parser.add_argument("--d_reg_every", type=int, default=0.0)

    parser.add_argument("--visual_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1000)


    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )



    parser.add_argument("--tensorboard", action="store_true")
    
    args = parser.parse_args()

    device = args.device
    
    args.start_iter = 0


    #save_script_path = os.path.join(args.exp_name, args.exp_disc)
    save_script_path = os.path.join(args.exp_name, args.exp_disc)
    if not os.path.exists(save_script_path):
        os.makedirs(save_script_path)
        #copy script to save_script_path
        root_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(os.path.join(root_path,'script.sh'), save_script_path)

    print("make directory:")
    
    if not os.path.exists(args.exp_name):
        os.mkdir(args.exp_name)
        print(f"create {args.exp_name}")
    if not os.path.exists(os.path.join(args.exp_name, args.exp_disc)):
        os.mkdir(os.path.join(args.exp_name, args.exp_disc))
        print(f"create {os.path.join(args.exp_name, args.exp_disc)}")


    print("load generator:", args.g_ckpt)
    g_ckpt = torch.load(args.g_ckpt, map_location=lambda storage, loc: storage)
    g_args = g_ckpt['args']
    
    args.size = g_args.size
    args.latent = g_args.latent
    args.n_mlp = g_args.n_mlp
    args.channel_multiplier = g_args.channel_multiplier
    
    generator = Generator().to(device)
    discriminator = Discriminator(channel_multiplier=args.channel_multiplier).to(device)
    # encoder = Encoder().to(device)
    encoder = Encoder_js().to(device)
    # encoder = ImageToLatent().to(device)

    # e_optim = optim.Adam(
    #     encoder.parameters(),
    #     lr=args.lr,
    #     betas=(0.9, 0.99),
    # )
    
    e_optim = optim.SGD(encoder.parameters(), lr=args.lr)

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr
    )
    
    generator.load_state_dict(g_ckpt["g_ema"])
    discriminator.load_state_dict(g_ckpt["d"])
    d_optim.load_state_dict(g_ckpt["d_optim"])
    
    if args.e_ckpt is not None:
        print("resume training:", args.e_ckpt)
        e_ckpt = torch.load(args.e_ckpt, map_location=lambda storage, loc: storage)

        encoder.load_state_dict(e_ckpt["e"])
        e_optim.load_state_dict(e_ckpt["e_optim"])
        discriminator.load_state_dict(e_ckpt["d"])
        d_optim.load_state_dict(e_ckpt["d_optim"])
        
        try:
            ckpt_name = os.path.basename(args.e_ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name.split('_')[-1])[0])
        except ValueError:
            pass     

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

    train(args, loader, encoder, generator, discriminator, e_optim, d_optim, device)
