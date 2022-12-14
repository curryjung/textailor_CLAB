import argparse
import math
import random
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from os.path import join as ospj
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from dataset_easy import tet_ganA, tet_ganB
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchvision import utils as tvutils
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

from OCR.demo import demo
from torch.utils.tensorboard import SummaryWriter

#from logger import TBLogger


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

def ocr_pred(args, c_demo_image, p_t, predict_text=False):
    # [1,1,64,256]?????? model ????????? ?????? 
    # [10,1,64,256]????????? ?????? ?????????..
    #c_demo_image = torch.cat([c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image, c_demo_image], dim=0)
    if predict_text==False:
        pred = demo(args,c_demo_image,p_t) #content image??? ????????? ??? ???????????? ??????
        return pred
    else:
        closs_preds, closs_target, pred = demo(args, c_demo_image, p_t, predict_text=True) #content image??? ????????? ??? ???????????? ??????
        return closs_preds, closs_target, pred

def c_loss(preds1, preds2):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss = criterion(preds1.view(-1, preds1.shape[-1]), preds2.contiguous().view(-1))
    return loss


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)
    #writer = TBLogger(args.dname_logger, args.dir_name)
    imsave_path = './sample/'+args.dir_name
    model_path = './checkpoint/'+args.dir_name
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
    ex_img_resize = functional.resize(ex_img, (256, 256))
    tvutils.save_image(
        ex_img,
        f'./sample/{args.dir_name}/style_fixed.png',
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )

    # c_image = input content image
    if not args.get_fixed_gray_text_by_cv2:
        image = Image.open('./gray_text/result/EARTH.png')
        tf = transforms.ToTensor()
        c_image = tf(image)  # c_image shape = [1,64,256]
        c_demo_image = c_image.unsqueeze(dim=0)  # shape [1,3,64,256]
        if not args.content_resnet:
            c_demo_image_gray = functional.rgb_to_grayscale(c_demo_image)
        else:
            c_demo_image_gray = c_demo_image
    else :
        import cv2
        image = cv2.imread('./gray_text/result/EARTH.png')
        gray_transform =  transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((64,256)), 
                        transforms.Normalize((0.5,), (0.5,))
                    ])
        c_image = gray_transform(image)  # c_image shape = [1,64,256]
        c_demo_image_gray = c_image.unsqueeze(dim=0)  # shape [1,1,64,256]


    tvutils.save_image(
        c_demo_image_gray,
        f'./sample/{args.dir_name}/content_fixed.png',
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )
    c_demo_image_gray = c_demo_image_gray.to(device)
    c_demo_image_gray = c_demo_image_gray.repeat(args.batch, 1, 1, 1)

    fixed_z = torch.randn(4, args.latent, device=device)

    var_dic = {}
    var_dic['random'] = []
    var_dic['encoder'] = []

    for idx in pbar:
        i = idx + args.start_iter

        generator.train()

        if i > args.iter:
            print("Done!")
            break

        # gray_text_img shape=[batch,1,64,256]
        # real_img shape=[batch,3,64,256]
        real_img, gray_text_img, label = next(loader)

        if args.return_dataset_pair:            
            real_img = real_img.to(device)
            real_img_resize = functional.resize(real_img, (256, 256))
            real_img_gray = functional.rgb_to_grayscale(real_img_resize)

            gray_text_img = gray_text_img.to(device)

            lower_label = [l.lower() for l in label]
            #label = [l.translate(str.maketrans('', '', string.punctuation)) for l in label]
            new_label = []
            for k in lower_label:
                for j in k:
                    if not j in args.character:
                        k = k.replace(j,'')
                if len(k) > 25:
                    k = k[:25]
                new_label.append(k)

            _,_,pred = ocr_pred(args, real_img_gray, new_label, predict_text=True)

        

            width_cell = real_img.shape[3]
            height_cell = real_img.shape[2]
            images = [real_img, gray_text_img]
            images = [tvutils.make_grid(image, nrow=1, normalize=True, range=(-1, 1)) for image in images]
            images = torch.cat(images, axis=2) * 255
            images = images.cpu().numpy().astype('uint8').transpose(1, 2, 0)  # H W C
            H, W, C = images.shape
            images_dtype = images.dtype


            # images = Image.fromarray(images)

            # H,W ,C = images.shape
            canvas_width = 256
            canvas = np.ones((H,canvas_width,C), images_dtype) * 255
            font = ImageFont.truetype('NanumGothicBold.ttf', 20)
            canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas)
            padding = 1
            centering = 5
            for ih, word in enumerate(new_label):
                offset = ih * (height_cell + padding) + centering
                draw.text((0, offset), word, fill='black', font=font, stroke_width=3, stroke_fill='white')

            images = np.concatenate([images, np.array(canvas)], axis=1)

            H,W ,C = images.shape
            canvas_width = 256
            canvas = np.ones((H,canvas_width,C), images_dtype) * 255
            font = ImageFont.truetype('NanumGothicBold.ttf', 20)
            canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas)
            padding = 1
            centering = 5
            for ih, word in enumerate(pred):
                offset = ih * (height_cell + padding) + centering
                draw.text((0, offset), word, fill='black', font=font, stroke_width=3, stroke_fill='white')

            images = np.concatenate([images, np.array(canvas)], axis=1)

            H,W ,C = images.shape

            canvas_height = 30
            canvas = np.ones((canvas_height, W, C), images_dtype) * 255
            font = ImageFont.truetype('NanumGothicBold.ttf', 20)
            canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas) 
            padding = 5
            centering = 50
            words = f'real,label,lower_label, pred_from_real'.split(',')
            for iw, word in enumerate(words):
                offset = iw * (width_cell + padding) + centering
                draw.text((offset, 0), word, fill='black', font=font, stroke_width=3, stroke_fill='white')
            images = np.concatenate([images, np.array(canvas)], axis=0)


            images = Image.fromarray(images)


            images.save(f'./sample/{args.dir_name}/{str(i).zfill(6)}.png') 
            

            continue

        # variance analysis
        if args.return_var:
            requires_grad(generator, False)
            requires_grad(discriminator, False)
            real_img = real_img.to(device)
            real_img_resize = functional.resize(real_img, (256, 256))

            gray_text_img = gray_text_img.to(device)

            # style encoder ?????? image
            _, var_encoder = generator(gray_text_img, real_img_resize, return_var=True)

            # random noise??? ?????? image
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            _, var_random = generator(gray_text_img, noise, random_style=True, return_var=True)

            var_dic['random'].append(var_encoder)
            var_dic['encoder'].append(var_random)

            continue

        real_img = real_img.to(device)
        real_img_resize = functional.resize(real_img, (256, 256))

        gray_text_img = gray_text_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # style encoder ?????? image
        fake_img, _ = generator(gray_text_img, real_img_resize)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        # random noise??? ?????? image
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(gray_text_img, noise, random_style=True)

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.tensor:
            writer.add_scalar("Loss/d_loss", d_loss, idx)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

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

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # G train
        # style encoder??? ?????? image
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_refguided, _ = generator(gray_text_img, real_img_resize)

        if args.augment:
            fake_refguided, _ = augment(fake_refguided, ada_aug_p)

        fake_pred = discriminator(fake_refguided)
        g_loss = g_nonsaturating_loss(fake_pred)

        fake_refguided, _ = generator(gray_text_img, real_img_resize)
        fake_refguided_gray = functional.rgb_to_grayscale(fake_refguided)

        # OCR ???????????? ?????? X, ????????? to ?????????, 25??? ?????? ??????
        label = [l.lower() for l in label]
        #label = [l.translate(str.maketrans('', '', string.punctuation)) for l in label]
        new_label = []
        for k in label:
            for j in k:
                if not j in args.character:
                    k = k.replace(j,'')
            if len(k) > 25:
                k = k[:25]
            new_label.append(k)

        closs_preds, closs_target, pred = ocr_pred(args, fake_refguided_gray, new_label, predict_text=True)
        # import pdb;pdb.set_trace()

        # if args.return_ocr_results:


        #     # width_cell = real_img.shape[3]
        #     # images = [real_img, gray_text_img, fake_refguided, fake_img_latentguided, t_fake_img_fixedz, t_fake_img_fixedref, t_fake_img_fixedz_g, t_fake_img_fixedref_g]
        #     # images = [tvutils.make_grid(image, nrow=1, normalize=True, range=(-1, 1)) for image in images]
        #     # images = torch.cat(images, axis=2) * 255
        #     # images = images.cpu().numpy().astype('uint8').transpose(1, 2, 0)  # H W C
        #     # H, W, C = images.shape

        #     # canvas_height = 30
        #     # canvas = np.ones((canvas_height, W, C), images.dtype) * 255
        #     # font = ImageFont.truetype('NanumGothicBold.ttf', 20)
        #     # canvas = Image.fromarray(canvas)
        #     # draw = ImageDraw.Draw(canvas)
        #     # padding = 5
        #     # centering = 50
        #     # words = 'real,content,train_refguided,train_latentguided,fixedz_gema,fixedref_gema,fixedz_g,gixedref_g'.split(',')
        #     # for iw, word in enumerate(words):
        #     #     offset = iw * (width_cell + padding) + centering
        #     #     draw.text((offset, 0), word, fill='black', font=font, stroke_width=3, stroke_fill='white')
        #     # images = np.concatenate([images, np.array(canvas)], axis=0)
        #     # images = Image.fromarray(images)
        #     # images.save(f'./sample/{args.dir_name}/{str(i).zfill(6)}.png') 

        ocr_loss = c_loss(closs_preds, closs_target)
        ocr_loss.requires_grad=False

        # image recon loss(??????)
        recon_loss = F.mse_loss(fake_refguided, real_img)

        # latent recon loss
        orig_latent = generator(gray_text_img, real_img_resize, latent_recon=True)  # real image ????????? style encoder latent
        latent_recon_img = functional.resize(fake_refguided, (256, 256))
        fake_latent = generator(gray_text_img, latent_recon_img, latent_recon=True)  # fake image ????????? style encoder latent
        recon_latent = F.mse_loss(orig_latent, fake_latent)

        # diversity sensitive loss
        g_loss = g_loss + args.recon_factor * recon_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # random noise??? ?????? image
        fake_img_latentguided, _ = generator(gray_text_img, noise, random_style=True)

        fake_pred = discriminator(fake_img_latentguided)
        g_loss = g_nonsaturating_loss(fake_pred)

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        if args.tensor:
            writer.add_scalar("Loss/g_loss", g_loss, idx)

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            fake_img_latentguided, latents = generator(gray_text_img, noise, random_style=True, return_latents=True)
            # fake_img, latents = generator(gray_text_img, real_img_resize, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img_latentguided, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img_latentguided[0, 0, 0, 0]

            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["recon_image"] = recon_loss
        loss_dict["recon_latent"] = recon_latent
        loss_dict["ocr_loss"] = ocr_loss
        loss_dict["g"] = g_loss

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description((
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
            ))

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
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

            if i % args.image_every == 0:
                with torch.no_grad():
                    g_ema.eval()

                    t_fake_img_fixedz, _ = g_ema(c_demo_image_gray, [fixed_z], random_style=True)
                    t_fake_img_fixedref, _ = g_ema(c_demo_image_gray, ex_img_resize)

                    generator.eval()

                    t_fake_img_fixedz_g, _ = generator(c_demo_image_gray, [fixed_z], random_style=True)
                    t_fake_img_fixedref_g, _ = generator(c_demo_image_gray, ex_img_resize)


                    width_cell = real_img.shape[3]
                    images = [real_img, gray_text_img, fake_refguided, fake_img_latentguided, t_fake_img_fixedz, t_fake_img_fixedref, t_fake_img_fixedz_g, t_fake_img_fixedref_g]
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
                    words = 'real,content,train_refguided,train_latentguided,fixedz_gema,fixedref_gema,fixedz_g,gixedref_g'.split(',')
                    for iw, word in enumerate(words):
                        offset = iw * (width_cell + padding) + centering
                        draw.text((offset, 0), word, fill='black', font=font, stroke_width=3, stroke_fill='white')
                    images = np.concatenate([images, np.array(canvas)], axis=0)
                    images = Image.fromarray(images)
                    images.save(f'./sample/{args.dir_name}/{str(i).zfill(6)}.png')

            if i % args.ckpt_every == 0:
                torch.save({"g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "args": args,
                            "ada_aug_p": ada_aug_p,
                            },
                           f'./checkpoint/{args.dir_name}/{str(i).zfill(6)}.ckpt',
                           )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--dataset', type=str, default=None, help='tet_gen or image_gru5k')
    parser.add_argument("--dataset_dir", type=str, default='/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset', help='datset directory')
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument('--dname_logger', type=str, default='exp_logs', help='root directory for logger')
    parser.add_argument('--dir_name', type=str, default='hi', help='?????? ??????')
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=8, help="number of the samples generated during training",)
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization",)
    parser.add_argument("--path_batch_shrink", type=int, default=2,
                        help="batch size reducing factor for the path length regularization (reduce memory consumption)",)
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization",)
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization",)
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training",)
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1",)
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--freeze", action="store_true", help="generator/discriminator freeze")
    parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation",)
    parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation",)
    parser.add_argument("--recon_factor", type=float, default=10.0, help="reconstruction loss facator",)
    parser.add_argument("--ada_length", type=int, default=500 * 1000,
                        help="target duraing to reach augmentation probability for adaptive augmentation",)
    parser.add_argument("--ada_every", type=int, default=256, help="probability update interval of the adaptive augmentation",)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--image_every", type=int, default=50, help="save images every",)
    parser.add_argument("--ckpt_every", type=int, default=10000, help="save checkpoints every",)
    parser.add_argument("--train_store", action="store_true")

    parser.add_argument("--get_fixed_gray_text_by_cv2", action="store_true")
    parser.add_argument("--return_dataset_pair", action="store_true")

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--ocr_batch_size', type=int, default=192, help='input batch size')

    """ Analysis """
    parser.add_argument('--return_var', action='store_true', help='return_var')
    parser.add_argument('--tensor', action='store_true', help='tensorboard')

    """OCR"""
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
    args.content_resnet = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    print('Count of using GPUs:', torch.cuda.device_count())

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    from model import Generator, Discriminator

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
        
    if args.dataset == "tet_gen":
        img_folder = args.dataset_dir + "/imagesA"
        test_label_path = args.dataset_dir + "/gt.txt"
        gray_text_folder = args.dataset_dir + "/gray_textA"        
    else:
        img_folder = args.dataset_dir+'/preprocessed'
        label_path = args.dataset_dir+'/label_dic.json'
        gray_text_folder = args.dataset_dir+'/gray_text'

    # dataset = IMGUR5K_Handwriting(args.img_folder, args.test_label_path, args.gray_text_folder, train=True)
    if args.dataset == "tet_gen":
        dataset = tet_ganA(img_folder, test_label_path, gray_text_folder, train=True)
    elif args.content_resnet:
        dataset = IMGUR5K_Handwriting(img_folder, label_path, gray_text_folder, train=True, content_resnet=True)
    else:
        dataset = IMGUR5K_Handwriting(img_folder, label_path, gray_text_folder, train=True)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
