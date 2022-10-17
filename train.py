import argparse
import math
import random
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from os.path import join as ospj
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
from dataset_easy import tet_ganA, tet_ganB
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

def preprocess_label(args,label):

    # OCR 특수문자 인식 X, 대문자 to 소문자, 25개 미만 글자
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
    return new_label


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, loader_extra = None):
    loader = sample_data(loader)
    loader_extra = sample_data(loader_extra)
    imsave_path = './sample/'+args.dir_name
    model_path = './checkpoint/'+args.dir_name   

    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

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

    # fixed z and content and style
    ex_style, ex_style_gray, _, ex_content_gray,_= next(loader)
    ex_style = ex_style.to(device)
    ex_style_gray = ex_style_gray.to(device)
    ex_content_gray = ex_content_gray.to(device)
    ex_style_resize = functional.resize(ex_style,(256,256))

    tvutils.save_image(
        ex_style,
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

    fixed_z = torch.randn(args.batch, args.latent, device=device)

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


    for idx in pbar:
        i = idx + args.start_iter

        generator.train()

        # print(generator.style_encoder.layer1[0].conv1.weight[0,0])

        if i > args.iter:
            print(f'estimated iteration({args.iter}) is finished')
            break
        
        if i % 2==0:
            style_img, style_gray_img, style_label, content_gray_img, content_label = next(loader)
        else:
            style_img, style_gray_img, style_label, content_gray_img, content_label = next(loader_extra)
        

        if args.return_dataset_pair:
            width_cell = style_img.shape[3]
            height_cell = style_img.shape[2]
            images = [style_img, style_gray_img, content_gray_img]
            images = [tvutils.make_grid(image, nrow=1, normalize=True, range=(-1, 1)) for image in images]
            images = torch.cat(images, axis=2) * 255
            images = images.cpu().numpy().astype('uint8').transpose(1, 2, 0)  # H W C
            H, W, C = images.shape
            images_dtype = images.dtype

            H,W ,C = images.shape

            canvas_height = 30
            canvas = np.ones((canvas_height, W, C), images_dtype) * 255
            font = ImageFont.truetype('NanumGothicBold.ttf', 20)
            canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas) 
            padding = 5
            centering = 50
            words = f'style_img,style_content({style_label}),random_content({content_label})'.split(',')
            for iw, word in enumerate(words):
                offset = iw * (width_cell + padding) + centering
                draw.text((offset, 0), word, fill='black', font=font, stroke_width=3, stroke_fill='white')
            images = np.concatenate([images, np.array(canvas)], axis=0)            

            images = Image.fromarray(images)

            images.save(os.path.join(imsave_path,'dataloader_sample', f'{i}.png'))
            
            continue

        style_img = style_img.to(device)
        style_gray_img = style_gray_img.to(device)
        content_gray_img = content_gray_img.to(device)


        #D train

        # style encoder 나온 image
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        fake_img, _ = generator(content_gray_img,functional.resize(style_img, (256, 256)))

        if args.augment:
            real_img_aug, _ = augment(style_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = style_img


        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)

        ## adv loss
        d_adv_loss = d_logistic_loss(real_pred, fake_pred)

        d_adv_loss = args.d_adv_loss_weight * d_adv_loss

        discriminator.zero_grad()
        d_adv_loss.backward()
        d_optim.step()

        loss_dict["d_img_guided"] = d_adv_loss.item()

        # random noise로 만든 image
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(content_gray_img, noise, random_style=True)

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        d_loss = args.d_loss_weight * d_loss

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        loss_dict["d_noise_guided"] = d_loss.item()


        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            style_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(style_img, ada_aug_p)
            else:
                real_img_aug = style_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, style_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss


        #G train
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        ## recon_loss
        recon_style_img, _ = generator(style_gray_img,functional.resize(style_img, (256, 256)))
        recon_loss = F.mse_loss(recon_style_img, style_img)


        fake_content_img, _ = generator(content_gray_img,functional.resize(style_img, (256, 256)))
        fake_content_img_gray = functional.rgb_to_grayscale(fake_content_img)

        fake_noise_img, _ = generator(content_gray_img, noise, random_style=True)

        ## ocr_loss of content image
        if args.use_ocr_loss_c:
            fake_content_img, _ = generator(content_gray_img,functional.resize(style_img, (256, 256)))
            fake_content_img_gray = functional.rgb_to_grayscale(fake_content_img)

            content_label = preprocess_label(args,content_label)

            closs_preds, closs_target, pred = ocr_pred(args, fake_content_img_gray, content_label, predict_text = True)

            ocr_loss_content = c_loss(closs_preds, closs_target)
        else :
            ocr_loss_content = torch.tensor(0.0, device=device)

        ## ocr_loss of style image
        if args.use_ocr_loss_s:
            recon_style_img_gray = functional.rgb_to_grayscale(recon_style_img)
            
            style_label = preprocess_label(args,style_label)

            sloss_preds, sloss_target, pred = ocr_pred(args, recon_style_img_gray, style_label, predict_text = True)

            ocr_loss_style = c_loss(sloss_preds, sloss_target)
        else :
            ocr_loss_style = torch.tensor(0.0, device=device)

        if args.late_ocr_adaptation_n > i:
            late_ocr_adaptation_weight = 0.0
        else:
            late_ocr_adaptation_weight = 1.0


        g_loss = args.recon_factor * recon_loss + late_ocr_adaptation_weight * args.ocr_loss_weight * (ocr_loss_content + ocr_loss_style)


        loss_dict["g_recon4"] = recon_loss.item()
        loss_dict["g_ocr_content5"] = ocr_loss_content.item()
        loss_dict["g_ocr_style6"] = ocr_loss_style.item()
        loss_dict["g456"]= g_loss.item()

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        recon_style_img, _ = generator(style_gray_img,functional.resize(style_img, (256, 256)))
        recon_loss = F.mse_loss(recon_style_img, style_img)


        fake_content_img, _ = generator(content_gray_img,functional.resize(style_img, (256, 256)))
        fake_content_img_gray = functional.rgb_to_grayscale(fake_content_img)

        fake_noise_img, _ = generator(content_gray_img, noise, random_style=True)

        ## adv loss
        g_adv_loss_recon = g_nonsaturating_loss(discriminator(recon_style_img))
        g_adv_loss_content = g_nonsaturating_loss(discriminator(fake_content_img))
        g_adv_loss_noise = g_nonsaturating_loss(discriminator(fake_noise_img))

        g_loss = args.g_adv_weight * (g_adv_loss_recon + g_adv_loss_content + g_adv_loss_noise) / 3

        # import pdb; pdb.set_trace()
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        loss_dict["g_adv_recon1"] = g_adv_loss_recon.item()
        loss_dict["g_adv_content2"] = g_adv_loss_content.item()
        loss_dict["g_adv_noise3"] = g_adv_loss_noise.item()
        loss_dict["g123"] = g_loss.item()

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            fake_img_latentguided, latents = generator(content_gray_img, noise, random_style=True, return_latents=True)
            # fake_img, latents = generator(gray_text_img, real_img_resize, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img_latentguided, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img_latentguided[0, 0, 0, 0]

            weighted_path_loss = args.g_adv_weight * weighted_path_loss
            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
        
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        if get_rank() == 0:
            pbar.set_description((
                f'g_123: {loss_reduced["g123"]:.4f}; ' #g_adv_recon1: {loss_reduced["g_adv_recon1"]:.4f}; g_adv_content2: {loss_reduced["g_adv_content2"]:.4f}; g_adv_noise3: {loss_reduced["g_adv_noise3"]:.4f}; '
                f'g_456: {loss_reduced["g456"]:.4f}; '# g_recon4: {loss_reduced["g_recon4"]:.4f}; g_ocr_content5: {loss_reduced["g_ocr_content5"]:.4f}; g_ocr_style6: {loss_reduced["g_ocr_style6"]:.4f}; '
                f'd_total: {loss_reduced["d_noise_guided"]+ loss_reduced["d_img_guided"]}'#, d_noise_guided: {loss_reduced["d_noise_guided"]:.4f}; d_img_guided: {loss_reduced["d_img_guided"]:.4f}; '
                #f'r1: {loss_reduced["r1"]:.4f}; path: {loss_reduced["path"]:.4f}; path_length: {loss_reduced["path_length"]:.4f}; mean_path_length: {mean_path_length_avg:.4f}; '

            ))

            if wandb and args.wandb:
                wandb.log(
                    {
                        "g_123": loss_reduced["g123"],
                        "g_adv_recon1": loss_reduced["g_adv_recon1"],
                        "g_adv_content2": loss_reduced["g_adv_content2"],
                        "g_adv_noise3": loss_reduced["g_adv_noise3"],
                        "g_456": loss_reduced["g456"],
                        "g_recon4": loss_reduced["g_recon4"],
                        "g_ocr_content5": loss_reduced["g_ocr_content5"],
                        "g_ocr_style6": loss_reduced["g_ocr_style6"],
                        "d_total": loss_reduced["d_noise_guided"]+ loss_reduced["d_img_guided"],
                        "d_noise_guided": loss_reduced["d_noise_guided"],
                        "d_img_guided": loss_reduced["d_img_guided"],
                        "r1": loss_reduced["r1"],
                        "path": loss_reduced["path"],
                        "path_length": loss_reduced["path_length"],
                        "mean_path_length": mean_path_length_avg,
                    }
                )

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


            if i % args.image_every == 0:
                with torch.no_grad():
                    # g_ema.eval()

                    generator.eval()

                    # t_fake_img_fixedz, _ = g_ema(c_demo_image_gray, [fixed_z], random_style=True)
                    # t_fake_img_fixedref, _ = g_ema(c_demo_image_gray, ex_style_resize)

                    generator.eval()

                    t_fake_img_fixedz_g, _ = generator(c_demo_image_gray, [fixed_z], random_style=True)
                    t_fake_img_fixedref_g, _ = generator(c_demo_image_gray, ex_style_resize)

                    # training
                    width_cell = style_img.shape[3]
                    images = [style_img, style_gray_img,recon_style_img,content_gray_img,fake_content_img,fake_noise_img]
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
                    words = 'style,style_gray,recon_style,content_gray,pred_content,noise_guided'.split(',')
                    for iw, word in enumerate(words):
                        offset = iw * (width_cell + padding) + centering
                        draw.text((offset, 0), word, fill='black', font=font, stroke_width=3, stroke_fill='white')
                    images = np.concatenate([images, np.array(canvas)], axis=0)
                    images = Image.fromarray(images)
                    tmp_dir = f'./sample/{args.dir_name}/train/'
                    if not os.path.exists(tmp_dir):
                        os.makedirs(tmp_dir)
                    images.save(tmp_dir + f'{str(i).zfill(6)}.png')    

                    #fixed
                    images = [ex_style, c_demo_image_gray, t_fake_img_fixedref_g, t_fake_img_fixedz_g]
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
                    words = 'style,content_gray,pred_content,noise_guided'.split(',')
                    for iw, word in enumerate(words):
                        offset = iw * (width_cell + padding) + centering
                        draw.text((offset, 0), word, fill='black', font=font, stroke_width=3, stroke_fill='white')
                    images = np.concatenate([images, np.array(canvas)], axis=0)
                    images = Image.fromarray(images)
                    tmp_dir = f'./sample/{args.dir_name}/fixed/'
                    if not os.path.exists(tmp_dir):
                        os.makedirs(tmp_dir)
                    images.save(tmp_dir + f'{str(i).zfill(6)}.png')   



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--late_ocr_adaptation_n', type=int, default=0)
    parser.add_argument('--dataset', type=str, default=None, help='tet_gen or image_gru5k')
    parser.add_argument("--dataset_dir", type=str, default='/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset', help='datset directory')
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument('--dname_logger', type=str, default='exp_logs', help='root directory for logger')
    parser.add_argument('--dir_name', type=str, default='hi', help='저장 이름')
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
    parser.add_argument("--d_adv_loss_weight", type=float, default=1.0, help="discriminator adv loss weight",)
    parser.add_argument("--d_loss_weight", type=float, default=1.0, help="discriminator loss weight",)
    parser.add_argument("--g_adv_weight", type=float, default=1.0, help="generator adv loss weight",)
    parser.add_argument("--ocr_loss_weight", type=float, default=1.0, help="ocr loss weight",)
    parser.add_argument("--use_ocr_loss_c", action="store_true", help="use ocr loss for content",)
    parser.add_argument("--use_ocr_loss_s", action="store_true", help="use ocr loss for style",)
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
        dataset = tet_ganA(img_folder, test_label_path, gray_text_folder, train=True,content_resnet=True)    
        dataset_extra = tet_ganB(img_folder, test_label_path, gray_text_folder, train=True,content_resnet=True)
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
    loader_extra = data.DataLoader(
        dataset_extra,
        batch_size=args.batch,
        sampler=data_sampler(dataset_extra, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.dir_name)
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, loader_extra = loader_extra)
