import argparse

import torch
from torchvision import utils
# from model import Generator
from model_previous import Generator
from tqdm import tqdm
import json
import os


def generate(args, g_ema, device, mean_latent):

    w_mean_std_dic={}
    w_mean_std_dic['mean'] = []
    w_mean_std_dic['std'] = []
    w_mean_std_dic['mean_mean']=0
    
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            if args.style_mix ==False:
                sample_z = torch.randn(args.sample, args.latent, device=device) #styles
                sample_c = torch.randn(args.sample, 16, device=device) #content

                if not args.save_w_std:
                    sample, _ = g_ema(
                        sample_c, [sample_z]
                    )
                else:
                    style_layer = g_ema.style
                    ws = style_layer(sample_z)

                    means = ws.mean(dim=1).to('cpu')
                    stds = ws.std(dim=1).to('cpu')

                    for mean, std in zip(means,stds):
                        w_mean_std_dic['mean'].append(float(mean.numpy()))
                        w_mean_std_dic['std'].append(float(std.numpy()))

                    w_mean_std_dic['mean_mean'] = sum(w_mean_std_dic['mean'])/len(w_mean_std_dic['mean'])
                    w_mean_std_dic['std_mean'] = sum(w_mean_std_dic['std'])/len(w_mean_std_dic['std'])

                    

                    with open('/home/sy/textailor_CLAB/w_mean_std.json', 'w') as f:
                        json.dump(w_mean_std_dic, f)
                
                if not args.save_w_std:
                    utils.save_image(
                        sample,
                        f"result/{str(i).zfill(6)}.png",
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )

            else: #style mixing
                source = torch.randn(args.sample, args.latent, device=device)
                target = torch.randn(args.sample, args.latent, device=device)

                sample_c = torch.randn(args.sample, 16, device=device)

                #coarse mixing : target과 source 바꿔서.. start_index = 2
                #middle mixing : start_index=2, finish_index=4
                #fine mixing : start_index=5
                mixing_result, _ = g_ema(
                    sample_c, 
                    [target], 
                    target_styles=[source],
                    start_index= 2,
                    style_mix = True
                    )
                source_result, _ = g_ema(
                    sample_c, 
                    [source]
                    )
                target_result, _ = g_ema(
                    sample_c, 
                    [target]
                    )
                utils.save_image(
                    mixing_result,
                    f"result/mixing{str(i).zfill(6)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    source_result,
                    f"result/source{str(i).zfill(6)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    target_result,
                    f"result/target{str(i).zfill(6)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                

def generate_fake_samples(args, g_ema, device, mean_latent):

    if not os.path.exists(args.save_dir):   
        os.makedirs(args.save_dir)

    with torch.no_grad():
        g_ema.eval()

        img_cnt = 0
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device) #styles
            sample_c = torch.randn(args.sample, 16, device=device) #content

            sample, _ = g_ema(
                sample_c, [sample_z]
            )

            #save sample separately, not in a grid
            #For example, if you want to save 1000 samples, you need to save 1000 images.

            for j in range(args.sample):
                utils.save_image(
                    sample[j],
                    f"{args.save_dir}/{str(img_cnt)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                img_cnt += 1

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument("--run_generate", action="store_true", help="generate samples")
    parser.add_argument("--run_generate_fake_samples", action="store_true", help="generate fake samples")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=16,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=500, help="number of images to be generated"
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
        default="/home/sy/textailor_CLAB/checkpoint/100000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--style_mix", action="store_true", help="style mixing"
    )

    parser.add_argument(
        "--save_w_std",type=bool, default=True,
    )
    parser.add_argument(
        "--save_dir",type=str, default="/home/sy/textailor_CLAB/result/fake_samples",
    )


    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator().to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    if args.run_generate:
        generate(args, g_ema, device, mean_latent)
    elif args.run_generate_fake_samples:
        generate_fake_samples(args, g_ema, device, mean_latent)
    



