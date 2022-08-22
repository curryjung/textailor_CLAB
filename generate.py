import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm


def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            if args.style_mix ==False:
                sample_z = torch.randn(args.sample, args.latent, device=device) #styles
                sample_c = torch.randn(args.sample, 16, device=device) #content

                sample, _ = g_ema(
                    sample_c, [sample_z]
                )

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
        "--pics", type=int, default=5, help="number of images to be generated"
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
        default="./checkpoint/080000.pt",
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

    generate(args, g_ema, device, mean_latent)
