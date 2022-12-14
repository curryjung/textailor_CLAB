#!/usr/bin/env python
import argparse
import json
import os
import random
import time

import torch
import torch.distributed as dist
from torchinfo import summary

try:
    from tqdm import tqdm
except:
    def tqdm(iterator, *args, **kwargs): return iterator

import data
import metrics
from model import StyleAwareDiscriminator
from model.augmentation import Augmentation, SimpleTransform
from mylib import misc, torch_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser = Augmentation.add_commandline_args(parser)
    parser = StyleAwareDiscriminator.add_commandline_args(parser)
    for k, v in metrics.__dict__.items():
        if "Evaluator" in k:
            parser = v.add_commandline_args(parser)

    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--extra-desc", nargs="+", default=[])

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--total-nimg", type=misc.str2int, default="5M")
    parser.add_argument("--batch-size", type=int, default=32)
    # Datasets.
    parser.add_argument("--train_content_dataset", default="../datasets/afhq/train")
    parser.add_argument("--train_style_dataset", default="../datasets/afhq/train")
    parser.add_argument("--eval-dataset", default="../datasets/afhq/val")
    parser.add_argument("--num-workers", type=int, default=4)
    # Logging.
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--snapshot_freq", type=int, default=5000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--save_freq", type=int, default=10000)
    # Evaluations.
    parser.add_argument("--evaluation", type=misc.str2bool, default=True)
    parser.add_argument("--fid-start-after", type=int, default=10000)
    # Misc.
    parser.add_argument("--allow-tf32", type=misc.str2bool, default=False)
    parser.add_argument("--cudnn-bench", type=misc.str2bool, default=True)

    parser.add_argument("--model_js", action='store_true', default=False)
    parser.add_argument("--num_patches", type=int, default=64)
    parser.add_argument("--intermediate_feat_num", type=int, default=8)
    parser.add_argument("--avg_pool",action='store_true', default=False)

    return parser.parse_args()


def option_from_args(args):
    if args.seed is None:
        args.seed = random.randint(0, 999)

    # Create output folder.
    assert os.path.isdir(args.train_content_dataset)  # TODO: support other formats.
    if "ffhq" in args.train_content_dataset:
        dataset = "ffhq"
    else:
        splits = "train", "test", "val", "eval", ""
        dataset = args.train_content_dataset.split("/")
        dataset = [x for x in dataset if x not in splits][-1]
        dataset = dataset.replace("_train", "").replace("_lmdb", "")  # NOTE for LSUNs.
    desc = dataset.replace("_", "") + f"-{args.image_size}x{args.image_size}"
    for d in args.extra_desc:
        desc += f"-{d}"

    prev_run_ids = []
    if os.path.isdir(args.out_dir):
        prev_run_ids = [int(x.split("-")[0]) for x in os.listdir(args.out_dir)]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(args.out_dir, f'{cur_run_id:03d}-{desc}')

    if args.load_size is None or args.load_size < args.image_size:
        args.load_size = args.image_size
    if args.crop_size is None:
        args.crop_size = args.image_size
    return args


def load_option(resume_dir):
    option_file = os.path.join(resume_dir, "option.json")
    assert os.path.isfile(option_file)
    with open(option_file, "r") as f:
        option_dict = json.load(f)
    option = argparse.Namespace(**option_dict)
    option.run_dir = resume_dir
    return option


def training_loop(model, opt, rank, world_size):
    assert opt.batch_size % world_size == 0
    start_step, cur_nimg = 0, 0
    start_time = time.time()

    checkpoint = torch_utils.load_checkpoint(opt.run_dir)
    if checkpoint is not None:
        start_step = checkpoint["step"]
        cur_nimg = checkpoint["nimg"]
        model.load(checkpoint)

    # import pdb; pdb.set_trace()

    content_datapipe = data.build_dataset(
        opt.train_content_dataset, Augmentation(**vars(opt)),
        seed=opt.seed, repeat=True,
    )
    style_datapipe = data.build_dataset(
        opt.train_style_dataset, Augmentation(**vars(opt)),
        seed=opt.seed, repeat=True,
    )

    content_dataloader = torch.utils.data.DataLoader(
        content_datapipe, batch_size=opt.batch_size // world_size,
        num_workers=opt.num_workers, pin_memory=True,
    )
    style_dataloader = torch.utils.data.DataLoader(
        style_datapipe, batch_size=opt.batch_size // world_size,
        num_workers=opt.num_workers, pin_memory=True,
    )

    content_dataiter = iter(content_dataloader)
    style_dataiter = iter(style_dataloader)

    eval_transform = SimpleTransform(opt.image_size)
    val_dataset = data.build_dataset(opt.eval_dataset, eval_transform)

    iters = opt.total_nimg // opt.batch_size
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(opt.run_dir)
            tb = True
        except ImportError:
            print("Cannot import 'tensorboard'. Skip tensorboard logging.")
            tb = False

        # Log model summary.
        with open(os.path.join(opt.run_dir, "log.txt"), "w") as f:
            lines = ["#Parameters\n"]
            for n, m in model.named_children():
                lines.append(f"{n}:\n{summary(m)}\n\n")
            f.writelines(lines)

        best_fid = float("inf")
        model.prepare_snapshot(val_dataset)

        #if opt.evaluation:
        #    knn_evaluator = metrics.KNNEvaluator(**vars(opt))
        #    mfid_evaluator = metrics.MeanFIDEvaluator(**vars(opt))

    for step in tqdm(range(start_step+1, iters+1)):
        content_xs = next(content_dataiter)
        style_xs = next(style_dataiter)

        content_xs = tuple(map(lambda x: x.to(rank, non_blocking=True), content_xs))
        style_xs = tuple(map(lambda x: x.to(rank, non_blocking=True), style_xs))

        model.set_input(step, content_xs, style_xs)
        loss_dict = model.training_step()
        # loss_dict = {"step":step}

        if rank != 0:
            continue 
        if step % opt.log_freq == 0:
            if tb:
                for k, v in loss_dict.items():
                    writer.add_scalar(k, v, step)
            elapsed = misc.readable_time(time.time() - start_time)
            loss_dict["elapsed"] = elapsed
            misc.report(loss_dict, run_dir=opt.run_dir, filename="log")

        if step % opt.snapshot_freq == 0:
            model.snapshot()

        cur_nimg = step * opt.batch_size
        is_best = False
        if opt.evaluation and (step % opt.eval_freq == 0):
        # if step % 2 == 0:
            result_dict = {"step": step, "nimg": f"{cur_nimg//10}k"}
            """
            if knn_evaluator.is_available():
                knn_dict = knn_evaluator.evaluate(model, step=step)
                result_dict.update(knn_dict)

            if mfid_evaluator.is_available() and step > opt.fid_start_after:
                fid_dict = mfid_evaluator.evaluate(model)
                is_best = fid_dict["mFID"] < best_fid
                best_fid = min(best_fid, fid_dict["mFID"])
                result_dict.update(fid_dict)
            """
            if tb:
                for k, v in result_dict.items():
                    if k == "step" or k == "nimg":
                        continue
                    print("key, value")
                    print(k, v)
                    # import pdb; pdb.set_trace()
                    writer.add_scalar("Metrics/"+k, v, step)
            misc.report(result_dict, run_dir=opt.run_dir, filename="metrics")

        if step%opt.save_freq==0 or step==iters or is_best:
            filename = f"networks-{cur_nimg//10:05d}k.pt"
            filename = os.path.join(opt.run_dir, filename)
            extra_state = {"step": step, "nimg": cur_nimg}
            state = model.get_state(ignore="recon_loss", **extra_state)
            torch.save(state, filename)


def main():
    args = parse_args()
    if args.resume is None:
        option = option_from_args(args)
    else:
        option = load_option(args.resume)
        print(f"resume training '{args.resume}'")

    if "LOCAL_RANK" not in os.environ.keys():
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"=> set cuda device = {rank}")
    torch.cuda.set_device(rank)
    print(f"=> allow tf32 = {option.allow_tf32}")
    torch.backends.cuda.matmul.allow_tf32 = option.allow_tf32
    torch.backends.cudnn.allow_tf32 = option.allow_tf32
    print(f"=> cuDNN benchmark = {option.cudnn_bench}")
    torch.backends.cudnn.benchmark = option.cudnn_bench

    if rank == 0:
        filename = os.path.join(option.run_dir, "code")
        misc.archive_python_files(os.getcwd(), filename)
        # Save training options.
        with open(os.path.join(option.run_dir, "option.json"), "w") as f:
            json.dump(vars(option), f, indent=4)

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        # Sync options across devices (required for seed and run_dir).
        broadcast_opts = [option]
        dist.broadcast_object_list(broadcast_opts)
        option = broadcast_opts[0]

    print(f"=> random seed = {option.seed}")
    torch_utils.set_seed(option.seed)
    
    if args.model_js:
        from model.model_js import StyleAwareDiscriminator

    
    model = StyleAwareDiscriminator(option)
        
    training_loop(model, option, rank, world_size)

if __name__ == "__main__":
    main()
