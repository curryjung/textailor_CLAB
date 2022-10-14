import os
import argparse
from dataset import IMGUR5K_Handwriting
from torch.utils import data
from OCR.demo import demo_accuracy

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def accuracy(args, loader):
    loader = sample_data(loader)
    final_acc=0
    for i in range(0, 3133):
        style_img, style_gray_img, style_label, content_gray_img, content_label, fake_img = next(loader)
        style_label_low = [l.lower() for l in style_label]
        pred = demo_accuracy(args, fake_img)

        acc = 0
        for i in range(0, args.batch):
            if style_label_low[i]==pred[i]:
                acc+=1

        final_acc+=acc
        print(final_acc)

    print(f"final : {(final_acc/50128)*100}")

    return acc

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--dataset_dir", type=str, default='/mnt/f06b55a9-977c-474a-bed0-263449158d6a/text_dataset/datasets/IMGUR5K-Handwriting-Dataset', help='datset directory')
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")

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

    img_folder = args.dataset_dir+'/preprocessed'
    label_path = args.dataset_dir+'/label_dic.json'
    gray_text_folder = args.dataset_dir+'/gray_text'
    fake_img_folder = '/mnt/f06b55a9-977c-474a-bed0-263449158d6a/textailor_CLAB/fake_image'

    # dataset = IMGUR5K_Handwriting(args.img_folder, args.test_label_path, args.gray_text_folder, train=True)
    dataset = IMGUR5K_Handwriting(img_folder, label_path, gray_text_folder, train=True, fake_img_folder=fake_img_folder)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=False, distributed=args.distributed),
        drop_last=True,
    )
    
    acc = accuracy(args, loader)