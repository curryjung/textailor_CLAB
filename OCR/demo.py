import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from OCR.utils import CTCLabelConverter, AttnLabelConverter
from OCR.dataset import RawDataset, AlignCollate
from OCR.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(args, img_tensor, p_t, predict_text = False):
    """ model configuration """
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    model = Model(args)
    #print('model input parameters', 64, 256, args.num_fiducial, args.input_channel, args.output_channel,
    #      args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
    #      args.SequenceModeling, args.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    #print('loading pretrained model from %s' % args.saved_model)
    model.load_state_dict(torch.load(args.saved_model, map_location=device))

    # predict
    # model.eval()
    # with torch.no_grad():
    ocr_batch_size = img_tensor.size(0)
    image = img_tensor.to(device)
    length_for_pred = torch.IntTensor([args.batch_max_length] * ocr_batch_size).to(device)
    text_for_pred = torch.LongTensor(ocr_batch_size, args.batch_max_length + 1).fill_(0).to(device)

    if 'CTC' in args.Prediction:
        preds = model(image, text_for_pred)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * ocr_batch_size)
        _, preds_index = preds.max(2)
        # preds_index = preds_index.view(-1)
        preds_str = converter.decode(preds_index, preds_size)

    else:
        preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    pred_list=[]

    for pred in preds_str:
        if 'Attn' in args.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_list.append(pred)

    text, length = converter.encode(p_t, batch_max_length=25)
    closs_preds = model(image, text[:, :-1], is_train=False)
    closs_target = text[:, 1:]
    
    if predict_text==False:
        return pred_list
    else:
        return closs_preds, closs_target, pred_list

def demo_accuracy(args, img_tensor):
    """ model configuration """
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    model = Model(args)
    #print('model input parameters', 64, 256, args.num_fiducial, args.input_channel, args.output_channel,
    #      args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
    #      args.SequenceModeling, args.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    #print('loading pretrained model from %s' % args.saved_model)
    model.load_state_dict(torch.load(args.saved_model, map_location=device))

    # predict
    model.eval()
    with torch.no_grad():
        ocr_batch_size = img_tensor.size(0)
        image = img_tensor.to(device)
        length_for_pred = torch.IntTensor([args.batch_max_length] * ocr_batch_size).to(device)
        text_for_pred = torch.LongTensor(ocr_batch_size, args.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in args.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * ocr_batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)

        else:
            preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
    
        pred_list=[]

        for pred in preds_str:
            if 'Attn' in args.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_list.append(pred)

    return pred_list

# 명령어 : CUDA_VISIBLE_DEVICES=0 python3 demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder demo_image/ --saved_model TPS-ResNet-BiLSTM-Attn.pth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--ocr_batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=256, help='the width of the input image')
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

    """ vocab / character number configuration """
    if args.sensitive:
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()

    demo(args)
