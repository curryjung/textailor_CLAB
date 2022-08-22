import os
import json
import cv2
import numpy as np

from PIL import Image

root_dir = "/hdd/datasets/IMGUR5K-Handwriting-Dataset"
dst_dir = os.path.join(root_dir, "preprocessed")

img_path = os.path.join(root_dir, "images")
label_path = os.path.join(root_dir,'dataset_info',"imgur5k_annotations.json")

with open(label_path) as f:
    annotation_file = json.load(f)

img_names = os.listdir(img_path)


img_cnt = 0

label_dic = {}


for j, img_name in enumerate(img_names):
    img_name = img_names[j]

    img_id = img_name.split('.')[0]

    if img_id not in annotation_file['index_to_ann_map'].keys():
        continue

    ann_id = annotation_file['index_to_ann_map'][img_id]
    annotations = [annotation_file['ann_id'][a_id] for a_id in ann_id]

    labels = [ann['word'] for ann in annotations if ann['word'] != '.']
    # x_center, y_center, width, height, angle
    boxes = [list(map(float, ann['bounding_box'].strip('[ ]').split(', ')))
                for ann in annotations if ann['word'] != '.']
    
    img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_COLOR)

    if img is None:
        continue

    for i in range(len(boxes)):

        box_targets = [cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), box[4] * (-1))) for box in boxes]
        box_targets = [np.concatenate((points.min(0), points.max(0)), axis=-1) for points in box_targets]
        box_target = np.int0(box_targets[i])      

        if len(labels[i]) < 2:
            print('len(labels[i]) < 2')
            continue
        elif box_targets[i][0] < 0 or box_targets[i][1] < 0 or box_targets[i][2] > img.shape[0] or box_targets[i][3] > img.shape[1]:
            print('boxes[i][0] < 0 or boxes[i][1] < 0 or boxes[i][2] > img.shape[1] or boxes[i][3] > img.shape[0]')     
            continue
        else:
            print(labels[i])
            save_name = str(img_cnt)+'.png'
            save_path = os.path.join(dst_dir, save_name)
            _, _, w, h, angle = boxes[i]

            # (x, y) coordinates of top left, top right, bottom right, bottom left corners

            # img_tmp = img[int(box_target[1]):int(box_target[3]), int(box_target[0]):int(box_target[2]),:]

            # img_tmp = Image.fromarray(img_tmp)
            # img_tmp = img_tmp.rotate(-angle)
            # img_tmp = np.array(img_tmp)

            # xc, yc = img_tmp.shape[1], img_tmp.shape[0]

            # img_tmp = img_tmp[int((yc-h)/2):int((yc+h)/2),int((xc-w)/2):int((xc+w)/2),:]

            # cv2.imwrite(save_path,img_tmp)
            label_dic[img_cnt]=labels[i]
            img_cnt += 1


with open(os.path.join(dst_dir, 'label_dic.json'), 'w') as f:
    json.dump(label_dic, f)




