'''
Decodes a PANDA-style dataset, passes bboxes into HRNet, and outputs
COCO-style annotations.
After running this, use panda_crop.py to randomly crop the results
and form them into a complete COCO dataset.

Use --json argument to specify location of PANDA bbox json file
'''

import json
import sys
import argparse
from tqdm import tqdm

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time

import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

from multiprocessing import Queue, Process, Manager


CTX = torch.device('cuda')

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17


def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)


def draw_bbox(bbox,img):
    """draw the detected bounding box on the image.
    :param img:
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = x1 + bbox[2]
    y2 = y1 + bbox[3]
    box = [(x1,y2), (x2,y1)]
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(bbox, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : [x y w h]
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

#    bottom_left_corner = box[0]
#    top_right_corner = box[1]
#    box_width = top_right_corner[0]-bottom_left_corner[0]
#    box_height = top_right_corner[1]-bottom_left_corner[1]
#    bottom_left_x = bottom_left_corner[0]
#    bottom_left_y = bottom_left_corner[1]

    top_right_x = bbox[0]
    top_right_y = bbox[1]
    box_width = bbox[2]
    box_height = bbox[3]

    center[0] = top_right_x + box_width * 0.5
    center[1] = top_right_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def convert_bbox(bbox, imgw, imgh):
    x1 = bbox['tl']['x'] * imgw
    y1 = bbox['tl']['y'] * imgh
    x2 = bbox['br']['x'] * imgw
    y2 = bbox['br']['y'] * imgh

    w = x2 - x1
    h = y2 - y1

    return [x1, y1, w, h]


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--json', type=str)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--showFps',action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg,args)

    path = args.json
    f = open(path)

    data = json.load(f)

    z = zip(data.keys(), data.values())
    z = list(z)

    images = []
    annotations = []
    ann_id = 1

    workers = 8
    worker_inputs = [[] for i in range(workers)]
    for i in range(len(z)):
        worker_inputs[i % workers].append((i, z[i]))

    # Multiprocess inputs and outputs
    # Must use managed queues; workers will not join with large outstanding queues otherwise
    procs = []
    m = Manager()
    images = m.Queue()
    annotations = m.Queue()

    for input in worker_inputs:
        p = Process(target=worker_process, args=(cfg, input, images, annotations))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    images.put('end')
    annotations.put('end')

    # Dump queues to lists to json
    images_list = []
    annos_list = []

    for i in iter(images.get, 'end'):
        images_list.append(i)
    for a in iter(annotations.get, 'end'):
        annos_list.append(a)

    json.dump(annos_list, open('annotations.json', 'w'))
    json.dump(images_list, open('images.json', 'w'))


def worker_process(cfg, input, images, annos):
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    process_id = input[0][0]
    ann_id = 0

    pbar = tqdm(total = len(input), position=process_id, desc=f'Worker {process_id}', smoothing=0)
    for idx, (filename, d) in input:
#        print(f'{idx}: {filename}')
        image_bgr = cv2.imread('path_to_images_of_primary_dataset' + filename) #Type the address to images folder of the primary dataset
        image = image_bgr[:, :, [2,1,0]]

        imgw = d['image size']['width']
        imgh = d['image size']['height']

        img_data = {'license': 1,
                    'file_name': filename,
                    'coco_url': '127.0.0.1',
                    'height': imgh,
                    'width': imgw,
                    'date_captured': '1970-01-01 00:00:00',
                    'flickr_url': '127.0.0.1',
                    'id': idx}
#        images.append(img_data)
        images.put(img_data)

        bboxes = [obj['rects']['full body'] for obj in d['objects list'] if obj['category'] == 'person']
        for bbox in bboxes:
            bbox = convert_bbox(bbox, imgw, imgh)
            ann = {'num_keypoints':17,
                   'image_id':idx,
                   'category_id':1,
                   'id':process_id*10000+ann_id,
                   'segmentation':[[bbox[0], bbox[1],
                                    bbox[0]+bbox[2], bbox[1],
                                    bbox[0]+bbox[2], bbox[1]+bbox[3],
                                    bbox[0], bbox[1]+bbox[3]]],
                   'bbox':bbox,
                   'area':bbox[2]*bbox[3],
                   'iscrowd':0}
            kps = []
            ann_id += 1

            center, scale = box_to_center_scale(bbox, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])

            image_pose = image.copy()

            try:
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
            except Exception as e:
                print(e)
                continue

            # pose_preds is shape [17][2]
            for i in range(len(pose_preds[0])):
                kps.append(float(pose_preds[0][i][0]))
                kps.append(float(pose_preds[0][i][1]))
                kps.append(2)
#                print(f'{pose_preds[0][i]}')

            draw_pose(pose_preds[0], image_bgr)

            ann.update({'keypoints':kps})
#            annotations.append(ann)
            annos.put(ann)
#            break

        pbar.update()
#        break

#        cv2.imwrite('out.jpg', image_bgr)
    pbar.close()


if __name__=='__main__':
    main()
