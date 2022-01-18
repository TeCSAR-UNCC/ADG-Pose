'''
Generate some basic quantitative measures about the character of a dataset:
how often people appear in an image, how large they are in that image.
Works on any COCO-style dataset.

Usage: python measure_bboxes.py {your_dataset_folder}/annotations/person_keypoints_{train or val}.json
'''

import json
import plotille
import sys
import collections
from tqdm import tqdm
from statistics import stdev


def check_bboxes(imgd, annd):
    detections = 0
    sizes = []

    imgw = imgd['width']
    imgh = imgd['height']
    img_area = imgw * imgh

    for a in annd:
        detections += 1
        w = a['bbox'][2]
        h = a['bbox'][3]
        area = w*h

        # Not sure why, but a few erroneous huge bboxes, so cap to 100%
        sizes.append(min(100 * area / img_area, 100))

    return [detections], sizes


def main():
    data = json.load(open(sys.argv[1]))

    image_data = data['images']
    anno_data = data['annotations']
    anno_data = [a for a in anno_data if a['category_id'] == 1]

    # Presort annotation data by images
    new_anno_data = collections.defaultdict(list)
    for a in anno_data:
        new_anno_data[a['image_id']].append(a)
    anno_data = new_anno_data

    detections = []
    sizes = []

    for imgd in tqdm(image_data):
#        annd = [a for a in anno_data if a['image_id'] == imgd['id']]
        annd = anno_data[imgd['id']]
        d, s = check_bboxes(imgd, annd)

        detections += d
        sizes += s

    print(data['info']['description'])
    print('Detections per image:')
    print(plotille.histogram(detections, x_min=0, x_max=max(detections)))
    avg = sum(detections) / len(detections)
    std = stdev(detections)
    print(f'avg: {avg:.2f}, std: {std:.2f}')

    print()
    print('Bbox size as percentage of image area per image:')
    print(plotille.histogram(sizes, x_min=0, x_max=max(sizes)))
    avg = sum(sizes) / len(sizes)
    std = stdev(sizes)
    print(f'avg: {avg:.2f}, std: {std:.2f}')


if __name__=='__main__':
    main()
