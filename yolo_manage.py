import os
import glob
import shutil
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imgaug as ia
import cv2
from imgaug import augmenters as iaa
from termcolor import colored
from sympy import *
import math
from scipy import misc
import random


def extract_labeled(from_dir, to_dir):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    os.chdir(from_dir)
    print(colored('Moving Completed Files From: ', None), colored(from_dir, 'blue'))
    for file_txt in glob.glob("*.txt"):
        shutil.move(f'{from_dir}/{file_txt}', f'{to_dir}/{file_txt}')
        file_img = file_txt.replace('.txt', '.jpg')
        shutil.move(f'{from_dir}/{file_img}', f'{to_dir}/{file_img}')
    print(colored('Finished Moving Files to: ', None), colored(to_dir, 'blue'))


def yolo_class_counter(obj_file, labeled_data):
    classes = open(obj_file, "r").read().split()
    class_count_dict = dict(zip(classes, [0] * len(classes)))
    os.chdir(labeled_data)
    for file_txt in glob.glob("*.txt"):
        try:
            class_num = int(open(f'{labeled_data}/{file_txt}', "r").read().split()[0])
            class_count_dict[classes[class_num]] = class_count_dict.get(classes[class_num], 0) + 1
        except IndexError:
            print(colored('Missing label for: ', None), colored(file_txt, 'red'))
    return class_count_dict


def data_viz(count):
    plt.figure(figsize=(12, 6))
    alphab = list(count.keys())
    frequencies = list(count.values())

    pos = np.arange(len(alphab))
    width = 1.0  # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(alphab)

    plt.bar(pos, frequencies, width, color='r')
    plt.show()


def move_odd_classes(obj_file, labeled_data):
    classes = open(obj_file, "r").read().split()
    class_count_dict = dict(zip(classes, [0] * len(classes)))
    os.chdir(labeled_data)
    for file_txt in glob.glob("*.txt"):
        try:
            class_num = int(open(f'{labeled_data}/{file_txt}', "r").read().split()[0])
            if class_num in [0, 2, 3, 4, 5, 7, 9, 10, 11, 12]:
                file_img = file_txt.replace('.txt', '.jpg')
                shutil.move(f'{labeled_data}/{file_img}', f'H:/UGA MASTERS/WIM_Project/data/odd/{file_img}')
        except IndexError:
            print(colored('Missing label for: ', None), colored(file_txt, 'red'))


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def im_aug_transpose_labels(constant_angle, num_im, angle, blur, color_var, path='data/sorted', outfile='data/aug_data', labeled=False):
    print(colored(f'Generating {num_im} Augmented Images for Every Item Scraped', 'blue'))
    # rotation angle
    if not os.path.exists(outfile):  # place to save augments and labels
        os.makedirs(outfile)
    for file in glob.glob(f"{path}/*.jpg"):  # only loop over images not labels
        for idxer in range(num_im):

            theta = random.randint(-1 * angle, angle) * (np.pi / 180)
            # aug options
            seq = iaa.Sequential([
                iaa.Affine(rotate=(theta * (180 / np.pi), theta * (180 / np.pi))),
                iaa.AdditiveGaussianNoise(scale=(0, blur)),
                iaa.Crop(percent=(0, 0.0)),
                iaa.Multiply((1 - color_var, 1 + color_var), per_channel=0.5)
            ])

            image = imageio.imread(f'{file}')
            img = cv2.imread(file)
            images = [image] * num_im
            h_im, w_im, c = image.shape
            images_aug = seq(images=images)

            txt_file = file.replace('.jpg', '.txt')  # read intial label
            # open label file fromated
            # <object-class> <x>  <y>  <absolute_ width> /<image_width>  <absolute_height> /<image_height>
            text = open(txt_file, "r").read().split()
            # convert box width and height to pixel space
            # list of [ [xc_bbox1, yc_bbox1] , [w_bbox1, h_bbox1] ]
            scaled_values = [[float(x) * w_im, float(y) * h_im] for x, y in [[text[1], text[2]], [text[3], text[4]]]]
            # generate bbox for the int label
            bbox1 = [[0, 0], [0, 0]]  # [[p1x, p1y], [p2x, p3y]]
            for point_idx, (sign) in enumerate([1, -1]):
                for idx in range(2):
                    # convert to bbox format e.g. point1x = xc - w/2
                    bbox1[point_idx][idx] = scaled_values[0][idx] - (sign * (scaled_values[1][idx])) / 2
            # finished converting to bbox notation

            # step 1 # solve systems of equations for true height and width of vehicle and true points
            r_angle = constant_angle * (np.pi / 180)
            h, w = symbols('h, w')
            eq1 = h * np.sin(r_angle) + w * np.cos(r_angle) - scaled_values[1][0]
            eq2 = h * np.cos(r_angle) + w * np.sin(r_angle) - scaled_values[1][1]
            h_true, w_true = nsolve([Eq(eq1), Eq(eq2)], [h, w], [1, 1])
            # true points
            points_true = [[bbox1[0][0] + h_true * np.sin(r_angle), bbox1[0][1]],
                           [bbox1[1][0] - h_true * np.sin(r_angle), bbox1[1][1]]]

            # step 2 # convert format and rotate
            # scale image to make the center (0,0)
            points = [[(x / w_im) * 2 - 1, ((y / h_im) * 2 - 1)] for x, y in points_true]
            # rotate
            points[0] = rotate([0, 0], points[0], theta)
            points[1] = rotate([0, 0], points[1], theta)
            # convert back
            points = [[(x + 1) * (w_im / 2), (((y + 1) * (h_im / 2)))] for x, y in points]

            # step 3 # add angle loss back to the width if theta > 0 or to the height if theta < 0
            if theta > 0:
                points[0][0] = points[0][0] - h_true * np.sin(theta + r_angle)
                points[1][0] = points[1][0] + h_true * np.sin(theta + r_angle)
            else:
                points[0][1] = points[0][1] + w_true * np.sin(theta + r_angle)
                points[1][1] = points[1][1] - w_true * np.sin(theta + r_angle)

            for idx, (scale) in enumerate([w_im, h_im]):
                val1, val2 = points[0][idx], points[1][idx]
                points[0][idx] = ((val1 + val2) / 2) / scale
                points[1][idx] = abs(val1 - val2) / scale

            YOLO_FORMAT = ''.join([f'{text[0]} '] + [f'{round(val, 6)} ' for val in points[0] + points[1]])
            print(YOLO_FORMAT)

            name = file.replace(path, outfile)
            name = name.replace('.jpg', '')
            cv2.imwrite(f'{name}-aug-{idxer}.jpg', cv2.cvtColor(images_aug[0], cv2.COLOR_BGR2RGB))
            with open(f'{name}-aug-{idxer}.txt', "w") as text_file:
                text_file.write(YOLO_FORMAT)

