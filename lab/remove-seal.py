#!/usr/bin/env python3
# coding: utf-8
# File: seal_removal.py
# Author: lxw
# Date: 7/25/18 8:56 AM
"""
去除图片中的印章
"""


import cv2
# import numpy as np
import os, sys
import warnings


warnings.filterwarnings("ignore")


def seal_removal(image_path, thresh=160, color=2):
    """
    :param thresh: 二值化的阈值(thresh越小印章去除的效果越好，但其他的横线会有很多的空缺口，
    可能会导致后面表格识别的准确度降低)
    :param color: 0: blue. 1: green. 2: red.
    :return: new_image_path
    """
    image = cv2.imread(image_path)  # shape: (1080, 1863, 3)

    channels = cv2.split(image)  # <list of ndarray>. shape of each ndarray in channels is (1080, 1863).
    # cv2.split(m[, mv]) → mv
    red_channel = channels[color]  # 0: blue. 1: green. 2: red.
    # cv2.imshow("red_channel", red_channel)

    # threshold the image, setting all fg pixels to 255 and all bg pixels to 0
    thresh_img = cv2.threshold(red_channel, thresh, 255, cv2.THRESH_BINARY)[1]  # thresh_img.shape: (1080, 1863).
    # cv2.threshold(src, thresh, maxval, type) → retval, dst

    # cv2.imshow("thresh_img", thresh_img)
    # cv2.waitKey(0)

    filepath = image_path.rsplit(".", 1)[0]  # image_path: "./images/demo.png"
    file_extension = os.path.basename(image_path).rsplit(".", 1)[1]
    new_image_path = "{0}_clean.{1}".format(filepath, file_extension).replace(" ", "_")
    cv2.imwrite(new_image_path, thresh_img)
    return new_image_path


if __name__ == "__main__":
    image_path = "./images/red-seal-1.png"
    if (len(sys.argv) > 1):
        image_path = "./images/"+str(sys.argv[1])

    print(image_path)
    seal_removal(image_path)
