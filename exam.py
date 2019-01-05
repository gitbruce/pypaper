import cv2
import numpy as np
import os, sys
import warnings
import ImageUtils
import matplotlib.pyplot as plt

folderpath = './newImages/'
filepath = folderpath + "IMG_20181205_215808.jpg"

def teacher_remove(image_path):
	src = cv2.imread(image_path)  # shape: (1080, 1863, 3)
	image = src.copy()
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# CLAHE (Contrast Limited Adaptive Histogram Equalization)
	ImageUtils.clahe_hsv(hsv)
	mask = ImageUtils.get_hsvmask(hsv, 160, 70, 70, 30)
	mask = ImageUtils.erode_dilate(mask, 3)

	result = ImageUtils.swap_color(image, mask, 255)
	# ImageUtils.show_images([src, result], 1)
	return result


def student_remove(image_path):
	src = cv2.imread(image_path)  # shape: (1080, 1863, 3)
	image = src.copy()
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# CLAHE (Contrast Limited Adaptive Histogram Equalization)
	ImageUtils.clahe_hsv(hsv)
	mask = ImageUtils.get_hsvmask(hsv, 220/2, 40, 20, 110/2)
	mask = ImageUtils.erode_dilate(mask, 3)

	result = ImageUtils.swap_color(image, mask, 255)
	ImageUtils.show_images([src, result], 1)


def print_remove(image_path):
	src = teacher_remove(image_path)
	image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

	ret1, th1 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

	ImageUtils.show_images([image, th1], 1)


def test(image_path):
	src = cv2.imread(image_path)  # shape: (1080, 1863, 3)
	image = src.copy()
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	#CLAHE (Contrast Limited Adaptive Histogram Equalization)
	ImageUtils.clahe_hsv(hsv)
	mask = ImageUtils.get_hsvmask(hsv, 170, 90, 64, 15)
	mask = ImageUtils.erode_dilate(mask)
	result = ImageUtils.swap_color(image, mask, 255)
	ImageUtils.show_images([src, mask, result], 3)


def test2():
	src = './newImages/IMG_20181205_215808.jpg'
	img = cv2.imread(src)
	img = ImageUtils.resize(img, 1000)

	Z = img.reshape((-1, 3))
	# convert to np.float32
	Z = np.float32(Z)
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 12
	ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	cv2.imshow('res2', res2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def main():
	# teacher_remove(filepath)
	# student_remove(filepath)
	print_remove(filepath)
	# test('./newImages/post-dominant-color.png')
	# test2()
	# rotate()


if __name__ == "__main__":
	main()