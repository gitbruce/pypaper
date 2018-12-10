import cv2
import numpy as np
import os, sys
import warnings
import ImageUtils
import matplotlib.pyplot as plt

folderpath = './newImages/'
filepath = folderpath + "IMG_20181205_215808.jpg"

def teacher_removal(image_path):
	"""
    :param thresh: 二值化的阈值(thresh越小印章去除的效果越好，但其他的横线会有很多的空缺口，
    可能会导致后面表格识别的准确度降低)
    :param color: 0: blue. 1: green. 2: red.
    :return: new_image_path
    """
	image = cv2.imread(image_path)  # shape: (1080, 1863, 3)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	teacher1_lower = np.array([280/2, 50, 0])
	teacher1_upper = np.array([360/2,255,255])
	teacher1_mask = cv2.inRange(hsv, teacher1_lower, teacher1_upper)
	teacher1_remove = cv2.bitwise_not(image, image, mask=teacher1_mask)

	teacher2_lower = np.array([320/2, 100,80])
	teacher2_upper = np.array([360/2,255,255])
	teacher2_mask = cv2.inRange(hsv, teacher2_lower, teacher2_upper)
	teacher2_remove = cv2.bitwise_not(image, image, mask=teacher2_mask)
	# teacher_remove = cv2.morphologyEx(teacher_remove, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
	ImageUtils.show_images([image, teacher1_remove, teacher2_remove], 1)

	# f = plt.figure()
	# f.add_subplot(1, 2, 1)
	# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# plt.imshow(img)
	# f.add_subplot(1, 2, 2)
	# plt.imshow(image)
	# plt.show(block=True)

def test(image_path):
	image = cv2.imread(image_path)  # shape: (1080, 1863, 3)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	threshhold = 20

	teacher1_lower = np.array([(182-10)/2, 47, 60])
	teacher1_upper = np.array([(182)/2, 255, 255])
	mask = cv2.inRange(hsv, teacher1_lower, teacher1_upper)
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.erode(mask, kernel, iterations=10)
	mask = cv2.dilate(mask, kernel, iterations=10)
	mask = cv2.dilate(mask, kernel, iterations=10)
	mask = cv2.erode(mask, kernel, iterations=10)

	imgEdited = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imgGray = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
	imgGray = cv2.bitwise_and(imgGray, imgEdited)
	imgEdited1 = cv2.bitwise_and(image, imgEdited)
	imgEdited2 = image - imgEdited1

	ImageUtils.show_images([image, imgEdited, imgGray, imgEdited1, imgEdited2], 1)

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


import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage


def rotate():
	filepath = './exam/'

	for filename in os.listdir(filepath):
		if all([not filename.endswith('.png'), not filename.endswith('.jpg')]):
			continue
		img = cv2.imread(filepath + filename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 50, 150, apertureSize=3)

		# 霍夫变换
		lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
		for rho, theta in lines[0]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1 = int(x0 + 1000 * (-b))
			y1 = int(y0 + 1000 * (a))
			x2 = int(x0 - 1000 * (-b))
			y2 = int(y0 - 1000 * (a))
		if x1 == x2 or y1 == y2:
			continue
		t = float(y2 - y1) / (x2 - x1)
		rotate_angle = math.degrees(math.atan(t))
		if rotate_angle > 45:
			rotate_angle = -90 + rotate_angle
		elif rotate_angle < -45:
			rotate_angle = 90 + rotate_angle
		rotate_img = ndimage.rotate(img, rotate_angle)
		cv2.imwrite(filepath + 'processed/' + filename, rotate_img)

def main():
	# teacher_removal(filepath)
	# test('./newImages/rectangle.png')
	# test2()
	rotate()


if __name__ == "__main__":
	main()