import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2_img_proc

debug = False
# debug = True

HYPER_THRESH_1 = 27
HYPER_THRESH_2 = 2
HYPER_KERNEL = 3

def resize(orgimg, longEdge=1000):
    oldHeight, oldWidth, _ = orgimg.shape
    dim = None
    if (oldHeight > oldWidth):
        r = longEdge / oldHeight
        dim = (int(oldWidth * r), longEdge)
    else:
        r = longEdge / oldWidth
        dim = (longEdge, int(oldHeight * r))
    img = cv2.resize(orgimg, dim, interpolation=cv2.INTER_AREA)
    # showImage(img, 'resize')
    return img


def showImage(orgimg, title='image'):
    if debug:
        cv2.imshow(title, orgimg)
        cv2.waitKey(0)

def drawHoughLines(orgimg):
    gray = cv2.cvtColor(orgimg, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray,50,150,apertureSize = 3)
    edges = cv2.Canny(gray, 10, 50, apertureSize=3)
    showImage(edges, "edges")

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    print("Len of lines:", len(lines))
    # print lines

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(orgimg, (x1, y1), (x2, y2), (0, 0, 255), 2)

        showImage(orgimg, "HoughLines")

def findPaper(orgimg, longEdge=1000):
    setup(longEdge)
    resizedImg = resize(orgimg, longEdge)

    img = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, HYPER_THRESH_1, HYPER_THRESH_2)
    thresh = th3

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (HYPER_KERNEL, HYPER_KERNEL))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(eroded, kernel, iterations=1)
    showImage(eroded, "Eroded Image");
    showImage(dilation, "dilation Image");

    image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours size: ", len(contours))

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    print(rect)
    showImage(resizedImg, 'final')


    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area < 6000:
    #         continue
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(resizedImg, [box], 0, (0, 0, 255), 2)
    #     showImage(resizedImg, 'final')

    cropimg = crop_minAreaRect(resizedImg, rect)
    showImage(cropimg, 'cropimg ')
    return cropimg


def findDetail(orgimg, longEdge=1000):
    setup(longEdge)
    resizedImg = resize(orgimg, longEdge)

    img = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(img,(5,5),0)
    # Sobel算子，X方向求梯度

    th3 = cv2.adaptiveThreshold(gaussian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = th3

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(eroded, kernel, iterations=1)
    eroded = cv2.erode(dilation, kernel, iterations=1)
    dilation = cv2.dilate(eroded, kernel, iterations=1)
    showImage(eroded, "Eroded Image");
    showImage(dilation, "dilation Image");

    image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours size: ", len(contours))

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    rect = cv2.minAreaRect(cnt)
    if debug:
        box = cv2.boxPoints(rect)
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        length = 5
        cv2.circle(resizedImg, leftmost, length, (255, 0, 0), 8)
        cv2.circle(resizedImg, rightmost, length, (0, 255, 0), 8)
        cv2.circle(resizedImg, topmost, length, (0, 0, 255), 8)
        cv2.circle(resizedImg, bottommost, length, (255, 255, 0), 8)

        print(rect)
        print(box)
        box = np.int0(box)
        cv2.drawContours(resizedImg, [box], 0, (0, 0, 255), 2)
        showImage(resizedImg, 'final')


    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area < 6000:
    #         continue
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(resizedImg, [box], 0, (0, 0, 255), 2)
    #     showImage(resizedImg, 'final')

    cropimg = crop_minAreaRect(resizedImg, rect)
    showImage(cropimg, 'cropimg ')
    return cropimg


def crop_minAreaRect(img, rect):
    center, size, angle = rect
    # rotate img
    rows,cols = img.shape[0], img.shape[1]
    if angle < -45.0:
        angle += 90.0
        width, height = size[0], size[1]
        size = (height, width)

    M = cv2.getRotationMatrix2D(center,angle,1.0)
    imgWidth, imgHeight = (img.shape[0], img.shape[1])
    rotated = cv2.warpAffine(img, M, (imgHeight, imgWidth), flags=cv2.INTER_CUBIC)
    sizeInt = (np.int0(size[0]), np.int0(size[1]))
    uprightRect = cv2.getRectSubPix(rotated, sizeInt, center)
    return uprightRect


def setup(longEdge):
    global HYPER_THRESH_1
    global HYPER_THRESH_2
    global HYPER_KERNEL
    if longEdge == 2000:
        HYPER_THRESH_1 = 27
        HYPER_THRESH_2 = 4
        HYPER_KERNEL = 3
    else:
        HYPER_THRESH_1 = 27
        HYPER_THRESH_2 = 2
        HYPER_KERNEL = 3

def main():
    filepath = './exam/train/'
    outputpath = filepath + 'processed/'
    for filename in os.listdir(outputpath):
        os.remove(outputpath+filename)
    for filename in os.listdir(filepath):
        if all([not filename.endswith('.png'), not filename.endswith('.jpg')]):
            continue
        orgimg = cv2.imread(filepath+filename)
        resizedImg = findPaper(orgimg, 2000)
        resizedImg = findDetail(resizedImg, 1000)
        cv2.imwrite(outputpath+filename, resizedImg)

        mser = cv2.MSER_create()
        gray = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)
        vis = gray.copy()
        regions, _ = mser.detectRegions(gray)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(vis, hulls, 1, (0, 255, 0))
        cv2.imshow('after', vis)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
