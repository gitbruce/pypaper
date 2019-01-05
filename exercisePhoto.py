import cv2
import os
from ImageUtils import resize

debug = False

HYPER_THRESH_1 = 27
HYPER_THRESH_2 = 2
HYPER_KERNEL = 3


def main():
    filepath = '/home/bruce/Documents/ShareDocs/processed/'
    outputpath = filepath + 'toprint/'
    for filename in os.listdir(outputpath):
        os.remove(outputpath+filename)
    for filename in os.listdir(filepath):
        if all([not filename.endswith('.png'), not filename.endswith('.jpg')]):
            continue
        orgimg = cv2.imread(filepath+filename)
        resizedImg = resize(orgimg, 2000)
        ret1, th1 = cv2.threshold(resizedImg, 200, 255, cv2.THRESH_TRUNC)
        cv2.imwrite(outputpath+filename, th1)

if __name__ == '__main__':
    main()