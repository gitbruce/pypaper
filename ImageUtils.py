import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_images(images, cols=1, opencv=True, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.


    gimp uses H = 0-360, S = 0-100 and V = 0-100. But OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        elif opencv:
            image = image[:,:,::-1]
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


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


def get_hsvmask(hsvsrc, h_low, s_low, v_low, threshold, s_high=None, v_high=None):
    h_high = h_low + threshold
    if not s_high:
        s_high = 255
    if not v_high:
        v_high = 255
    if h_high > 180:
        mask_lower1 = np.array([h_low, s_low, v_low], np.uint8)
        mask_upper1 = np.array([180, s_high, v_high], np.uint8)
        mask1 = cv2.inRange(hsvsrc, mask_lower1, mask_upper1)
        mask_lower2 = np.array([0, s_low, v_low], np.uint8)
        mask_upper2 = np.array([(h_high-180), s_high, v_high], np.uint8)
        mask2 = cv2.inRange(hsvsrc, mask_lower2, mask_upper2)
        return mask1 + mask2
    else:
        mask_lower = np.array([h_low, s_low, v_low], np.uint8)
        mask_upper = np.array([h_low + threshold, s_high, v_high], np.uint8)
        mask = cv2.inRange(hsvsrc, mask_lower, mask_upper)
        return mask


def erode_dilate(src, kernelsize=3, iteration=1, byReference=True):
    if byReference:
        copy = src
    else:
        copy = src.copy()
    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    # copy = cv2.erode(copy, kernel, iterations=iteration)
    copy = cv2.dilate(copy, kernel, iterations=iteration)
    copy = cv2.erode(copy, kernel, iterations=iteration)
    copy = cv2.dilate(copy, kernel, iterations=iteration)
    # mask = ImageUtils.swap_color(image, mask, 255)
    # erosion = ImageUtils.swap_color(image, erosion, 255)
    # dilation = ImageUtils.swap_color(image, dilation, 255)
    # opening = ImageUtils.swap_color(image, opening, 255)
    # closing = ImageUtils.swap_color(image, closing, 255)
    # gradient = ImageUtils.swap_color(image, gradient, 255)
    return copy


def swap_color(src, mask, destColor, byReference=True):
    if byReference:
        copy = src
    else:
        copy = src.copy()
    copy[mask > 0] = destColor
    return copy


def clahe_hsv(hsvsrc, clipLimit=2.0, tileGridSize=8):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    hsvsrc[:, 0, :] = clahe.apply(hsvsrc[:, 0, :])
    return hsvsrc
