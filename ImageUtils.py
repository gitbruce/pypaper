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
