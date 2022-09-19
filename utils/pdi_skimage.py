import matplotlib.pyplot as plt
import numpy as np
from skimage import (
    filters,
    color,
    exposure,
    segmentation,
    feature
)

def rgb2gray(image):
    """ Returns gray image.

        Arguments:
            image {numpy.array} -- RGB image
        Returns:
            gray_image {numpy.array} -- Grayscale image
    """
    return color.rgb2gray(image)

def rgb2hsv(image):
    """ Returns HSV image.

        Arguments:
            image {numpy.array} -- RGB image
        Returns:
            hsv_image {numpy.array} -- HSV image
    """
    return color.rgb2hsv(image)

def rgb2lab(image):
    """ Returns Lab image.

        Arguments:
            image {numpy.array} -- RGB image
        Returns:
            lab_image {numpy.array} -- Lab image
    """
    return color.rgb2lab(image)

def equalize_hist(hist):
    """ Returns equalized hist.

        Arguments:
            hist {numpy.array} -- Histogram
        Returns:
            equalized_hist {numpy.array} -- Equalized Histogram
    """
    return exposure.equalize_hist(hist)

def equalize_image(image):
    """ Returns equalized image.

        Arguments:
            image {numpy.array} -- RGB image
        Returns:
            equalized_image {numpy.array} -- Equalized image
    """
    hsv = rgb2hsv(image)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    return color.hsv2rgb(hsv)

def median_filter(gray_image):
    """ Returns filtered image with median filter.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
        Returns:
            gray_image_filtered {numpy.array} -- Grayscale image filtered
    """
    return filters.median(gray_image)

def gaussian_filter(gray_image, sigma: int):
    """ Returns filtered image with gaussian filter.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
            sigma {number} -- Standard Deviation to gaussian filter
        Returns:
            gray_image_filtered {numpy.array} -- Grayscale image filtered
    """
    return filters.gaussian(gray_image,  sigma=sigma, channel_axis=True)

def otsu(image):
    """ Returns binarized image with otsu threshold.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
        Returns:
            image_binarized {numpy.array} -- Image binarized
    """
    return filters.threshold_otsu(image)

def niblack(gray_image):
    """ Returns binarized image with niblack threshold.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
        Returns:
            image_binarized {numpy.array} -- Image binarized
    """
    return filters.threshold_niblack(gray_image)

def sauvola(gray_image):
    """ Returns binarized image with sauvola threshold.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
        Returns:
            image_binarized {numpy.array} -- Image binarized
    """
    return filters.threshold_sauvola(gray_image)

def sobel(gray_image):
    """ Returns segmented image with sobel edge detection.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
        Returns:
            image_binarized_edge {numpy.array} -- Image binarized with sobel edge detection
    """
    return filters.sobel(gray_image)

def canny(image, sigma: int):
    """ Returns binarized image with canny edge detection.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
            sigma {float} -- Standard Deviation to gaussian filter
        Returns:
            image_binarized {numpy.array} -- Image binarized with canny edge detection
    """
    return feature.canny(image, sigma=sigma)

def active_countour(gray_image, name: str, x1: int, x2: int, r: int):
    """ Save segmented image with active contours in memory.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
            name {str} -- Name of file to  be saved
            x1 {int} -- center first point
            x2 {int} -- center second point
            r {int} -- radius of the circle
    """
    x1 = x1 + r*np.cos(np.linspace(0, 2*np.pi, 500))
    x2 = x2 + r*np.sin(np.linspace(0, 2*np.pi, 500))
    snake = np.array([x1, x2]).T
    img_snake = segmentation.active_contour(gray_image, snake)

    plt.imshow(gray_image)
    plt.plot(img_snake[:, 0], img_snake[:, 1], '-b', lw=5)
    plt.savefig(name)
    plt.close()

def chan_vese(gray_image, max_iter: int = 100):
    """ Returns segmented image with chan-vese segmentation.

        Arguments:
            gray_image {numpy.array} -- Grayscale image
            max_iter {int} -- Max Iteractor
        Returns:
            image_segmented {numpy.array} -- Image segmented with canny edge detection
    """
    return segmentation.chan_vese(gray_image, max_iter=max_iter, extended_output=True)

def mark_boundaries(image, n_segment: int =100, compactness: float =1):
    """ Returns segmented image with mark boundaries.

        Arguments:
            image {numpy.array} -- RGB image
            n_segment {int} -- Approximate number of labels
            compactness {float} -- Balances color proximity and space proximity
        Returns:
            image_segmented {numpy.array} -- Image segmented with mark boundaries
    """
    seg = segmentation.slic(image, n_segments=n_segment, compactness=compactness)
    return segmentation.mark_boundaries(image, seg)

def slic(image, n_segment: int =50, compactness: float =10):
    """ Returns segmented image with ative clusterization slic.

        Arguments:
            image {numpy.array} -- RGB image
            n_segment {int} -- Approximate number of labels
            compactness {float} -- Balances color proximity and space proximity
        Returns:
            image_segmented {numpy.array} -- Image segmented with ative clusterization slic
    """
    seg = segmentation.slic(image, n_segments=n_segment, compactness=compactness)
    color.label2rgb(seg, image, kind = 'avg')
    return image

def felzenszwalb(image, scale: float =2, sigma: float =5, min_size: int =100):
    """ Returns segmented image with felzenszwalb.

        Arguments:
            image {numpy.array} -- RGB image
            scale {float} -- clusters proportion
            sigma {float} -- Stardard Deviation to gaussian filter
            min_size {int} -- Minimun component size
        Returns:
            image_segmented {numpy.array} -- Image segmented with felzenszwalb
    """
    seg = segmentation.felzenszwalb(image, scale = scale, sigma= sigma, min_size= min_size)
    return segmentation.mark_boundaries(image, seg)
