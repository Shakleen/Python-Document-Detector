import cv2
import math
from scipy import ndimage
import numpy as np

class RotationCorrector:
    def __init__(self, output_process = False):
        self.output_process = output_process

    def __call__(self, image):
        img_before = image.copy()
        
        img_edges = cv2.Canny(img_before, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            img_edges, 
            1, 
            math.pi / 90.0, 
            100, 
            minLineLength = 100,
            maxLineGap = 5
        )
        print("Number of lines found:", len(lines))
        
        def get_angle(line): 
            x1, y1, x2, y2 = line[0]
            return math.degrees(math.atan2(y2 - y1, x2 - x1))

        median_angle = np.median(np.array([get_angle(line) for line in lines]))
        img_rotated = ndimage.rotate(
            img_before, 
            median_angle, 
            cval = 255,
            reshape = False
        )

        print("Angle is {}".format(median_angle))
        
        if self.output_process: 
            cv2.imwrite('output/10. tab_extract rotated.jpg', img_rotated) 

        return img_rotated


class Resizer:
    """Resizes image.

    Params
    ------
    image   is the image to be resized
    height  is the height the resized image should have. Width is changed by similar ratio.

    Returns
    -------
    Resized image
    """
    def __init__(self, height = 1280, output_process = False):
        self._height = height
        self.output_process = output_process


    def __call__(self, image):
        if image.shape[0] <= self._height: return image
        ratio = round(self._height / image.shape[0], 3)
        width = int(image.shape[1] * ratio)
        dim = (width, self._height)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
        if self.output_process: cv2.imwrite('output/resized.jpg', resized)
        return resized


class OtsuThresholder:
    """Thresholds image by using the otsu method

    Params
    ------
    image   is the image to be Thresholded

    Returns
    -------
    Thresholded image
    """
    def __init__(self, thresh1 = 0, thresh2 = 255, output_process = False):
        self.output_process = output_process
        self.thresh1 = thresh1
        self.thresh2 = thresh2


    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T_, thresholded = cv2.threshold(image, self.thresh1, self.thresh2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.output_process: cv2.imwrite('output/thresholded.jpg', thresholded)
        return thresholded


class FastDenoiser:
    """Denoises image by using the fastNlMeansDenoising method

    Params
    ------
    image       is the image to be Thresholded
    strength    the amount of denoising to apply

    Returns
    -------
    Denoised image
    """
    def __init__(self, strength = 7, output_process = False):
        self._strength = strength
        self.output_process = output_process


    def __call__(self, image):
        temp = cv2.fastNlMeansDenoising(image, h = self._strength)
        if self.output_process: cv2.imwrite('output/denoised.jpg', temp)
        return temp


class Closer:
    def __init__(self, kernel_size = 3, iterations = 10, output_process = False):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process


    def __call__(self, image):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self._kernel_size, self._kernel_size)
        )
        closed = cv2.morphologyEx(
            image, 
            cv2.MORPH_CLOSE, 
            kernel,
            iterations = self._iterations
        )

        if self.output_process: cv2.imwrite('output/closed.jpg', closed)
        return closed


class Opener:
    def __init__(self, kernel_size = 3, iterations = 25, output_process = False):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process


    def __call__(self, image):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self._kernel_size, self._kernel_size)
        )
        opened = cv2.morphologyEx(
            image, 
            cv2.MORPH_OPEN,
            kernel,
            iterations = self._iterations 
        )

        if self.output_process: cv2.imwrite('output/opened.jpg', opened)
        return opened


class EdgeDetector:
    def __init__(self, output_process = False):
        self.output_process = output_process


    def __call__(self, image, thresh1 = 50, thresh2 = 150, apertureSize = 3):
        edges = cv2.Canny(image, thresh1, thresh2, apertureSize = apertureSize)
        if self.output_process: cv2.imwrite('output/edges.jpg', edges)
        return edges