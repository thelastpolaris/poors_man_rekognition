import abc
import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        pass

    @abc.abstractmethod
    def process(self, data):
        pass

class ResizeImage(Preprocessor):
    def __init__(self, min_width, min_height):
        self._min_width = min_width
        self._min_height = min_height

    def process(self, image):
        height = image.shape[0]
        width = image.shape[1]

        if width > self._min_width or height > self._min_height:
            ratio = self._min_width/width
            image = cv2.resize(image, dsize=(int(width*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)

        return image

class InvertColors(Preprocessor):
    def process(self, image):
        return np.invert(image)

class Lambda(Preprocessor):
    def __init__(self, lambda_func):
        self._lambda_func = lambda_func

    def process(self, data):
        return self._lambda_func(data)
