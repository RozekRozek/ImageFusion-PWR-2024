from PIL import Image
from ConfigurationProvider import configurationProvider
import os 
import cv2
import numpy as np

class ImageProvider():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ImageProvider, cls).__new__(cls)
        return cls.instance
        
    def __init__(self):
        self.__config = configurationProvider.GetConfiguration("ImageProvider")
        self.MriBuffer = _ImageBuffer(self.__config.DefaultPath, self.__config.CtFolderName)
        self.CtBuffer = _ImageBuffer(self.__config.DefaultPath, self.__config.MriFolderName)
        
class _ImageBuffer():
    def __init__(self, defaultPath, folderName):
        self.folderPath = os.path.join(defaultPath, folderName)
        self.__allImages = sorted(os.listdir(self.folderPath))
        self.__imageCount = len(self.__allImages)
        self.__imageIndex = 0
        
    def ShiftIndex(self, direction):
        self.__imageIndex = self.__imageIndex + direction
        if self.__imageIndex == -1: self.__imageIndex = self.__imageCount - 1
        elif self.__imageIndex >= self.__imageCount: self.__imageIndex = 0

    def GetCurrentImage(self):
        return Image.open(os.path.join(self.folderPath, self.__allImages[self.__imageIndex]))
    
    def GetCurrentImageCV2(self):
        return cv2.imread(os.path.join(self.folderPath, self.__allImages[self.__imageIndex]))
    
__IMAGE_PROVIDER = ImageProvider()
imageProvider = __IMAGE_PROVIDER