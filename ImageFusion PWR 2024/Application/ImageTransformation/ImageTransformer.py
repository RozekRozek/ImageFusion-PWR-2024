from Presentation.Components.ImageDisplay.ImageDisplayLarge import ImageDisplayLarge
from Application.ImageTransformation.Transformations import *
from Application.ImageProvider import imageProvider

class ImageTransformer():
    def __new__(cls, largeImageDisplay):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ImageTransformer, cls).__new__(cls)
            return cls.instance
        
    def __init__(self, largeImageDisplay):
        self.mriImageBuffer = imageProvider.MriBuffer
        self.ctImageBuffer = imageProvider.CtBuffer
        self.largeImageDisplay : ImageDisplayLarge = largeImageDisplay
        self.currentTransformation = QNetworkFuseSilu
        self.stretchWhitescale = False
        
    def FuzeImages(self):
        self.largeImageDisplay.LoadImage(self._executeTransformation(self.currentTransformation))
        
    def _executeTransformation(self, transformation):
        pilImage, cv2Image =  transformation(self.mriImageBuffer.GetCurrentImageCV2(),
                              self.ctImageBuffer.GetCurrentImageCV2())

        return ContrastStretching(cv2Image) if self.stretchWhitescale else (pilImage, cv2Image)
    
    def SwitchContrastStretching(self):
        self.stretchWhitescale = not self.stretchWhitescale
        self.FuzeImages()
        
    def SwapFuzionMethod(self, fusionMethod):
        self.currentTransformation = fusionMethod
        self.FuzeImages()