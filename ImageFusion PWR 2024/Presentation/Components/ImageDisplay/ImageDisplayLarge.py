from PIL import Image
from ConfigurationProvider import configurationProvider
import customtkinter

class ImageDisplayLarge(customtkinter.CTkLabel):
    def __new__(cls, master, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ImageDisplayLarge, cls).__new__(cls)
            cls.master = master
        return cls.instance
    
    def GetInstance(cls):
        return cls.instance
    
    def __init__(self, master):
        self._config = configurationProvider.GetConfiguration("ImageDisplayLarge")
        self.Image = customtkinter.CTkImage(
            light_image = Image.open("Images/placeholder.png"),
            dark_image = Image.open("Images/placeholder.png"),
            size = (self._config.width, self._config.height))
        
        super().__init__(
            self.master,
            image = self.Image,
            width = self._config.width,
            height = self._config.height,
            text=""
            )
        
    def place(self):
        coordinate = self._config.placement
        super().place(**{"x" : coordinate.x, "y" : coordinate.y})
        
    def LoadImage(self, image_bundle):
        self.PilImage, CV2Image = image_bundle
        self.CV2Image = CV2Image
        self.Image = customtkinter.CTkImage(
            light_image = self.PilImage,
            dark_image = self.PilImage,
            size = (self._config.width, self._config.height))
        super().configure(image = self.Image)
        super().__setattr__("image", self.Image) 
        
    def GetCurrentImageCV2(self):
        return self.CV2Image