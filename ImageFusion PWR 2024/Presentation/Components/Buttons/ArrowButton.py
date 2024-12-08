import customtkinter
from PIL import Image
from Presentation.Components.ImageDisplay.ImageDisplaySmall import ImageDisplaySmall
from Application.ImageTransformation.ImageTransformer import ImageTransformer

class ArrowButton(customtkinter.CTkButton): 
    def __init__(self, master, configuration, imageDisplays : list[ImageDisplaySmall], direction, imageTransformer,
                 qualityAssesionPanel, **kwargs):
        self._config = configuration
        self.imageDisplays = imageDisplays
        self.direction = direction
        self.qualityAssesionPanel = qualityAssesionPanel
        self.imageTransformer : ImageTransformer = imageTransformer
        super().__init__(
            master,
            width = self._config.width,
            height = self._config.height,
            image = customtkinter.CTkImage(
                light_image = Image.open(self._config.imagePath),
                dark_image = Image.open(self._config.imagePath),
                size = (30, 30)),  
            command = self.shiftImages,
            text = None,
        )
        
    def place(self):
        coordinate = self._config.placement
        super().place(**{"x" : coordinate.x, "y" : coordinate.y})
        
    def shiftImages(self):
        for imageDisplay in self.imageDisplays:
            imageDisplay : ImageDisplaySmall
            imageDisplay.ShiftImage(self.direction)
        self.imageTransformer.FuzeImages()
        self.qualityAssesionPanel.RefreshQuality()