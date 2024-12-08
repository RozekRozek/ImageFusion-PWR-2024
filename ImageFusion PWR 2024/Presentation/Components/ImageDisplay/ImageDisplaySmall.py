from Application.ImageProvider import imageProvider
import customtkinter

class ImageDisplaySmall(customtkinter.CTkLabel):
    def __init__(self, master, configuration, imageOrigin, **kwargs):
        self._config = configuration
        self.imageBuffer = imageProvider.MriBuffer if imageOrigin == "MRI" else imageProvider.CtBuffer
        self.Image = customtkinter.CTkImage(
            light_image = self.imageBuffer.GetCurrentImage(),
            dark_image = self.imageBuffer.GetCurrentImage(),
            size = (self._config.width, self._config.height))
        
        super().__init__(
            master,
            image = self.Image,
            width = self._config.width,
            height = self._config.height,
            text=""
            )
        
    def place(self):
        coordinate = self._config.placement
        super().place(**{"x" : coordinate.x, "y" : coordinate.y})
        
    def ShiftImage(self, direction):
        self.imageBuffer.ShiftIndex(direction)
        self.Image = customtkinter.CTkImage(
            light_image = self.imageBuffer.GetCurrentImage(),
            dark_image = self.imageBuffer.GetCurrentImage(),
            size = (self._config.width, self._config.height))
        super().configure(image = self.Image)
        super().__setattr__("image", self.Image)