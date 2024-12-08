import customtkinter

class ImageSettingsbutton(customtkinter.CTkButton):
    def __init__(self, master, configuration,  **kwargs):
        self._config = configuration
        super().__init__(
            master,
            width = self._config.width,
            height = self._config.height
        )
        
    def place(self):
        coordinate = self._config.placement
        super().place(**{"x" : coordinate.x, "y" : coordinate.y})