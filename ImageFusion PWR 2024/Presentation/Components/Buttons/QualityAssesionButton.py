import customtkinter
from Presentation.Components.Panels.QualityAssesionPanel import QualityAssesionWindow
from ConfigurationProvider import configurationProvider

class QualityAssesionButton(customtkinter.CTkButton):
    def __init__(self, master, topLevel,  **kwargs):
        self.topLevel = topLevel
        self._config = configurationProvider.GetConfiguration("QualityAssesionButton")
        super().__init__(
            master,
            text =  self._config.text,
            width = self._config.width,
            height = self._config.height,
            command = self.ShowQualityAssesment
        )
        
    def place(self):
        coordinate = self._config.placement
        super().place(**{"x" : coordinate.x, "y" : coordinate.y})
        
    def ShowQualityAssesment(self):
        if self.topLevel is None or not self.topLevel.winfo_exists():
            self.topLevel = QualityAssesionWindow(self, master = self.master) 
            self.topLevel.focus() 
        else:
            self.topLevel.focus()  