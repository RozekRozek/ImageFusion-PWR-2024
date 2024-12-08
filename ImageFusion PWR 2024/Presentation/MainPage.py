import customtkinter
from Presentation.Components.ImageDisplay.ImageDisplayLarge import ImageDisplayLarge 
from Presentation.Components.ImageDisplay.ImageDisplaySmall import ImageDisplaySmall
from Presentation.Components.Buttons.ImageSettingsButton import ImageSettingsbutton
from Presentation.Components.Buttons.ArrowButton import ArrowButton
from Presentation.Components.Panels.QualityAssesionPanel import QualityAssesionPanel
from Application.ImageTransformation.ImageTransformer import ImageTransformer
from ConfigurationProvider import configurationProvider
from Application.ImageTransformation.Transformations import TRANSFORMATIONS_DICT

class MainPage(customtkinter.CTk):
    def __init__(self, configuration, **kwargs):
        super().__init__()
        self.title("ImageFusion PWR 2024")
        self.geometry(configuration.geometry)
        
        imageDisplayLarge = ImageDisplayLarge(self)
        imageDisplaySmallLeft = ImageDisplaySmall(
            self, configurationProvider.GetConfiguration("ImageDisplaySmallLeft"), "MRI", **kwargs)
        imageDisplaySmallRight = ImageDisplaySmall(
            self, configurationProvider.GetConfiguration("ImageDisplaySmallRight"), "CT", **kwargs)
        
        imageTransformer = ImageTransformer(imageDisplayLarge)
        imageTransformer.FuzeImages()
        
        def _FuzionFunctionSwappingCommand(value):
                imageTransformer.SwapFuzionMethod(TRANSFORMATIONS_DICT[value])
                qualityAssesionPanel.RefreshQuality()
                
        options = list(TRANSFORMATIONS_DICT.keys())
        fusionMethodComboBox = customtkinter.CTkComboBox(self, values=options, command=_FuzionFunctionSwappingCommand)
        fusionMethodComboBox.set("Model Fuzji")
        fusionMethodComboBox.place(
            x = configurationProvider.GetConfiguration("FuzeMethodSelector").placement.x,
            y = configurationProvider.GetConfiguration("FuzeMethodSelector").placement.y,
        )
    
        smallImageDisplays = [imageDisplaySmallRight, imageDisplaySmallLeft]

        qualityAssesionPanel = QualityAssesionPanel(self)
        
        def _ContrastStretchingCommand():
                imageTransformer.SwitchContrastStretching()
                qualityAssesionPanel.RefreshQuality()
                
        contrastStretchingSwitch = customtkinter.CTkSwitch(
            self,
            configurationProvider.GetConfiguration("ContrastStretchingSwitch").width,
            configurationProvider.GetConfiguration("ContrastStretchingSwitch").height,
            text = "Rozciąganie kontrastu",
            command = _ContrastStretchingCommand)
        
        contrastStretchingSwitch.place(
            x = configurationProvider.GetConfiguration("ContrastStretchingSwitch").placement.x,
            y = configurationProvider.GetConfiguration("ContrastStretchingSwitch").placement.y,
        )
        
        arrowButtonDown = ArrowButton(
            self,
            configurationProvider.GetConfiguration("ArrowButtonDown"),
            smallImageDisplays, -1, imageTransformer,
            qualityAssesionPanel, **kwargs)
        
        arrowButtonUp = ArrowButton(
            self,
            configurationProvider.GetConfiguration("ArrowButtonUp"),
            smallImageDisplays, 1, imageTransformer,
            qualityAssesionPanel, **kwargs)
        
        imageDisplayLarge.place()
        imageDisplaySmallLeft.place()
        imageDisplaySmallRight.place()
        
        arrowButtonDown.place()
        arrowButtonUp.place()
        
        qualityAssesionPanel.place()