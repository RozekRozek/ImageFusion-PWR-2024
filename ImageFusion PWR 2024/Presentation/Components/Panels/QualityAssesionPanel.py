import customtkinter as ctk
from ConfigurationProvider import configurationProvider
from Application.ImageProvider import imageProvider
from Presentation.Components.ImageDisplay.ImageDisplayLarge import ImageDisplayLarge
import numpy as np
import cv2
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity
from skimage import measure

CONFIG = configurationProvider.GetConfiguration("QualityAssesionPanel")

TEXTBOXHEIGHT = 30
TEXTBOXWIDTH = 330

MSE_PREFIX_PATTERN = "Błąd średniokwadratowy: Mri -> {} | CT -> {}"
MI_PREFIX_PATTERN = "Informacja wzajemna: Mri -> {} | CT -> {}"
EPI_PREFIX_PATTERN = "EPI: Mri -> {} | CT -> {}"
CORELLATION_PREFIX_PATTERN = "Współczynnik korelacji: Mri -> {} | CT -> {}"
PSNR_PREFIX_PATTERN = "PSNR: Mri -> {} | CT -> {}"
SSIM_PREFIX_PATTERN = "SSIM: Mri -> {} | CT -> {}"

class QualityAssesionPanel(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            CONFIG.width,
            CONFIG.height,
            **kwargs)
        self.MriImageBuffer = imageProvider.MriBuffer
        self.CtImageBuffer = imageProvider.CtBuffer 
        self.QualityHolder = QualityHolder(self.MriImageBuffer, self.CtImageBuffer)
        
        self.mseText = ctk.CTkLabel(
            self, width=TEXTBOXWIDTH, height=TEXTBOXHEIGHT, text ="mse", anchor = 'w')
        self.mseText.place(x = 7, y = 21)
        
        self.epiText = ctk.CTkLabel(
            self, width=TEXTBOXWIDTH, height=TEXTBOXHEIGHT, text ="entropy", anchor = 'w')
        self.epiText.place(x = 7, y = 65)

        self.psnrText = ctk.CTkLabel(
            self, width=TEXTBOXWIDTH, height=TEXTBOXHEIGHT, text ="psnr", anchor = 'w')
        self.psnrText.place(x = 7, y = 109)

        self.correlationText = ctk.CTkLabel(
            self, width=TEXTBOXWIDTH, height=TEXTBOXHEIGHT, text ="core", anchor = 'w')
        self.correlationText.place(x = 7, y = 153)
        
        self.mutualInformationText = ctk.CTkLabel(
            self, width=TEXTBOXWIDTH, height=TEXTBOXHEIGHT, text ="core", anchor = 'w')
        self.mutualInformationText.place(x = 7, y = 197)
        
        self.ssimText = ctk.CTkLabel(
            self, width=TEXTBOXWIDTH, height=TEXTBOXHEIGHT, text ="core", anchor = 'w')
        self.ssimText.place(x = 7, y = 241)
        
        self.RefreshQuality()
        
    def RefreshQuality(self):
        self.QualityHolder.Refresh()
        self.UpdateText()
        
    def UpdateText(self):
        self.mseText.configure(text = MSE_PREFIX_PATTERN.format(
            f"{self.QualityHolder.meanSquaredError[0]:.2f}",
            f"{self.QualityHolder.meanSquaredError[1]:.2f}"))
        
        self.epiText.configure(text = EPI_PREFIX_PATTERN.format(
            f"{self.QualityHolder.EPI[0]:.2f}",
            f"{self.QualityHolder.EPI[1]:.2f}"))
        
        self.mutualInformationText.configure(text = MI_PREFIX_PATTERN.format(
            f"{self.QualityHolder.mutualInformation[0]:.2f}",
            f"{self.QualityHolder.mutualInformation[1]:.2f}"))
        
        self.correlationText.configure(text = CORELLATION_PREFIX_PATTERN.format(
            f"{self.QualityHolder.corelationCoeficcient[0]:.2f}",
            f"{self.QualityHolder.corelationCoeficcient[1]:.2f}"))
        
        self.psnrText.configure(text = PSNR_PREFIX_PATTERN.format(
            f"{self.QualityHolder.PSNR[0]:.2f}",
            f"{self.QualityHolder.PSNR[1]:.2f}"))
        
        self.ssimText.configure(text = SSIM_PREFIX_PATTERN.format(
            f"{self.QualityHolder.SSIM[0]:.2f}",
            f"{self.QualityHolder.SSIM[1]:.2f}"))
        
    def place(self):
        coordinate = CONFIG.placement
        super().place(**{"x" : coordinate.x, "y" : coordinate.y})
    
class QualityHolder():
    def __init__(self, mriImageBuffer, ctImageBuffer):
        self.mriImageBuffer = mriImageBuffer
        self.ctImageBuffer = ctImageBuffer
        self.mutualInformation = 7
        self.meanSquaredError = 1
        self.entropy = 2
        self.corelationCoeficcient = 3
        self.PSNR = 4
        self.EPI = 8
        
    def Refresh(self):
        mriImage = cv2.cvtColor(self.mriImageBuffer.GetCurrentImageCV2(), cv2.COLOR_BGR2GRAY)
        ctImage = cv2.cvtColor(self.ctImageBuffer.GetCurrentImageCV2(), cv2.COLOR_BGR2GRAY)
        fuzedImage = cv2.cvtColor(cv2.resize(
            ImageDisplayLarge.GetInstance(ImageDisplayLarge).GetCurrentImageCV2(), (256,256)),cv2.COLOR_BGR2GRAY)
        
        self.meanSquaredError = QualityHolder._GetMSE(mriImage, ctImage, fuzedImage)
        self.mutualInformation = QualityHolder._GetMI(mriImage, ctImage, fuzedImage)
        self.EPI = QualityHolder._GetEPI(mriImage, ctImage, fuzedImage)
        self.corelationCoeficcient = QualityHolder._GetCorrelationCoef(mriImage, ctImage, fuzedImage)
        self.PSNR = self._GetPsnr()
        self.SSIM = QualityHolder._GetSSIM(mriImage, ctImage, fuzedImage)
        
    def _GetMSE(mriImage, ctImage, fuzedImage):
        fuzedImage = fuzedImage.astype('float')
        mriImage = mriImage.astype('float')
        ctImage = ctImage.astype('float')

        mse_mri_fuzed = np.mean((fuzedImage - mriImage)**2) 
        mse_ct_fuzed = np.mean((fuzedImage - ctImage)**2) 

        return (mse_mri_fuzed, mse_ct_fuzed)
                     
    def _GetCorrelationCoef(mriImage, ctImage, fuzedImage):
            mriFlat = mriImage.ravel()
            ctFlat = ctImage.ravel()
            fuzedFlat = fuzedImage.ravel()
            
            correlation_mat_mri_fuzed = np.corrcoef(mriFlat, fuzedFlat)
            correlation_mat_ct_fuzed = np.corrcoef(ctFlat, fuzedFlat)
            
            return (correlation_mat_mri_fuzed[0,1], correlation_mat_ct_fuzed[0,1])
    
    def _GetPsnr(self):
        psnr_mri = 20 * np.log10(255 / np.sqrt(self.meanSquaredError[0]))
        psnr_ct = 20 * np.log10(255 / np.sqrt(self.meanSquaredError[1]))
        
        return (psnr_mri, psnr_ct)
         
    def _GetMI(mriImage, ctImage, fuzedImage):
            histMriFuzed, _, _ = np.histogram2d(mriImage.ravel(), fuzedImage.ravel(), bins=10)
            histCtFuzed, _, _ = np.histogram2d(ctImage.ravel(), fuzedImage.ravel(), bins=10)
            
            return ( mutual_info_score(None, None, contingency = histMriFuzed),
                    mutual_info_score(None, None, contingency = histCtFuzed))
            
    def _GetSSIM(mriImage, ctImage, fuzedImage):
        ssim_mri = structural_similarity(mriImage, fuzedImage)
        ssim_ct = structural_similarity(ctImage, fuzedImage)
        return (ssim_mri, ssim_ct)
    
    # def _GetEPI(mriImage, ctImage, fuzedImage):
    #         mriEdges = cv2.Sobel(mriImage, cv2.CV_64F, 1, 1, ksize=3)
    #         ctEdges = cv2.Sobel(ctImage, cv2.CV_64F, 1, 1, ksize=3)
    #         fuzed_edges = cv2.Sobel(fuzedImage, cv2.CV_64F, 1, 1, ksize=3)   
            
    #         mri_edges_flat = mriEdges.ravel()
    #         ctEdges_flat = ctEdges.ravel()
    #         fuzed_edges_flat = fuzed_edges.ravel()
            
    #         return  (np.corrcoef(mri_edges_flat, fuzed_edges_flat)[0,1],
    #                  np.corrcoef(ctEdges_flat, fuzed_edges_flat)[0,1])
    
    def _GetEPI(mriImage, ctImage, fuzedImage):
        mriEdges = np.sqrt(cv2.Sobel(mriImage, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                        cv2.Sobel(mriImage, cv2.CV_64F, 0, 1, ksize=3)**2)
        ctEdges = np.sqrt(cv2.Sobel(ctImage, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                        cv2.Sobel(ctImage, cv2.CV_64F, 0, 1, ksize=3)**2)
        fuzedEdges = np.sqrt(cv2.Sobel(fuzedImage, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                            cv2.Sobel(fuzedImage, cv2.CV_64F, 0, 1, ksize=3)**2)

        mriEdges = mriEdges / (np.max(mriEdges) + 1e-10)
        ctEdges = ctEdges / (np.max(ctEdges) + 1e-10)
        fuzedEdges = fuzedEdges / (np.max(fuzedEdges) + 1e-10)

        numeratorMRI = np.sum(mriEdges * fuzedEdges)
        denominatorMRI = np.sqrt(np.sum(mriEdges**2) * np.sum(fuzedEdges**2))
        epiMRI = np.abs(numeratorMRI / denominatorMRI) if denominatorMRI != 0 else 0

        numeratorCT = np.sum(ctEdges * fuzedEdges)
        denominatorCT = np.sqrt(np.sum(ctEdges**2) * np.sum(fuzedEdges**2))
        epiCT = np.abs(numeratorCT / denominatorCT) if denominatorCT != 0 else 0

        return (epiMRI if epiMRI >= 0 else 0, epiCT if epiCT >= 0 else 0)