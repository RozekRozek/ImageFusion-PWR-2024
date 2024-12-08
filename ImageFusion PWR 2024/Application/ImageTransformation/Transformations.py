from PIL import Image
import numpy as np
from Application.ImageTransformation.Fusers.QTable import QTable
from Application.ImageTransformation.Fusers.QNetwork_50_50 import QNetwork_50_50
from Application.ImageTransformation.Fusers.QNetwork_50_50_Silu import QNetwork_50_50_Silu
from Application.ImageTransformation.Fusers.DQN_cnn import DQN_CNN
from Application.ImageTransformation.Fusers.DQN_cnn_fixed import DQN_CNN_FIXED
import cv2
import torch

qTableFuzer = QTable("SavedModels//e.npy")
qNetworkFuser_50_50 = QNetwork_50_50(2500, 3, "SavedModels//dqn_model_50_50_4_4.pth", 50)
qNetworkFuser_50_50_Silu = QNetwork_50_50_Silu(2500, 3, "SavedModels//dqn_model_50_50_silu_larger.pth", 50)

cnnFuser = DQN_CNN(input_channels=1, action_dim=3)
cnnFuser.load_state_dict(torch.load("SavedModels//dqn_cnn_model.pth"))
cnnFuser.eval()

cnnFuserSmooth = DQN_CNN_FIXED(input_channels=1, action_dim=3)
cnnFuserSmooth.load_state_dict(torch.load("SavedModels//dqn_improved_model.pth"))
cnnFuserSmooth.eval()

def PixelAveraging(imageLeft, imageRight):
    fusedImage = (imageLeft + imageRight) / 2.0
    fusedImage = np.clip(fusedImage, 0, 255).astype(np.uint8)

    pilImage = _convertCVIntoPIL(fusedImage)

    return (pilImage, fusedImage)

def QTableFuse(imageMri, imageCt):
    fusedImage = qTableFuzer.fuse_images(imageMri, imageCt)

    pilImage = _convertCVIntoPIL(fusedImage)
    return (pilImage, fusedImage)

def QNetworkFuse(imageMri, imageCt):
    fusedImage = qNetworkFuser_50_50.fuse(imageMri, imageCt)
    fusedImage = cv2.resize(fusedImage, (256, 256))
    
    pilImage = _convertCVIntoPIL(fusedImage)
    return (pilImage, fusedImage)
    
def QNetworkFuseSilu(imageMri, imageCt):
    fusedImage = qNetworkFuser_50_50_Silu.fuse(imageMri, imageCt)
    fusedImage = cv2.resize(fusedImage, (256, 256))
    
    pilImage = _convertCVIntoPIL(fusedImage)
    return (pilImage, fusedImage)

def CNNFuse(imageMri, imageCt):
    fusedImage = cv2.cvtColor(cnnFuser.fuse_images(imageMri, imageCt), cv2.COLOR_GRAY2BGR)
    fusedImage = cv2.resize(fusedImage, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    pilImage = _convertCVIntoPIL(fusedImage)
    return (pilImage, fusedImage)

def CNNFuseSmoothing(imageMri, imageCt):
    fusedImage = cv2.cvtColor(cnnFuserSmooth.fuse_images(imageMri, imageCt), cv2.COLOR_GRAY2BGR)
    fusedImage = cv2.resize(fusedImage, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    pilImage = _convertCVIntoPIL(fusedImage)
    return (pilImage, fusedImage)

def ContrastStretching(fusedImage):
    minVal, maxVal = np.min(fusedImage), np.max(fusedImage)
    stretchedImage = ((fusedImage - minVal) / (maxVal - minVal) * 255).astype(np.uint8)
    
    return (_convertCVIntoPIL(stretchedImage), stretchedImage)

def _convertCVIntoPIL(cvImage):
    return Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))

TRANSFORMATIONS_DICT = {
    "QNetwork" : QNetworkFuse,
    "QNetwork - Silu" : QNetworkFuseSilu,
    "QTableFuse" : QTableFuse,
    "PixelAveraging" : PixelAveraging,
    "CNN DQN": CNNFuse,
    "CNN DQN Smooth": CNNFuseSmoothing
}