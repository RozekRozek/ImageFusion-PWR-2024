import torch
import numpy as np
import torch.nn as nn

class QNetwork_50_50_Silu:
    def __init__(self, inputDim, actionDim, modelPath, imageShape):
        self.zerosState = np.zeros(imageShape* imageShape)
        self.actions = [0, 0.5, 1]
        self.model = self.load_model(inputDim, actionDim, modelPath )
        
    def step(self, action, img1, img2):
        alpha = self.actions[action]
        self.fused_image = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)
        return self.fused_image

    def load_model(self, inputDim, actionDim, modelPath):
        model = DQN(inputDim, actionDim)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        return model

    def fuse(self, img1, img2):
        stateTensor = torch.FloatTensor(self.zerosState).unsqueeze(0)

        with torch.no_grad():
            qValues = self.model(stateTensor)
            action = torch.argmax(qValues).item()

        fusedImage = self.step(action, img1, img2)
        return fusedImage

class DQN(nn.Module):
    def __init__(self,  inputDim, outputDim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear( inputDim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 25)
        self.fc4 = nn.Linear(25, outputDim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = nn.functional.silu(self.fc3(x))
        return self.fc4(x)