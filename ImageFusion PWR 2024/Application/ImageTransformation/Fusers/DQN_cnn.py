import torch
import cv2
import numpy as np
from torch import nn

class DQN_CNN(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = nn.functional.adaptive_avg_pool2d(x, (6, 6))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def fuse_images(self, img1, img2):
        actions = [0, 0.5, 1]

        img1 = cv2.cvtColor(cv2.resize(img1, (50, 50)), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.resize(img2, (50, 50)), cv2.COLOR_BGR2GRAY)

        img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            state = torch.zeros_like(img1)  
            Q_values = self(state)
            action = torch.argmax(Q_values).item()

        alpha = actions[action]

        fused_image = (alpha * img1 + (1 - alpha) * img2).squeeze(0).squeeze(0).cpu().numpy()
        fused_image = fused_image.astype(np.uint8)

        return fused_image