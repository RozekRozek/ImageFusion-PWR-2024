import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from skimage.metrics import structural_similarity as ssim
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(DQN, self).__init__()
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

class ImageFusionEnv:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.fusedImage = torch.zeros_like(img1, device=device)
        self.actions = [0, 0.5, 1]

    def reset(self):
        self.fusedImage = torch.zeros_like(self.img1, device=device)
        return self.fusedImage.cpu().numpy()

    def step(self, action):
        alpha = self.actions[action]
        self.fusedImage = (alpha * self.img1 + (1 - alpha) * self.img2).to(torch.uint8)
        reward = self.calculate_reward(self.fusedImage)
        return self.fusedImage.cpu().numpy(), reward

    def calculate_reward(self, fusedImage):
        fusedImageCpu = np.squeeze(fusedImage.cpu().numpy())
        sq_im1 = np.squeeze(self.img1.cpu().numpy())
        sq_im2 = np.squeeze(self.img2.cpu().numpy())
        return ssim(sq_im1, fusedImageCpu, data_range=255) + ssim(sq_im2, fusedImageCpu, data_range=255)

class DQNAgent:
    def __init__(self, input_channels, action_dim):
        self.model = DQN(input_channels, action_dim).to(device)
        self.target_model = DQN(input_channels, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9
        self.epsilon = 0.1
        self.updateTargetFrequency = 100
        self.steps = 0
        self.scaler = GradScaler()
        self.action_dim = action_dim

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                QValues = self.model(state)
                return torch.argmax(QValues).item()

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in batch:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward = torch.FloatTensor([reward]).to(device)
            
            with torch.no_grad():
                with autocast():
                    targetQValue = reward + self.gamma * torch.max(self.target_model(next_state))

            with autocast():
                QValue = self.model(state)[0][action]
                loss = self.criterion(QValue, targetQValue)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.steps += 1
        if self.steps % self.updateTargetFrequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

def main():
    path = "./DataPlaceholder"
    mriImages = os.listdir("./DataPlaceholder/MRI")
    ctImages = os.listdir("./DataPlaceholder/CT")

    agent = DQNAgent(input_channels=1, action_dim=3)
    for index, image in enumerate(mriImages):
        img1 = cv2.imread(os.path.join(path, "MRI", mriImages[index]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(path, "CT", ctImages[index]), cv2.IMREAD_GRAYSCALE)
        
        img1 = cv2.resize(img1, (50, 50))
        img2 = cv2.resize(img2, (50, 50))
        img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).to(device)
        img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).to(device)
        
        env = ImageFusionEnv(img1, img2)
        for episode in range(4):
            state = env.reset()
            totalReward = 0
            
            for z in range(4):
                action = agent.choose_action(state)
                nextState, reward = env.step(action)
                agent.store_experience(state, action, reward, nextState)
                agent.update(batch_size=32)
                totalReward += reward
                state = nextState

            print(f"Episode {index}/{len(mriImages)} {episode + 1}, Total Reward: {totalReward}")

    agent.save_model('../SavedModels/dqn_cnn_model.pth')

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
