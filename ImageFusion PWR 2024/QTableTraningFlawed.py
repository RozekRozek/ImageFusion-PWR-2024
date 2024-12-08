import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import os

class ImageFusionEnv:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.fusedImage = np.zeros_like(img1)

    def reset(self):
        self.fusedImage = np.zeros_like(self.img1)
        return self.fusedImage

    def step(self, action):
        h, w = self.fusedImage.shape
        for i in range(h):
            for j in range(w):
                if action == 0:
                    self.fusedImage[i, j] = self.img1[i, j]
                elif action == 1:
                    self.fusedImage[i, j] = self.img2[i, j]
        reward = self.calculateReward(self.fusedImage)
        return self.fusedImage, reward

    def calculateReward(self, fusedImage):
        return ssim(self.img1, fusedImage) + ssim(self.img2, fusedImage)

class QLearningAgent:
    def __init__(self, actions):
        self.qTable = np.zeros((1, len(actions)))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def chooseAction(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.qTable[0])

    def update(self, state, action, reward):
        bestNextAction = np.argmax(self.qTable[0])
        tdTarget = reward + self.gamma * self.qTable[0][bestNextAction]
        tdDelta = tdTarget - self.qTable[0][action]
        self.qTable[0][action] += self.alpha * tdDelta
        
    def saveModel(self, filepath):
        np.save(filepath, self.qTable)
        
def main():
    path = "./DataPlaceholder"
    mriImages = os.listdir("./DataPlaceholder/MRI")
    ctImages = os.listdir("./DataPlaceholder/CT")

    agent = QLearningAgent(actions=[0, 1])

    imagesCount = len(mriImages)
    episodes = 10
    lapses = 4
    steps = 0
    totalSteps = lapses * imagesCount * episodes
    for index, _ in enumerate(mriImages):
        img1 = cv2.imread(os.path.join(path, "MRI", mriImages[index]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(path, "CT", ctImages[index]), cv2.IMREAD_GRAYSCALE)
        
        env = ImageFusionEnv(img1, img2)
        for episode in range(episodes):
            state = env.reset()
            totalReward = 0
            
            for _ in range(lapses):
                action = agent.chooseAction()
                _, reward = env.step(action)
                agent.update(state, action, reward)
                totalReward += reward
                steps += 1
            print(f"Percent Completed {steps / totalSteps:.2f}%, Total Reward: {totalReward}")
            
    agent.saveModel('SavedModels/e')

if __name__ == '__main__':
    main()
