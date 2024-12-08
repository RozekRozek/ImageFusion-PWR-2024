import numpy as np
import os

class QTable:
    def __init__(self, modelPath):
        self.qTable = None
        self.modelPath = modelPath
        self.load_model()

    def load_model(self):
        if os.path.exists(self.modelPath):
            self.qTable = np.load(self.modelPath)

    def predict_action(self):
        action = np.argmax(self.qTable[0])
        return action

    def fuse_images(self, img1, img2):
        action = self.predict_action()
        fusedImage = np.zeros_like(img1)

        h, w, _ = np.shape(fusedImage)
        for i in range(h):
            for j in range(w):
                if action == 0:
                    fusedImage[i, j] = img1[i, j]
                elif action == 1:
                    fusedImage[i, j] = img2[i, j]

        return fusedImage