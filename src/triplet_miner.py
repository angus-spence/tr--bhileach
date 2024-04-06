from typing import Union
from dataclasses import dataclass
import os
import random

import torch
import torchvision as tv

import numpy as np
import matplotlib.pyplot as plt

root = os.path.join(os.getcwd(), 'src/data/')

@dataclass	
class CosineTripletMiner:
    difficulty: float = 0.25
    
    def __post_init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.coss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def __call__(self, batch) -> np.ndarray:
        matx = self._cosine_similarity_matrix(batch)
        mapping = self._miner(matx)
        for i in range(len(batch)):
            a, b = batch[i], batch[int(mapping[i])]
            fig, axs, = plt.subplots(1, 2)
            axs[0].imshow(a[0].cpu().numpy(), cmap='gray')
            axs[1].imshow(b[0].cpu().numpy(), cmap='gray')
            plt.show()

    def _cosine_similarity_matrix(self, batch) -> np.ndarray:
        n = batch.shape[0]
        cos_sim = np.zeros((n, n))
        x = batch.to(self.device)
        for i in range(n):
            for j in range(n):
                cos_sim[i, j] = np.sqrt(np.mean(self.coss(x[i], x[j]).cpu().numpy()**2))
        return cos_sim

    def _miner(self, csm: np.ndarray) -> np.ndarray:
        """
        csm: np.ndarray
            Cosine similarity matrix
        """
        mapping = np.zeros(csm.shape[0])
        for i in range(csm.shape[1]):
            x = np.where(csm[i] > np.percentile(csm[i], self.difficulty))[0]
            if len(x) < 1: x = sorted(x)[int(len(x)*self.difficulty)]
            mapping[i] = random.choice(x)
        return mapping

if __name__ == "__main__":
    miner = CosineTripletMiner(difficulty=0.5)