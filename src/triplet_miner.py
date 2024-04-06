from typing import Union
from dataclasses import dataclass
import os

import torch
import torchvision as tv

import numpy as np
import matplotlib.pyplot as plt

root = os.path.join(os.getcwd(), 'src/data/')

@dataclass	
class CosineTripletMiner:
    difficulty: float = 0.5
    
    def __post_init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.coss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def __call__(self, batch) -> np.ndarray:
        matx = self._cosine_similarity_matrix(batch)
        idx_match = self._miner(matx)
        return
    
    def _cosine_similarity_matrix(self, batch) -> np.ndarray:
        n = batch.shape[0]
        cos_sim = np.zeros((n, n))
        x = batch.to(self.device)
        for i in range(n):
            for j in range(n):
                cos_sim[i, j] = np.sqrt(np.mean(self.coss(x[i], x[j]).cpu().numpy()**2))
        return cos_sim

    def _miner(self, cos_sim_matrix: np.ndarray) -> np.ndarray:
        for i in range(cos_sim_matrix.shape[1]):
            #TODO: Implement miner

if __name__ == "__main__":
    miner = CosineTripletMiner(difficulty=0.5)