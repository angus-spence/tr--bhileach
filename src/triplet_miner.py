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
    data: torch.utils.data.Dataset
    dataloader: torch.utils.data.DataLoader
    root: str
    train: bool = True
    download: bool = True
    batch_size: int = 1024
    transform: tv.transforms.Compose = tv.transforms.ToTensor()

    def __post_init__(self):
        self.data(root=self.root, 
                  train=self.train, 
                  download=self.download, 
                  transform=self.transform)
        self.dataloader(dataset=self.data,
                        batch_size=self.batch_size,
                        shuffle=True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.coss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def _cosine_similarity(self) -> np.ndarray:
        cos_sim = np.zeros((self.batch_size, self.batch_size))
        for x, x_trgt in self.dataloader:
            for i in self.batch_size:
                for j in self.batch_size:
                    cos_sim[i, j] = self.coss(x[i], x[j])
            plt.matshow(cos_sim)
            plt.show()

    def __call__(self, targets: list) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        return
    
if __name__ == "__main__":
    miner = CosineTripletMiner(data=tv.datasets.MNIST,
                               dataloader=torch.utils.data.DataLoader, 
                               root=root)
    miner._cosine_similarity()
    print(miner.device)