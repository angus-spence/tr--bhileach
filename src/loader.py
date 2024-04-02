import os

from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

root = os.path.join(os.getcwd(), 'src/data/')
train = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(root=root, train=False, download=True)

trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

for data, target in trainloader:
    plt.matshow(data[0][0], cmap='gray')
    plt.show()