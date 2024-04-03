from nn import ResNet50, TripletLossFunction

import os

from torchvision import datasets, transforms
import torch

EPOCHS = 50

def main() -> None:
    root = os.path.join(os.getcwd(), 'src/data/')
    train = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root=root, train=False, download=True)

    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu') 
    
    model = ResNet50()
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-7)
    criterion = TripletLossFunction(margin=1).to(device)

    running_loss = []
    for epoch in range(EPOCHS):
        for data, target in trainloader:
            print(data[0])
            print(target)
            

if __name__ == "__main__":
    main()