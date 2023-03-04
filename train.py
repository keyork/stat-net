import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from torch.optim.lr_scheduler import StepLR

from model import Net, StatNet
from mnist import MNIST


def train():
    # dataset & dataloader
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    
    training_dataset = MNIST(
        root="./data", train=True, download=True, transform=img_transforms
    )
    
    testing_dataset = MNIST(
        root="./data", train=False, download=True, transform=img_transforms
    )
    
    

    train_scale = 0.7
    batch_size = 64



    train_num = int(len(training_dataset) * train_scale)
    valid_num = len(training_dataset) - train_num
    training_data, validing_data = random_split(
        training_dataset,
        [train_num, valid_num],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        validing_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        testing_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = Net()
    

    # # 初始化权重
    # for layer in network.modules():
    #     if isinstance(layer, nn.Conv2d):
    #         nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    #         nn.init.constant_(layer.bias, 0)

    # weights_path = "./demo"
    # network.load_state_dict(torch.load(weights_path))

    network = network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    size = len(train_dataloader.dataset)

    for t in range(10):
        print(f"Epoch {t+1}\n-------------------------------")
        network.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = network(X)
            _, result = torch.max(pred.data, 1)
            # print(pred, y)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # valid
        val_size = len(valid_dataloader.dataset)
        loss = 0
        corr = 0
        network.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(valid_dataloader):
                X, y = X.to(device), y.to(device)

                # Compute prediction error
                pred = network(X)
                _, result = torch.max(pred.data, 1)
                
                
                
                corr += (result == y).sum()
                batch_loss = loss_fn(pred, y)
                loss += batch_loss.item()
            print("loss={:5f}, acc={:5f}".format(loss / val_size, corr / val_size))
    # test
    test_size = len(test_dataloader.dataset)
    loss = 0
    corr = 0
    network.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = network(X)
            _, result = torch.max(pred.data, 1)
            corr += (result == y).sum()
            batch_loss = loss_fn(pred, y)
            loss += batch_loss.item()
        print("loss={:5f}, acc={:5f}".format(loss / test_size, corr / test_size))

if __name__ == '__main__':
    train()
    