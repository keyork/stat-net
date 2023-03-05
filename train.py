import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from model import Net, StatNet
from mnist import MNIST

train_scale = 0.5 # 0.5, 0.166666666667, 0.08333333333333, 0.0166666666667, 0.008333333333333, 0.00166666666667
batch_size = 64
epochs = 10

batch_gap = 20
# 100, 500 - 1; 1000 - 2; 5000 - 5; 10000 - 10; 30000, full - 20

# is_stat = False
# train_log = './log/mlpnet-exp2-30000.npz'
# weights_path_head = './weights/exp2/30000-mlpnet'

is_stat = True
train_log = './log/statnet-exp2-30000.npz'
weights_path_head = './weights/exp2/30000-statnet'


def train():
    
    train_loss_list = np.array([])
    valid_loss_list = np.array([])
    train_acc_list = np.array([])
    valid_acc_list = np.array([])
    valid_time_list = np.array([])
    test_loss = np.array([])
    test_acc = np.array([])
    
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
    if is_stat:
        network = StatNet()
    else:
        network = Net()
    
    # network.load_state_dict(torch.load(weights_path))

    network = network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    size = len(train_dataloader.dataset)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        network.train()
        corr = 0
        loss_log = 0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = network(X)
            _, result = torch.max(pred.data, 1)
            
            loss = loss_fn(pred, y)
            loss_log += loss.item()
            corr += (result == y).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (batch+1) % batch_gap == 0:
                loss, current = loss_log, batch * batch_size + len(X)
                print(f"loss: {loss/current:>7f} acc: {corr/current:>7f}  [{current:>5d}/{size:>5d}]")
                train_loss_list = np.append(train_loss_list, loss/current)
                train_acc_list = np.append(train_acc_list, (corr/current).cpu())
        valid_time_list = np.append(valid_time_list, (t+1)*batch)
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
            valid_loss_list = np.append(valid_loss_list, loss / val_size)
            valid_acc_list = np.append(valid_acc_list, (corr / val_size).cpu())
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
        test_loss = np.append(test_loss, loss / test_size)
        test_acc = np.append(test_acc, (corr / test_size).cpu())
    
    torch.save(
            network.state_dict(),
            "{}-acc-{:5f}.pth".format(weights_path_head, (corr / test_size).cpu())
        )
    
    np.savez(train_log, train_loss = train_loss_list, train_acc = train_acc_list,
            valid_loss = valid_loss_list, valid_acc = valid_acc_list, valid_x = valid_time_list,
            test_loss = test_loss, test_acc = test_acc)

if __name__ == '__main__':
    train()
    