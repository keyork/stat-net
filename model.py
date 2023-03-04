import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.neck = nn.Sequential(nn.Linear(1024 * 4, 10))

    def forward(self, x):
        x = torch.flatten(x[:, :, :, 0], 1)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        x4 = self.classifier4(x)
        x_neck = torch.cat((x1, x2, x3, x4), 1)
        x = self.neck(x_neck)
        return x


class StatNet(nn.Module):
    def __init__(self):
        super(StatNet, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1024), nn.ReLU(inplace=True), nn.Dropout(p=0.5)
        )
        self.neck = nn.Sequential(nn.Linear(1024 * 4, 10))

    def forward(self, x):
        x1 = torch.flatten(x[:, :, :, 0], 1)
        x2 = torch.flatten(x[:, :, :, 1], 1)
        x3 = torch.flatten(x[:, :, :, 2], 1)
        x4 = torch.flatten(x[:, :, :, 3], 1)
        x1 = self.classifier1(x1)
        x2 = self.classifier2(x2)
        x3 = self.classifier3(x3)
        x4 = self.classifier4(x4)
        x_neck = torch.cat((x1, x2, x3, x4), 1)
        x = self.neck(x_neck)
        return x


if __name__ == "__main__":
    model = StatNet()
    demo_input = torch.rand(1, 1, 28, 28)
    # print("input:")
    # print(demo_input)
    y = model(demo_input)
    print(y)
