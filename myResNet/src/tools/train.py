import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import configparser
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import random
import torchvision
import sys
from torchvision import transforms

sys.path.append(os.getcwd())
from src.lib.model import TestModel
from src.lib.trainer import training

seed = 116

def test_ResNet(device):
    #seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    mnist_train = torchvision.datasets.MNIST(root="/Users/isoda/codes/git/2d_prediction/datasets", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root="/Users/isoda/codes/git/2d_prediction/datasets", train=False, download=True, transform=transform)

    train_dataloader = DataLoader(
        mnist_train,
        batch_size=300,
        shuffle=True
    )

    test_dataloader = DataLoader(
        mnist_test,
        batch_size=300,
        shuffle=False
    )

    # mymodel = TestModel()
    torchmodel = torchvision.models.resnet50(pretrained=False)
    torchmodel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    torchmodel.fc = nn.Linear(512 * 4, 10)

    # mymodel.to('cpu')
    # optimizer = optim.Adam(
    #     params = mymodel.parameters()
    # )
    # loss, acc = training(mymodel, optimizer, train_dataloader, test_dataloader, 3, device)
    # print("--------------------------")
    # print("mymodel final loss and acc")
    # print("--------------------------")
    # print("loss:{} acc:{}".format(loss,acc))
    
    torchmodel.to('cpu')
    optimizer = optim.Adam(
        params = torchmodel.parameters()
    )
    loss, acc = training(torchmodel, optimizer, train_dataloader, test_dataloader, 3, device)
    print("--------------------------")
    print("torchmodel final loss and acc")
    print("--------------------------")
    print("loss:{} acc:{}".format(loss,acc))

if __name__ == '__main__':
    test_ResNet('cpu')
