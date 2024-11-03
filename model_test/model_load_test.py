import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from lib import calculate_normalize as CalNorm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 50
BATCH_SIZE = 64

trainset = datasets.CIFAR100('./.data',
                             train=True,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ]))
testset = datasets.CIFAR100('./.data',
                             train=False,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ]))
mean_, std_ = CalNorm(trainset)
print("mean = {}\nstd = {}".format(mean_, std_))


trainset = datasets.CIFAR100('./.data',
                             train=True,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_, std_)
                             ]))
testset = datasets.CIFAR100('./.data',
                             train=False,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean_, std_)
                             ]))
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3136, 625)
        self.fc2 = nn.Linear(625, 100)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, traget) in enumerate(train_loader):
        data, target = data.to(DEVICE), traget.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

    
PATH = './.models/CNN_Model.pt'

# cuda 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# 저장된 모델을 불러옵니다.
model = CNN().to(DEVICE)  # 모델 구조는 동일하게 정의되어야 합니다.
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()  # 평가 모드로 전환합니다.

# 모델 평가
test_loss, test_accuracy = evaluate(model, test_loader)
print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))