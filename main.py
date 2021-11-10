import torch.optim as optim
import torch
from net import Net
from utils import *
from PIL import Image
import os
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as tt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

emotionsDict = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6,
}

curr = os.path.dirname(__file__)

# check if the processed data exists, otherwise, recreate it
if not os.path.exists(os.path.join(curr, r'processed/')):
    processFacesFromPictures()


# train = TrainingData()

training_data = datasets.ImageFolder(root=os.path.join(curr, r'fer2013/train/'),
                                     transform=transforms.ToTensor())

# print(training_data, training_data.targets)

testing_data = datasets.ImageFolder(root=os.path.join(curr, r'fer2013/validation/'),
                                    transform=transforms.ToTensor())

training_set = DataLoader(training_data, batch_size=10, shuffle=True)
testing_set = DataLoader(testing_data, batch_size=10, shuffle=True)

net = Net()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):  # 3 full passes over the data
    for idx, data in enumerate(training_set):  # `data` is a batch of data
        # X is the batch of features, y is the batch of targets.
        sys.stdout.flush()
        sys.stdout.write("\bCurrent epoch: %s/%s. Current progress: %s %%\r" %
                         (str(epoch), str(32), (str(math.ceil((idx / len(training_set)) * 100)))))
        X, y = data  # X is the batch of features, y is the batch of targets.
        # print(X.shape, y.shape)
        # sets gradients to 0 before loss calc. You will do this likely every step.
        net.zero_grad()
        # print(X[0])
        # pass in the reshaped batch (recall they are 28x28 atm)
        output = net(X)
        # print('SHAPE', y.shape, torch.unsqueeze(y, 3).shape)
        # calc and grab the loss value
        preds = torch.argmax(output, dim=1)
        # print(X, y)
        # print(preds.shape, output.shape, y.shape, y.unsqueeze(1).shape)
        # print(preds, output, y)
        # print(y.unsqueeze(1), y.unsqueeze(1).shape)
        # loss = F.nll_loss(output, y)
        for idx, i in enumerate(output):
            loss = F.nll_loss(torch.argmax(i), y[idx])
            loss.backward()  # apply this loss backwards thru the network's parameters
        # loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!


correct = 0
total = 0

with torch.no_grad():
    for data in testing_set:
        X, y = data
        output = net(X)
        for idx, i in enumerate(output):
            print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

accuracy = round(correct/total, 3)
print("Accuracy: ", accuracy)

# trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
if accuracy > 0.2 and not os.path.exists('v1_model.pth'):
    torch.save(net.state_dict(), 'v1_model.pth')
