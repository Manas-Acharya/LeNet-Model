import cv2 as cv
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import optim
import matplotlib.pyplot as plt





#### getting data ####  

train_data = datasets.MNIST('mnist_data', train= True, download=True, transform=ToTensor())

test_data = datasets.MNIST('mnist_data', train= False, download=True, transform=ToTensor())

train_dl = DataLoader(train_data, batch_size=64)

test_dl = DataLoader(test_data, batch_size=64)


### view first image #### 
# plt.imshow(train_data[0][0][0], cmap='gray')
# plt.show()
# print(type(train_data[0][0][0]))

### check for GPU #### 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten  = nn.Flatten
        self.LeNet_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(6, 16 , 5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),


            nn.Flatten(),
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)

        )

    def forward(self, x):
    
        # x = self.flatten(x)
        logits = self.LeNet_layers(x)
        return logits

model = LeNet() 

batch_size = 64
loss_fn = nn.CrossEntropyLoss()


def train(dataloader, model, epoch=3, lr=1e-3, device = device):
    
    size = len(dataloader.dataset)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for batch, (images, label) in enumerate(dataloader):
        images = images.to(device)
        label = label.to(device)

        ## predictin and loss##
        pred = model(images)
        loss = loss_fn(pred, label)

        ## backpropagation##
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 ==0:
            loss = loss.item()
            current = batch * batch_size + len(images)
            print(f"loss : {loss:>7f}, [{current:5d}/ {size:5d}] ")


def test(dataloader, model):
    model.to(device)
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            label = label.to(device)
            pred = model(images)
            test_loss += loss_fn(pred, label).item()
            correct += torch.sum(pred.argmax(1)==label).type(torch.float).item()

    test_loss /= batch_size
    correct /= size
    print(f"Test error: \n Accuracy: {correct*100:>3f}%, avg loss: {test_loss:>7f}")

epoch = 10
for t in range (epoch):
    print(f"Epoch {t+1} \n -----------")
    train(train_dl, model, device)
    test(test_dl, model)
print("Done")


