import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#Create a Fully Connected NN

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Set Device to see if Cuda exists
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784
lr = 0.01
num_classes = 10
batch_size = 64
num_epochs = 5

#Load data and Initialise network
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size, num_classes).to(device=device)
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#Train Network
for epoch in range(num_epochs):
    print("Running epoch {}\n".format(epoch+1))
    for batch_idx, (data, targets) in enumerate(train_loader):
        #Run model on GPU if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        #Flatten data from 28x28 to 784x1
        data = data.reshape(data.shape[0], -1)

        #forward pass
        scores = model(data)
        loss = lossFunction(scores, targets)

        #backward pass
        optimizer.zero_grad()
        loss.backward()

        #gradient descent with adam optimizer
        optimizer.step()

#Evaluate Network

def check_acc(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    #run same as Train network, but without backward pass and grad calculations
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        correct_rate = (num_correct*100)/num_samples

        if loader.dataset.train:
            print("Accuracy of train data is {}%".format(correct_rate))
        else:
            print("Accuracy of test data is {}%".format(correct_rate))

check_acc(train_loader, model)
check_acc(test_loader, model)