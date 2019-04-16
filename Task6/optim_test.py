import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#import adabound
import matplotlib.pyplot as plt

batch_size=40
test_batch_size=40
devices=[0,0,0,0]
epochs=1
log_interval=100
momentum=0.9
learning_rate=0.01
loss_list=[[],[],[],[]]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, flag):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        loss_list[flag].append(loss.data.cpu().numpy())




def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)


model_SGD = Net().to(devices[0])
model_Adam = Net().to(devices[1])
model_RMSprop = Net().to(devices[2])
model_Momentum = Net().to(devices[3])
SGD = optim.SGD(model_SGD.parameters(), lr=learning_rate)
Adam = optim.Adam(model_Adam.parameters(), lr=learning_rate, betas=(0.9, 0.99))
RMSprop = optim.RMSprop(model_RMSprop.parameters(), lr=learning_rate, alpha=0.9)
Momentum = optim.SGD(model_Momentum.parameters(), lr=learning_rate, momentum=momentum)


for epoch in range(1, epochs + 1):
    print("---SGD---")
    train(model_SGD, devices[0], train_loader, SGD, epoch,0)
    test(model_SGD, devices[0], test_loader)
    print("---Adam---")
    train(model_Adam, devices[1], train_loader, Adam, epoch,1)
    test(model_Adam, devices[1], test_loader)
    print("---RMSprop---")
    train(model_RMSprop, devices[2], train_loader, RMSprop, epoch,2)
    test(model_RMSprop, devices[2], test_loader)
    print("---Momentum---")
    train(model_Momentum, devices[3], train_loader, Momentum, epoch,3)
    test(model_Momentum, devices[3], test_loader)
    print("---------")
labels = ['SGD', 'Adam', 'RMSprop', 'Momentum']
for i, l_his in enumerate(loss_list):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 3))
plt.show()
