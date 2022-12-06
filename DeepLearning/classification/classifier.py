import torch
import torchvision
import torchvision.transforms as transforms
import wandb
import numpy as np

wandb.login() #  To be able to log metrics to projects

batches = []

for i in range(6):
    batches.append(2**(i+2))

sweep_config = {
    'project' : 'classification-test',
    'entity' : 'projectcarla',
    'method': 'random',
    'metric' : {
        'name': 'loss',
        'goal': 'minimize',
    },
    'parameters' : {
        'epochs': {
            'value': 10,
            },
        'lr': {
            'min': 0.00001,
            'max': 0.01,
            },
        'batch_size': {
            'values' : batches,
            },
    }
}
sweep_id = wandb.sweep(sweep_config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Testing whether a GPU (cuda device) is available. If there is one, use this, if not use the CPU.
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./DeepLearning/classification/data', train=True,
                                        download=True, transform=transform)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
#                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch.optim as optim
def train(config = None):
    with wandb.init(config=config): #initialize a new wandb run
        config = wandb.config

        net = Net().to(device)

        # trainset = torchvision.datasets.CIFAR10(root='./DeepLearning/classification/data', train=True,
        #                             download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                            shuffle=True, num_workers=8)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)


        for epoch in range(config.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()


                if i % 200 == 199:    # print every 200 mini-batches
                    wandb.log({"loss": running_loss / 200, "epoch": epoch})
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0

        print('Finished Training')


        PATH = './DeepLearning/classification/cifar_net.pth'
        torch.save(net.state_dict(), PATH)


wandb.agent(sweep_id, train, count=20)