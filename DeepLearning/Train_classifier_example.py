import torch.optim as optim #Optimizer package in pytorch
from Classifier_network import Net

criterion = nn.CrossEntropyLoss()

#Function in which we train the model,
def train(config = None):
  #with wandb.init(config=sweep_config, entity="m-a"):
    #config = wandb.config
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device) #Put model on GPU
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=1e-5) #Define optimizer with w&b config
    for epoch in range(config.epochs):  # go over the dataset multiple times

        running_loss = 0.0
        train_total = 0
        train_correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the input
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #wandb.log({"loss": loss.item(), "epoch": epoch})
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            scores, predictions = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += int(sum(predictions == labels))
            if i % 1000 == 999:    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        acc = round((train_correct / train_total) * 100, 2) #Calculate training accuracy
        #wandb.log({'Train Accuracy': acc})
        correct = 0
        total = 0
        # Calculate test accuracy and save the model
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        testacc = 100 * correct // total
        print(f'Accuracy of the network on the 10000 test images: {testacc} %')
        #wandb.log({'Test Accuracy': testacc})
        #PATH = './cifar_net.pth'
        #torch.save(net.state_dict(), PATH) #Save model
print('Finished Training')