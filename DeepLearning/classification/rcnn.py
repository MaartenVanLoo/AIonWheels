
import torch
import torchvision

version = torch.version.__version__[:5]
print('torch version is {}'.format(version))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Testing whether a GPU (cuda device) is available. If there is one, use this, if not use the CPU.
print(torch.__version__)
print(torch.cuda.is_available())
print(device)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn().to(device)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)


# for epoch in range(config.epochs):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()


#         if i % 200 == 199:    # print every 200 mini-batches
#             wandb.log({"loss": running_loss / 200, "epoch": epoch})
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
#             running_loss = 0.0

# print('Finished Training')

# # For training
# images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
# boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
# labels = torch.randint(1, 91, (4, 11))
# images = list(image for image in images)
# targets = []
# for i in range(len(images)):
#     d = {}
#     d['boxes'] = boxes[i]
#     d['labels'] = labels[i]
#     targets.append(d)
# output = model(images, targets)


# For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)

# optionally, if you want to export the model to ONNX:
# torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
