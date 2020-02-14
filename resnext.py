from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader, Dataset


class model_ft(nn.Module):
    def __init__(self, model):
        super(model_ft, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 257)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 2048)
        x = self.classifier(x)

        return x


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.module.state_dict())
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['TrainSet', 'TestSet']:
            if phase == 'TrainSet':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            #    # 前1次迭代不更新卷积权重
            #    if epoch == 0 and phase == 'TrainSet':
            #       for p in model.features.parameters():
            #           p.requires_grad = False

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'TrainSet'):

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'TrainSet':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #      if epoch == 0 and phase == 'TrainSet':
            #          for p in model.features.parameters():
            #              p.requires_grad = True

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'TestSet' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 多gpu模型权重要这么读！！
                #    best_model_wts = copy.deepcopy(model.module.state_dict())
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'resnet_best.pkl')
            if phase == 'TestSet':
                scheduler.step(epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return 0


if __name__ == '__main__':
    data_transforms = {
        'TrainSet': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'TestSet': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'C:/Users/27518/Desktop/resnet'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['TrainSet', 'TestSet']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=16, num_workers=4)
                   for x in ['TrainSet', 'TestSet']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['TrainSet', 'TestSet']}

    class_names = image_datasets['TrainSet'].classes

    device = torch.device("cuda:0")
    # device = torch.device("cuda")
    model_ft = model_ft(models.resnext101_32x8d(pretrained=True))
    # MLP层的权重也是预训练过的，直接用初始化的loss会崩（原因未知）
    model_ft.load_state_dict(torch.load('MLP_best.pkl'), strict=False)

    model_ft = model_ft.to(device)
    #  model_ft = nn.DataParallel(model_ft, device_ids=[0, 1])
    # 用DataParrallel要加 .module
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD([
        {'params': model_ft.features.parameters()},
        {'params': model_ft.classifier.parameters(), 'lr': 0.001}], lr=0.0001, momentum=0.9)
    # lr还可以调整

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor=0.3, verbose=True)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=300)
