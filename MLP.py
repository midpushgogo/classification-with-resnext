# coding=utf-8
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


class npDataset(Dataset):
    def __init__(self,name):
        self.Data = np.load(name + '.npy')
        self.Label = np.load(name + '_labels'+'.npy')
        print(self.Data.shape)
        print(self.Label.shape)
    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label

    def __len__(self):
        return len(self.Data)
'''
Epoch 31/299
----------
TrainSet Loss: 0.2542 Acc: 0.9247
TestSet Loss: 0.6376 Acc: 0.8630
SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
class model_MLP(nn.Module):
    def __init__(self):

        super(model_MLP, self).__init__()

        self.classifier = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(0.5),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ELU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, 257)
        )
    def forward(self, x):
        x=x.view(-1,2048)
        x = self.classifier(x)
    
        return x





Epoch 58/299
----------
TrainSet Loss: 0.4151 Acc: 0.9027
TestSet Loss: 1.0365 Acc: 0.8626
SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
class model_MLP(nn.Module):
    def __init__(self):

        super(model_MLP, self).__init__()

        self.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 257)
        )
    def forward(self, x):
        x=x.view(-1,2048)
        x = self.classifier(x)
 
        return x
        
        
Epoch 31/299 (稳定性更好)
----------
TrainSet Loss: 0.1228 Acc: 0.9639
TestSet Loss: 0.6085 Acc: 0.8673
Epoch    31: reducing learning rate of group 0 to 1.0000e-03.
class model_MLP(nn.Module):
    def __init__(self):

        super(model_MLP, self).__init__()

        self.classifier = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(0.5),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, 257)
        )
    def forward(self, x):
        x=x.view(-1,2048)
        x = self.classifier(x)
  
        return x


Epoch 80/299
----------
TrainSet Loss: 0.1828 Acc: 0.9438
TestSet Loss: 0.5888 Acc: 0.8696

class model_MLP(nn.Module):
    def __init__(self):

        super(model_MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 257)
        )
    def forward(self, x):
        x=x.view(-1,2048)
        x = self.classifier(x)

        return x
'''

# best model
class model_MLP(nn.Module):
    def __init__(self):

        super(model_MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 257)
        )
    def forward(self, x):
        x=x.view(-1,2048)
        x = self.classifier(x)

        return x


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'TestSet' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'MLP_best.pkl')
            if phase=='TestSet':
                scheduler.step(epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return


if __name__ == '__main__':


    datasets = {x: npDataset(x)
                      for x in ['TrainSet', 'TestSet']}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], shuffle=True, batch_size=128)
                   for x in ['TrainSet', 'TestSet']}
    dataset_sizes = {x: len(datasets[x]) for x in ['TrainSet', 'TestSet']}


    device = torch.device("cuda:0")

    model_ft = model_MLP()

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)

  #  optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, factor=0.2 ,verbose=True)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=300)
