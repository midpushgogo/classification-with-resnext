from __future__ import print_function, division
from  logger import Logger
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
    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label

    def __len__(self):
        return len(self.Data)

class model_ft(nn.Module):
    def __init__(self,model):

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
        x=x.view(-1,2048)
        x = self.classifier(x)
    #    return nn.functional.softmax(x,dim=0)
        return x
if __name__ == '__main__':

    model_ft=model_ft(models.resnext101_32x8d(pretrained=False))
    model_ft.load_state_dict(torch.load('resnet_best.pkl'),strict=True)

    data_transforms = {

        'TestSet': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = 'C:/Users/王泽灏/Desktop/resnet'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in [ 'TestSet']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=16, num_workers=4)
                   for x in [ 'TestSet']}
    dataset_sizes = {x: len(image_datasets[x]) for x in [ 'TestSet']}

    class_names = image_datasets['TestSet'].classes

    device = torch.device("cuda:0")

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for phase in ['TestSet']:

        model_ft.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0


        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)



            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'TrainSet'):

                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)


            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]


        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

