# -*- coding: utf-8 -*-
from __future__ import print_function, division

from torchsummary import summary

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


class model_bn(nn.Module):
    def __init__(self, model):

        super(model_bn, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == '__main__':

    data_transforms = {
        'TrainSet': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        #    transforms.RandomRotation(10),
        #    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
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


    data_dir = '.'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['TrainSet', 'TestSet']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=32, num_workers=4)
                   for x in ['TrainSet', 'TestSet']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['TrainSet', 'TestSet']}

    class_names = image_datasets['TrainSet'].classes

    device = torch.device("cuda:0")

    model_ft = model_bn(models.resnext101_32x8d(pretrained=True))


    for param in model_ft.parameters():
        param.requires_grad = False

    model_ft = model_ft.to(device)


    for phase in ['TrainSet', 'TestSet']:

        model_ft.eval()  # Set model to evaluate mode
        flag=0
        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
           



            outputs = model_ft(inputs).cpu()

            outputs = outputs.data.numpy()
            labels=labels.numpy()
            print(outputs.shape)
            print(labels.shape)
            if flag == 0:
                res=outputs
                res2=labels
                flag=1

            else:
                res = np.concatenate((res, outputs), axis=0)
                res2= np.concatenate((res2, labels), axis=0)

        print(res.shape)
        print(res2.shape)
        np.save(phase+'.npy',res)
        np.save(phase+'_labels.npy',res2)

