# -*- coding: utf-8 -*-
"""FoodSim.ipynb

Original CoLab file located at
    https://colab.research.google.com/drive/1eQdYR6uS9eP5H8blY5kfJ6nOGAh3nGhJ

Use a Siamese network with a triplet margin loss to identify food pictures with similar ´tastes' in the triplets.

"""

# Initial importing

import numpy as np
import random
import os, glob

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision as tvision
from torchvision import datasets, transforms
from PIL import Image

import matplotlib.pyplot as plt

PROJ_DIR = '/content/task3'
PIC_DIR = os.path.join(PROJ_DIR, 'food')
PIC_SIZE = 128
BATCH_SIZE = 32

"""### Triplet dataset and utilities"""

from torchvision.transforms.transforms import ToTensor

def read_triplet(fname, line_idx):
    with open(os.path.join(PROJ_DIR, fname), 'r') as fread:
        fread.seek(line_idx*LINE_STRIDE)
        triplet = fread.readline().strip().split(' ')
    return triplet

def get_triplet_img(fdir, triplet):
    imgs = [trans_train(tvision.io.read_image(os.path.join(fdir, f'{fname}.jpg')).to(torch.float)) \
            for fname in triplet]
    return imgs


class PicDataset(Dataset):
    """Used for reading pictures as dataset."""

    def __init__(self, dir, fnames, transform=None) -> None:
        super().__init__()
        self.dir = dir
        self.fnames = fnames
        self.transform = transform
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        fname = self.fnames[index]
        img = Image.open(os.path.join(self.dir, fname))
        return self.transform(img)


class TripletVectorDataset(Dataset):
    """Used for reading triplet vectors as dataset."""

    def __init__(self, vector_folder, triplet_text, transform=None) -> None:
        super().__init__()
        self.vector_folder = vector_folder
        self.triplet_text = triplet_text
        self.transform = transform
        self.lines = self.count_lines()
    
    def __len__(self):
        return self.lines
    
    def __getitem__(self, index):
        triplet_idx = read_triplet(self.triplet_text, index)
        vectors = [torch.load(os.path.join(self.vector_folder, f'{idx}.pt')) \
                for idx in triplet_idx]
        return vectors[0], vectors[1], vectors[2]
    
    def count_lines(self):
        with open(os.path.join(PROJ_DIR, self.triplet_text), 'r') as fread:
            n_lines = sum([1 for line in fread])
        return n_lines


class FCNetwork(nn.Module):
    """Additional FC NN superimposed on RESNET outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.acf = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.2)
    
    def forward(self, x):
        z = self.drop1(x)
        z = self.fc1(z)
        z = self.acf(z)
        z = self.fc2(z)
        z = self.acf(z)
        z = self.fc3(z)
        return z


class TripletWrapper(nn.Module):
    """Change any network to triplet network."""

    def __init__(self, single_model) -> None:
        super().__init__()
        self.single_model = single_model
    
    def forward(self, x_anchor, x_pos, x_neg):
        o_anchor = self.single_model(x_anchor)
        o_pos = self.single_model(x_pos)
        o_neg = self.single_model(x_neg)
        return o_anchor, o_pos, o_neg



"""Build & Train on Pretrained Convolutional Layers"""

# Load RESNET model, and set last layer to identity

model_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
for name, param in model_resnet.named_parameters():
    param.requires_grad = False
model_resnet.fc = nn.Identity()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_resnet.to(device)

trans_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_list = os.listdir(PIC_DIR)
img_list = sorted([img_name for img_name in img_list if img_name[-3:] == 'jpg'])
img_ds = PicDataset(PIC_DIR, img_list, transform=trans_resnet)
img_loader = DataLoader(img_ds, batch_size=64, shuffle=False, num_workers=2)

# Precalculate all outputs of resnet

with torch.no_grad():
    model_resnet.eval()
    for i, imgs in enumerate(img_loader):
        idx = slice(i*64, i*64 + imgs.size(dim=0))
        fnames = img_list[idx]
        resnet_output = model_resnet(imgs.to(device))
        for j, fname in enumerate(fnames):
            torch.save(resnet_output[j], os.path.join(PROJ_DIR, 'food_resnet50', fname[:-4] + '.pt'))
        if (i + 1) % 15 == 0:
            print(f'{(i+1)*64} images saved with resnet-50 output.')


# Building triplet NN on top of the pretrained model output

ds_resnet_train = TripletVectorDataset(os.path.join(PROJ_DIR, 'food_resnet50'), 'train_triplets.txt')
loader_resnet_train = DataLoader(ds_resnet_train, shuffle=False, batch_size=BATCH_SIZE, num_workers=0)

net_train = FCNetwork()
triplet_refit = TripletWrapper(net_train)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_train = net_train.to(device)

criterion = nn.TripletMarginLoss(margin=0.5)
optimizer = optim.Adam(net_train.parameters(), lr=1e-3)

# Train the added FC layers

epochs = 15
print_every = 400
validations = list()
accuracies = list()
OUT_DIR = os.path.join(PROJ_DIR, 'outputs')

for epoch in range(epochs):

    running_loss = 0.0
    validation_loss = 0.0
    count_correct = 0
    count_total = 0

    for i, data in enumerate(loader_resnet_train):
    
        if i <= len(ds_resnet_train) // BATCH_SIZE - 100:

            net_train.train(True)

            i_ach = data[0].to(device)
            i_pos = data[1].to(device)
            i_neg = data[2].to(device)

            optimizer.zero_grad()

            o_ach, o_pos, o_neg = triplet_refit(i_ach, i_pos, i_neg)
            loss = criterion(o_ach, o_pos, o_neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % print_every == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_every:.3f}')
                running_loss = 0.0
        
        else:

            net_train.eval()
    
            with torch.no_grad():
                i_ach = data[0].to(device)
                i_pos = data[1].to(device)
                i_neg = data[2].to(device)

                o_ach, o_pos, o_neg = triplet_refit(i_ach, i_pos, i_neg)
                loss = criterion(o_ach, o_pos, o_neg)
                d_pos = torch.sum((o_ach - o_pos)**2, dim=1)
                d_neg = torch.sum((o_ach - o_neg)**2, dim=1)
                count_correct += (d_neg > d_pos).sum().item()
                count_total += d_pos.size()[0]

                validation_loss += loss.item()
    
    ave_valid = validation_loss / 100
    accuracy = count_correct / count_total
    if ((not validations) or ave_valid < validations[-1]) or ((not accuracies) or accuracy > accuracies[-1]):
        torch.save(net_train.state_dict(), os.path.join(OUT_DIR, 'temp_resnet50_f3dropout', f'temp_ep{epoch}.pth'))
    print(f'Epoch {epoch + 1} validation loss: {ave_valid:.3f}, accuracy: {accuracy:.3f}')
    validations.append(ave_valid)
    accuracies.append(accuracy)


"""Output and evaluations"""

OUT_DIR = os.path.join(PROJ_DIR, 'outputs')

# Select a network
# net = triplet_refit
# net_save = net_train
net_save = FCNetwork()
net_save.load_state_dict(torch.load(os.path.join(OUT_DIR, 'temp_resnet50_f3dropout', 'temp_ep8.pth')))
net = TripletWrapper(net_save)

# Select a network label
# This label will be used to save the network and its predictions
net_label = 'resnet50_f3e8_p0.2'

# Save the trained model
torch.save(net_save.state_dict(), os.path.join(OUT_DIR, f'net_{net_label}.pth'))

# Recall on training set

fname_train_recall = 'train_recall_' + net_label + '.txt'

with open(os.path.join(OUT_DIR, fname_train_recall), 'w') as fwrite:
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(loader_resnet_train):
            i_ach, i_pos, i_neg = data[0].to(device), data[1].to(device), data[2].to(device)
            o_ach, o_pos, o_neg = net(i_ach, i_pos, i_neg)
            d_pos = torch.sum((o_ach - o_pos)**2, dim=1)
            d_neg = torch.sum((o_ach - o_neg)**2, dim=1)
            print(*tuple(((d_neg > d_pos)*1).tolist()), sep='\n', file=fwrite)
            if (i+1) % 100 == 0:
                print(f'{i + 1:5d} batches recalled.')

with open(os.path.join(OUT_DIR, fname_train_recall), 'r') as fread:
    predict = [int(line.strip()) for line in fread]
    accuracy = sum(predict)/len(predict)

print(f'Accuracy = {accuracy}')

# Predict on test set

triplet_test_ds = TripletVectorDataset(os.path.join(PROJ_DIR, 'food_resnet50'), 'test_triplets.txt')
triplet_test_loader = DataLoader(triplet_test_ds, shuffle=False, batch_size=BATCH_SIZE, num_workers=0)

fname_test_predict = 'test_predict_' + net_label + '.txt'

with open(os.path.join(OUT_DIR, fname_test_predict), 'w') as fwrite:
    net_save.eval()
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(triplet_test_loader):
            i_ach, i_pos, i_neg = data[0].to(device), data[1].to(device), data[2].to(device)
            o_ach, o_pos, o_neg = net(i_ach, i_pos, i_neg)
            d_pos = torch.sum((o_ach - o_pos)**2, dim=1)
            d_neg = torch.sum((o_ach - o_neg)**2, dim=1)
            # print(*tuple(list(zip([i]*len(d_neg), ((d_neg > d_pos)*1).tolist()))), sep='\n', file=fwrite)
            print(*tuple(((d_neg > d_pos)*1).tolist()), sep='\n', file=fwrite)
            if (i+1) % 100 == 0:
                print(f'{i + 1:5d} batches predicted')


'''Ensemble multiple decisions
The multiple decisions from multiple models are pre-calculated from changing the previous parameters
which are not completely archived in this code'''

with open('./outputs/test_predict_resnet50_f3e3_p0.2.txt', 'r') as f1:
    with open('./outputs/test_predict_resnet18_f3e10_p0.2.txt', 'r') as f2:
        with open('./outputs/test_predict_resnet50_f3Le9_p0.2.txt', 'r') as f3:
            predict1 = np.array([int(line.strip()) for line in f1], dtype=int)
            predict2 = np.array([int(line.strip()) for line in f2], dtype=int)
            predict3 = np.array([int(line.strip()) for line in f3], dtype=int)

print(sum(predict1 == predict2)/predict1.size, sum(predict2 == predict3)/predict2.size, sum(predict3 == predict1)/predict3.size)

with open('./outputs/test_ensemble_3_resnets.txt', 'w') as fwrite:
    predicts = np.stack([predict1, predict2, predict3], axis=1)
    decisions = np.ones(predict1.size, dtype=int)*(np.sum(predicts, axis=1) > 1)
    print(*tuple(list(decisions)), sep='\n', file=fwrite)

