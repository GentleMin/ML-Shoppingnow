# -*- coding: utf-8 -*-
"""

Original file is located at
    https://colab.research.google.com/drive/1gQ0QX_9OIADFhTjyzDiE_1LEPyiC3oPY

# HOMO-LUMO prediction of chemicals
"""

import numpy as np
import pandas as pd
import random
import os, glob

import torch
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

import matplotlib.pyplot as plt

DAT_DIR = "/content/data"
OUT_DIR = "/content/output"
BATCH_SIZE = 1024

"""## 1. Pretrain network with pretrain dataset and LUMO labels"""

feat_lumo = pd.read_csv(os.path.join(DAT_DIR, "pretrain_features.csv.zip"))
feat_lumo = torch.tensor(feat_lumo.iloc[:, 2:].values.astype(np.float32))
labl_lumo = pd.read_csv(os.path.join(DAT_DIR, "pretrain_labels.csv.zip"))
labl_lumo = torch.tensor(labl_lumo["lumo_energy"].values.astype(np.float32))

ds_lumo = TensorDataset(feat_lumo, labl_lumo)
ds_lumo_train, ds_lumo_valid = torch.utils.data.random_split(ds_lumo, 
    [int(0.9*len(ds_lumo)), int(0.1*len(ds_lumo))], torch.Generator().manual_seed(0))
loader_lumo_train = DataLoader(ds_lumo_train, batch_size=BATCH_SIZE)
loader_lumo_valid = DataLoader(ds_lumo_valid, batch_size=BATCH_SIZE)

class lumo_network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500,200),
            nn.ReLU(),
            nn.Linear(200,75),
            nn.ReLU(),
            nn.Linear(75,24),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self,inputs):
        features = self.encoder(inputs)
        pred = self.predictor(features)
        # gap = self.gap(pred)
        
        return pred


class homo_lumo_gap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500,200),
            nn.ReLU(),
            nn.Linear(200,75),
            nn.ReLU(),
            nn.Linear(75,24),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.pred_gap = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, inputs):
        features = self.encoder(inputs)
        medium = self.predictor(features)
        gap = self.pred_grad(medium)
        return gap


"""Training on large LUMO dataset"""
 
lumo_net = lumo_network()
pretrained_dict = torch.load(os.path.join(DAT_DIR, "autoencoder.pth"), torch.device("cpu"))
pretrained_dict = pretrained_dict.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in lumo_net.state_dict()}
# 2. overwrite entries in the existing state dict
lumo_state = lumo_net.state_dict()
lumo_state.update(pretrained_dict)
lumo_net.load_state_dict(lumo_state)

optimizer = optim.Adam(lumo_net.parameters(), lr=3e-3)
criterion = nn.MSELoss(reduction="sum")

for layer in lumo_net.encoder.parameters():
    layer.requires_grad = False

for name, param in lumo_net.named_parameters():
    print(name, param.requires_grad)

epochs = 30
print_every = 10

epochs = 40
validations = list()

for epoch in range(epochs):

    running_loss = 0.0
    lumo_net.train(True)

    for i, (data, lumo_labl) in enumerate(loader_lumo_train):
        optimizer.zero_grad()
        lumo_pred = lumo_net(data)
        loss = criterion(lumo_labl, torch.squeeze(lumo_pred))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    valid_loss = 0.0
    lumo_net.eval()

    with torch.no_grad():
        for data, lumo_labl in loader_lumo_valid:
            lumo_pred = lumo_net(data)
            loss = criterion(lumo_labl, torch.squeeze(lumo_pred))
            valid_loss += loss.item()
    
    print(f"[{epoch+1:2d}] Training loss {np.sqrt(running_loss/len(ds_lumo_train)):6.3f}, Validation loss {np.sqrt(valid_loss/len(ds_lumo_valid)):6.3f}")
    if epoch >= 10 and np.sqrt(valid_loss/len(ds_lumo_valid)) < validations[-1]:
        torch.save(lumo_net.state_dict(), os.path.join(OUT_DIR, f"temp_{epoch+1}.pth"))
        validations.append(np.sqrt(valid_loss/len(ds_lumo_valid)))
    elif epoch < 10:
        validations.append(np.sqrt(valid_loss/len(ds_lumo_valid)))
    else:
        validations.append(validations[-1])

"""### Save network"""

torch.save(lumo_net.state_dict(), os.path.join(OUT_DIR, "lumo_net_ac4-bl24-pred5d2_e40.pth"))

"""## 2. Transfer the network to HOMO-LUMO gap dataset"""

feat_gap = pd.read_csv(os.path.join(DAT_DIR, "train_features.csv.zip"))
feat_gap = torch.tensor(feat_gap.iloc[:, 2:].values.astype(np.float32))
labl_gap = pd.read_csv(os.path.join(DAT_DIR, "train_labels.csv.zip"))
labl_gap = torch.tensor(labl_gap["homo_lumo_gap"].values.astype(np.float32))

ds_gap = TensorDataset(feat_gap, labl_gap)
ds_gap_train, ds_gap_valid = torch.utils.data.random_split(ds_gap, 
    [int(0.9*len(ds_gap)), int(0.1*len(ds_gap))], torch.Generator().manual_seed(0))
loader_gap_train = DataLoader(ds_gap_train, batch_size=128)
loader_gap_valid = DataLoader(ds_gap_valid, batch_size=128)

gap_net = homo_lumo_gap()

# Load pretrained encoder, and disable gradient
pretrained_dict = torch.load(os.path.join(OUT_DIR, "lumo_net_ac4-bl24-pred5d2_e40_v2.pth"))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in gap_net.state_dict()}
gap_net_state = gap_net.state_dict()
gap_net_state.update(pretrained_dict)
gap_net.load_state_dict(gap_net_state)

# Freeze encoder
for param in gap_net.encoder.parameters():
    param.requires_grad = False

# Freeze part of predictor
for param in gap_net.predictor.parameters():
    param.requires_grad = False

# optimizer = optim.Adam(gap_net.lumo_predictor[idx_layer[-1]].parameters(), lr=3e-2, weight_decay=1e-4)
optimizer = optim.Adam(gap_net.parameters(), lr=1e-2, weight_decay=1e-3)
criterion = nn.MSELoss()
critvalid = nn.MSELoss(reduction="sum")

for name, param in gap_net.named_parameters():
    print(name, param.requires_grad)

"""### Training"""

epochs = 286

for epoch in range(epochs):

    running_loss = 0.0
    gap_net.train(True)
    # gap_net.predictor[4].eval()

    for i, (data, gap_labl) in enumerate(loader_gap_train):
        optimizer.zero_grad()
        gap_pred = gap_net(data)
        loss = criterion(gap_labl, torch.squeeze(gap_pred))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / (i+1)

    valid_loss = 0.0
    gap_net.eval()

    with torch.no_grad():
        for data, gap_labl in loader_gap_valid:
            gap_pred = gap_net(data)
            loss = critvalid(gap_labl, torch.squeeze(gap_pred))
            valid_loss += loss.item()

    print(f"[{epoch+1:2d}] Training loss {np.sqrt(train_loss):6.3f}, Validation loss {np.sqrt(valid_loss/len(ds_gap_valid)):6.3f}")


"""## 3. Predict HOMO-LUMO gap"""

feat_test_frame = pd.read_csv(os.path.join(DAT_DIR, "test_features.csv.zip"))
feat_test = torch.tensor(feat_test_frame.iloc[:, 2:].values.astype(np.float32))

ds_test = TensorDataset(feat_test)
loader_test = DataLoader(ds_test, batch_size=1024)

pred_test = list()
gap_net.eval()
with torch.no_grad():
    for i, data in enumerate(loader_test):
        gap_pred = gap_net(data[0])
        pred_test.append(torch.squeeze(gap_pred).numpy())
pred_test = np.concatenate(pred_test)

df_output = pd.DataFrame(data={"Id": feat_test_frame.Id, "y": pred_test})

df_output.to_csv(os.path.join(OUT_DIR, "test_predict_ac5-bl24-pred5drop_gap-retrain2_e30-286.csv"), index=False)

