# -*- coding: utf-8 -*-
"""JTM_HOMO_LUMO.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gQ0QX_9OIADFhTjyzDiE_1LEPyiC3oPY

# HOMO-LUMO prediction of chemicals
"""

from google.colab import drive
# drive.flush_and_unmount()
drive.mount('/content/drive/')

!cp -r /content/drive/MyDrive/Task4/data /content/data
!cp -r /content/drive/MyDrive/Task4/output /content/output

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

class LumoNetwork(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU()
        )

        self.lumo_predictor = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        lumo = self.lumo_predictor(z)
        return lumo

lumo_net = LumoNetwork()
optimizer = optim.Adam(lumo_net.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction="sum")

"""### Training"""

epochs = 30
print_every = 10

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
    
    print(f"[{epoch+1}:2d] Training loss {np.sqrt(running_loss/len(ds_lumo_train)):6.3f}, Validation loss {np.sqrt(valid_loss/len(ds_lumo_valid)):6.3f}")

"""### Save network"""

torch.save(lumo_net.state_dict(), os.path.join(OUT_DIR, "lumo_net_fc3-bl4-fc4_e30.pth"))

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

class HomoLumoGapNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU()
        )
        self.homo_lumo_gap_predictor = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        gap = self.homo_lumo_gap_predictor(z)
        return gap

gap_net = HomoLumoGapNetwork()

# Load pretrained encoder, and disable gradient
pretrained_state = torch.load(os.path.join(OUT_DIR, "lumo_net_fc3-bl4-fc4_e30.pth"))
with torch.no_grad():
    for i, layer in enumerate(gap_net.encoder):
        if i % 2 == 0:
            layer.weight.copy_(pretrained_state[f"encoder.{i}.weight"])
            layer.bias.copy_(pretrained_state[f"encoder.{i}.bias"])
            layer.requires_grad = False

optimizer = optim.Adam(gap_net.homo_lumo_gap_predictor.parameters(), lr=1e-3)
criterion = nn.MSELoss()
critvalid = nn.MSELoss(reduction="sum")

gap_net

"""### Training"""

epochs = 40

for epoch in range(epochs):

    running_loss = 0.0
    gap_net.train(True)

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

df_output.to_csv(os.path.join(OUT_DIR, "test_predict_fc3-bl4-fc4_e30.csv"), index=False)

