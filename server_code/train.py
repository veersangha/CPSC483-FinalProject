#!/home/vs398/miniconda3/envs/pytorch_env/bin/python
# import the pytorch library into environment and check its version
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, Data, DataLoader

from torch.nn import Linear
from torch.nn import Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from main import *
from torchmetrics.classification import AUROC
from models import GCN_TC
from preprocessing import EcgDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## files for the data on the server
prefix = '/mnt/yale-ecg-signals/numpy_rp/'
ecg_df = '/mnt/yale-ecg-signals/numpy_rp/ecg_df.csv'
ecg_df_test = '/mnt/yale-ecg-signals/numpy_rp/ecg_df_test_new.csv'
dataset = EcgDataset(prefix,ecg_df)
test_dataset = EcgDataset(prefix,ecg_df_test,test=True)

small_test_dataset = test_dataset[0:1000]

train_loader = DataLoader(dataset,batch_size=64,shuffle=False,num_workers=40)
test_loader = DataLoader(test_dataset,batch_size=64, shuffle=False,num_workers = 40)

## here we can change the model to which we want to run our test on. 
model = GCN_TC(in_channels=dataset[0].num_node_features, hidden_channels=500, out_channels=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

loss_func = torch.nn.BCELoss()



def train(model, loader, optimizer, loss_func):
    '''
    Function taht trains a given model with the data on the server
    '''
    loss = 0
    model.train()

    all_labels = []
    all_preds = []
    for i, data in enumerate(loader):
        data = data.to(device)
        model = model.to(device)
        
        pred = model(data.x, data.edge_index, data.batch)
        pred = pred.to(device) 
        target = data.y
        target = target.unsqueeze(1)
        target = target.float()

        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .01)
        all_labels.append(target)
        all_preds.append(pred)
    all_labels = torch.flatten(torch.cat(all_labels))
    all_preds = torch.flatten(torch.cat(all_preds))
    print(all_preds[0:5])

    auroc = AUROC(task='binary')
    auroc_score = auroc(all_preds,all_labels)
    return model, auroc_score

def test(model, loader):
    '''
    Our test function for training and validating our models
    '''
    model.eval()

    all_labels = []
    all_preds = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch)
        pred = pred.to(device)   
        target = data.y
        target = target.unsqueeze(1)
        target = target.float()

        all_labels.append(target)
        all_preds.append(pred)
    all_labels = torch.flatten(torch.cat(all_labels))
    all_preds = torch.flatten(torch.cat(all_preds)) 
    print(all_preds[0:5])   
    auroc = AUROC(task='binary')
    auroc_score = auroc(all_preds,all_labels)
    return auroc_score  # Derive ratio of correct predictions.

epochs = 100


# Train our model here 

#model= torch.nn.DataParallel(model)
model.to(device)
for epoch in range(1, epochs):
    model, train_auroc = train(model, train_loader, optimizer, loss_func)
    test_auc = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Train AUROC: {train_auroc:.3f} Test AUROC: {test_auc:.3f}')
