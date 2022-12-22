#!/home/vs398/miniconda3/envs/pytorch_env/bin/python

# import the pytorch library into environment and check its version
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, Data, DataLoader
from tqdm import tqdm
#print("Using torch", torch.__version__)
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
#conda install pyg -c pyg
import numpy as np
import matplotlib.mlab as mlab
from scipy.signal import hilbert
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import resample
torch.cuda.is_available()
import numpy as np
import matplotlib.mlab as mlab
from scipy.signal import hilbert
from sklearn.feature_selection import mutual_info_regression

def gen_edges_corr(x, weighted=True):
    adj = np.corrcoef(x.T)
    adj[range(adj.shape[0]), range(adj.shape[0])] = 0
    avg = np.sum(adj) / (adj.shape[0] * adj.shape[0] - adj.shape[0])
    zeros_index = np.argwhere(adj <= avg)
    adj[zeros_index[:, 0], zeros_index[:, 1]] = 0

    edge_index = np.argwhere(adj != 0).T

    if weighted:
        edge_weight = adj[edge_index[0, :], edge_index[1, :]].reshape(-1, 1)
        return edge_index, edge_weight
    else:
        return edge_index

def gen_edges_mi(x, weighted=True):
    adj = mutual_info_regression(x, x, discrete_features=[False])
    adj[range(adj.shape[0]), range(adj.shape[0])] = 0
    avg = np.sum(adj) / (adj.shape[0] * adj.shape[0] - adj.shape[0])
    zeros_index = np.argwhere(adj <= avg)
    adj[zeros_index[:, 0], zeros_index[:, 1]] = 0

    edge_index = np.argwhere(adj != 0).T

    if weighted:
        edge_weight = adj[edge_index[0, :], edge_index[1, :]].reshape(-1, 1)
        return edge_index, edge_weight
    else:
        return edge_index

def gen_edges_cg(x):
    samples, channels = x.shape
    edge_index = [[i, j] for i in range(channels) for j in range(channels)
                  if i != j]
    edge_index = np.asarray(edge_index).T
    return edge_index


def gen_features_raw(x):
    # x = x[range(0, x.shape[0], 2), :]
    features = x.T
    return features


def gen_data_list(data, label, edge_type='corr'):
    data_list = []
    for trial in range(data.shape[0]):
        trial_data = data[trial, ...]
        trial_label = label[trial]

        # generate edge index and node features
        if edge_type == 'corr':
            edge_index, edge_weight = gen_edges_corr(trial_data)
        elif edge_type == 'mi':
            edge_index, edge_weight = gen_edges_corr(trial_data)
        elif edge_type == 'cg':
            edge_index = gen_edges_cg(trial_data)
            edge_weight = np.zeros((edge_index.shape[-1], 1))

        x = gen_features_raw(trial_data)

        edge_index = torch.from_numpy(edge_index).long()
        edge_weight = torch.from_numpy(edge_weight).float()
        x = torch.from_numpy(x).float()

        graph_data = Data(x=x, edge_index=edge_index,
                          y=trial_label, edge_attr=edge_weight)
        data_list.append(graph_data)
    return data_list



class EcgDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        super(EcgDataset,self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
      
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]  
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index,row in tqdm(self.data.iterrows()):
            # Read data from `raw_path`.
            label = self._get_labels(row["Male"])
            data_npy = np.load(self.root + '/'+row['np_fileID'])
            data_npy = data_npy[:2500]
            data_npy = resample(data_npy, 500, axis=0)
            data_npy =data_npy[:,0:12]
            data = gen_data_list(np.expand_dims(data_npy,0),label,edge_type='corr')
            data = data[0]

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if self.test:
                torch.save(data, os.path.join(self.processed_dir, 
                        f'data_test_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, 
                        f'data_{index}.pt'))
    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
        
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

prefix = '/mnt/yale-ecg-signals/numpy_rp/'
ecg_df = '/mnt/yale-ecg-signals/numpy_rp/ecg_df.csv'
ecg_df_test = '/mnt/yale-ecg-signals/numpy_rp/ecg_df_test.csv'
dataset = EcgDataset(prefix,ecg_df)
dataset_test = EcgDataset(prefix,ecg_df_test,test=True)