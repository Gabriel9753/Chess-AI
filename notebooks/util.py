import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import re
import numpy as np
from tqdm import tqdm
# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ChessDataset import ChessDataset
from torch.optim import lr_scheduler
import optuna

data_path = '../data/01_raw/small_all_data.csv'
AMOUNT = 500_000

def get_device():
    """Returns the device to be used for training and inference."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def eval_to_number(x):
    int_value = 0
    try:
        int_value = int(x)
    except ValueError as e:
        if x.startswith('#'):
            int_value = 20100 - int(x[2:]) * 100 if x[1] == '+' else -20100 + int(x[2:]) * 100
            int_value = int(int_value)
        else:
            print(f'{e} for {x}')
    return int_value

def load_data():
    """Loads the data from the csv file and returns a pandas dataframe."""
    df = pd.read_csv(data_path)
    chess_data = df.copy()
    chess_data = chess_data.sample(AMOUNT)
    chess_data['Evaluation'] = chess_data['Evaluation'].apply(eval_to_number)
    # normalize the evaluation column to be between -10 and 10
    scaler = MinMaxScaler(feature_range=(-20, 20))
    chess_data['Normalized Evaluation'] = scaler.fit_transform(chess_data['Evaluation'].values.reshape(-1, 1))
    return chess_data

def get_datalaoder(data, batch_size):
    """Returns a dataloader for the given data and batch size."""
    # split chess_data in train and test set
    train_data = data.sample(frac=0.8)
    val_data = data.drop(train_data.index)

    train_dataset = ChessDataset(train_data)
    val_dataset = ChessDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader