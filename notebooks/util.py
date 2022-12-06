import math
import re
from functools import partial

import numpy as np
import optuna
import pandas as pd
# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna.trial import TrialState
from sklearn.preprocessing import MinMaxScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# import copy
import copy
from ChessDataset import ChessDataset
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from lookahead import Lookahead
import settings

import pyarrow.parquet as pq
SAMPLES = settings.SAMPLES
BATCH_SIZE = settings.BATCH_SIZE
LEARNING_RATE = settings.LEARNING_RATE


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

def process_data(df):
    df['Evaluation'] = df['Evaluation'].apply(eval_to_number)
    # normalize the evaluation column to be between -10 and 10
    scaler = MinMaxScaler(feature_range=(-50, 50))
    df['Normalized Evaluation'] = scaler.fit_transform(df['Evaluation'].values.reshape(-1, 1))
    
    # remove data in normalized evaluation that is not between -2 and 2
    df_only_between = df[(df['Normalized Evaluation'] > -5) & (df['Normalized Evaluation'] < 5)]
    scaler = MinMaxScaler(feature_range=(-30, 30))
    df_only_between['Normalized Evaluation'] = scaler.fit_transform(df_only_between['Normalized Evaluation'].values.reshape(-1, 1))
    
    df_not_between = df[(df['Normalized Evaluation'] <= -5) | (df['Normalized Evaluation'] >= 5)]
    df_not_between = df_not_between[(df_not_between['Normalized Evaluation'] <= -30) | (df_not_between['Normalized Evaluation'] >= 30)]
    
    df = pd.concat([df_only_between, df_not_between])
    return df

def load_data(path_to_data, amount_of_samples, batch_size, split):
    """Loads the data from the csv file and returns a pandas dataframe."""
    parquet_file = pq.ParquetFile(path_to_data)
    FEN = []
    Evaluation = []
    AMOUNT_TO_READ = amount_of_samples
    READ_BATCH_SIZE = batch_size

    with tqdm(total=int(AMOUNT_TO_READ/READ_BATCH_SIZE), desc=split) as pbar:
        for idx, batch in enumerate(parquet_file.iter_batches(batch_size=READ_BATCH_SIZE), start=1):
            batch_df = batch.to_pandas()
            FEN.extend(batch_df["FEN"])
            Evaluation.extend(batch_df["Evaluation"])
            pbar.update()
            if idx * READ_BATCH_SIZE >= AMOUNT_TO_READ:
                break

    df = pd.DataFrame(zip(FEN, Evaluation), columns=["FEN", "Evaluation"])
    df = process_data(df)
    return df

def get_datalaoder(df, split):
    """Returns a dataloader for the given data and batch size."""
    dataset = ChessDataset(df)
    if split == "train":
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=settings.NUM_WORKERS)
    else:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=settings.NUM_WORKERS)
    
    return loader

# ------------------- Main -------------------

def create_model():
    DEVICE = get_device()
    in_features = 13 * 8 * 8
    layers = []
    for i in range(settings.N_LAYERS):
        out_features = settings.N_UNITS_PER_LAYER[i]
        p = settings.DROPOUT_PER_LAYER[i]
        
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 1))
    
    model = nn.Sequential(*layers).to(DEVICE)
    print(model)
    return model

def endless_iter(data_loader):
    while True:
        for data in data_loader:
            yield data
            
def calc_metrics(preds, labels):
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    correct_team = []
    correct_white = []
    correct_black = []
    
    for pred, label in zip(preds, labels):
        if pred[0] < 0 and label[0] < 0:
            correct_black.append(1)
            correct_team.append(1)
        elif pred[0] >= 0 and label[0] >= 0:
            correct_white.append(1)
            correct_team.append(1)
        else:
            correct_white.append(0)
            correct_black.append(0)
            correct_team.append(0)
            
    return {
        "correct_team": np.mean(correct_team),
        "correct_white": np.mean(correct_white),
        "correct_black": np.mean(correct_black)
    }
    
def eval(_model, _eval_loader, criterion):
    DEVICE = get_device()
    eval_loss = 0.0
    nb_eval_steps = 0
    val_preds_list = []
    val_labels_list = []
    _model.eval()
    with torch.no_grad():
        for data, target in tqdm(_eval_loader, desc="Evaluating", leave=False, smoothing=0):
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            output = _model(data.float())
            val_labels_list.append(target.detach().cpu().numpy())
            val_preds_list.append(output.detach().cpu().numpy())
            loss = criterion(output, target)
            eval_loss += loss.item()
            nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    results = calc_metrics(val_preds_list, val_labels_list)
    results["loss"] = eval_loss
    return results

def train(model, train_loader, val_loader, lookahead, criterion, optimizer, scheduler, samples, _twriter, val_interval):
    DEVICE = get_device()
    model.train()
    
    best_loss = np.inf
    best_model = None
    best_metrics = None
    train_preds_list = []
    train_labels_list = []
    with tqdm(total=int(samples/BATCH_SIZE), desc="Training") as pbar:
        for batch_idx, (data, target) in enumerate(endless_iter(train_loader), start=1):
            current_samples = int(batch_idx * BATCH_SIZE)
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            lookahead.zero_grad()
            
            _twriter.add_scalar("lr", optimizer.param_groups[0]["lr"], batch_idx)
            outputs = model(data.float())
            loss = criterion(outputs, target)
            if batch_idx % 100 == 0:
                pbar.set_postfix({'Train-Loss': f'{loss.item():.5f}'})
                _twriter.add_scalar('Train/Loss', loss.item(), batch_idx)
            if (batch_idx % 500) >= 300:
                train_preds_list.append(outputs.detach().cpu().numpy())
                train_labels_list.append(target.detach().cpu().numpy())
                
            if batch_idx % 500 == 0:
                # calc metrics
                results_train = calc_metrics(train_preds_list, train_labels_list)
                for k, v in results_train.items():
                    _twriter.add_scalar(f"train/{k}", v, batch_idx)
                train_preds_list = []
                train_labels_list = []
            loss.backward()
            # optimizer.step()
            lookahead.step()
            try:
                scheduler.step()
            except:
                print(f"{batch_idx}: Scheduler failed")
            # Log training metrics to Tensorboard
            pbar.update()
            
            # Validate model in regular intervals
            if batch_idx % max(1, val_interval) == 0:
                # Check if val loader actually contains examples
                if val_loader is not None and len(val_loader) > 0:
                    res = eval(model, val_loader, criterion)
                    eval_loss = res["loss"]
                    model.train()
                    for k, v in res.items():
                        _twriter.add_scalar(f"val/{k}", v, batch_idx)

                    if eval_loss < best_loss:
                        print(f'Sample {batch_idx}: old = {best_loss} | new = {eval_loss}')
                        best_loss = eval_loss
                        best_model = copy.deepcopy(model)
                        best_metrics = res
            _twriter.flush()
            if current_samples >= samples:
                break
    return best_model, best_metrics

def main():
    model = create_model()
    VAL_INTERVAL = SAMPLES // BATCH_SIZE // 10
    
    cur_date = datetime.now().date()
    cur_time = str(datetime.now().time()).split('.')[0]
    RUN_PATH = f"runs/test_{cur_date}_{cur_time}"
    os.makedirs(RUN_PATH, exist_ok=False)
    _twriter = SummaryWriter(log_dir=RUN_PATH)
    train_df = load_data(settings.PATH_TO_TRAIN_DATA, settings.TRAIN_SAMPLES, settings.LOAD_BATCH_SIZE, "train")
    val_df = load_data(settings.PATH_TO_VAL_DATA, settings.VAL_SAMPLES, settings.LOAD_BATCH_SIZE, "val")
    
    train_loader = get_datalaoder(train_df, "train")
    val_loader = get_datalaoder(val_df, "val")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lookahead = Lookahead(optimizer, k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=SAMPLES//BATCH_SIZE, epochs=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=SAMPLES//BATCH_SIZE//20, eta_min=LEARNING_RATE/100)
    criterion = nn.MSELoss()
    hparams = {
        'iterations': SAMPLES,
        'train_batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
    }

    best_model, best_metrics = train(model, train_loader, val_loader, lookahead, criterion, optimizer, scheduler, SAMPLES, _twriter, VAL_INTERVAL)
    _twriter.add_hparams(hparams, metric_dict=best_metrics)
    # save model
    torch.save(best_model.state_dict(), os.path.join("models/", 'model.pt'))
    _twriter.close()
    
if __name__ == "__main__":
    main()