import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import f1_score
from net1d import Net1D
from inception import Inception, InceptionBlock

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 7
FOLD_NUM = 5
FILE_PATH = "PATH"
LEARNING_RATE = 0.002
BATCH_SIZE = 1
MAX_EPOCHS = 100
DROPOUT_RATE = 0.5

print(f"Using {DEVICE} device")

# label encoding
def label_change(label):
    return [1 if i in [1, 2, 3, 4] else 0 for i in label]

# for class imbalance
def label_weight(label):
    unique, counts = np.unique(label, return_counts=True)
    normed_weights = [1 - (x / sum(counts)) for x in counts]
    return torch.FloatTensor(normed_weights).to(DEVICE)

# load data
def load_data(file_list):
    data_path = "PATH/DATA/"
    label_path = "PATH/LABEL/"
    
    data = []
    labels = []
    
    for file in file_list:
        data.append(np.load(data_path + file).reshape(-1, 730800, 3))
        label = np.load(label_path + file)
        labels.append(np.array(label_change(label)).reshape(-1, 812))
    
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Data shape: {data.shape}, Label shape: {labels.shape}")
    return data, labels

# for k-fold validation
def k_fold(k, fold_num, file_path):
    subject_list = sorted(os.listdir(file_path))
    samples_in_fold = int(round(len(subject_list) / k))
    
    if len(subject_list) % samples_in_fold != 0:
        supplement_num = samples_in_fold - (len(subject_list) % samples_in_fold)
        subject_list += subject_list[:supplement_num]
    
    test_subject_set = subject_list[samples_in_fold*(fold_num-1):samples_in_fold*fold_num]
    train_subject_set = [x for x in subject_list if x not in test_subject_set]
    
    return train_subject_set, test_subject_set

# feature load
def load_features(file_list):
    path = "PATH/FEATURE/"
    data = np.concatenate([np.load(path + file).reshape(-1, 812, 16) for file in file_list], axis=0)
    
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    
    return normalized_data


# Main (load data)
train_subject_set, test_subject_set = k_fold(K_FOLDS, FOLD_NUM, FILE_PATH)
print("Train subjects:", train_subject_set)
print("Test subjects:", test_subject_set)

# Load and prepare training data (dataset)
train_data, train_labels = load_data(train_subject_set)
train_weights = label_weight(train_labels)

train_data = torch.FloatTensor(train_data.transpose(0, 2, 1))
train_labels = torch.FloatTensor(train_labels)

train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# Load and prepare test data (dataset)
test_data, test_labels = load_data(test_subject_set)

test_data = torch.FloatTensor(test_data.transpose(0, 2, 1))
test_labels = torch.FloatTensor(test_labels)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Load and prepare features
train_features = load_features(train_subject_set)
test_features = load_features(test_subject_set)

train_features = torch.from_numpy(train_features.transpose(0, 2, 1)).to(DEVICE)
test_features = torch.from_numpy(test_features.transpose(0, 2, 1)).to(DEVICE)

