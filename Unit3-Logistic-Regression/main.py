import sys
sys.path.append("..")

import pandas as pd
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.LogisticRegression import LogisticRegression
from trains.logistic_classification_train import train, train_with_standardization
from data_sets.SimpleDataset import SimpleDataset
from accuracies.logistic_regression import compute_accuracy

def dataset_load(path):
    df = pd.read_csv(os.path.join(os.getcwd(), path), header=None)
    return df

def unit_3_exercise(df):
    
    X_features = df[[0, 1, 2, 3]].values
    y_labels = df[4].values
    print(X_features.shape)
    print(np.bincount(y_labels))

    train_size = int(X_features.shape[0]*0.80)
    validation_size = X_features.shape[0] - train_size

    dataset = SimpleDataset(X_features, y_labels)

    torch.manual_seed(1)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=10,
        shuffle=True,
    )

    validation_loader = DataLoader(
        dataset=val_set,
        batch_size=10,
        shuffle=False,
    )

    # Compute the mean (for standardization)
    train_mean = torch.zeros(X_features.shape[1])
    for x, y in train_loader:
        train_mean += x.sum(dim=0)
    train_mean /= len(train_set)

    # Compute the standard deviation (for standardization)
    train_std = torch.zeros(X_features.shape[1])
    for x, y in train_loader:
        train_std += ((x - train_mean)**2).sum(dim=0)

    train_std = torch.sqrt(train_std / (len(train_set)-1))

    print("Feature means:", train_mean)
    print("Feature std. devs:", train_std)

    lr=0.5
    num_epochs=2

    model = LogisticRegression(num_features=4)
    print("==========Training Without Standardization==========")
    train(model, train_loader, lr, num_epochs)

    print("==========Training Period ended==========")

    train(model, validation_loader, lr, num_epochs)
    print("==========Validation Period ended==========")
    print()
    
    print(f"Training accuracy: {compute_accuracy(model, train_loader)*100:.2f}%")
    print(f"Validation accuracy: {compute_accuracy(model, validation_loader)*100:.2f}%")

    print("==========Training With Standardization==========")
    std_model = LogisticRegression(num_features=4)
    train_with_standardization(std_model, train_loader, train_mean, train_std, lr, 20)

    print("==========Training Period ended==========")

    train_with_standardization(std_model, validation_loader, train_mean, train_std, lr, 20)
    print("==========Validation Period ended==========")
    print()
    
    print(f"Training accuracy: {compute_accuracy(std_model, train_loader, train_mean, train_std, True)*100:.2f}%")
    print(f"Validation accuracy: {compute_accuracy(std_model, validation_loader, train_mean, train_std, True)*100:.2f}%")




try:
    with open("data_banknote_authentication.txt", "r") as f:
        data= json.load(f)
        if not os.path.exists("actual_data.csv"):
            with open("actual_data.csv", "w") as f:
                for line in data['payload']['blob']['rawLines']:
                    f.write(f"{line}\n")
except: 
    print("Error importing the data")

df = dataset_load("actual_data.csv")
df.head()
unit_3_exercise(df)