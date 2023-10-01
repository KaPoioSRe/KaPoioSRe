import sys
sys.path.append("..")

import torch
import random
from models.perceptron import Perceptron
from accuracies.binary_classification import accuracy_between_predicted_and_actual
from trains.binary_classification_train import train
import pandas as pd
import numpy as np
import os
import json

def random_initialization_of_X_Y(feature_count=5, train_examples=100, epochs=5):
    ppn = Perceptron(num_features=feature_count)

    X_train = torch.tensor([[random.random() for _ in range(feature_count)] for i in range(train_examples)])
    Y_train = torch.tensor([random.randint(0,1) for _ in range(train_examples)])
    train(ppn, X_train, Y_train, epochs)
    train_acc = accuracy_between_predicted_and_actual(ppn, X_train, Y_train)
    print(f"Accuracy: {train_acc*100}%")

def dataset_load(path, sep="\t"):
    df = pd.read_csv(os.path.join(os.getcwd(), path), sep=sep)
    return df

def unit_2():
    try:
        with open("perceptron_toydata-truncated.txt", "r") as f:
            data= json.load(f)
            if not os.path.exists("actual_data.csv"):
                with open("actual_data.csv", "w") as f:
                    for line in data['payload']['blob']['rawLines']:
                        f.write(f"{line}\n")
        df = dataset_load("actual_data.csv")

        X_train = df[["x1", "x2"]].values
        y_train = df["label"].values
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_train = X_train.to(torch.float32)

        ppn = Perceptron(num_features=2)

        train(ppn, X_train, y_train, epochs=5)
        train_acc = accuracy_between_predicted_and_actual(ppn, X_train, y_train)

        print(f"Accuracy: {train_acc*100}%")
        return ppn
    
    except: 
        print("Error importing the data")

def unit_2_exercise_2(model):
    X_data = torch.tensor([
        [-1.0, -2.0],
        [-3.0, 4.5],
        [5.0, 6.0]
    ])

    
    for x in X_data:
        print(model.forward(x))

trained_model = unit_2()
unit_2_exercise_2(trained_model)