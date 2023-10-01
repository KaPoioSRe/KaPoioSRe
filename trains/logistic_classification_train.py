import torch
import torch.nn.functional as F

from useful_functions.feature_normalization import z_score_standardize

# model has to implement torch.nn.Module
def train(model, data_loader, learning_rate, epochs, train_mean=None, train_std=None):
    torch.manual_seed(123)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

    for epoch in range(epochs):
        
        model = model.train()
        for batch_idx, (features, class_labels) in enumerate(data_loader):
            
            # Standardize if mean and standard deviation are given
            if train_mean and train_std:
                features = z_score_standardize(features, train_mean, train_std)
            probas = model(features)
            
            loss = F.binary_cross_entropy(probas, class_labels.view(probas.shape))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 20: # log every 20th batch
                print(f'Epoch: {epoch+1:03d}/{epochs:03d}'
                    f' | Batch {batch_idx:03d}/{len(data_loader):03d}'
                    f' | Loss: {loss:.2f}')
                
def train_with_standardization(model, data_loader, train_mean, train_std, learning_rate, epochs):
    torch.manual_seed(123)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

    for epoch in range(epochs):

        model = model.train()
        for batch_idx, (features, class_labels) in enumerate(data_loader):

            features = z_score_standardize(features, train_mean, train_std)
            probas = model(features)
            
            loss = F.binary_cross_entropy(probas, class_labels.view(probas.shape))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 20: # log every 20th batch
                print(f'Epoch: {epoch+1:03d}/{epochs:03d}'
                    f' | Batch {batch_idx:03d}/{len(data_loader):03d}'
                    f' | Loss: {loss:.2f}')