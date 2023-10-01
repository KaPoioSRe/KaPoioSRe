import torch
import torch.nn.functional as F

def manual_cross_entropy(net_inputs, y_labels):
    activations = torch.softmax(net_inputs, dim=1)
    y_onehot = F.one_hot(y_labels)
    train_losses = - torch.sum(torch.log(activations) * y_onehot, dim=1)
    avg_loss = torch.mean(train_losses)
    return avg_loss

def pytorch_cross_entropy(net_inputs, y_labels):
    return F.cross_entropy(net_inputs, y_labels)