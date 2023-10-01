import torch

def compute_accuracy(model, dataloader, train_mean=0, train_std=0, standardization=False):

    model = model.eval()
    
    correct = 0.0
    total_examples = 0
    
    for idx, (features, class_labels) in enumerate(dataloader):
        
        if standardization:
            def standardize(df, train_mean, train_std):
                return (df - train_mean) / train_std
            features = standardize(features, train_mean, train_std)

        with torch.no_grad():
            probas = model(features)
        
        pred = torch.where(probas > 0.5, 1, 0)
        lab = class_labels.view(pred.shape).to(pred.dtype)

        compare = lab == pred
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples