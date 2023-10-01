import torch

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, num_features, num_classes=1):
        super().__init__()
        if num_classes==0:
            raise ValueError("Number of classes cannot be 0")
        self.classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        logits = self.linear(x)
        if self.classes == 1:
            probas = torch.sigmoid(logits)
        else:
            probas = torch.nn.functional.softmax(logits)
        return probas