import torch

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features)
        self.bias = torch.tensor(0.)

    def forward(self, x):
        weighted_sum_z = torch.matmul(x, self.weights) + self.bias

        prediction = torch.where(weighted_sum_z > 0., 1., 0.)
        return prediction


    def update(self, x, true_y):
        prediction = self.forward(x)        
        error = true_y - prediction
        
        self.bias = self.bias + error
        self.weights += error * x

        return error