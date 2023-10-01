import torch 

a = torch.tensor([1.2, 5.1, -4.6])
b = torch.tensor([-2.1, 3.1, 5.5])

print(a.dot(b))

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

print(A.matmul(B))

import random 

random.seed(123)

b=0.
X = [[random.random() for _ in range(1000)] for i in range(500)]
w = [random.random() for _ in range(1000)]
X[10][10] = 'a'

def my_func(X, w, b):
    outputs = []
    for j, x in enumerate(X):
        output = b
        for i, (x_j, w_j) in enumerate(zip(x, w)):
            try:
                output += x_j * w_j
            except: 
                import pdb; pdb.post_mortem()
        outputs.append(output)
    return outputs

print(my_func(X, w ,b))