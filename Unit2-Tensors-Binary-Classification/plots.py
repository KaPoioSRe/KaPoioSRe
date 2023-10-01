def plot_boundary_binary_classification_two_features(model):
    w1, w2 = model.weights[0], model.weigths[1]
    b = model.bias

    x1_min = -20
    x2_min = (-(w1 * x1_min) -b) / w2

    
    x1_max = -20
    x2_max = (-(w1 * x1_max) -b) / w2

    return x1_min, x1_max, x2_min, x2_max

