def accuracy_between_predicted_and_actual(model, all_x, all_y):
    correct = 0.0

    if not hasattr(model, 'forward') or not callable(getattr(model, 'forward')):
        print("Error: model does not have a callable 'forward' method.")
        return
    
    for x, y in zip(all_x, all_y):
        prediction = model.forward(x)
        correct += int(prediction==y)
    
    return correct / len(all_y)