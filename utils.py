import random
import numpy as np
import torch
import csv

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_results(model_name, training_times, inference_times, accuracy):
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Total Training Time', 'Total Inference Time', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({
            'Model': model_name,
            'Total Training Time': sum(training_times),
            'Total Inference Time': sum(inference_times),
            'Accuracy': accuracy
        })