import random
import numpy as np
import torch
import csv
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_results(model_name, training_times, inference_times, accuracy):
    file_exists = os.path.isfile('results.csv')
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Total Training Time', 'Total Inference Time', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnamesfieldnames=fieldnames)
        
        # Write header only if file does not exist or is empty
        if not file_exists or os.stat('results.csv').st_size == 0:
            writer.writeheader()
        
        writer.writerow({
            'Model': model_name,
            'Total Training Time': sum(training_times),
            'Total Inference Time': sum(inference_times),
            'Accuracy': accuracy
        })

def load_model(model_class, checkpoint_path):
    model = model_class(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        hidden_dim=512,
        num_layers=6,
        dropout=0.1
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model