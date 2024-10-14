import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os
from collections import OrderedDict
# Import models
from models.vit import VisionTransformer
from models.performer_relu import PerformerReLUTransformer
from models.performer_exp import PerformerExpTransformer
from models.performer_f_theta import PerformerLearnableTransformer
import pickle
from utils import set_seed, log_results

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models with adjusted parameters for efficiency
    vit_model = VisionTransformer(
        img_size=32,
        patch_size=8,            # Increased patch size from 4 to 8
        in_channels=3,
        num_classes=10,
        embed_dim=128,           # Reduced embedding dimension from 256 to 128
        num_heads=4,             # Reduced number of heads from 8 to 4
        hidden_dim=256,          # Reduced hidden dimension from 512 to 256
        num_layers=4,            # Reduced number of layers from 6 to 4
        dropout=0.1
    ).to(device)

    performer_relu_model = PerformerReLUTransformer(
        img_size=32,
        patch_size=8,            # Increased patch size from 4 to 8
        in_channels=3,
        num_classes=10,
        embed_dim=128,           # Reduced embedding dimension from 256 to 128
        num_heads=4,             # Reduced number of heads from 8 to 4
        hidden_dim=256,          # Reduced hidden dimension from 512 to 256
        num_layers=4,            # Reduced number of layers from 6 to 4
        dropout=0.1
    ).to(device)

    performer_exp_model = PerformerExpTransformer(
        img_size=32,
        patch_size=8,            # Increased patch size from 4 to 8
        in_channels=3,
        num_classes=10,
        embed_dim=128,           # Reduced embedding dimension from 256 to 128
        num_heads=4,             # Reduced number of heads from 8 to 4
        hidden_dim=256,          # Reduced hidden dimension from 512 to 256
        num_layers=4,            # Reduced number of layers from 6 to 4
        num_features=128,        # Adjusted num_features to match reduced embed_dim
        dropout=0.1
    ).to(device)

    performer_learnable_model = PerformerLearnableTransformer(
        img_size=32,
        patch_size=8,            # Increased patch size from 4 to 8
        in_channels=3,
        num_classes=10,
        embed_dim=128,           # Reduced embedding dimension from 256 to 128
        num_heads=4,             # Reduced number of heads from 8 to 4
        hidden_dim=256,          # Reduced hidden dimension from 512 to 256
        num_layers=4,            # Reduced number of layers from 6 to 4
        num_features=128,        # Adjusted num_features to match reduced embed_dim
        dropout=0.1
    ).to(device)



    # Define optimizers
    optimizer_vit = optim.AdamW(vit_model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_relu = optim.AdamW(performer_relu_model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_exp = optim.AdamW(performer_exp_model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_learnable = optim.AdamW(performer_learnable_model.parameters(), lr=0.001, weight_decay=0.01)

    # Define schedulers
    scheduler_vit = optim.lr_scheduler.CosineAnnealingLR(optimizer_vit, T_max=50)
    scheduler_relu = optim.lr_scheduler.CosineAnnealingLR(optimizer_relu, T_max=50)
    scheduler_exp = optim.lr_scheduler.CosineAnnealingLR(optimizer_exp, T_max=50)
    scheduler_learnable = optim.lr_scheduler.CosineAnnealingLR(optimizer_learnable, T_max=50)

    # Models dictionary
    models = {
        "ViT": {
            "model": vit_model,
            "optimizer": optimizer_vit,
            "scheduler": scheduler_vit,
        },
        "Performer-ReLU": {
            "model": performer_relu_model,
            "optimizer": optimizer_relu,
            "scheduler": scheduler_relu,
        },
        "Performer-exp": {
            "model": performer_exp_model,
            "optimizer": optimizer_exp,
            "scheduler": scheduler_exp,
        },
        "Performer-Learnable": {
            "model": performer_learnable_model,
            "optimizer": optimizer_learnable,
            "scheduler": scheduler_learnable,
        },
    }

    for model_name, components in models.items():
        model = components["model"]
        optimizer = components["optimizer"]
        scheduler = components["scheduler"]

        # Initialize variables
        start_epoch = 0
        best_val_accuracy = 0

        # Check if a checkpoint exists
        checkpoint_path = f'best_{model_name}.pth'
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint for {model_name} from '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with model and optimizer states
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_accuracy = checkpoint['best_val_accuracy']
                print(f"Resuming {model_name} from epoch {start_epoch} with best validation accuracy {best_val_accuracy:.2f}%")
            elif isinstance(checkpoint, (dict, OrderedDict)):
                # Checkpoint is just the model's state_dict
                model.load_state_dict(checkpoint)
                start_epoch = 0
                best_val_accuracy = 0
                print(f"Loaded model state_dict for {model_name}. Starting training from epoch {start_epoch}.")
            else:
                print(f"Unexpected checkpoint format for {model_name}. Starting from scratch.")
        else:
            print(f"\nNo checkpoint found for {model_name}, starting from scratch.")

        # Train the model
        print(f"\nTraining {model_name}")
        train_model(
            model,
            optimizer,
            scheduler,
            model_name,
            start_epoch=start_epoch,
            best_val_accuracy=best_val_accuracy
        )

def train_model(model, optimizer, scheduler, model_name, start_epoch=0, best_val_accuracy=0):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    print(f"using device: {device}, num_workers: {num_workers}")

    # Set seed for reproducibility
    set_seed(42)

    # Define data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # Create data loaders
    batch_size = 256
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Define criterion
    criterion = nn.CrossEntropyLoss()

    
    num_epochs = 10
    training_times = []
    inference_times = []

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_time = time.time() - start_time
        training_times.append(epoch_time)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100. * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        running_val_loss = 0
        correct = 0
        total = 0
        inference_start = time.time()

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        inference_time = time.time() - inference_start
        inference_times.append(inference_time)

        val_loss = running_val_loss / len(test_loader.dataset)
        val_accuracy = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"{model_name} Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Train Acc: {train_accuracy:.2f}% "
              f"Val Acc: {val_accuracy:.2f}% "
              f"Train Time: {epoch_time:.2f}s "
              f"Infer Time: {inference_time:.2f}s")

        # Update scheduler
        scheduler.step()

        # Save checkpoint if current epoch's validation accuracy is better
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
            }
            torch.save(checkpoint, f'best_{model_name}.pth')

    # Log results
    log_results(model_name, training_times, inference_times, best_val_accuracy)

    # Save metrics for visualization
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    with open(f'{model_name}_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

if __name__ == "__main__":
    main()