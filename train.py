import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import models
from models.vit import VisionTransformer
from models.performer_relu import PerformerReLUTransformer
from models.performer_exp import PerformerExpTransformer
from models.performer_f_theta import PerformerLearnableTransformer


# Import utility functions
from utils import set_seed, log_results

import os



def main():
    # Import utility functions
    from utils import set_seed, log_results

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    print(f"Using device: {device}, num_workers: {num_workers}")

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

    # Define criterion and optimizers
    criterion = nn.CrossEntropyLoss()

    optimizer_vit = optim.AdamW(vit_model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_relu = optim.AdamW(performer_relu_model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_exp = optim.AdamW(performer_exp_model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_learnable = optim.AdamW(performer_learnable_model.parameters(), lr=0.001, weight_decay=0.01)

    # Optionally, define learning rate schedulers
    scheduler_vit = optim.lr_scheduler.CosineAnnealingLR(optimizer_vit, T_max=50)
    scheduler_relu = optim.lr_scheduler.CosineAnnealingLR(optimizer_relu, T_max=50)
    scheduler_exp = optim.lr_scheduler.CosineAnnealingLR(optimizer_exp, T_max=50)
    scheduler_learnable = optim.lr_scheduler.CosineAnnealingLR(optimizer_learnable, T_max=50)

    num_epochs = 10

    # Training function
    def train_model(model, optimizer, scheduler, model_name):
        best_val_accuracy = 0
        training_times = []
        inference_times = []

        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            running_loss = 0

            for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_time = time.time() - start_time
            training_times.append(epoch_time)

            # Validation
            model.eval()
            correct = 0
            total = 0
            inference_start = time.time()

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            val_accuracy = 100. * correct / total
            print(f"{model_name} Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {running_loss / len(train_loader.dataset):.4f} "
                  f"Val Accuracy: {val_accuracy:.2f}% "
                  f"Training Time: {epoch_time:.2f}s "
                  f"Inference Time: {inference_time:.2f}s")

            # Update scheduler
            scheduler.step()

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), f'best_{model_name}.pth')

        # Log results
        log_results(model_name, training_times, inference_times, best_val_accuracy)

    # Train and evaluate ViT
    print("\nTraining Vision Transformer")
    train_model(vit_model, optimizer_vit, scheduler_vit, "ViT")

    # Train and evaluate Performer-ReLU
    print("\nTraining Performer-ReLU")
    train_model(performer_relu_model, optimizer_relu, scheduler_relu, "Performer-ReLU")

    # Train and evaluate Performer-exp
    print("\nTraining Performer-exp")
    train_model(performer_exp_model, optimizer_exp, scheduler_exp, "Performer-exp")

    # Train and evaluate Performer-fθ (Learnable)
    print("\nTraining Performer-Learnable")
    train_model(performer_learnable_model, optimizer_learnable, scheduler_learnable, "Performer-Learnable")

if __name__ == "__main__":
    main()