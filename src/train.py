import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Import models
from src.model.cnn import CNN
from src.model.transfer import ResNet50Transfer, EfficientNetB0Transfer

def get_args():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection models")
    parser.add_argument("--data_dir", type=str, default="Dataset", help="Path to PlantVillage Dataset folder")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["cnn", "resnet50", "efficientnet_b0"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="plant_disease_model_latest.pt", help="Path to save the best model weights")
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' not found.")
        print("Please download the PlantVillage dataset from Mendeley: https://data.mendeley.com/datasets/tywbtsjrjv/1")
        print("Extract it and place the leaf categories under a folder (e.g. 'Dataset/')")
        return
        
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(args.data_dir)
    num_classes = len(full_dataset.classes)
    print(f"Loaded dataset from {args.data_dir} with {len(full_dataset)} images across {num_classes} classes.")
    
    # Split: 80% train, 10% validation, 10% test
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])
    
    # Apply specific transforms
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform
    test_set.dataset.transform = val_transform
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Select Model
    print(f"Initializing {args.model} model...")
    if args.model == "cnn":
        model = CNN(num_classes)
    elif args.model == "resnet50":
        model = ResNet50Transfer(num_classes, pretrained=True)
    elif args.model == "efficientnet_b0":
        model = EfficientNetB0Transfer(num_classes, pretrained=True)
        
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tune: adjust learning rates for transfer learning
    if args.model != "cnn":
        # Lower learning rate for pre-trained feature extractor, higher for new classifier
        if args.model == "resnet50":
            classifier_params = model.model.fc.parameters()
            backbone_params = [p for name, p in model.model.named_parameters() if not name.startswith("fc")]
        else:  # efficientnet_b0
            classifier_params = model.model.classifier.parameters()
            backbone_params = [p for name, p in model.model.named_parameters() if not name.startswith("classifier")]
            
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": classifier_params, "lr": args.lr}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training Loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        duration = time.time() - t0
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {duration:.1f}s")
              
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"==> Saved new best model checkpoint to {args.save_path} (Val Acc: {val_acc:.4f})")
            
    # Load best model for testing
    print("\nTraining completed. Evaluating best model on independent test split...")
    model.load_state_dict(torch.load(args.save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    # Plot curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    
    plot_path = f"{args.model}_training_curves.png"
    plt.savefig(plot_path)
    print(f"Saved training curves plot to {plot_path}")

if __name__ == "__main__":
    main()
