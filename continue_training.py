# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Import preprocessing module - modified for better compatibility with Colab
# The dataset structure is already preset in the environment when this script runs
# try:
#     # First try direct import (when run in same directory)
#     from PreProcessing_dataset import FaceDataset, process_casia_webface
#     print("✓ Successfully imported preprocessing modules")
# except ImportError:
#     try:
#         # For Colab environment where modules might be in different locations
#         import sys
#         # Add the directory containing PreProcessing_dataset.py to Python's path
#         module_path = os.path.abspath(os.path.dirname(__file__))
#         if module_path not in sys.path:
#             sys.path.append(module_path)
        
#         # Now try importing again
#         from PreProcessing_dataset import FaceDataset, process_casia_webface
#         print("✓ Successfully imported preprocessing modules via path adjustment")
#     except ImportError:
#         # Last resort - try to load dynamically if files are in same directory
#         import importlib.util
        
#         # Path to the preprocessing module
#         preprocessing_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
#                                          "PreProcessing_dataset.py")
        
#         if os.path.exists(preprocessing_path):
#             # Load module dynamically
#             spec = importlib.util.spec_from_file_location("PreProcessing_dataset", preprocessing_path)
#             preprocessing = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(preprocessing)
            
#             # Get the needed components
#             FaceDataset = preprocessing.FaceDataset
#             process_casia_webface = preprocessing.process_casia_webface
#             print("✓ Successfully imported preprocessing modules dynamically")
#         else:
#             print("Error: PreProcessing_dataset.py must be in the same directory!")
#             print("Please make sure PreProcessing_dataset.py is available before running this script.")
#             raise ImportError("PreProcessing_dataset.py not found")

# Set up GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def continue_training(model_path='/content/models/best_face_model.pth', additional_epochs=5, lr=0.00003):
    """
    Continue training a previously trained model for additional epochs.
    
    Parameters:
    - model_path: Path to the saved model checkpoint
    - additional_epochs: Number of additional epochs to train
    - lr: Learning rate for continued training (should be lower than initial training)
    """
    print("-" * 50)
    print(f"CONTINUING TRAINING FOR {additional_epochs} MORE EPOCHS")
    print("-" * 50)
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please ensure you have trained a model first using train_model.py")
        return False
    
    # Load previous training history if it exists
    history_path = os.path.join(os.path.dirname(model_path), 'training_history.pkl')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            prev_train_losses = history['train_losses']
            prev_val_losses = history['val_losses']
            prev_train_accs = history['train_accs']
            prev_val_accs = history['val_accs']
            print(f"✓ Loaded previous training history: {len(prev_train_losses)} epochs")
        except Exception as e:
            print(f"⚠️ Could not load training history: {e}")
            prev_train_losses, prev_val_losses = [], []
            prev_train_accs, prev_val_accs = [], []
    else:
        print("⚠️ No previous training history found, starting fresh history tracking")
        prev_train_losses, prev_val_losses = [], []
        prev_train_accs, prev_val_accs = [], []
    
    # Settings for training
    data_path = "/content/datasets/casia-webface"
    output_dir = "/content/models"
    batch_size = 16  # Adjusted for GPU memory
    workers = 2  # Number of data loading workers
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process dataset if needed
    print("\n📦 Step 1: Checking dataset")
    processed_path = process_casia_webface(data_path)
    if not processed_path:
        print("❌ Failed to process dataset. Exiting.")
        return False

    # Data transforms
    print("\n🔄 Step 2: Setting up data transforms")
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # More augmentation for continued training
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # More augmentation
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset and dataloaders
    print("\n📊 Step 3: Creating datasets and dataloaders")
    dataset = FaceDataset(processed_path, data_transforms)
    if len(dataset) == 0:
        print("❌ Error: No valid images found!")
        return False

    # Get the label map from the dataset
    label_map = dataset.label_map
    num_classes = len(label_map)
    print(f"✓ Training with {num_classes} identities")
    
    # Split into train and validation - use the same seed as initial training
    torch.manual_seed(42)  # Important: use same seed to keep validation set consistent
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    torch.manual_seed(torch.initial_seed())  # Reset seed

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Load the model from checkpoint
    print(f"\n🔄 Step 4: Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if the checkpoint contains a model with a classifier
        if 'num_classes' in checkpoint:
            saved_num_classes = checkpoint['num_classes']
            print(f"✓ Found model with {saved_num_classes} classes")
        else:
            # Default to the number of classes in the current dataset
            saved_num_classes = num_classes
            print("⚠️ Could not determine number of classes from checkpoint.")
        
        # Initialize the model with the same architecture
        model = InceptionResnetV1(
            pretrained='vggface2',
            classify=True,
            num_classes=saved_num_classes
        ).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully. Previous training epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # If number of classes is different, adjust the classifier
        if saved_num_classes != num_classes:
            print(f"⚠️ Number of classes changed from {saved_num_classes} to {num_classes}")
            print("⚠️ Rebuilding classifier layer to match new dataset...")
            # Replace the classifier layer with a new one
            model.classifier = nn.Linear(512, num_classes).to(device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Falling back to a new model...")
        model = InceptionResnetV1(
            pretrained='vggface2',
            classify=True,
            num_classes=num_classes
        ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # If optimizer state was saved, load it (optional)
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading optimizer state: {e}. Creating new optimizer.")
            optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )

    # Initialize tracking variables
    best_acc = checkpoint.get('accuracy', 0.0)  # Get previous best accuracy if available
    print(f"🏆 Previous best accuracy: {best_acc:.4f}")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Training loop
    print(f"\n🚀 Step 5: Continuing training for {additional_epochs} more epochs")
    for epoch in range(additional_epochs):
        print(f'\nEpoch {epoch+1}/{additional_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{additional_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.float() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu().numpy())
        print(f'📈 Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating epoch {epoch+1}/{additional_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = running_loss / len(val_dataset)
        val_epoch_acc = running_corrects.float() / len(val_dataset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.cpu().numpy())
        print(f'📊 Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}')
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_epoch_acc)

        # Save best model (only if it improves)
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            # Save as continued model
            torch.save({
                'epoch': checkpoint.get('epoch', 0) + epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_epoch_acc,
                'num_classes': num_classes
            }, os.path.join(output_dir, 'best_face_model.pth'))
            print(f"🏆 New best model saved with accuracy: {best_acc:.4f}")
        else:
            print(f"📉 No improvement. Current best: {best_acc:.4f}")

    # Save final model and label map
    print("\n💾 Step 6: Saving final model and plotting results")
    torch.save({
        'epoch': checkpoint.get('epoch', 0) + additional_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': best_acc,
        'num_classes': num_classes
    }, os.path.join(output_dir, 'final_face_model.pth'))

    # Combine previous and current training history
    combined_train_losses = prev_train_losses + train_losses
    combined_val_losses = prev_val_losses + val_losses
    combined_train_accs = prev_train_accs + train_accs
    combined_val_accs = prev_val_accs + val_accs
    
    # Save combined training history
    history = {
        'train_losses': combined_train_losses,
        'val_losses': combined_val_losses,
        'train_accs': combined_train_accs,
        'val_accs': combined_val_accs
    }
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    # Plot full training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(combined_train_losses, label='Train Loss')
    plt.plot(combined_val_losses, label='Val Loss')
    if len(prev_train_losses) > 0:
        plt.axvline(x=len(prev_train_losses)-0.5, color='r', linestyle='--', label='Continuation Point')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(combined_train_accs, label='Train Acc')
    plt.plot(combined_val_accs, label='Val Acc')
    if len(prev_train_accs) > 0:
        plt.axvline(x=len(prev_train_accs)-0.5, color='r', linestyle='--', label='Continuation Point')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_full.png'))
    
    # Also display the recent training segment
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Recent Training Loss (Continued)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Recent Training Accuracy (Continued)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_recent.png'))
    plt.show()  # Show plots in Colab

    print(f"\n✅ Training continuation complete!")
    print(f"🏆 Final best accuracy: {best_acc:.4f}")
    print(f"💾 Model saved to {os.path.join(output_dir, 'best_face_model.pth')}")
    print(f"📊 Total training history now spans {len(combined_train_losses)} epochs")
    print("\nYou can now use this model with the Face Recognition application!")
    return True

# Make it easy to run directly in a Colab cell
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Continue training a face recognition model')
    parser.add_argument('--model', type=str, default='/content/models/best_face_model.pth',
                        help='Path to the model checkpoint to continue training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of additional epochs to train')
    parser.add_argument('--lr', type=float, default=0.00003,
                        help='Learning rate for continued training')
    
    # Parse known args to handle Colab environment
    args, _ = parser.parse_known_args()
    
    # Call the function with the parsed arguments
    continue_training(args.model, args.epochs, args.lr)
