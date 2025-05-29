# Configuration - adjust as needed
ADDITIONAL_EPOCHS = 1  # Number of additional epochs to train
MODEL_PATH = '/content/models/final_face_model.pth'  # Path to the previously trained model
DATASET_PATH = '/content/datasets/processed_casia'  # Path to the processed dataset

# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1  # Only used to recreate same architecture
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import sys
import importlib.util

# Set up GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Handle PreProcessing_dataset.py import issues
def import_preprocessing():
    """
    Import the preprocessing module dynamically to handle cases when running
    directly after dataset download and preprocessing
    """
    # First try: Direct import if the module is in the path
    try:
        # Try direct import
        from PreProcessing_dataset import FaceDataset, process_casia_webface
        print("✓ Successfully imported preprocessing modules")
        return FaceDataset, process_casia_webface
    except ImportError:
        print("Could not directly import PreProcessing_dataset. Trying alternative methods...")

    # Second try: Execute from current directory
    preprocessing_path = os.path.join(os.getcwd(), "PreProcessing_dataset.py")
    if os.path.exists(preprocessing_path):
        print(f"Found preprocessing script at: {preprocessing_path}")
        # Execute the preprocessing script directly
        with open(preprocessing_path, 'r') as f:
            preprocessing_code = f.read()
        
        # Create namespace for execution
        preprocessing_namespace = {'__file__': preprocessing_path}
        exec(preprocessing_code, preprocessing_namespace)
        
        # Extract the needed classes and functions
        FaceDataset = preprocessing_namespace.get('FaceDataset')
        process_casia_webface = preprocessing_namespace.get('process_casia_webface')
        
        if FaceDataset and process_casia_webface:
            print("✓ Successfully imported preprocessing modules from file")
            return FaceDataset, process_casia_webface

    # Third try: Define the necessary classes and functions directly here
    print("⚠️ Implementing preprocessing functions directly")
    
    # Include minimal versions of the preprocessing functions
    class FaceDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.label_map = {}

            label_idx = 0
            print("Loading dataset...")

            # Process directory structure
            print(f"Processing directory: {data_dir}")
            for identity in tqdm(sorted(os.listdir(data_dir))):
                identity_dir = os.path.join(data_dir, identity)
                if not os.path.isdir(identity_dir):
                    continue

                # Get image files
                import glob
                image_files = glob.glob(os.path.join(identity_dir, "*.jpg"))
                if len(image_files) < 5:  # Filter identities with < 5 images
                    continue

                # Assign label
                if identity not in self.label_map:
                    self.label_map[identity] = label_idx
                    label_idx += 1

                # Limit to 100 images per identity
                for img_path in image_files[:100]:
                    self.image_paths.append(img_path)
                    self.labels.append(self.label_map[identity])

            print(f"Loaded {len(self.image_paths)} images with {len(self.label_map)} identities")

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]

            try:
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                image = Image.new('RGB', (160, 160), color=(0, 0, 0))

            if self.transform:
                image = self.transform(image)

            return image, label
    
    def process_casia_webface(dataset_dir, output_dir='/content/datasets/processed_casia'):
        """Simple version that assumes dataset is already processed"""
        # Check if the output directory exists and seems to have data
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"✓ Using existing processed dataset at {output_dir}")
            return output_dir
            
        # Check if dataset_dir itself is already in the right format
        if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
            # Check if it contains subdirectories (identities)
            subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            if len(subdirs) > 0:
                print(f"✓ Dataset directory appears to already be processed: {dataset_dir}")
                return dataset_dir
                
        print("❌ Could not find processed dataset. Please run PreProcessing_dataset.py first.")
        return None
    
    return FaceDataset, process_casia_webface

def continue_training(model_path=MODEL_PATH, additional_epochs=ADDITIONAL_EPOCHS, lr=0.00003, dataset_path=DATASET_PATH):
    """
    Continue training a previously trained model for additional epochs.
    This picks up exactly where the previous training left off.
    
    Parameters:
    - model_path: Path to the saved model checkpoint from previous training
    - additional_epochs: Number of additional epochs to train
    - lr: Learning rate for continued training (should be lower than initial training)
    - dataset_path: Path to the processed dataset
    """
    print("-" * 80)
    print(f"CONTINUING TRAINING FROM EXISTING MODEL: {model_path}")
    print(f"ADDING {additional_epochs} MORE EPOCHS TO PREVIOUS TRAINING")
    print("-" * 80)
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Previous model not found at {model_path}")
        print("Please ensure you have trained a model first using train_model.py")
        return False
    
    # Import preprocessing module
    try:
        FaceDataset, process_casia_webface = import_preprocessing()
    except Exception as e:
        print(f"❌ Error importing preprocessing modules: {e}")
        print("Please make sure PreProcessing_dataset.py has been executed first")
        return False
    
    # Check if dataset path is valid or find an alternative
    if not dataset_path or not os.path.exists(dataset_path):
        # Let's look for processed dataset in common locations
        dataset_locations = [
            '/content/datasets/processed_casia',  # Default processed dataset path
            '/content/datasets/casia-webface',    # Original dataset path
            '/content/processed_casia',           # Alternative path
        ]
        
        processed_path = None
        for location in dataset_locations:
            if os.path.exists(location):
                processed_path = location
                print(f"✓ Found dataset at: {processed_path}")
                break
        
        if not processed_path:
            print("❌ Could not find dataset. Please specify the correct path.")
            data_path = input("Enter dataset path (or press Enter to exit): ")
            if not data_path:
                return False
            processed_path = data_path
    else:
        processed_path = dataset_path
        print(f"✓ Using dataset at: {processed_path}")
    
    # Output directory for saving models and results
    output_dir = os.path.dirname(model_path)
    if not output_dir:
        output_dir = "/content/models"
    
    # Load previous training history if it exists
    history_path = os.path.join(output_dir, 'training_history.pkl')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            prev_train_losses = history['train_losses']
            prev_val_losses = history['val_losses']
            prev_train_accs = history['train_accs']
            prev_val_accs = history['val_accs']
            print(f"✓ Loaded previous training history: {len(prev_train_losses)} epochs already completed")
        except Exception as e:
            print(f"⚠️ Could not load training history: {e}")
            prev_train_losses, prev_val_losses = [], []
            prev_train_accs, prev_val_accs = [], []
    else:
        print("⚠️ No previous training history found, starting fresh history tracking")
        prev_train_losses, prev_val_losses = [], []
        prev_train_accs, prev_val_accs = [], []
    
    # Settings for training
    batch_size = 16  # Adjusted for GPU memory
    workers = 2  # Number of data loading workers
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Split into train and validation - IMPORTANT: use the same seed as initial training
    torch.manual_seed(42)  # Important: use same seed to keep validation set consistent
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    torch.manual_seed(torch.initial_seed())  # Reset seed

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Load the EXACT model from previous training checkpoint
    print(f"\n🔄 Step 4: Loading previously trained model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get information about the model from checkpoint
        if 'num_classes' not in checkpoint:
            print("⚠️ Warning: Checkpoint doesn't contain num_classes information")
            saved_num_classes = num_classes
        else:
            saved_num_classes = checkpoint['num_classes']
            print(f"✓ Found model with {saved_num_classes} classes")
            
        # Check if the dataset has changed
        if saved_num_classes != num_classes:
            print(f"⚠️ Warning: Number of classes in dataset ({num_classes}) differs from model ({saved_num_classes})")
            print("⚠️ This may cause issues if class IDs have changed")
        
        # Initialize model with the SAME architecture as before
        model = InceptionResnetV1(
            pretrained='vggface2',  # This is just to initialize the architecture
            classify=True,          # We need the classification layer
            num_classes=saved_num_classes
        ).to(device)
        
        # Load EXACT weights from previous training (this is the critical part)
        print("✓ Loading weights from previous training run")
        model.load_state_dict(checkpoint['model_state_dict'])

        # This confirms we're using the previously trained model, not the pretrained one
        print("✓ Successfully loaded previously trained model weights")
        previous_epoch = checkpoint.get('epoch', 0)
        print(f"✓ Previous training completed {previous_epoch} epoch(s) according to checkpoint")

        # Check if the model is in eval mode and set to train
        if not model.training:
            print("✓ Setting model to training mode")
            model.train()
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Cannot continue training without a valid model. Exiting.")
        return False
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # If optimizer state was saved, load it (this is crucial for proper continuation)
    if 'optimizer_state_dict' in checkpoint:
        try:
            print("✓ Loading optimizer state from previous training")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Update the learning rate for continued training
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = lr
                print(f"✓ Updated learning rate: {old_lr} → {lr}")
        except Exception as e:
            print(f"⚠️ Error loading optimizer state: {e}. Creating new optimizer.")
            optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        print("⚠️ No optimizer state found in checkpoint. Creating new optimizer.")
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
    best_acc = checkpoint.get('accuracy', 0.0)  # Get previous best accuracy
    print(f"🏆 Previous best accuracy: {best_acc:.4f}")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Continue the training loop from previous epoch
    previous_epoch = checkpoint.get('epoch', 0)
    start_epoch = previous_epoch + 1
    end_epoch = start_epoch + additional_epochs
    
    print(f"\n🚀 Step 5: Previously trained up to epoch {previous_epoch}")
    print(f"🚀 Continuing training for {additional_epochs} more epochs ({start_epoch} to {end_epoch-1})")
    
    for epoch in range(start_epoch, end_epoch):
        print(f'\nEpoch {epoch} (Additional: {epoch-start_epoch+1}/{additional_epochs})')
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"Training epoch {epoch}"):
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
            for inputs, labels in tqdm(val_loader, desc=f"Validating epoch {epoch}"):
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
            # Save with the latest epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_epoch_acc,
                'num_classes': saved_num_classes
            }, os.path.join(output_dir, 'best_face_model.pth'))
            print(f"🏆 New best model saved with accuracy: {best_acc:.4f}")
        else:
            print(f"📉 No improvement. Current best: {best_acc:.4f}")

    # Save final model
    print("\n💾 Step 6: Saving final model and plotting results")
    torch.save({
        'epoch': end_epoch - 1,  # Last epoch completed
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': best_acc,
        'num_classes': saved_num_classes
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
    
    print(f"✓ Saved training history covering {len(combined_train_losses)} total epochs")

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
    plt.title(f'Recent Training Loss (Epochs {start_epoch}-{end_epoch-1})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Recent Training Accuracy (Epochs {start_epoch}-{end_epoch-1})')
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
    # Check if we need to install any missing dependencies
    try:
        import facenet_pytorch
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "facenet-pytorch", "tqdm", "matplotlib", "Pillow"])
    
    import argparse
    parser = argparse.ArgumentParser(description='Continue training a face recognition model')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='Path to the model checkpoint to continue training')
    parser.add_argument('--epochs', type=int, default=ADDITIONAL_EPOCHS,
                        help='Number of additional epochs to train')
    parser.add_argument('--lr', type=float, default=0.00003,
                        help='Learning rate for continued training')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH,
                        help='Path to the processed dataset')
    
    # Parse known args to handle Colab environment
    args, _ = parser.parse_known_args()
    
    # Call the function with the parsed arguments
    continue_training(args.model, args.epochs, args.lr, args.dataset)
