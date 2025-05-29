import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import pickle

# Set up GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom dataset class (unchanged)
class FaceDataset(Dataset):
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
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            image = Image.new('RGB', (160, 160), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

# Modified process_casia_webface function
def process_casia_webface(dataset_dir, output_dir='/content/datasets/processed_casia'):
    record_file = os.path.join(dataset_dir, 'train.rec')
    if os.path.exists(record_file):
        print("Found RecordIO format, converting to images...")
        try:
            import numpy as np
            np.bool = bool  # Patch to fix deprecated np.bool issue
            import mxnet as mx
            idx_file = os.path.join(dataset_dir, 'train.idx')
            if not os.path.exists(idx_file):
                print(f"Error: Index file {idx_file} not found")
                return None

            os.makedirs(output_dir, exist_ok=True)
            record = mx.recordio.MXIndexedRecordIO(idx_file, record_file, 'r')
            keys = list(record.keys)

            # Initialize counters
            success_count = 0
            failed_count = 0

            # Process each image with error handling
            for i in tqdm(range(len(keys))):
                key = keys[i]
                try:
                    item = record.read_idx(key)
                    header, img = mx.recordio.unpack_img(item)  # This is where the error occurs
                    label = int(header.label)
                    identity_dir = os.path.join(output_dir, str(label))
                    os.makedirs(identity_dir, exist_ok=True)
                    img_path = os.path.join(identity_dir, f"{key}.jpg")
                    if cv2.imwrite(img_path, img):  # Check if write succeeds
                        success_count += 1
                    else:
                        print(f"Failed to write image {key}")
                        failed_count += 1
                except Exception as e:
                    print(f"Error processing image {key}: {e}")
                    failed_count += 1

            # Print summary
            print(f"Conversion complete. {success_count} images saved to {output_dir}, {failed_count} images skipped due to errors.")
            return output_dir
        except Exception as e:
            print(f"Error initializing RecordIO processing: {e}")
            return None
    else:
        print("No RecordIO files found. Assuming images are already extracted.")
        return dataset_dir

# Training function (adjusted epochs)
def train_model():
    # Hardcoded arguments for Colab
    data_path = "/content/datasets/casia-webface"
    output_dir = "/content/models"
    batch_size = 16  # Reduced for Colab
    epochs = 10  # Reduced from 20 to 10 as per user's request
    lr = 0.0001
    workers = 2  # Reduced for Colab

    # Process dataset
    processed_path = process_casia_webface(data_path)
    if not processed_path:
        print("Failed to process dataset. Exiting.")
        return

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset and dataloaders
    dataset = FaceDataset(processed_path, data_transforms)
    if len(dataset) == 0:
        print("Error: No valid images found!")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Model setup
    num_classes = len(dataset.label_map)
    print(f"Training with {num_classes} identities")
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    os.makedirs(output_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.float() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu().numpy())
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating epoch {epoch+1}/{epochs}"):
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
        print(f'Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}')

        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_epoch_acc,
                'num_classes': num_classes
            }, os.path.join(output_dir, 'best_face_model.pth'))
            print(f"Saved best model with accuracy: {best_acc:.4f}")

    # Save final model and label map
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': best_acc,
        'num_classes': num_classes
    }, os.path.join(output_dir, 'final_face_model.pth'))

    with open(os.path.join(output_dir, 'label_map.pkl'), 'wb') as f:
        pickle.dump(dataset.label_map, f)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))

    print(f"Training complete. Final model saved at {os.path.join(output_dir, 'final_face_model.pth')}")
    print(f"Best accuracy: {best_acc:.4f}")

# Run training
train_model()