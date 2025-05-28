# FaceSync: Deep Learning Face Recognition System

FaceSync is a comprehensive face recognition system built with PyTorch that allows you to train your own face recognition model from scratch and use it for face verification and authentication. The system leverages deep learning techniques and provides both Google Colab and local execution environments.

![FaceSync Logo](https://img.shields.io/badge/FaceSync-Face%20Recognition-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)

## 📋 System Overview

FaceSync leverages the following components:
1. **Dataset Download**: Automated download of face recognition datasets
2. **Preprocessing**: Extracting and preparing face images for training
3. **Model Training**: Fine-tuning a deep neural network on your dataset
4. **Training Extension**: Continuing training of existing models for better performance
5. **Face Recognition**: Real-time face verification and authentication

## 🔍 Requirements

The complete list of requirements is in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## 🚀 Google Colab Setup

Follow these steps to run the entire FaceSync system in Google Colab:

### Step 1: Upload the Scripts to Google Colab

Create a new Colab notebook and upload all Python files to the notebook environment:
- `download_dataset.py`
- `PreProcessing_dataset.py`
- `train_model.py`
- `continue_training.py` (optional, for extending training)
- `faceid_app.py`
- `kaggle.json` (for dataset download)

### Step 2: Set Up Cells for Each Component

#### Cell 1: Install Dependencies

```python
!pip install torch torchvision facenet-pytorch opencv-python tqdm matplotlib mtcnn kaggle mxnet pillow
```

#### Cell 2: Download Dataset

```python
# Upload kaggle.json
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json file here

# Execute the dataset downloader
!python download_dataset.py --dataset casia-webface
```

#### Cell 3: Preprocess the Dataset

```python
# Import and run the preprocessing
from PreProcessing_dataset import process_casia_webface
processed_path = process_casia_webface('/content/datasets/casia-webface')
print(f"Dataset preprocessed at: {processed_path}")
```

#### Cell 4: Train the Model

```python
# Import the preprocessing module to make it available for training
from PreProcessing_dataset import FaceDataset, process_casia_webface

# Run the training process (initial training)
!python train_model.py
```

#### Cell 5 (Optional): Continue Training to Improve Model

```python
# After initial training completes, you can continue training for more epochs
!python continue_training.py --epochs 5 --lr 0.00003
```

#### Cell 6: Run Face Recognition Application

```python
# Run the Face ID application in Colab mode
!python faceid_app.py
```

## 💻 Local Execution

To run FaceSync on your local machine after training on Colab:

1. First download the trained model from Colab:
   - In Colab, run the Face ID app and use option 5 to download the model files
   - Save them to your local `models` directory

2. Then run the face recognition app locally:
   ```bash
   python faceid_app.py
   ```

## 📊 System Components in Detail

### 1. Dataset Download (`download_dataset.py`)

This script downloads the CASIA-WebFace dataset (or alternative datasets) using the Kaggle API:

```python
# Example usage
python download_dataset.py --dataset casia-webface
```

Options:
- `--dataset`: Choose from `casia-webface`, `vggface2`, or `lfw`

### 2. Dataset Preprocessing (`PreProcessing_dataset.py`)

This script prepares the raw dataset for training by:
- Converting from RecordIO format if needed
- Filtering out low-quality samples
- Creating a PyTorch dataset class

```python
# Example usage
from PreProcessing_dataset import process_casia_webface
processed_path = process_casia_webface('path/to/dataset')
```

### 3. Model Training (`train_model.py`)

This script trains the face recognition model on the preprocessed dataset:

```python
# Example usage
python train_model.py
```

Key features:
- Fine-tunes a pre-trained neural network
- Uses GPU acceleration
- Saves checkpoints during training
- Generates training history plots

### 4. Training Extension (`continue_training.py`)

This script continues training a previously trained model for additional epochs:

```python
# Example usage
python continue_training.py --model /content/models/best_face_model.pth --epochs 5
```

Key features:
- Loads a previously trained model and continues training
- Uses a lower learning rate for fine-tuning
- Applies additional data augmentation to prevent overfitting
- Maintains full training history across multiple training sessions
- Visualizes progress with comprehensive plots

### 5. Face Recognition Application (`faceid_app.py`)

This script provides a user interface for face registration and verification:

```python
# Example usage
python faceid_app.py
```

Features:
- Register new faces with multiple samples
- Verify faces in real-time
- Manage registered users
- Works in both Colab (with file uploads) and local environments (with webcam)

## 🔧 Customization

You can customize various aspects of the system:
- Change the dataset by using a different dataset option
- Adjust training hyperparameters in train_model.py
- Continue training for more epochs with continue_training.py
- Modify the verification threshold in faceid_app.py
- Train on your own dataset by organizing it in the required format

## 🛠 Troubleshooting

Common issues and solutions:
- **Kaggle API errors**: Ensure your kaggle.json file is properly set up
- **CUDA out of memory**: Reduce batch size in train_model.py
- **Face detection failures**: Ensure proper lighting and face positioning
- **Model loading errors**: Make sure the model files are in the correct directory
- **Training errors**: Try continuing training with a lower learning rate

## 🔗 Model Architecture

FaceSync uses a neural network architecture based on the Inception-ResNet design for face recognition. The model is trained to generate embeddings that place faces of the same person close together in the embedding space, while maintaining distance between different identities.

## 📈 Training Strategy

For optimal results, we recommend:
1. Initial training: 5-10 epochs with learning rate 0.0001
2. Continued training: 5-10 additional epochs with learning rate 0.00003
3. Fine-tuning: If needed, 3-5 more epochs with learning rate 0.00001

## 📜 License

This project is made available for educational and research purposes.

## 🙏 Acknowledgements

- FaceNet paper by Google
- The CASIA-WebFace dataset
- PyTorch and OpenCV communities
