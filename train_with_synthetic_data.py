#!/usr/bin/env python3
"""
Example training script showing how to use synthetic rural driving data
This demonstrates the concept - you'd adapt this to your specific model architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RuralDrivingDataset(Dataset):
    """Dataset class for rural driving images"""
    
    def __init__(self, image_dir, transform=None, use_synthetic=True):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.use_synthetic = use_synthetic
        
        # Load image paths
        self.image_paths = []
        self.labels = []
        
        if use_synthetic:
            # Load synthetic images
            synthetic_dir = Path("generated_images")
            if synthetic_dir.exists():
                for img_path in synthetic_dir.glob("*.png"):
                    self.image_paths.append(img_path)
                    # Create synthetic labels (steering angle, throttle, brake)
                    # In practice, you'd have actual labels or generate them
                    self.labels.append(self.generate_synthetic_label(img_path))
                
                logger.info(f"Loaded {len(self.image_paths)} synthetic images")
        
        # Load real data if available
        real_data_dir = Path("real_driving_data")
        if real_data_dir.exists():
            for img_path in real_data_dir.glob("*.png"):
                self.image_paths.append(img_path)
                # Load real labels from corresponding JSON files
                label_path = img_path.with_suffix('.json')
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        label_data = json.load(f)
                    self.labels.append([
                        label_data.get('steering', 0.0),
                        label_data.get('throttle', 0.5),
                        label_data.get('brake', 0.0)
                    ])
                else:
                    # Default label if no real label available
                    self.labels.append([0.0, 0.5, 0.0])
            
            logger.info(f"Loaded {len([p for p in self.image_paths if 'real_driving_data' in str(p)])} real images")
    
    def generate_synthetic_label(self, img_path):
        """Generate synthetic driving labels based on image analysis"""
        # This is a simplified example - in practice you'd use more sophisticated methods
        
        # Extract some basic features from filename or image analysis
        filename = img_path.stem
        
        # Simulate different driving scenarios based on synthetic image characteristics
        if "curve" in filename.lower() or "turn" in filename.lower():
            steering = np.random.uniform(-0.3, 0.3)  # More steering for curves
            throttle = np.random.uniform(0.3, 0.6)   # Slower for curves
        elif "straight" in filename.lower():
            steering = np.random.uniform(-0.1, 0.1)  # Minimal steering
            throttle = np.random.uniform(0.5, 0.8)   # Faster on straights
        else:
            # Default rural driving behavior
            steering = np.random.uniform(-0.2, 0.2)
            throttle = np.random.uniform(0.4, 0.7)
        
        brake = 0.0  # Minimal braking for rural roads
        
        return [steering, throttle, brake]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label

class RuralDrivingCNN(nn.Module):
    """Simple CNN for rural driving control prediction"""
    
    def __init__(self, input_channels=3, output_size=3):
        super(RuralDrivingCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        # Calculate the size of flattened features
        # For 1024x1024 input: ((((1024-8)/4+1-4)/2+1-3)/2+1-3)/2+1 = 30
        self.feature_size = 256 * 30 * 30  # Approximate
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size)  # [steering, throttle, brake]
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        
        # Apply appropriate activations
        steering = torch.tanh(x[:, 0:1])      # Steering: -1 to 1
        throttle = torch.sigmoid(x[:, 1:2])   # Throttle: 0 to 1
        brake = torch.sigmoid(x[:, 2:3])      # Brake: 0 to 1
        
        return torch.cat([steering, throttle, brake], dim=1)

def train_model(use_synthetic_data=True, epochs=50, batch_size=16):
    """Train the rural driving model"""
    
    logger.info(f"Training model {'WITH' if use_synthetic_data else 'WITHOUT'} synthetic data")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize for faster training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = RuralDrivingDataset(
        image_dir="generated_images",
        transform=transform,
        use_synthetic=use_synthetic_data
    )
    
    if len(dataset) == 0:
        logger.error("No training data found!")
        return None
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create model
    model = RuralDrivingCNN(input_channels=3, output_size=3).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        logger.info(f'Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}')
        
        scheduler.step()
    
    # Save model
    model_name = f"rural_driving_{'with_synthetic' if use_synthetic_data else 'baseline'}.pth"
    model_path = Path("models") / model_name
    model_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'RuralDrivingCNN',
        'use_synthetic_data': use_synthetic_data,
        'training_epochs': epochs,
        'final_loss': avg_loss
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    return model, model_path

def evaluate_model(model_path, test_data_dir="test_images"):
    """Evaluate trained model"""
    logger.info(f"Evaluating model: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = RuralDrivingCNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup test data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test on some sample images
    test_dir = Path(test_data_dir)
    if not test_dir.exists():
        logger.warning(f"Test directory {test_dir} not found")
        return
    
    with torch.no_grad():
        for img_path in list(test_dir.glob("*.png"))[:5]:  # Test on first 5 images
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            prediction = model(image_tensor)
            steering, throttle, brake = prediction[0].cpu().numpy()
            
            logger.info(f"Image: {img_path.name}")
            logger.info(f"  Predicted - Steering: {steering:.3f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}")

def main():
    """Main training function"""
    logger.info("Rural Driving Model Training with Synthetic Data")
    logger.info("="*50)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Train baseline model (without synthetic data)
    logger.info("\n--- Training Baseline Model ---")
    baseline_model, baseline_path = train_model(use_synthetic_data=False, epochs=10)
    
    # Train model with synthetic data
    logger.info("\n--- Training Model with Synthetic Data ---")
    synthetic_model, synthetic_path = train_model(use_synthetic_data=True, epochs=10)
    
    # Evaluate both models
    if baseline_path:
        logger.info("\n--- Evaluating Baseline Model ---")
        evaluate_model(baseline_path)
    
    if synthetic_path:
        logger.info("\n--- Evaluating Synthetic Data Model ---")
        evaluate_model(synthetic_path)
    
    logger.info("\nTraining completed! Models saved in 'models/' directory")
    logger.info("You can now use these models in the CARLA test script")

if __name__ == "__main__":
    main()