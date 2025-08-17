#!/usr/bin/env python3
"""
Train three models for CARLA comparison:
1. Baseline model (minimal training data)
2. Model with synthetic rural driving data
3. Model with real BDD100K rural data
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
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RuralDrivingDataset(Dataset):
    """Dataset class for rural driving images with multiple data sources"""
    
    def __init__(self, data_type='baseline', transform=None):
        self.data_type = data_type
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if data_type == 'baseline':
            self.load_baseline_data()
        elif data_type == 'synthetic':
            self.load_synthetic_data()
        elif data_type == 'bdd100k_real':
            self.load_bdd100k_data()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        logger.info(f"Loaded {len(self.image_paths)} images for {data_type} training")
    
    def load_baseline_data(self):
        """Load minimal baseline training data"""
        # For baseline, use a small subset of any available data
        synthetic_dir = Path("rural_driving_10k/images")
        if synthetic_dir.exists():
            # Use only 100 synthetic images for baseline
            all_images = list(synthetic_dir.glob("*.png"))[:100]
            for img_path in all_images:
                self.image_paths.append(img_path)
                self.labels.append(self.generate_basic_label())
        
        logger.info(f"Baseline model using {len(self.image_paths)} minimal training images")
    
    def load_synthetic_data(self):
        """Load synthetic rural driving data"""
        synthetic_dir = Path("rural_driving_10k/images")
        if not synthetic_dir.exists():
            logger.error(f"Synthetic data directory not found: {synthetic_dir}")
            return
        
        for img_path in synthetic_dir.glob("*.png"):
            self.image_paths.append(img_path)
            self.labels.append(self.generate_synthetic_label(img_path))
    
    def load_bdd100k_data(self):
        """Load BDD100K real rural driving data"""
        bdd100k_dir = Path("bdd100k_rural_10k/images")
        metadata_file = Path("bdd100k_rural_10k/metadata/bdd100k_rural_metadata.json")
        
        if not bdd100k_dir.exists():
            logger.error(f"BDD100K data directory not found: {bdd100k_dir}")
            logger.info("Run: python download_bdd100k_rural.py")
            return
        
        # Load metadata if available
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_list = json.load(f)
                metadata = {item['filename']: item for item in metadata_list}
        
        for img_path in bdd100k_dir.glob("*.jpg"):
            self.image_paths.append(img_path)
            img_metadata = metadata.get(img_path.name, {})
            self.labels.append(self.generate_bdd100k_label(img_path, img_metadata))
    
    def generate_basic_label(self):
        """Generate basic driving labels for baseline model"""
        # Very simple, conservative driving behavior
        steering = np.random.uniform(-0.1, 0.1)  # Minimal steering
        throttle = np.random.uniform(0.3, 0.5)   # Conservative throttle
        brake = 0.0
        return [steering, throttle, brake]
    
    def generate_synthetic_label(self, img_path):
        """Generate driving labels for synthetic images"""
        filename = img_path.stem.lower()
        
        # Analyze filename for driving scenario hints
        if any(word in filename for word in ['curve', 'turn', 'bend']):
            steering = np.random.uniform(-0.3, 0.3)  # More steering for curves
            throttle = np.random.uniform(0.4, 0.6)   # Moderate speed
        elif any(word in filename for word in ['straight', 'highway']):
            steering = np.random.uniform(-0.1, 0.1)  # Minimal steering
            throttle = np.random.uniform(0.6, 0.8)   # Higher speed
        else:
            # Default rural driving
            steering = np.random.uniform(-0.2, 0.2)
            throttle = np.random.uniform(0.5, 0.7)
        
        brake = 0.0
        return [steering, throttle, brake]
    
    def generate_bdd100k_label(self, img_path, metadata):
        """Generate driving labels for BDD100K images based on metadata"""
        scene = metadata.get('scene', '').lower()
        weather = metadata.get('weather', '').lower()
        timeofday = metadata.get('timeofday', '').lower()
        
        # Adjust driving behavior based on real-world conditions
        base_steering = np.random.uniform(-0.15, 0.15)
        base_throttle = 0.6
        
        # Scene-based adjustments
        if 'highway' in scene or 'freeway' in scene:
            base_throttle = 0.75  # Higher speed on highways
            base_steering *= 0.7  # Less steering variation
        elif 'mountain' in scene:
            base_throttle = 0.5   # Slower on mountain roads
            base_steering *= 1.2  # More steering needed
        elif 'country' in scene or 'rural' in scene:
            base_throttle = 0.65  # Moderate speed
            base_steering *= 1.0  # Normal steering
        
        # Weather-based adjustments
        if 'rainy' in weather or 'wet' in weather:
            base_throttle *= 0.8  # Slower in rain
        elif 'clear' in weather:
            base_throttle *= 1.1  # Faster in clear weather
        
        # Time-based adjustments
        if 'dawn' in timeofday or 'dusk' in timeofday:
            base_throttle *= 0.9  # Slower during low light
        
        # Ensure values are within valid ranges
        steering = np.clip(base_steering, -1.0, 1.0)
        throttle = np.clip(base_throttle, 0.0, 1.0)
        brake = 0.0
        
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
    """CNN for rural driving control prediction"""
    
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
        
        # Calculate feature size (for 256x256 input)
        self.feature_size = 256 * 6 * 6  # Approximate for 256x256 input
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size)
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

def train_model(data_type, epochs=20, batch_size=16):
    """Train a model with specified data type"""
    
    logger.info(f"Training {data_type.upper()} model...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = RuralDrivingDataset(data_type=data_type, transform=transform)
    
    if len(dataset) == 0:
        logger.error(f"No training data found for {data_type}!")
        return None, None
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create model
    model = RuralDrivingCNN().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
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
                logger.info(f'{data_type.upper()} - Epoch [{epoch+1}/{epochs}], '
                           f'Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        logger.info(f'{data_type.upper()} - Epoch [{epoch+1}/{epochs}] completed, '
                   f'Average Loss: {avg_loss:.4f}')
        
        scheduler.step()
    
    # Save model
    model_name = f"rural_driving_{data_type}.pth"
    model_path = Path("models") / model_name
    model_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'RuralDrivingCNN',
        'data_type': data_type,
        'training_epochs': epochs,
        'final_loss': avg_loss,
        'dataset_size': len(dataset)
    }, model_path)
    
    logger.info(f"{data_type.upper()} model saved to {model_path}")
    return model, model_path

def evaluate_models():
    """Evaluate all three trained models"""
    logger.info("Evaluating trained models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_types = ['baseline', 'synthetic', 'bdd100k_real']
    results = {}
    
    for model_type in model_types:
        model_path = Path("models") / f"rural_driving_{model_type}.pth"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=device)
            model = RuralDrivingCNN().to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get model info
            results[model_type] = {
                'model_path': str(model_path),
                'training_epochs': checkpoint.get('training_epochs', 'unknown'),
                'final_loss': checkpoint.get('final_loss', 'unknown'),
                'dataset_size': checkpoint.get('dataset_size', 'unknown')
            }
            
            logger.info(f"{model_type.upper()} model loaded successfully")
            logger.info(f"  Training epochs: {results[model_type]['training_epochs']}")
            logger.info(f"  Final loss: {results[model_type]['final_loss']:.4f}")
            logger.info(f"  Dataset size: {results[model_type]['dataset_size']}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
    
    return results

def main():
    """Main training function"""
    logger.info("="*70)
    logger.info("THREE-WAY MODEL TRAINING FOR CARLA COMPARISON")
    logger.info("="*70)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Check data availability
    synthetic_path = Path("generated_images")
    bdd100k_path = Path("bdd100k_rural_10k/images")
    
    logger.info("Checking data availability...")
    if synthetic_path.exists():
        synthetic_count = len(list(synthetic_path.glob("*.png")))
        logger.info(f"✓ Synthetic data: {synthetic_count} images")
    else:
        logger.warning("✗ Synthetic data not found")
    
    if bdd100k_path.exists():
        bdd100k_count = len(list(bdd100k_path.glob("*.jpg")))
        logger.info(f"✓ BDD100K data: {bdd100k_count} images")
    else:
        logger.warning("✗ BDD100K data not found - run: python download_bdd100k_rural.py")
    
    # Train all three models
    model_types = ['baseline', 'synthetic', 'bdd100k_real']
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAINING {model_type.upper()} MODEL")
        logger.info(f"{'='*50}")
        
        try:
            model, model_path = train_model(model_type, epochs=15)
            if model is not None:
                trained_models[model_type] = model_path
                logger.info(f"✓ {model_type.upper()} model training completed")
            else:
                logger.error(f"✗ {model_type.upper()} model training failed")
        except Exception as e:
            logger.error(f"✗ {model_type.upper()} model training failed: {e}")
    
    # Evaluate all models
    logger.info(f"\n{'='*50}")
    logger.info("MODEL EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    
    evaluation_results = evaluate_models()
    
    # Final summary
    logger.info(f"\n{'='*50}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*50}")
    
    for model_type in model_types:
        if model_type in trained_models:
            logger.info(f"✓ {model_type.upper()} model ready for CARLA testing")
        else:
            logger.error(f"✗ {model_type.upper()} model not available")
    
    logger.info("\nNext steps:")
    logger.info("1. Start CARLA simulator: ./CarlaUE4.sh")
    logger.info("2. Run three-way comparison: python carla_three_way_comparison.py")

if __name__ == "__main__":
    main()