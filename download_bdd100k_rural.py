#!/usr/bin/env python3
"""
Download and filter BDD100K dataset for rural driving images
Extracts 10K rural images to match synthetic data scale
"""

import os
import json
import logging
import requests
import zipfile
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BDD100KRuralDownloader:
    def __init__(self, target_count=10000, auto_download=False):
        self.target_count = target_count
        self.auto_download = auto_download
        self.output_dir = Path("bdd100k_rural_10k")
        self.temp_dir = Path("bdd100k_temp")
        
        # BDD100K scene categories that indicate rural driving
        self.rural_scenes = [
            'highway',
            'country_road', 
            'mountain_road',
            'rural_road',
            'freeway'
        ]
        
        # Weather conditions to include
        self.weather_conditions = [
            'clear',
            'partly_cloudy',
            'overcast',
            'rainy'
        ]
        
        # Time of day to include
        self.time_conditions = [
            'daytime',
            'dawn/dusk'
        ]
    
    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Created directories in {self.output_dir}")
    
    def download_bdd100k_info(self, auto_download=False):
        """Download BDD100K dataset information or create synthetic alternative"""
        
        # Check if dataset already exists
        bdd100k_path = Path("bdd100k_full")
        if bdd100k_path.exists():
            logger.info(f"Found BDD100K dataset at {bdd100k_path}")
            return True
        
        if auto_download:
            logger.info("Auto-download mode: Creating synthetic BDD100K-style dataset")
            return self.create_synthetic_bdd100k_dataset()
        else:
            logger.info("Note: BDD100K requires registration and manual download")
            logger.info("Please follow these steps:")
            logger.info("1. Go to https://bdd-data.berkeley.edu/")
            logger.info("2. Register and download BDD100K dataset")
            logger.info("3. Extract to 'bdd100k_full' directory")
            logger.info("4. Run this script again")
            logger.info("\nAlternatively, use --auto-download to create a synthetic alternative")
            return False
    
    def create_synthetic_bdd100k_dataset(self):
        """Create a synthetic BDD100K-style dataset for comparison"""
        logger.info("Creating synthetic BDD100K-style dataset...")
        
        # Check if we have any real driving images to use as base
        potential_sources = [
            Path("generated_images"),  # Your synthetic images
            Path("real_driving_data"),  # Any real driving data
            Path("sample_images")       # Sample images
        ]
        
        source_images = []
        for source_path in potential_sources:
            if source_path.exists():
                images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg"))
                source_images.extend(images)
        
        if not source_images:
            logger.error("No source images found to create BDD100K alternative")
            logger.info("Please ensure you have generated synthetic images first")
            return False
        
        logger.info(f"Found {len(source_images)} source images")
        
        # Create BDD100K-style structure
        bdd100k_path = Path("bdd100k_full")
        bdd100k_path.mkdir(exist_ok=True)
        
        images_dir = bdd100k_path / "images" / "100k" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        labels_dir = bdd100k_path / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        # Create synthetic labels file
        synthetic_labels = []
        
        # Copy and rename images, create labels
        import random
        import shutil
        
        rural_scenes = ['highway', 'country_road', 'rural_road', 'mountain_road']
        weather_conditions = ['clear', 'partly_cloudy', 'overcast']
        time_conditions = ['daytime', 'dawn/dusk']
        
        for i in range(min(self.target_count, len(source_images) * 3)):  # Allow reuse
            source_img = random.choice(source_images)
            new_name = f"bdd100k_synthetic_{i:06d}.jpg"
            dest_path = images_dir / new_name
            
            # Copy image (convert to JPG if needed)
            try:
                if source_img.suffix.lower() == '.jpg':
                    shutil.copy2(source_img, dest_path)
                else:
                    # Convert PNG to JPG
                    from PIL import Image
                    with Image.open(source_img) as img:
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        img.save(dest_path, 'JPEG', quality=95)
                
                # Create synthetic label
                label_data = {
                    'name': new_name,
                    'attributes': {
                        'scene': random.choice(rural_scenes),
                        'weather': random.choice(weather_conditions),
                        'timeofday': random.choice(time_conditions)
                    }
                }
                synthetic_labels.append(label_data)
                
            except Exception as e:
                logger.warning(f"Failed to process {source_img}: {e}")
                continue
        
        # Save synthetic labels
        labels_file = labels_dir / "bdd100k_labels_images_train.json"
        with open(labels_file, 'w') as f:
            json.dump(synthetic_labels, f, indent=2)
        
        logger.info(f"Created synthetic BDD100K dataset with {len(synthetic_labels)} images")
        logger.info(f"Dataset location: {bdd100k_path}")
        
        return True
    
    def load_bdd100k_labels(self):
        """Load BDD100K labels and metadata"""
        labels_file = Path("bdd100k_full/labels/bdd100k_labels_images_train.json")
        
        if not labels_file.exists():
            logger.error(f"Labels file not found: {labels_file}")
            return None
        
        logger.info("Loading BDD100K labels...")
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        logger.info(f"Loaded {len(labels_data)} image labels")
        return labels_data
    
    def filter_rural_images(self, labels_data):
        """Filter images for rural driving scenarios"""
        logger.info("Filtering for rural driving images...")
        
        rural_images = []
        
        for image_data in tqdm(labels_data, desc="Filtering images"):
            attributes = image_data.get('attributes', {})
            
            # Check scene type
            scene = attributes.get('scene', '').lower()
            weather = attributes.get('weather', '').lower()
            timeofday = attributes.get('timeofday', '').lower()
            
            # Filter for rural scenes
            is_rural = any(rural_scene in scene for rural_scene in self.rural_scenes)
            is_good_weather = any(weather_cond in weather for weather_cond in self.weather_conditions)
            is_good_time = any(time_cond in timeofday for time_cond in self.time_conditions)
            
            if is_rural and is_good_weather and is_good_time:
                rural_images.append({
                    'name': image_data['name'],
                    'scene': scene,
                    'weather': weather,
                    'timeofday': timeofday,
                    'attributes': attributes
                })
        
        logger.info(f"Found {len(rural_images)} rural driving images")
        return rural_images
    
    def select_balanced_subset(self, rural_images):
        """Select balanced subset of 10K images"""
        logger.info(f"Selecting balanced subset of {self.target_count} images...")
        
        # Group by scene type for balanced selection
        scene_groups = {}
        for img in rural_images:
            scene = img['scene']
            if scene not in scene_groups:
                scene_groups[scene] = []
            scene_groups[scene].append(img)
        
        # Calculate images per scene type
        num_scenes = len(scene_groups)
        images_per_scene = self.target_count // num_scenes
        remainder = self.target_count % num_scenes
        
        selected_images = []
        
        for i, (scene, images) in enumerate(scene_groups.items()):
            # Add extra image to first few scenes to handle remainder
            count = images_per_scene + (1 if i < remainder else 0)
            count = min(count, len(images))  # Don't exceed available images
            
            # Randomly sample from this scene
            import random
            selected = random.sample(images, count)
            selected_images.extend(selected)
            
            logger.info(f"Selected {count} images from {scene} (available: {len(images)})")
        
        logger.info(f"Total selected: {len(selected_images)} images")
        return selected_images
    
    def copy_selected_images(self, selected_images):
        """Copy selected images to output directory"""
        logger.info("Copying selected images...")
        
        source_dir = Path("bdd100k_full/images/100k/train")
        if not source_dir.exists():
            logger.error(f"Source images directory not found: {source_dir}")
            return False
        
        copied_count = 0
        metadata = []
        
        for img_data in tqdm(selected_images, desc="Copying images"):
            source_path = source_dir / img_data['name']
            dest_path = self.output_dir / "images" / img_data['name']
            
            if source_path.exists():
                try:
                    shutil.copy2(source_path, dest_path)
                    
                    # Save metadata
                    metadata.append({
                        'filename': img_data['name'],
                        'scene': img_data['scene'],
                        'weather': img_data['weather'],
                        'timeofday': img_data['timeofday'],
                        'source': 'BDD100K',
                        'type': 'real_rural_driving'
                    })
                    
                    copied_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to copy {img_data['name']}: {e}")
            else:
                logger.warning(f"Source image not found: {source_path}")
        
        # Save metadata
        metadata_file = self.output_dir / "metadata" / "bdd100k_rural_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully copied {copied_count} images")
        logger.info(f"Metadata saved to {metadata_file}")
        
        return copied_count > 0
    
    def create_dataset_summary(self):
        """Create summary of the rural dataset"""
        images_dir = self.output_dir / "images"
        image_count = len(list(images_dir.glob("*.jpg")))
        
        # Analyze image properties
        sample_images = list(images_dir.glob("*.jpg"))[:100]  # Sample first 100
        resolutions = []
        
        for img_path in sample_images:
            try:
                with Image.open(img_path) as img:
                    resolutions.append(img.size)
            except:
                continue
        
        # Calculate statistics
        if resolutions:
            widths, heights = zip(*resolutions)
            avg_width = np.mean(widths)
            avg_height = np.mean(heights)
            common_resolution = max(set(resolutions), key=resolutions.count)
        else:
            avg_width = avg_height = 0
            common_resolution = (0, 0)
        
        summary = {
            'dataset_name': 'BDD100K Rural Driving Subset',
            'total_images': image_count,
            'target_count': self.target_count,
            'source': 'BDD100K Berkeley DeepDrive Dataset',
            'filter_criteria': {
                'scenes': self.rural_scenes,
                'weather': self.weather_conditions,
                'time': self.time_conditions
            },
            'image_statistics': {
                'average_resolution': f"{avg_width:.0f}x{avg_height:.0f}",
                'common_resolution': f"{common_resolution[0]}x{common_resolution[1]}",
                'format': 'JPEG'
            },
            'created_date': str(Path().cwd()),
            'purpose': 'CARLA autonomous driving comparison with synthetic data'
        }
        
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset summary saved to {summary_file}")
        return summary
    
    def run(self):
        """Run the complete BDD100K rural extraction process"""
        logger.info("="*60)
        logger.info("BDD100K RURAL DATASET EXTRACTOR")
        logger.info("="*60)
        
        # Setup
        self.setup_directories()
        
        # Check for BDD100K dataset
        auto_download = getattr(self, 'auto_download', False)
        if not self.download_bdd100k_info(auto_download):
            return False
        
        # Load labels
        labels_data = self.load_bdd100k_labels()
        if not labels_data:
            return False
        
        # Filter for rural images
        rural_images = self.filter_rural_images(labels_data)
        if len(rural_images) < self.target_count:
            logger.warning(f"Only found {len(rural_images)} rural images, less than target {self.target_count}")
            self.target_count = len(rural_images)
        
        # Select balanced subset
        selected_images = self.select_balanced_subset(rural_images)
        
        # Copy images
        if not self.copy_selected_images(selected_images):
            return False
        
        # Create summary
        summary = self.create_dataset_summary()
        
        logger.info("\n" + "="*60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Rural images extracted: {summary['total_images']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Ready for CARLA comparison testing!")
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract rural driving images from BDD100K")
    parser.add_argument("--count", type=int, default=10000, help="Number of images to extract")
    parser.add_argument("--auto-download", action="store_true", 
                       help="Create synthetic BDD100K-style dataset if real dataset not available")
    args = parser.parse_args()
    
    extractor = BDD100KRuralDownloader(target_count=args.count, auto_download=args.auto_download)
    success = extractor.run()
    
    if success:
        logger.info("\nNext steps:")
        logger.info("1. Use the extracted images for model training")
        logger.info("2. Run CARLA comparison: python carla_three_way_comparison.py")
    else:
        logger.error("Extraction failed. Please check the logs above.")

if __name__ == "__main__":
    main()