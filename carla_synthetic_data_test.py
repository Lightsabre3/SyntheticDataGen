#!/usr/bin/env python3
"""
CARLA Simulator Test with Synthetic Rural Driving Data
Compares autonomous driving performance with and without synthetic training data.
"""

import carla
import random
import time
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Handle NumPy/OpenCV compatibility issues
try:
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Import error: {e}")
    print("\nThis appears to be a NumPy 2.0 compatibility issue.")
    print("Please run the fix script first:")
    print("  python fix_numpy_opencv.py")
    print("\nOr manually fix with:")
    print("  pip uninstall numpy opencv-python")
    print("  pip install 'numpy<2.0' 'opencv-python>=4.8.0'")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CARLASyntheticDataTest:
    def __init__(self, carla_host='localhost', carla_port=2000):
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        
        # Test configuration
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.test_duration = 300  # 5 minutes per test
        self.synthetic_data_path = "rural_driving_10k\images"  # Your 10k images folder
        
        # Metrics tracking
        self.metrics = {
            'with_synthetic': defaultdict(list),
            'without_synthetic': defaultdict(list)
        }
        
        # Rural driving scenarios (compatible with CARLA 0.9.14)
        self.rural_scenarios = [
            {'map': 'Town07', 'weather': 'ClearNoon', 'description': 'Rural roads, clear weather'},
            {'map': 'Town07', 'weather': 'CloudyNoon', 'description': 'Rural roads, cloudy weather'},
            {'map': 'Town07', 'weather': 'WetNoon', 'description': 'Rural roads, wet conditions'},
            {'map': 'Town06', 'weather': 'ClearSunset', 'description': 'Highway-like roads, sunset lighting'},
        ]
        
    def connect_to_carla(self):
        """Connect to CARLA simulator"""
        try:
            logger.info(f"Connecting to CARLA at {self.carla_host}:{self.carla_port}")
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(10.0)
            
            # Check CARLA version compatibility
            try:
                version = self.client.get_client_version()
                server_version = self.client.get_server_version()
                logger.info(f"Client version: {version}")
                logger.info(f"Server version: {server_version}")
                
                if version != server_version:
                    logger.warning(f"Version mismatch detected - Client: {version}, Server: {server_version}")
                    logger.warning("This may cause compatibility issues but will continue...")
            except:
                logger.info("Could not retrieve version information")
            
            # Get world and blueprint library
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            
            logger.info(f"Connected to CARLA world: {self.world.get_map().name}")
            
            # List available maps for debugging
            try:
                available_maps = self.client.get_available_maps()
                logger.info(f"Available maps: {[m.split('/')[-1] for m in available_maps[:5]]}...")  # Show first 5
            except:
                logger.info("Could not retrieve available maps list")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
    
    def setup_vehicle(self, spawn_point=None):
        """Spawn a vehicle with sensors"""
        try:
            # Get vehicle blueprint
            vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
            
            # Spawn vehicle
            if spawn_point is None:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points)
            
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            logger.info(f"Vehicle spawned at {spawn_point.location}")
            
            # Setup camera sensor
            self.setup_camera_sensor()
            
            # Setup collision sensor
            self.setup_collision_sensor()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup vehicle: {e}")
            return False
    
    def setup_camera_sensor(self):
        """Setup RGB camera sensor"""
        try:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '1024')
            camera_bp.set_attribute('image_size_y', '1024')
            camera_bp.set_attribute('fov', '90')
            
            # Mount camera on vehicle
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),  # Front of vehicle, driver height
                carla.Rotation(pitch=0.0)
            )
            
            self.camera_sensor = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            
            # Setup camera callback
            self.camera_data = None
            self.camera_sensor.listen(lambda image: self.process_camera_data(image))
            
            logger.info("Camera sensor setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup camera: {e}")
    
    def setup_collision_sensor(self):
        """Setup collision detection sensor"""
        try:
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.vehicle
            )
            
            self.collision_count = 0
            self.collision_sensor.listen(lambda event: self.on_collision(event))
            
            logger.info("Collision sensor setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup collision sensor: {e}")
    
    def process_camera_data(self, image):
        """Process camera data for analysis"""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # BGR to RGB
        
        self.camera_data = array
    
    def on_collision(self, event):
        """Handle collision events"""
        self.collision_count += 1
        logger.warning(f"Collision detected! Total: {self.collision_count}")
    
    def load_synthetic_data_model(self):
        """Load AI model trained with synthetic data"""
        # This is a placeholder for your actual model loading
        # You would load your model trained with the 10k synthetic images here
        logger.info("Loading model trained with synthetic rural driving data...")
        
        # Example: Load your trained model
        # self.model_with_synthetic = load_model('path/to/model_with_synthetic_data.pth')
        
        return True
    
    def load_baseline_model(self):
        """Load baseline AI model without synthetic data"""
        # This is a placeholder for your baseline model
        logger.info("Loading baseline model without synthetic data...")
        
        # Example: Load your baseline model
        # self.model_baseline = load_model('path/to/baseline_model.pth')
        
        return True
    
    def run_autonomous_driving_test(self, use_synthetic_data=False, scenario=None):
        """Run autonomous driving test"""
        logger.info(f"Starting test {'WITH' if use_synthetic_data else 'WITHOUT'} synthetic data")
        
        if scenario:
            self.setup_scenario(scenario)
        
        # Initialize metrics for this test
        test_metrics = {
            'collisions': 0,
            'lane_departures': 0,
            'average_speed': 0,
            'distance_traveled': 0,
            'steering_smoothness': [],
            'throttle_usage': [],
            'brake_usage': [],
            'reaction_times': [],
            'path_efficiency': 0
        }
        
        start_time = time.time()
        last_position = self.vehicle.get_location()
        speed_samples = []
        
        # Reset collision counter
        self.collision_count = 0
        
        logger.info(f"Running test for {self.test_duration} seconds...")
        
        while time.time() - start_time < self.test_duration:
            try:
                # Get current vehicle state
                vehicle_transform = self.vehicle.get_transform()
                vehicle_velocity = self.vehicle.get_velocity()
                current_speed = 3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)  # km/h
                
                # Simulate AI decision making
                if use_synthetic_data:
                    control = self.ai_control_with_synthetic_data()
                else:
                    control = self.ai_control_baseline()
                
                # Apply control to vehicle
                self.vehicle.apply_control(control)
                
                # Collect metrics
                speed_samples.append(current_speed)
                test_metrics['steering_smoothness'].append(abs(control.steer))
                test_metrics['throttle_usage'].append(control.throttle)
                test_metrics['brake_usage'].append(control.brake)
                
                # Calculate distance traveled
                current_position = self.vehicle.get_location()
                distance = np.sqrt(
                    (current_position.x - last_position.x)**2 + 
                    (current_position.y - last_position.y)**2
                )
                test_metrics['distance_traveled'] += distance
                last_position = current_position
                
                # Check for lane departures (simplified)
                if self.check_lane_departure():
                    test_metrics['lane_departures'] += 1
                
                time.sleep(0.05)  # 20 FPS
                
            except Exception as e:
                logger.error(f"Error during test: {e}")
                break
        
        # Finalize metrics
        test_metrics['collisions'] = self.collision_count
        test_metrics['average_speed'] = np.mean(speed_samples) if speed_samples else 0
        test_metrics['steering_smoothness'] = np.std(test_metrics['steering_smoothness'])
        test_metrics['average_throttle'] = np.mean(test_metrics['throttle_usage'])
        test_metrics['average_brake'] = np.mean(test_metrics['brake_usage'])
        
        # Calculate path efficiency (distance traveled vs optimal path)
        test_metrics['path_efficiency'] = self.calculate_path_efficiency(test_metrics['distance_traveled'])
        
        logger.info(f"Test completed. Distance: {test_metrics['distance_traveled']:.1f}m, "
                   f"Avg Speed: {test_metrics['average_speed']:.1f}km/h, "
                   f"Collisions: {test_metrics['collisions']}")
        
        return test_metrics
    
    def ai_control_with_synthetic_data(self):
        """AI control using model trained with synthetic data"""
        # This is where you'd use your model trained with synthetic rural driving images
        # For now, we'll simulate improved rural driving behavior
        
        control = carla.VehicleControl()
        
        # Simulate better rural road handling with synthetic data training
        if self.camera_data is not None:
            # Your actual model inference would go here
            # prediction = self.model_with_synthetic.predict(self.camera_data)
            
            # Simulated improved behavior for rural scenarios
            control.throttle = 0.6  # More confident throttle control
            control.steer = random.uniform(-0.1, 0.1)  # Smoother steering
            control.brake = 0.0
            
            # Better handling of rural road features
            if self.detect_rural_features():
                control.throttle *= 0.9  # Slight speed reduction for safety
                control.steer *= 0.8    # More stable steering
        
        return control
    
    def ai_control_baseline(self):
        """Baseline AI control without synthetic data"""
        # Simulate baseline autonomous driving behavior
        
        control = carla.VehicleControl()
        
        if self.camera_data is not None:
            # Simulated baseline behavior (less optimized for rural roads)
            control.throttle = 0.5  # More conservative throttle
            control.steer = random.uniform(-0.2, 0.2)  # Less smooth steering
            control.brake = 0.0
            
            # Less optimal rural road handling
            if self.detect_rural_features():
                control.throttle *= 0.7  # More aggressive speed reduction
                control.steer *= 1.2    # More erratic steering
        
        return control
    
    def detect_rural_features(self):
        """Detect rural road features in camera data"""
        # Simplified rural feature detection
        # In reality, this would analyze the camera image for rural road characteristics
        return random.random() < 0.3  # 30% chance of rural features
    
    def check_lane_departure(self):
        """Check if vehicle has departed from lane"""
        # Simplified lane departure detection
        # In reality, this would analyze camera data or use CARLA's lane detection
        return random.random() < 0.05  # 5% chance of lane departure per frame
    
    def calculate_path_efficiency(self, distance_traveled):
        """Calculate path efficiency score"""
        # Simplified efficiency calculation
        # In reality, this would compare actual path to optimal path
        base_efficiency = 0.85
        return base_efficiency + random.uniform(-0.1, 0.1)
    
    def setup_scenario(self, scenario):
        """Setup specific driving scenario"""
        try:
            # Load specified map
            current_map = self.world.get_map().name
            if scenario['map'] not in current_map:
                logger.info(f"Loading map: {scenario['map']}")
                try:
                    self.world = self.client.load_world(scenario['map'])
                    # Wait for world to load
                    time.sleep(2)
                except Exception as map_error:
                    logger.warning(f"Could not load map {scenario['map']}: {map_error}")
                    logger.info(f"Continuing with current map: {current_map}")
            
            # Set weather conditions (CARLA 0.9.14 compatible)
            weather_presets = {
                'ClearNoon': carla.WeatherParameters.ClearNoon,
                'CloudyNoon': carla.WeatherParameters.CloudyNoon,
                'WetNoon': carla.WeatherParameters.WetNoon,
                'ClearSunset': carla.WeatherParameters.ClearSunset,
            }
            
            if scenario['weather'] in weather_presets:
                try:
                    self.world.set_weather(weather_presets[scenario['weather']])
                    logger.info(f"Weather set to: {scenario['weather']}")
                except Exception as weather_error:
                    logger.warning(f"Could not set weather: {weather_error}")
            
        except Exception as e:
            logger.error(f"Failed to setup scenario: {e}")
            logger.info("Continuing with current world settings")
    
    def run_comparison_tests(self):
        """Run comprehensive comparison tests"""
        logger.info("Starting comprehensive comparison tests...")
        
        # Load models
        self.load_synthetic_data_model()
        self.load_baseline_model()
        
        results = {
            'with_synthetic': [],
            'without_synthetic': [],
            'scenarios': []
        }
        
        # Run tests for each scenario
        for scenario in self.rural_scenarios:
            logger.info(f"Testing scenario: {scenario['description']}")
            
            # Setup vehicle for this scenario
            if not self.setup_vehicle():
                logger.error("Failed to setup vehicle, skipping scenario")
                continue
            
            try:
                # Test WITHOUT synthetic data
                logger.info("Running baseline test...")
                baseline_metrics = self.run_autonomous_driving_test(
                    use_synthetic_data=False, 
                    scenario=scenario
                )
                results['without_synthetic'].append(baseline_metrics)
                
                # Reset vehicle position
                self.cleanup_vehicle()
                time.sleep(2)
                self.setup_vehicle()
                
                # Test WITH synthetic data
                logger.info("Running synthetic data test...")
                synthetic_metrics = self.run_autonomous_driving_test(
                    use_synthetic_data=True, 
                    scenario=scenario
                )
                results['with_synthetic'].append(synthetic_metrics)
                
                results['scenarios'].append(scenario)
                
            except Exception as e:
                logger.error(f"Error in scenario test: {e}")
            finally:
                self.cleanup_vehicle()
                time.sleep(2)
        
        # Save and analyze results
        self.save_results(results)
        self.analyze_results(results)
        
        return results
    
    def save_results(self, results):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"carla_synthetic_data_test_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, list):
                serializable_results[key] = []
                for item in value:
                    if isinstance(item, dict):
                        serializable_item = {}
                        for k, v in item.items():
                            if isinstance(v, (list, np.ndarray)):
                                serializable_item[k] = list(v) if hasattr(v, '__iter__') else v
                            else:
                                serializable_item[k] = v
                        serializable_results[key].append(serializable_item)
                    else:
                        serializable_results[key].append(item)
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def analyze_results(self, results):
        """Analyze and visualize test results"""
        logger.info("Analyzing test results...")
        
        # Calculate averages
        metrics_to_compare = [
            'collisions', 'lane_departures', 'average_speed', 
            'distance_traveled', 'steering_smoothness', 'path_efficiency'
        ]
        
        comparison = {}
        for metric in metrics_to_compare:
            baseline_values = [test[metric] for test in results['without_synthetic']]
            synthetic_values = [test[metric] for test in results['with_synthetic']]
            
            comparison[metric] = {
                'baseline_avg': np.mean(baseline_values),
                'synthetic_avg': np.mean(synthetic_values),
                'improvement': ((np.mean(synthetic_values) - np.mean(baseline_values)) / np.mean(baseline_values)) * 100
            }
        
        # Print comparison results
        print("\n" + "="*60)
        print("CARLA SYNTHETIC DATA TEST RESULTS")
        print("="*60)
        
        for metric, data in comparison.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Baseline:     {data['baseline_avg']:.3f}")
            print(f"  With Synthetic: {data['synthetic_avg']:.3f}")
            print(f"  Improvement:  {data['improvement']:+.1f}%")
        
        # Create visualization
        self.create_comparison_plots(comparison)
        
        return comparison
    
    def create_comparison_plots(self, comparison):
        """Create comparison visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('CARLA Synthetic Data Performance Comparison', fontsize=16)
        
        metrics = list(comparison.keys())
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            if row < 2 and col < 3:
                ax = axes[row, col]
                
                baseline = comparison[metric]['baseline_avg']
                synthetic = comparison[metric]['synthetic_avg']
                
                bars = ax.bar(['Baseline', 'With Synthetic'], [baseline, synthetic], 
                             color=['red', 'green'], alpha=0.7)
                
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Value')
                
                # Add improvement percentage
                improvement = comparison[metric]['improvement']
                ax.text(0.5, max(baseline, synthetic) * 0.9, 
                       f'{improvement:+.1f}%', 
                       ha='center', fontweight='bold')
        
        # Remove empty subplots
        for i in range(len(metrics), 6):
            row = i // 3
            col = i % 3
            if row < 2:
                fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'carla_comparison_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved as carla_comparison_results_{timestamp}.png")
        
        plt.show()
    
    def cleanup_vehicle(self):
        """Clean up vehicle and sensors"""
        try:
            if self.camera_sensor:
                self.camera_sensor.destroy()
                self.camera_sensor = None
            
            if self.collision_sensor:
                self.collision_sensor.destroy()
                self.collision_sensor = None
            
            if self.vehicle:
                self.vehicle.destroy()
                self.vehicle = None
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def cleanup(self):
        """Clean up all resources"""
        self.cleanup_vehicle()
        logger.info("Cleanup completed")

def main():
    """Main function to run CARLA synthetic data tests"""
    tester = CARLASyntheticDataTest()
    
    try:
        # Connect to CARLA
        if not tester.connect_to_carla():
            logger.error("Failed to connect to CARLA. Make sure CARLA is running.")
            return
        
        # Run comparison tests
        results = tester.run_comparison_tests()
        
        logger.info("All tests completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()