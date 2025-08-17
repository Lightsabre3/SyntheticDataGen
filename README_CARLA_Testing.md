# CARLA Synthetic Data Testing

This setup allows you to test the impact of your 10K generated rural driving images on autonomous driving performance in CARLA simulator.

## Overview

The testing framework compares two scenarios:
1. **Baseline**: AI model trained without synthetic data
2. **Enhanced**: AI model trained with your 10K synthetic rural driving images

## Prerequisites

### 1. CARLA Simulator
- Download CARLA 0.9.15 from: https://github.com/carla-simulator/carla/releases
- Extract and ensure CarlaUE4.sh (Linux/Mac) or CarlaUE4.exe (Windows) is accessible

### 2. Your Generated Images
- Ensure your 10K generated images are in a folder called `generated_images/`
- Images should be in PNG or JPG format

### 3. Python Environment
- Python 3.8+
- CUDA-capable GPU recommended (but not required)

## Quick Start

### 1. Setup Environment
```bash
# Run the setup script
python setup_carla_test.py
```

This will:
- Check CARLA installation
- Install required Python packages
- Create necessary directories
- Test CARLA connection

### 2. Start CARLA Server
```bash
# Linux/Mac
./CarlaUE4.sh

# Windows
CarlaUE4.exe

# For headless mode (no graphics)
./CarlaUE4.sh -RenderOffScreen
```

### 3. Train Models (Optional)
```bash
# Train both baseline and synthetic data models
python train_with_synthetic_data.py
```

### 4. Run Comparison Tests
```bash
# Run the full comparison test
python carla_synthetic_data_test.py

# Or use the convenience script
./run_carla_test.sh
```

## Test Scenarios

The framework tests 4 different rural driving scenarios:

1. **Town07 - Clear Weather**: Rural roads with perfect visibility
2. **Town07 - Cloudy Weather**: Rural roads with overcast conditions  
3. **Town07 - Wet Conditions**: Rural roads with rain/wet surfaces
4. **Town10HD - Sunset**: Highway driving with challenging lighting

## Metrics Measured

### Primary Metrics
- **Collisions**: Number of crashes during test
- **Lane Departures**: Times vehicle left the lane
- **Average Speed**: Mean driving speed (km/h)
- **Distance Traveled**: Total distance covered (meters)
- **Path Efficiency**: How optimal the driving path was

### Secondary Metrics
- **Steering Smoothness**: Standard deviation of steering inputs
- **Throttle Usage**: Average throttle application
- **Brake Usage**: Average brake application
- **Reaction Times**: Response time to road events

## Expected Results

With good synthetic data, you should see improvements in:
- ✅ **Fewer collisions** in rural scenarios
- ✅ **Better lane keeping** on country roads
- ✅ **Smoother steering** through curves
- ✅ **More appropriate speed** for rural conditions
- ✅ **Better handling** of rural road features

## File Structure

```
├── carla_synthetic_data_test.py    # Main test script
├── carla_test_config.json          # Configuration file
├── setup_carla_test.py             # Setup and installation script
├── train_with_synthetic_data.py    # Model training example
├── run_carla_test.sh               # Convenience run script
├── generated_images/               # Your 10K synthetic images
├── models/                         # Trained model files
├── carla_test_results/            # Test results and logs
└── README_CARLA_Testing.md        # This file
```

## Customization

### Adding Your Own Model
Replace the placeholder model loading in `carla_synthetic_data_test.py`:

```python
def load_synthetic_data_model(self):
    # Load your actual trained model
    self.model_with_synthetic = torch.load('path/to/your/model.pth')
    return True
```

### Custom Scenarios
Edit `carla_test_config.json` to add your own test scenarios:

```json
{
  "map": "Town01",
  "weather": "HardRainNoon", 
  "description": "Urban rain test",
  "spawn_points": "urban"
}
```

### Different Metrics
Modify the metrics collection in the `run_autonomous_driving_test()` method to track additional performance indicators.

## Troubleshooting

### CARLA Connection Issues
```bash
# Check if CARLA is running
ps aux | grep CarlaUE4

# Test connection manually
python -c "import carla; client = carla.Client('localhost', 2000); print(client.get_world().get_map().name)"
```

### Memory Issues
- Reduce batch size in training
- Use smaller image resolution
- Enable CARLA's low-quality mode: `./CarlaUE4.sh -quality-level=Low`

### Performance Issues
- Use GPU acceleration if available
- Reduce test duration in config
- Run fewer scenarios simultaneously

## Results Analysis

After testing, you'll get:

1. **JSON Results File**: Detailed metrics for each test
2. **Comparison Plots**: Visual comparison charts
3. **Console Summary**: Quick overview of improvements

Example output:
```
CARLA SYNTHETIC DATA TEST RESULTS
==================================

Collisions:
  Baseline:       2.750
  With Synthetic: 1.250
  Improvement:    -54.5%

Lane Departures:
  Baseline:       8.500
  With Synthetic: 4.750
  Improvement:    -44.1%
```

## Next Steps

1. **Analyze Results**: Review which scenarios showed the most improvement
2. **Iterate on Data**: Generate more targeted synthetic images for weak areas
3. **Real-World Testing**: Validate improvements with real vehicle testing
4. **Model Refinement**: Use CARLA results to improve your training approach

## Support

For issues with:
- **CARLA**: Check CARLA documentation and GitHub issues
- **This Framework**: Review logs in `carla_test_results/` directory
- **Model Training**: Adjust hyperparameters in training script

## License

This testing framework is provided as-is for research and development purposes.