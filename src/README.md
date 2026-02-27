# Confocal Image Segmentation Package

A comprehensive Python package for analyzing confocal microscopy images, including segmentation, circularity analysis, and circle fitting.

## Features

- **Image Preprocessing**: Median filtering, thresholding, morphological operations
- **Segmentation**: Watershed segmentation with customizable parameters
- **Circularity Analysis**: Calculate and filter regions by circularity
- **Circle Fitting**: Multiple algorithms for fitting circles to segmented objects
- **Visualization**: Comprehensive plotting and Napari integration
- **Batch Processing**: Complete pipeline for automated analysis

## Installation

```bash
# Install required dependencies
pip install numpy matplotlib scikit-image scipy napari czifile
```

## Quick Start

```python
from confocal_segmentation import analyze_image, load_czi_file

# Load your CZI file
image = load_czi_file('path/to/your/file.czi')

# Run complete analysis
results = analyze_image(image, circularity_threshold=0.8)

# Access results
segmented_labels = results['segmentation']['labels_filtered']
circularities = results['circularities']
circles = results['circles']
```

## Modules

### 1. `circularity.py` - Circularity Analysis

Calculate and filter regions based on circularity:

```python
from circularity import calculate_circularity, filter_labels_by_circularity

# Calculate circularity for all regions
circularities = calculate_circularity(labels)

# Filter regions with circularity > 0.8
filtered_labels = filter_labels_by_circularity(labels, circularities, min_circularity=0.8)
```

**Key Functions:**
- `calculate_circularity()`: Calculate circularity for labeled regions
- `filter_labels_by_circularity()`: Filter regions by circularity threshold
- `advanced_circularity_filter()`: Multi-criteria filtering
- `visualize_circularity_distribution()`: Plot circularity distribution

### 2. `circle_fitting.py` - Circle Fitting

Fit circles to segmented objects using various algorithms:

```python
from circle_fitting import complete_circle_fitting_workflow

# Fit circles using algebraic method
circles, metrics = complete_circle_fitting_workflow(labels, image, method='algebraic')
```

**Available Methods:**
- `simple`: Fast, uses equivalent diameter
- `area`: Based on area calculation
- `least_squares`: Most accurate for irregular shapes
- `algebraic`: Fast and accurate for near-circular objects
- `contour`: Uses object boundaries only

### 3. `image_processing.py` - Image Preprocessing

Complete segmentation pipeline:

```python
from image_processing import complete_segmentation_pipeline

# Run complete segmentation
results = complete_segmentation_pipeline(
    image,
    median_size=12,
    threshold_method='li',
    morph_disk_size=16,
    watershed_min_distance=15
)
```

**Pipeline Steps:**
1. Median filtering
2. Thresholding (Li, Otsu, or manual)
3. Morphological operations
4. Connected component labeling
5. Watershed segmentation
6. Size filtering

### 4. `utils.py` - Utility Functions

Common utilities and complete analysis:

```python
from utils import analyze_image, create_summary_report

# Complete analysis with all steps
results = analyze_image(
    image,
    segmentation_params={...},
    circularity_threshold=0.8,
    circle_fitting_method='algebraic'
)

# Create summary report
create_summary_report(results['segmentation'], results['circularities'], results['circles'])
```

## Usage Examples

### Basic Segmentation

```python
import numpy as np
from confocal_segmentation import *

# Load image
image = load_czi_file('data/sample.czi')

# Run segmentation
results = complete_segmentation_pipeline(image)
segmented_labels = results['labels_filtered']

# View in Napari
viewer = create_napari_viewer()
add_to_napari(viewer, image, name='Original')
add_to_napari(viewer, segmented_labels, name='Segmented')
```

### Circularity Analysis

```python
# Calculate circularity
circularities = calculate_circularity(segmented_labels)

# Visualize distribution
visualize_circularity_distribution(circularities, threshold=0.8)

# Filter by circularity
circular_labels = filter_labels_by_circularity(
    segmented_labels, circularities, min_circularity=0.8
)
```

### Circle Fitting

```python
# Fit circles using different methods
circles_simple = fit_circles_area_based(labels)
circles_accurate = fit_circles_algebraic(labels)

# Get quality metrics
metrics = get_circle_metrics(labels, circles_accurate)

# Visualize results
visualize_circle_fits(image, labels, circles_accurate, method_name='Algebraic')
```

### Complete Analysis Pipeline

```python
# One-line complete analysis
results = analyze_image(
    image,
    segmentation_params={
        'median_size': 12,
        'threshold_method': 'li',
        'morph_disk_size': 16
    },
    circularity_threshold=0.8,
    circle_fitting_method='algebraic',
    visualize=True
)

# Access all results
segmented = results['segmentation']['labels_filtered']
circularities = results['circularities']
circles = results['circles']
```

## Parameters Guide

### Segmentation Parameters

- `median_size`: Size of median filter (default: 12)
- `threshold_method`: 'li', 'otsu', 'mean' (default: 'li')
- `morph_disk_size`: Size of morphological disk (default: 16)
- `morph_operation`: 'closing', 'opening', 'erosion', 'dilation' (default: 'closing')
- `connectivity`: 1 or 2 for labeling (default: 1)
- `watershed_min_distance`: Minimum distance between peaks (default: 15)

### Circularity Parameters

- `min_circularity`: Minimum circularity threshold (0-1, default: 0.8)
- `min_area`: Minimum region area in pixels (default: 100)
- `max_area`: Maximum region area in pixels (optional)

### Circle Fitting Parameters

- `method`: 'simple', 'area', 'least_squares', 'algebraic', 'contour'
- For droplets: use 'area' or 'algebraic'
- For irregular shapes: use 'least_squares'
- For speed: use 'simple' or 'area'

## File Structure

```
confocal_segmentation/
├── __init__.py              # Package initialization
├── circularity.py           # Circularity analysis functions
├── circle_fitting.py       # Circle fitting algorithms
├── image_processing.py     # Image preprocessing functions
├── utils.py                # Utility functions
├── main.py                 # Example usage script
└── README.md               # This file
```

## Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scikit-image`: Image processing
- `scipy`: Scientific computing
- `napari`: Interactive image viewer
- `czifile`: CZI file reading

## Examples

See `main.py` for complete examples demonstrating all functionality.

## Contributing

This package is designed for confocal microscopy analysis. Feel free to extend it for your specific needs.

## License

MIT License - see LICENSE file for details.



