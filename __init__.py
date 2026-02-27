"""
Confocal Image Segmentation Package

A comprehensive package for analyzing confocal microscopy images,
including segmentation, circularity analysis, and circle fitting.

Modules:
- circularity: Circularity calculation and filtering
- circle_fitting: Circle fitting algorithms
- image_processing: Image preprocessing and segmentation
- utils: Utility functions and common imports
"""

from .circularity import (
    calculate_circularity,
    filter_labels_by_circularity,
    advanced_circularity_filter,
    quick_circularity_filter,
    visualize_circularity_distribution,
    print_circularity_stats
)

from .circle_fitting import (
    fit_circles_simple,
    fit_circles_area_based,
    fit_circles_least_squares,
    fit_circles_algebraic,
    fit_circles_contour_based,
    visualize_circle_fits,
    get_circle_metrics,
    create_circle_mask,
    complete_circle_fitting_workflow
)

from .image_processing import (
    preprocess_image,
    threshold_image,
    create_binary_mask,
    morphological_operations,
    label_regions,
    watershed_segmentation,
    filter_by_size,
    complete_segmentation_pipeline,
    visualize_segmentation_pipeline,
    print_segmentation_stats
)

from .utils import (
    load_czi_file,
    create_napari_viewer,
    add_to_napari,
    analyze_image,
    create_summary_report
)

__version__ = "1.0.0"
__author__ = "Confocal Analysis Team"
__email__ = "your.email@example.com"

__all__ = [
    # Circularity functions
    'calculate_circularity',
    'filter_labels_by_circularity',
    'advanced_circularity_filter',
    'quick_circularity_filter',
    'visualize_circularity_distribution',
    'print_circularity_stats',
    
    # Circle fitting functions
    'fit_circles_simple',
    'fit_circles_area_based',
    'fit_circles_least_squares',
    'fit_circles_algebraic',
    'fit_circles_contour_based',
    'visualize_circle_fits',
    'get_circle_metrics',
    'create_circle_mask',
    'complete_circle_fitting_workflow',
    
    # Image processing functions
    'preprocess_image',
    'threshold_image',
    'create_binary_mask',
    'morphological_operations',
    'label_regions',
    'watershed_segmentation',
    'filter_by_size',
    'complete_segmentation_pipeline',
    'visualize_segmentation_pipeline',
    'print_segmentation_stats',
    
    # Utility functions
    'load_czi_file',
    'create_napari_viewer',
    'add_to_napari',
    'analyze_image',
    'create_summary_report'
]



