"""
Image processing functions for confocal microscopy data.

This module provides functions for preprocessing confocal images,
including filtering, thresholding, morphological operations,
and watershed segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.morphology import disk, closing, opening, erosion, dilation
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def create_binary_mask(image, threshold=None, method='threshold_li'):
    """
    Create binary mask from image.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    threshold : float, optional
        Manual threshold value
    method : str
        Thresholding method if threshold is None ('threshold_li', 'threshold_otsu', 'threshold_mean')
        
    Returns:
    --------
    binary_mask : ndarray
        Binary mask
    threshold_used : float
        Threshold value used
    """
    if threshold is None:
        # Apply automatic thresholding based on method
        if method == 'threshold_li':
            threshold = filters.threshold_li(image)
        elif method == 'threshold_otsu':
            threshold = filters.threshold_otsu(image)
        elif method == 'threshold_mean':
            threshold = filters.threshold_mean(image)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")
    
    binary_mask = image > threshold
    return binary_mask, threshold


def watershed_segmentation(labels, min_distance=24, disk_size=24):
    """
    Apply watershed segmentation to labeled regions.
    
    Parameters:
    -----------
    labels : ndarray
        Labeled image
    min_distance : int
        Minimum distance between peaks
    disk_size : int
        Size of disk for peak detection
        
    Returns:
    --------
    labels_ws : ndarray
        Watershed segmented labels
    """
    # Calculate distance transform
    distance = ndi.distance_transform_edt(labels > 0)
    
    # Find local maxima
    coords = peak_local_max(
        distance, 
        footprint=disk(disk_size), 
        min_distance=min_distance, 
        labels=labels
    )
    
    # Create markers
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    
    # Apply watershed
    labels_ws = watershed(-distance, markers, mask=labels)
    
    return labels_ws


def filter_by_size(labels, min_size=None, max_size=None, method='mean'):
    """
    Filter labeled regions by size.
    
    Parameters:
    -----------
    labels : ndarray
        Labeled image
    min_size : int, optional
        Minimum size in pixels
    max_size : int, optional
        Maximum size in pixels
    method : str
        Method for automatic threshold ('mean', 'li', 'otsu')
        
    Returns:
    --------
    filtered_labels : ndarray
        Size-filtered labels
    """
    # Get region sizes
    unique_labels, sizes = np.unique(labels, return_counts=True)
    
    # Set automatic thresholds if not provided
    if min_size is None:
        if method == 'mean':
            min_size = filters.threshold_mean(sizes)
        elif method == 'li':
            min_size = filters.threshold_li(sizes)
        elif method == 'otsu':
            min_size = filters.threshold_otsu(sizes)
        else:
            min_size = 0
    
    # Create filtered labels
    filtered_labels = labels.copy()
    
    for label_id, size in zip(unique_labels, sizes):
        remove = False
        
        if min_size is not None and size < min_size:
            remove = True
        if max_size is not None and size > max_size:
            remove = True
            
        if remove:
            filtered_labels[filtered_labels == label_id] = 0
    
    return filtered_labels


def print_segmentation_stats(results):
    """
    Print statistics about the segmentation results.
    
    Parameters:
    -----------
    results : dict
        Results from complete_segmentation_pipeline
    """
    print("Segmentation Pipeline Statistics:")
    print(f"Threshold used: {results['threshold']:.3f}")
    print(f"Initial regions: {len(np.unique(results['labels'])) - 1}")
    print(f"After watershed: {len(np.unique(results['labels_ws'])) - 1}")
    print(f"Final regions: {len(np.unique(results['labels_filtered'])) - 1}")
    
    # Size statistics
    if 'labels_filtered' in results:
        unique_labels, sizes = np.unique(results['labels_filtered'], return_counts=True)
        sizes = sizes[sizes > 0]  # Remove background
        if len(sizes) > 0:
            print(f"Region size range: {sizes.min()} - {sizes.max()} pixels")
            print(f"Mean region size: {sizes.mean():.1f} pixels")
            print(f"Median region size: {np.median(sizes):.1f} pixels")
