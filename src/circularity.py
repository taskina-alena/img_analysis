"""
Circularity analysis and filtering functions for image segmentation.

This module provides functions to calculate circularity metrics for labeled regions
and filter regions based on circularity thresholds.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage import filters


def calculate_circularity(labels):
    """
    Calculate circularity for each labeled region.
    
    Circularity = 4π * area / perimeter²
    Perfect circle has circularity = 1.0
    
    Parameters:
    -----------
    labels : ndarray
        Labeled image where each region has a unique integer value
        
    Returns:
    --------
    circularities : dict
        Dictionary mapping label_id to circularity value
    """
    props = regionprops(labels)
    circularities = {}
    
    for prop in props:
        if prop.area > 0:  # Skip background
            # Calculate circularity
            perimeter = prop.perimeter
            if perimeter > 0:
                circularity = 4 * np.pi * prop.area / (perimeter ** 2)
            else:
                circularity = 0
                
            circularities[prop.label] = circularity
    
    # Ensure background has circularity 0
    circularities[0] = 0
    
    return circularities


def circ_filter(labels, 
                circularity_filter='li', 
                circularity_threshold=None,
                size_filter='mean', 
                size_threshold=None,
                min_size=None, 
                max_size=None,
                min_circularity=None,
                max_circularity=None):
    """
    Comprehensive filtering function for labels based on circularity and size.
    
    Parameters:
    -----------
    labels : ndarray
        Labeled image
    circularity_filter : str
        Method for automatic circularity threshold ('threshold_li', 'threshold_otsu', 'threshold_mean', 'manual', 'none')
    circularity_threshold : float, optional
        Manual circularity threshold (used if circularity_filter='manual')
    size_filter : str
        Method for automatic size threshold ('mean', 'li', 'otsu', 'manual', 'none')
    size_threshold : float, optional
        Manual size threshold (used if size_filter='manual')
    min_size : int, optional
        Minimum size in pixels
    max_size : int, optional
        Maximum size in pixels
    min_circularity : float, optional
        Minimum circularity (0-1)
    max_circularity : float, optional
        Maximum circularity (0-1)
        
    Returns:
    --------
    filtered_labels : ndarray
        Filtered labels
    stats : dict
        Statistics about filtering results
    """
    
    # Calculate circularities
    circularities = calculate_circularity(labels)
    props = regionprops(labels)
    
    # Get region sizes
    unique_labels, sizes = np.unique(labels, return_counts=True)
    sizes_dict = dict(zip(unique_labels, sizes))
    
    # Initialize statistics
    stats = {
        'total_regions': len(props),
        'filtered_regions': 0,
        'removed_by_circularity': 0,
        'removed_by_size': 0,
        'circularity_threshold_used': None,
        'size_threshold_used': None,
        'circularity_stats': [],
        'size_stats': []
    }
    
    # Determine circularity threshold
    if circularity_filter == 'manual':
        if circularity_threshold is None:
            raise ValueError("circularity_threshold must be provided when circularity_filter='manual'")
        circ_thresh = circularity_threshold
    elif circularity_filter != 'none':
        # Get circularity values for automatic thresholding
        circ_values = [v for v in circularities.values() if v > 0]
        if len(circ_values) > 0:
            if circularity_filter == 'li':
                circ_thresh = filters.threshold_li(np.array(circ_values))
            elif circularity_filter == 'otsu':
                circ_thresh = filters.threshold_otsu(np.array(circ_values))
            elif circularity_filter == 'mean':
                circ_thresh = filters.threshold_mean(np.array(circ_values))
            elif circularity_filter == 'yen':
                circ_thresh = filters.threshold_yen(np.array(circ_values))
            else:
                raise ValueError(f"Unknown circularity filter method: {circularity_filter}")
        else:
            circ_thresh = 0
    else:
        circ_thresh = 0
    
    # Determine size threshold
    if size_filter == 'manual':
        if size_threshold is None:
            raise ValueError("size_threshold must be provided when size_filter='manual'")
        size_thresh = size_threshold
    elif size_filter != 'none':
        # Get size values for automatic thresholding
        size_values = sizes[sizes > 0]  # Exclude background
        if len(size_values) > 0:
            if size_filter == 'mean':
                size_thresh = filters.threshold_mean(size_values)
            elif size_filter == 'li':
                size_thresh = filters.threshold_li(size_values)
            elif size_filter == 'otsu':
                size_thresh = filters.threshold_otsu(size_values)
            elif size_filter == 'yen':
                size_thresh = filters.threshold_yen(size_values)
            else:
                raise ValueError(f"Unknown size filter method: {size_filter}")
        else:
            size_thresh = 0
    else:
        size_thresh = 0
    
    # Store thresholds used
    stats['circularity_threshold_used'] = circ_thresh
    stats['size_threshold_used'] = size_thresh
    
    # Create filtered labels
    filtered_labels = labels.copy()
    
    for prop in props:
        if prop.area == 0:  # Skip background
            continue
            
        label_id = prop.label
        area = prop.area
        circularity = circularities.get(label_id, 0)
        
        # Apply filters
        remove = False
        reasons = []
        
        # Circularity filtering
        if min_circularity is not None and circularity < min_circularity:
            remove = True
            reasons.append('min_circularity')
            stats['removed_by_circularity'] += 1
        elif max_circularity is not None and circularity > max_circularity:
            remove = True
            reasons.append('max_circularity')
            stats['removed_by_circularity'] += 1
        elif circularity_filter != 'none' and circularity < circ_thresh:
            remove = True
            reasons.append('circularity_threshold')
            stats['removed_by_circularity'] += 1
        
        # Size filtering
        if min_size is not None and area < min_size:
            remove = True
            reasons.append('min_size')
            stats['removed_by_size'] += 1
        elif max_size is not None and area > max_size:
            remove = True
            reasons.append('max_size')
            stats['removed_by_size'] += 1
        elif size_filter != 'none' and area < size_thresh:
            remove = True
            reasons.append('size_threshold')
            stats['removed_by_size'] += 1
        
        if remove:
            filtered_labels[filtered_labels == label_id] = 0
        else:
            stats['filtered_regions'] += 1
            stats['circularity_stats'].append({
                'label': label_id,
                'circularity': circularity,
                'area': area,
                'perimeter': prop.perimeter
            })
            stats['size_stats'].append({
                'label': label_id,
                'area': area,
                'circularity': circularity
            })
    
    return filtered_labels, stats


def print_circ_filter_stats(stats):
    """
    Print statistics from circ_filter function.
    
    Parameters:
    -----------
    stats : dict
        Statistics dictionary from circ_filter
    """
    print("Circularity and Size Filtering Results:")
    print("=" * 45)
    print(f"Total regions: {stats['total_regions']}")
    print(f"Filtered regions: {stats['filtered_regions']}")
    print(f"Removed by circularity: {stats['removed_by_circularity']}")
    print(f"Removed by size: {stats['removed_by_size']}")
    
    if stats['circularity_threshold_used'] is not None:
        print(f"Circularity threshold used: {stats['circularity_threshold_used']:.3f}")
    if stats['size_threshold_used'] is not None:
        print(f"Size threshold used: {stats['size_threshold_used']:.1f} pixels")
    
    if stats['circularity_stats']:
        circ_values = [s['circularity'] for s in stats['circularity_stats']]
        print(f"\nCircularity statistics (filtered regions):")
        print(f"  Range: {min(circ_values):.3f} - {max(circ_values):.3f}")
        print(f"  Mean: {np.mean(circ_values):.3f}")
        print(f"  Median: {np.median(circ_values):.3f}")
    
    if stats['size_stats']:
        size_values = [s['area'] for s in stats['size_stats']]
        print(f"\nSize statistics (filtered regions):")
        print(f"  Range: {min(size_values)} - {max(size_values)} pixels")
        print(f"  Mean: {np.mean(size_values):.1f} pixels")
        print(f"  Median: {np.median(size_values):.1f} pixels")


