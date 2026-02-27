"""
Utility functions and common imports for confocal image segmentation.

This module provides common imports, utility functions, and helper methods
used across the segmentation pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import czifile
from napari.viewer import Viewer

# Image processing imports
from skimage import filters
from skimage.morphology import disk, closing, opening, erosion, dilation
from skimage.segmentation import watershed
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import footprint_rectangle as rectangle
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import morphology
from .ellipse_analysis import ellipse_polygon
'''
# Import our custom modules
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
'''

def load_czi_file(filepath, channel=0, timepoint=0): #add timepoint
    """
    Load CZI file and extract specific channel and timepoint.
    
    Parameters:
    -----------
    filepath : str
        Path to CZI file
    channel : int
        Channel index to extract
    timepoint : int
        Timepoint index to extract
        
    Returns:
    --------
    image : ndarray
        Extracted image
    """
    I = czifile.imread(filepath)
    I = I[0, 0, channel, 0, :, :, 0]
    return I


def plot_histogram(data, bins=100, title="Histogram", **kwargs):
    """
    Plot histogram of data.
    
    Parameters:
    -----------
    data : ndarray
        Data to plot
    bins : int
        Number of bins
    title : str
        Plot title
    **kwargs : dict
        Additional arguments for plt.hist
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data.flatten(), bins=bins, **kwargs)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def print_array_info(array, name="Array"):
    """
    Print information about an array.
    
    Parameters:
    -----------
    array : ndarray
        Array to analyze
    name : str
        Array name for display
    """
    print(f"{name} Information:")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")
    print(f"  Min: {array.min()}")
    print(f"  Max: {array.max()}")
    print(f"  Mean: {array.mean():.3f}")
    print(f"  Std: {array.std():.3f}")
    if array.dtype in [np.int32, np.int64]:
        unique_vals = np.unique(array)
        print(f"  Unique values: {len(unique_vals)} (range: {unique_vals.min()} - {unique_vals.max()})")


def save_results(results, filename_prefix="results"):
    """
    Save segmentation results to files.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from segmentation pipeline
    filename_prefix : str
        Prefix for saved files
    """
    import os
    
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            filename = f"{filename_prefix}_{key}.npy"
            np.save(filename, value)
            print(f"Saved {key} to {filename}")


def load_results(filename_prefix="results"):
    """
    Load segmentation results from files.
    
    Parameters:
    -----------
    filename_prefix : str
        Prefix for files to load
        
    Returns:
    --------
    results : dict
        Loaded results dictionary
    """
    import os
    import glob
    
    results = {}
    pattern = f"{filename_prefix}_*.npy"
    files = glob.glob(pattern)
    
    for filename in files:
        key = filename.replace(f"{filename_prefix}_", "").replace(".npy", "")
        results[key] = np.load(filename)
        print(f"Loaded {key} from {filename}")
    
    return results


def create_summary_report(results, circularities=None, circles=None):
    """
    Create a summary report of segmentation results.
    
    Parameters:
    -----------
    results : dict
        Segmentation results
    circularities : dict, optional
        Circularity values
    circles : list, optional
        Fitted circles
    """
    print("=" * 50)
    print("SEGMENTATION SUMMARY REPORT")
    print("=" * 50)
    
    # Basic statistics
    if 'labels_filtered' in results:
        labels = results['labels_filtered']
        unique_labels = np.unique(labels)
        n_regions = len(unique_labels) - 1  # Exclude background
        
        print(f"Number of segmented regions: {n_regions}")
        
        if n_regions > 0:
            # Size statistics
            sizes = []
            for label_id in unique_labels:
                if label_id > 0:
                    size = np.sum(labels == label_id)
                    sizes.append(size)
            
            if sizes:
                sizes = np.array(sizes)
                print(f"Region size range: {sizes.min()} - {sizes.max()} pixels")
                print(f"Mean region size: {sizes.mean():.1f} pixels")
                print(f"Median region size: {np.median(sizes):.1f} pixels")
    
    # Circularity statistics
    if circularities:
        circ_values = [v for v in circularities.values() if v > 0]
        if circ_values:
            print(f"\nCircularity statistics:")
            print(f"  Mean circularity: {np.mean(circ_values):.3f}")
            print(f"  Median circularity: {np.median(circ_values):.3f}")
            print(f"  Range: {min(circ_values):.3f} - {max(circ_values):.3f}")
    
    # Circle fitting statistics
    if circles:
        radii = [circle[2] for circle in circles]
        print(f"\nCircle fitting results:")
        print(f"  Number of fitted circles: {len(circles)}")
        print(f"  Radius range: {min(radii):.1f} - {max(radii):.1f} pixels")
        print(f"  Mean radius: {np.mean(radii):.1f} pixels")
        print(f"  Median radius: {np.median(radii):.1f} pixels")
    
    print("=" * 50)


def get_region_properties(labels, properties=None):
    """
    Get region properties for labeled image.
    
    Parameters:
    -----------
    labels : ndarray
        Labeled image
    properties : list, optional
        List of properties to extract
        
    Returns:
    --------
    props_table : dict
        Dictionary of region properties
    """
    if properties is None:
        properties = [
            'label', 'area', 'perimeter', 'centroid', 
            'eccentricity', 'solidity', 'extent'
        ]
    
    return regionprops_table(labels, properties=properties)


# Convenience function for complete analysis
def analyze_image(image, 
                 segmentation_params=None,
                 circularity_threshold=0.8,
                 circle_fitting_method='algebraic',
                 visualize=True):
    """
    Complete analysis pipeline: segmentation + circularity + circle fitting.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    segmentation_params : dict, optional
        Parameters for segmentation pipeline
    circularity_threshold : float
        Circularity threshold for filtering
    circle_fitting_method : str
        Method for circle fitting
    visualize : bool
        Whether to show visualizations
        
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    # Default segmentation parameters
    if segmentation_params is None:
        segmentation_params = {
            'median_size': 12,
            'threshold_method': 'li',
            'morph_disk_size': 16,
            'morph_operation': 'closing',
            'connectivity': 1,
            'watershed_min_distance': 15,
            'watershed_disk_size': 12,
            'size_filter_method': 'mean'
        }
    
    # Run segmentation pipeline
    seg_results = complete_segmentation_pipeline(image, **segmentation_params)
    
    # Calculate circularity
    circularities = calculate_circularity(seg_results['labels_filtered'])
    
    # Filter by circularity
    labels_circular = filter_labels_by_circularity(
        seg_results['labels_filtered'], circularities, circularity_threshold
    )
    
    # Fit circles
    circles, metrics = complete_circle_fitting_workflow(
        labels_circular, image if visualize else None, circle_fitting_method
    )
    
    # Compile results
    results = {
        'segmentation': seg_results,
        'circularities': circularities,
        'labels_circular': labels_circular,
        'circles': circles,
        'circle_metrics': metrics
    }
    
    # Create summary report
    create_summary_report(seg_results, circularities, circles)
    
    return results


def plot_label_ellipse(raw_image, label_image, label_id,
                       padding=30, ax=None, figsize=(6, 6),
                       ellipse_color='yellow', major_color='red',
                       minor_color='cyan', cmap='gray', linewidth=1.5):
    """Plot a zoomed-in view of a labeled object with its fitted ellipse and axes.

    Parameters
    ----------
    raw_image : np.ndarray, shape (H, W)
        Single frame of the raw microscopy image.
    label_image : np.ndarray, shape (H, W)
        Connected component label image (same frame).
    label_id : int
        Which label to visualize.
    padding : int
        Pixels of context around the object bounding box.
    ax : matplotlib Axes, optional
        If None, creates a new figure.
    figsize : tuple
        Figure size (only used if ax is None).
    ellipse_color, major_color, minor_color : str
        Colors for ellipse outline and axes.
    cmap : str
        Colormap for the raw image.
    linewidth : float
        Line width for ellipse and axes.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    mask = label_image == label_id
    if mask.sum() == 0:
        raise ValueError(f"Label {label_id} not found in label image")

    prop = regionprops(mask.astype(np.uint8))[0]
    cy, cx = prop.centroid
    r_major = prop.major_axis_length / 2
    r_minor = prop.minor_axis_length / 2
    angle = prop.orientation

    rmin, cmin, rmax, cmax = prop.bbox
    H, W = raw_image.shape
    rmin = max(0, rmin - padding)
    cmin = max(0, cmin - padding)
    rmax = min(H, rmax + padding)
    cmax = min(W, cmax + padding)

    outline, major_ax, minor_ax = ellipse_polygon(
        cy, cx, r_major, r_minor, angle, return_axes=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.imshow(raw_image[rmin:rmax, cmin:cmax], cmap=cmap,
              extent=[cmin, cmax, rmax, rmin])
    ax.plot(outline[:, 1], outline[:, 0], color=ellipse_color,
            linewidth=linewidth, label='ellipse')
    ax.plot(major_ax[:, 1], major_ax[:, 0], color=major_color,
            linewidth=linewidth, label=f'major r={r_major:.1f}')
    ax.plot(minor_ax[:, 1], minor_ax[:, 0], color=minor_color,
            linewidth=linewidth, label=f'minor r={r_minor:.1f}')

    ax.set_xlim(cmin, cmax)
    ax.set_ylim(rmax, rmin)
    ax.set_aspect('equal')

    return fig, ax


def visualize_ellipses(ellipse_data, n_images, view, tag='',
                       edge_color='yellow', edge_width=2,
                       show_axes=True, major_color='red', minor_color='cyan'):
    """Add tracked ellipses to napari viewer, one shapes layer per scene.

    Optionally overlays major and minor axis lines.

    Parameters
    ----------
    ellipse_data : list of tuples
        Each tuple: (t, cy, cx, r_major, r_minor, angle, image_idx, label_id).
    n_images : int
        Number of scenes/images.
    view : napari.Viewer
        Napari viewer instance.
    tag : str
        Name tag for the shapes layers.
    edge_color : str
        Ellipse outline color.
    edge_width : float
        Ellipse outline width.
    show_axes : bool
        If True, overlay major and minor axis lines.
    major_color, minor_color : str
        Colors for axis lines.
    """
    for j_view in range(n_images):
        ellipses_j = [e for e in ellipse_data if e[6] == j_view]
        if not ellipses_j:
            continue
        shapes = []
        axis_lines = []
        axis_colors = []
        for t, cy, cx, r_maj, r_min, angle, _, _ in ellipses_j:
            pts_2d, major, minor = ellipse_polygon(
                cy, cx, r_maj, r_min, angle, return_axes=True)
            pts_3d = np.column_stack([np.full(len(pts_2d), t), pts_2d])
            shapes.append(pts_3d)
            if show_axes:
                axis_lines.append(np.column_stack([np.full(2, t), major]))
                axis_colors.append(major_color)
                axis_lines.append(np.column_stack([np.full(2, t), minor]))
                axis_colors.append(minor_color)
        view.add_shapes(shapes, shape_type='polygon',
                        edge_color=edge_color, face_color='transparent',
                        edge_width=edge_width,
                        name=f'ellipses {tag} scene {j_view}')
        if show_axes and axis_lines:
            view.add_shapes(axis_lines, shape_type='line',
                            edge_color=axis_colors, face_color='transparent',
                            edge_width=1,
                            name=f'axes {tag} scene {j_view}')
