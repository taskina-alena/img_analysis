"""
Optimized circle fitting - self-contained, no dependencies on circle_fitting.py

Key improvements:
- Vectorized operations for pairing
- Batch processing with progress tracking
- CSV export functionality
- Self-contained implementation
"""

import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import warnings
from skimage import measure


class CondensateDropletPair:
    """Simple container for a paired condensate and water-in-oil droplet."""
    
    def __init__(self, condensate_id: int, wod_id: int,
                 condensate_radius: float, wod_radius: float,
                 condensate_center: Tuple[float, float],
                 wod_center: Tuple[float, float],
                 pixel_size: Optional[Dict] = None):
        self.condensate_id = condensate_id
        self.wod_id = wod_id
        self.condensate_radius = condensate_radius
        self.wod_radius = wod_radius
        self.condensate_center = condensate_center
        self.wod_center = wod_center
        self.pixel_size = pixel_size
        self.distance = np.sqrt((condensate_center[0] - wod_center[0])**2 + 
                               (condensate_center[1] - wod_center[1])**2)
    
    @property
    def radius_ratio(self) -> float:
        """Ratio of condensate radius to WOD radius."""
        return self.condensate_radius / self.wod_radius if self.wod_radius > 0 else 0
    
    @property
    def volume_ratio(self) -> float:
        """Ratio of condensate volume to WOD volume (assumes spheres)."""
        return (self.condensate_radius / self.wod_radius) ** 3 if self.wod_radius > 0 else 0
    
    def get_volume_pixels(self, is_2d: bool = False) -> Tuple[float, float]:
        """Calculate volumes in pixel units."""
        if is_2d:
            condensate_vol = np.pi * self.condensate_radius ** 2
            wod_vol = np.pi * self.wod_radius ** 2
        else:
            condensate_vol = (4/3) * np.pi * self.condensate_radius ** 3
            wod_vol = (4/3) * np.pi * self.wod_radius ** 3
        return condensate_vol, wod_vol
    
    def get_volume_physical(self, is_2d: bool = False) -> Tuple[float, float]:
        """Calculate volumes in physical units (µm² or µm³)."""
        if self.pixel_size is None:
            raise ValueError("pixel_size must be set")
        
        avg_pixel_size = (self.pixel_size['x_um'] + self.pixel_size['y_um']) / 2
        cond_r_um = self.condensate_radius * avg_pixel_size
        wod_r_um = self.wod_radius * avg_pixel_size
        
        if is_2d:
            condensate_vol = np.pi * cond_r_um ** 2
            wod_vol = np.pi * wod_r_um ** 2
        else:
            if cond_r_um <= 35:
                condensate_vol = (4/3) * np.pi * cond_r_um ** 3
            else:
                condensate_vol = 70 *np.pi * cond_r_um ** 2
            if wod_r_um <= 35:
                wod_vol = (4/3) * np.pi * wod_r_um ** 3
            else:
                wod_vol = 70 *np.pi * wod_r_um ** 2
        
        return condensate_vol, wod_vol


def fit_circles_area_based(labels: np.ndarray) -> List[Tuple]:
    """
    Fit circles based on area: radius = sqrt(area/pi)
    Simple and fast.
    """
    props = measure.regionprops(labels)
    circles = []
    
    for prop in props:
        cy, cx = prop.centroid
        radius = np.sqrt(prop.area / np.pi)
        circles.append((cy, cx, radius, prop.label))
    
    return circles


def fit_circles_simple(labels: np.ndarray) -> List[Tuple]:
    """
    Fit circles using equivalent diameter from regionprops.
    Fast but approximate.
    """
    props = measure.regionprops(labels)
    circles = []
    
    for prop in props:
        cy, cx = prop.centroid
        radius = prop.equivalent_diameter / 2
        circles.append((cy, cx, radius, prop.label))
    
    return circles


def fit_circles_robust(labels: np.ndarray, 
                      methods: List[str] = ['area', 'simple']) -> List[Tuple]:
    """
    Try multiple circle fitting methods with automatic fallback.
    
    Parameters:
    -----------
    labels : ndarray
        Labeled image
    methods : list of str
        Methods to try in order
        
    Returns:
    --------
    circles : list of tuples
        Fitted circles (cy, cx, radius, label_id)
    """
    for method in methods:
        try:
            if method == 'area':
                return fit_circles_area_based(labels)
            elif method == 'simple':
                return fit_circles_simple(labels)
            elif method in ['algebraic', 'contour']:
                # Import from main module only if needed
                from scripts.circle_fitting import (
                    fit_circles_algebraic, 
                    fit_circles_contour_based
                )
                if method == 'algebraic':
                    return fit_circles_algebraic(labels)
                else:
                    return fit_circles_contour_based(labels)
        except Exception as e:
            warnings.warn(f"Method '{method}' failed: {e}")
            continue
    
    raise RuntimeError("All circle fitting methods failed")


def pair_condensates_with_droplets_vectorized(cond_circles: List[Tuple],
                                               wod_circles: List[Tuple],
                                               pixel_size: Optional[Dict] = None) -> List[CondensateDropletPair]:
    """
    Vectorized version of pairing - faster for many condensates.
    
    Parameters:
    -----------
    cond_circles : list of tuples
        Condensate circles (cy, cx, r, label)
    wod_circles : list of tuples
        WOD circles (cy, cx, r, label)
    pixel_size : dict, optional
        Physical pixel size
        
    Returns:
    --------
    pairs : list of CondensateDropletPair
    """
    pairs = []
    
    if not cond_circles or not wod_circles:
        return pairs
    
    # Convert condensates to array for vectorization
    cond_array = np.array([[c[0], c[1]] for c in cond_circles])  # Centers only
    
    for cy_wod, cx_wod, r_wod, wod_id in wod_circles:
        # Vectorized distance calculation for all condensates at once
        distances = np.sqrt((cond_array[:, 0] - cy_wod)**2 + 
                           (cond_array[:, 1] - cx_wod)**2)
        
        # Find condensates inside this WOD
        contained_indices = np.where(distances <= r_wod)[0]
        
        for idx in contained_indices:
            cy_cond, cx_cond, r_cond, cond_id = cond_circles[idx]
            
            # Create pair with correct parameter names
            pair = CondensateDropletPair(
                condensate_id=cond_id,
                wod_id=wod_id,
                condensate_radius=r_cond,
                wod_radius=r_wod,
                condensate_center=(cy_cond, cx_cond),
                wod_center=(cy_wod, cx_wod),
                pixel_size=pixel_size
            )
            
            # Distance is already calculated in __init__
            # But override with the exact value we calculated
            pair.distance = distances[idx]
            
            pairs.append(pair)
    
    return pairs


def analyze_paired_volumes(pairs: List[CondensateDropletPair],
                          is_2d: bool = True,
                          use_physical_units: bool = True) -> Dict:
    """
    Analyze volumes of paired condensates and droplets.
    
    Parameters:
    -----------
    pairs : list of CondensateDropletPair
        List of paired objects
    is_2d : bool, default=True
        If True, calculate areas (2D). If False, assume 3D spheres.
    use_physical_units : bool, default=True
        If True, use physical units (requires pixel_size in pairs)
        
    Returns:
    --------
    results : dict
        Dictionary containing volume statistics
    """
    condensate_volumes = []
    wod_volumes = []
    volume_ratios = []
    radius_ratios = []
    
    for pair in pairs:
        if use_physical_units and pair.pixel_size is not None:
            cond_vol, wod_vol = pair.get_volume_physical(is_2d=is_2d)
        else:
            cond_vol, wod_vol = pair.get_volume_pixels(is_2d=is_2d)
        
        condensate_volumes.append(cond_vol)
        wod_volumes.append(wod_vol)
        volume_ratios.append(pair.volume_ratio)
        radius_ratios.append(pair.radius_ratio)
    
    # Determine unit
    if use_physical_units:
        unit = 'µm²' if is_2d else 'µm³'
    else:
        unit = 'pixels²' if is_2d else 'pixels³'
    
    results = {
        'condensate_volumes': np.array(condensate_volumes),
        'wod_volumes': np.array(wod_volumes),
        'volume_ratios': np.array(volume_ratios),
        'radius_ratios': np.array(radius_ratios),
        'mean_volume_ratio': np.mean(volume_ratios) if volume_ratios else 0,
        'std_volume_ratio': np.std(volume_ratios) if volume_ratios else 0,
        'mean_radius_ratio': np.mean(radius_ratios) if radius_ratios else 0,
        'std_radius_ratio': np.std(radius_ratios) if radius_ratios else 0,
        'unit': unit,
        'n_pairs': len(pairs)
    }
    
    return results


def complete_paired_analysis_fast(condensate_labels: np.ndarray,
                                   wod_labels: np.ndarray,
                                   pixel_size: Optional[Dict] = None,
                                   method: str = 'contour',
                                   is_2d: bool = True) -> Tuple[List[CondensateDropletPair], Dict]:
    """
    Fast version of paired analysis - no visualization, optimized pairing.
    
    Parameters:
    -----------
    condensate_labels : ndarray
        Labeled segmentation of condensates
    wod_labels : ndarray
        Labeled segmentation of water-in-oil droplets
    pixel_size : dict, optional
        Physical pixel size with 'x_um' and 'y_um' keys
    method : str, default='area'
        Circle fitting method (area is fastest)
    is_2d : bool, default=True
        2D or 3D analysis
        
    Returns:
    --------
    pairs : list of CondensateDropletPair
    results : dict
        Volume analysis results
    """
    print(f"Fitting circles to condensates using '{method}' method...")
    cond_circles = fit_circles_robust(condensate_labels, methods=[method, 'area', 'simple'])
    print(f"  Found {len(cond_circles)} condensates")
    
    print(f"\nFitting circles to WODs using '{method}' method...")
    wod_circles = fit_circles_robust(wod_labels, methods=[method, 'area', 'simple'])
    print(f"  Found {len(wod_circles)} WODs")
    
    # Vectorized pairing
    print("\nPairing condensates with WODs...")
    pairs = pair_condensates_with_droplets_vectorized(cond_circles, wod_circles, pixel_size)
    print(f"  Found {len(pairs)} pairs")
    
    # Analyze volumes
    use_physical = pixel_size is not None
    results = analyze_paired_volumes(pairs, is_2d=is_2d, 
                                    use_physical_units=use_physical)
    
    return pairs, results


def batch_process_files(file_list: List[str],
                       condensate_func,
                       wod_func,
                       pixel_size: Optional[Dict] = None,
                       method: str = 'contour',
                       show_progress: bool = True) -> List[Dict]:
    """
    Process multiple files sequentially with progress tracking.
    
    Parameters:
    -----------
    file_list : list of str
        List of file paths
    condensate_func : callable
        Function to process condensates: func(path, ch=0) -> labels
    wod_func : callable
        Function to process WODs: func(path, ch=1) -> labels
    pixel_size : dict, optional
        Physical pixel size with 'x_um' and 'y_um' keys
    method : str, default='area'
        Circle fitting method ('area', 'simple')
    show_progress : bool
        Show progress bar
        
    Returns:
    --------
    results_list : list of dict
        Results for each file
    """
    results_list = []
    
    iterator = tqdm(file_list, desc="Processing files") if show_progress else file_list
    
    for file_path in iterator:
        print(f"Processing {file_path}...")
        try:
            # Process both channels
            condensate_labels = condensate_func(file_path, ch=0)
            wod_labels = wod_func(file_path, ch=1)
            
            # Analyze pairs
            pairs, results = complete_paired_analysis_fast(
                condensate_labels, wod_labels, 
                pixel_size=pixel_size, method=method, is_2d=True
            )
            
            # Store results
            results['file'] = os.path.basename(file_path)
            results['n_condensates'] = len(np.unique(condensate_labels)) - 1
            results['n_wods'] = len(np.unique(wod_labels)) - 1
            results['pairs'] = pairs
            
            results_list.append(results)
            
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            results_list.append({
                'file': os.path.basename(file_path),
                'error': str(e)
            })
    
    return results_list


def export_batch_results_to_csv(results_list: List[Dict], 
                                 output_file: str,
                                 is_2d: bool = True):
    """
    Export batch processing results to CSV.
    
    Parameters:
    -----------
    results_list : list of dict
        Results from batch_process_files
    output_file : str
        Output CSV file path
    is_2d : bool
        Whether volumes are 2D (area) or 3D
    """
    import pandas as pd
    
    data = []
    
    for result in results_list:
        if 'error' in result:
            continue
            
        file_name = result['file']
        
        if result.get('n_pairs', 0) > 0:
            # Add each pair as a row
            for pair in result.get('pairs', []):
                # Calculate volumes using the pair's methods
                if pair.pixel_size is not None:
                    cond_vol, wod_vol = pair.get_volume_physical(is_2d=is_2d)
                else:
                    cond_vol, wod_vol = pair.get_volume_pixels(is_2d=is_2d)
                
                data.append({
                    'file': file_name,
                    'condensate_id': pair.condensate_id,
                    'wod_id': pair.wod_id,
                    'condensate_radius': pair.condensate_radius,
                    'wod_radius': pair.wod_radius,
                    'condensate_volume': cond_vol,
                    'wod_volume': wod_vol,
                    'volume_ratio': pair.volume_ratio,
                    'radius_ratio': pair.radius_ratio,
                    'distance': pair.distance
                })
        else:
            # Add file with no pairs
            data.append({
                'file': file_name,
                'condensate_id': np.nan,
                'wod_id': np.nan,
                'condensate_radius': np.nan,
                'wod_radius': np.nan,
                'condensate_volume': np.nan,
                'wod_volume': np.nan,
                'volume_ratio': np.nan,
                'radius_ratio': np.nan,
                'distance': np.nan
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} rows to {output_file}")
    
    return df


def print_batch_summary(results_list: List[Dict]):
    """
    Print summary of batch processing results.
    
    Parameters:
    -----------
    results_list : list of dict
        Results from batch_process_files
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    
    total_files = len(results_list)
    successful = sum(1 for r in results_list if 'error' not in r)
    failed = total_files - successful
    
    print(f"\nTotal files: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        # Aggregate statistics
        all_volume_ratios = []
        all_condensate_vols = []
        all_wod_vols = []
        total_pairs = 0
        
        for result in results_list:
            if 'error' not in result and result.get('n_pairs', 0) > 0:
                all_volume_ratios.extend(result['volume_ratios'])
                all_condensate_vols.extend(result['condensate_volumes'])
                all_wod_vols.extend(result['wod_volumes'])
                total_pairs += result['n_pairs']
        
        if total_pairs > 0:
            print(f"\nTotal pairs found: {total_pairs}")
            print(f"\nAggregate Statistics:")
            print(f"  Volume ratio: {np.mean(all_volume_ratios):.4f} ± {np.std(all_volume_ratios):.4f}")
            print(f"  Condensate volume: {np.mean(all_condensate_vols):.2f} ± {np.std(all_condensate_vols):.2f}")
            print(f"  WOD volume: {np.mean(all_wod_vols):.2f} ± {np.std(all_wod_vols):.2f}")
    
    print("="*60)


# Simple wrapper for easier usage
def process_file_pair(file_path, condensate_func, wod_func, 
                     pixel_size=None, method='area', is_2d=True):
    """
    Process a single file and return results.
    
    Parameters:
    -----------
    file_path : str
        Path to CZI file
    condensate_func : callable
        Function to process condensates
    wod_func : callable
        Function to process WODs
    pixel_size : dict, optional
        Physical pixel size
    method : str
        Circle fitting method
    is_2d : bool
        2D or 3D analysis
        
    Returns:
    --------
    pairs : list
        List of pairs
    results : dict
        Volume analysis results
    """
    condensate_labels = condensate_func(file_path, ch=0)
    wod_labels = wod_func(file_path, ch=1)
    
    pairs, results = complete_paired_analysis_fast(
        condensate_labels, wod_labels,
        pixel_size=pixel_size,
        method=method,
        is_2d=is_2d
    )
    
    return pairs, results
