"""
Ellipse analysis utilities for tracking and fitting aspect ratio decay.

This module provides functions for:
- Generating rotated ellipse polygons with axis endpoints
- Tracking aspect ratios of labeled objects over time
- Fitting exponential decay of aspect ratios (relaxation time measurement)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.measure import regionprops


def ellipse_polygon(cy, cx, r_major, r_minor, angle, n_points=50,
                    return_axes=False):
    """Generate rotated ellipse as polygon points, optionally with axis endpoints.

    Parameters
    ----------
    cy, cx : float
        Center coordinates (row, col).
    r_major, r_minor : float
        Semi-major and semi-minor axis lengths.
    angle : float
        Orientation angle in radians (regionprops convention).
    n_points : int
        Number of polygon vertices.
    return_axes : bool
        If True, also return major and minor axis endpoints.

    Returns
    -------
    polygon : np.ndarray, shape (n_points, 2)
        Ellipse outline as (y, x) coordinates.
    major_axis : np.ndarray, shape (2, 2), only if return_axes=True
        Start and end points of the major axis [[y0,x0],[y1,x1]].
    minor_axis : np.ndarray, shape (2, 2), only if return_axes=True
        Start and end points of the minor axis [[y0,x0],[y1,x1]].
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    theta = np.linspace(0, 2 * np.pi, n_points)
    y_local = r_major * np.cos(theta)
    x_local = r_minor * np.sin(theta)
    y_rot = cos_a * y_local - sin_a * x_local + cy
    x_rot = sin_a * y_local + cos_a * x_local + cx
    polygon = np.column_stack([y_rot, x_rot])

    if not return_axes:
        return polygon

    major_axis = np.array([
        [cy - cos_a * r_major, cx - sin_a * r_major],
        [cy + cos_a * r_major, cx + sin_a * r_major],
    ])
    minor_axis = np.array([
        [cy + sin_a * r_minor, cx - cos_a * r_minor],
        [cy - sin_a * r_minor, cx + cos_a * r_minor],
    ])
    return polygon, major_axis, minor_axis


def linear(t, log_A, inv_tau):
    """Linear model for log(AR - 1) decay: log_A - inv_tau * t."""
    return log_A - inv_tau * t


def track_aspect_ratios(label_stacks, label_ids, t_starts, t_len, dx,
                        char_l_timepoint=1):
    """
    Track aspect ratios of labeled objects over time across multiple image stacks.

    Parameters
    ----------
    label_stacks : list of np.ndarray
        Label images per scene, each with shape (T, H, W).
    label_ids : list of list/array
        Object IDs to track, per scene.
    t_starts : list of list/array
        Start timepoint for each object, per scene.
    t_len : int
        Number of timepoints to track from each t_start.
    dx : float
        Pixel size for converting characteristic length to physical units.
    char_l_timepoint : int
        Which dt offset to use for characteristic length (default 1 = second frame).

    Returns
    -------
    aspect_ratios : np.ndarray, shape (t_len, n_images, max_labels)
    ellipse_data : list of tuples (t, cy, cx, r_major, r_minor, angle, image_idx, label_id)
    char_lengths : np.ndarray, shape (n_images, max_labels)
    """
    n_images = len(label_ids)
    max_labels = max(len(ids) for ids in label_ids)

    aspect_ratios = np.full((t_len, n_images, max_labels), np.nan, dtype=np.float32)
    ellipse_data = []
    char_lengths = np.full((n_images, max_labels), np.nan, dtype=np.float32)

    for j, (ids, starts) in enumerate(zip(label_ids, t_starts)):
        for i, (label_id, t_start) in enumerate(zip(ids, starts)):
            if np.isnan(label_id):
                continue
            for dt in range(t_len):
                t = t_start + dt
                if t >= len(label_stacks[j]):
                    print(f'scene {j}, label {label_id}: abrupt end at t={t}')
                    break

                mask = label_stacks[j][int(t)] == label_id
                if mask.sum() == 0:
                    print(f'scene {j}, label {label_id}: empty mask at t={t}')
                    continue

                prop = regionprops(mask.astype(np.uint8))[0]
                cy, cx = prop.centroid
                r_major = prop.major_axis_length / 2
                r_minor = prop.minor_axis_length / 2
                angle = prop.orientation

                aspect_ratios[dt, j, i] = r_major / r_minor if r_minor > 0 else np.nan
                ellipse_data.append((t, cy, cx, r_major, r_minor, angle, j, label_id))

                if dt == char_l_timepoint:
                    char_lengths[j, i] = (r_major + r_minor) * dx

    return aspect_ratios, ellipse_data, char_lengths


def _fit_ar_decay(y, times):
    """Fit exponential decay of aspect ratio: log(AR - 1) = log_A - inv_tau * t.

    Parameters
    ----------
    y : np.ndarray, shape (n_timepoints,)
        Aspect ratio values (should be > 1 for valid fit).
    times : np.ndarray, shape (n_timepoints,)
        Time values in seconds.

    Returns
    -------
    inv_tau : float
        Inverse relaxation time (1/s). NaN if fit fails.
    popt : tuple or None
        Fit parameters (log_A, inv_tau), or None if fit fails.
    """
    if np.any(np.isnan(y)) or np.any(y <= 1):
        return np.nan, None
    try:
        popt, _ = curve_fit(linear, times, np.log(y - 1))
        return popt[1], popt
    except Exception:
        return np.nan, None


def fit_single_label(aspect_ratios, scene_idx, label_idx, delta_t,
                     plot=True, axes=None, plot_log=True, label=None,
                     color=None):
    """Fit aspect ratio decay for a single label, optionally plotting.

    Parameters
    ----------
    aspect_ratios : np.ndarray, shape (t_len, n_images, max_labels)
    scene_idx : int
        Scene (image) index.
    label_idx : int
        Label index within the scene.
    delta_t : float
        Time between frames in seconds.
    plot : bool
        If True, plot AR vs time (and log if plot_log=True) with fit.
    axes : matplotlib Axes or array of 2 Axes, optional
        Single Axes for AR-only, or array of 2 for AR + log.
        If None and plot=True, creates new figure.
    plot_log : bool
        If True, plot log(AR-1) on a second axes panel.
    label : str, optional
        Legend label for this curve.
    color : str, optional
        Color for data and fit lines. None uses matplotlib default cycle.

    Returns
    -------
    inv_tau : float
    popt : tuple or None
    """
    y = aspect_ratios[1:, scene_idx, label_idx]
    times = np.arange(len(y)) * delta_t
    inv_tau, popt = _fit_ar_decay(y, times)

    if plot:
        if axes is None:
            if plot_log:
                _, axes = plt.subplots(1, 2, figsize=(10, 4))
            else:
                _, axes = plt.subplots(figsize=(6, 4))

        fit_color = 'grey' if color is None else color
        ax_ar = axes[0] if plot_log else axes
        ax_ar.plot(times, y, 'o-', alpha=0.8, label=label, color=color)
        ax_ar.set_xlabel('time (s)')
        ax_ar.set_ylabel('aspect ratio')

        if popt is not None:
            t_fit = np.linspace(times[0], times[-1], 50)
            ax_ar.plot(t_fit, np.exp(linear(t_fit, *popt)) + 1, '--', c=fit_color, alpha=0.4)

        if plot_log:
            log_y = np.log(y - 1) if np.all(y > 1) else np.full_like(y, np.nan)
            axes[1].plot(times, log_y, 'o-', alpha=0.8, label=label, color=color)
            axes[1].set_xlabel('time (s)')
            axes[1].set_ylabel('log(aspect ratio - 1)')
            if popt is not None:
                axes[1].plot(t_fit, linear(t_fit, *popt), '--', c=fit_color, alpha=0.4)

    return inv_tau, popt


def fit_aspect_ratio_bulk(aspect_ratios, delta_t, plot=False):
    """Fit aspect ratio decay for all scenes and labels.

    Parameters
    ----------
    aspect_ratios : np.ndarray, shape (t_len, n_images, max_labels)
    delta_t : float
        Time between frames in seconds.
    plot : bool
        If True, creates (n_scenes x 2) subplot grid showing all fits.

    Returns
    -------
    inv_taus : np.ndarray, shape (n_images, max_labels)
    """
    n_images = aspect_ratios.shape[1]
    max_labels = aspect_ratios.shape[2]
    inv_taus = np.full((n_images, max_labels), np.nan)

    if plot:
        fig, axs = plt.subplots(n_images, 2, figsize=(12, 4.5 * n_images))
        if n_images == 1:
            axs = axs[np.newaxis, :]

    for j in range(n_images):
        for i in range(max_labels):
            row_axes = axs[j] if plot else None
            inv_tau, _ = fit_single_label(
                aspect_ratios, scene_idx=j, label_idx=i, delta_t=delta_t,
                plot=plot, axes=row_axes
            )
            inv_taus[j, i] = inv_tau

    if plot:
        plt.tight_layout()

    print("tau values (s):")
    print(1 / inv_taus)

    return inv_taus
