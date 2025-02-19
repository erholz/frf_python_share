#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 04:36:32 2024

@author: Nick
"""

import numba
import numpy as np

@numba.njit()
def interpolate_with_max_gap(orig_x,
                             orig_y,
                             target_x,
                             max_gap=np.inf,
                             orig_x_is_sorted=False,
                             target_x_is_sorted=False):
    """
    Interpolate data linearly with maximum gap. If there is
    larger gap in data than `max_gap`, the gap will be filled
    with np.nan.

    The input values should not contain NaNs.

    Parameters
    ---------
    orig_x: np.array
        The input x-data
    orig_y: np.array
        The input y-data
    target_x: np.array
        The output x-data; the data points in x-axis that
        you want the interpolation results from.
    max_gap: float
        The maximum allowable gap in `orig_x` inside which
        interpolation is still performed. Gaps larger than
        this will be filled with np.nan in the output `target_y`.
    orig_x_is_sorted: boolean, default: False
        If True, the input data `orig_x` is assumed to be monotonically
        increasing. Some performance gain if you supply sorted input data.
    target_x_is_sorted: boolean, default: False
        If True, the input data `target_x` is assumed to be 
        monotonically increasing. Some performance gain if you supply
        sorted input data.

    Returns
    ------
    target_y: np.array
        The interpolation results.
    """
    if not orig_x_is_sorted:
        # Sort to be monotonous wrt. input x-variable.
        idx = orig_x.argsort()
        orig_x = orig_x[idx]
        orig_y = orig_y[idx]

    if not target_x_is_sorted:
        target_idx = target_x.argsort()
        # Needed for sorting back the data.
        target_idx_for_reverse = target_idx.argsort()
        target_x = target_x[target_idx]

    target_y = np.empty(target_x.size)
    idx_orig = 0
    orig_gone_through = False

    for idx_target, x_new in enumerate(target_x):

        # Grow idx_orig if needed.
        while not orig_gone_through:

            if idx_orig + 1 >= len(orig_x):
                # Already consumed the orig_x; no more data
                # so we would need to extrapolate
                orig_gone_through = True
            elif x_new > orig_x[idx_orig + 1]:
                idx_orig += 1
            else:
                # x_new <= x2
                break

        if orig_gone_through:
            target_y[idx_target] = np.nan
            continue

        x1 = orig_x[idx_orig]
        y1 = orig_y[idx_orig]
        x2 = orig_x[idx_orig + 1]
        y2 = orig_y[idx_orig + 1]

        if x_new < x1:
            # would need to extrapolate to left
            target_y[idx_target] = np.nan
            continue

        delta_x = x2 - x1

        if delta_x > max_gap:
            target_y[idx_target] = np.nan
            continue

        delta_y = y2 - y1

        if delta_x == 0:
            target_y[idx_target] = np.nan
            continue

        k = delta_y / delta_x

        delta_x_new = x_new - x1
        delta_y_new = k * delta_x_new
        y_new = y1 + delta_y_new

        target_y[idx_target] = y_new

    if not target_x_is_sorted:
        return target_y[target_idx_for_reverse]
    return target_y