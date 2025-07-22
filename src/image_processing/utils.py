"""
utils.py

Script containing general utility functions for image processing tasks.

Written by Nathan Crossley, 2025.

"""

from scipy import signal
import numpy as np


def find_highest_peak(y: list[float]) -> int:
    """Finds x index of highest peak in input y-array.

    Args:
        y (list[float]): y-array for peak detection.

    Returns:
        int: idx of highest peak in input y-array.
    """

    # find all peaks, specifying height to get height info in props
    peaks, props = signal.find_peaks(y, height=0)

    # extract x index of highest peak
    peak_idx = peaks[np.argmax(props["peak_heights"])]

    return peak_idx
