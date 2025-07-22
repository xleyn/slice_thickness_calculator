"""
peak.py

Script defines subclass of XY, the Peak class, to store profile information for an individual peak.
During initialisation, the global peak of the profile is detected, along with the peak FWHM and an
associated binarised profile.

Written by Nathan Crossley, 2025.

"""

import numpy as np

from src.image_processing.utils import find_highest_peak
from src.image_processing.entities.xy import XY


class Peak(XY):
    """Subclass of XY, that stores a profile information for an
    individual peak. After instantiation, the true peak is detected,
    the peak FWHM calculated and a binarised profile derived.
    """

    def __init__(self, *args: list[int | float]):
        """Initialises Peak object, detecting highest peak,
        calculating peak FWHM and determining binarised profile.
        """

        # find index of highest peak and associated height
        self.peak_idx = find_highest_peak(self.y)
        self.peak_y = self.y[self.peak_idx]

        # find half heights for left and right halfs of peak, based on peak height and background heights
        self.half_height_left = self.y[0] + (self.peak_y - self.y[0]) / 2
        self.half_height_right = self.y[-1] + (self.peak_y - self.y[-1]) / 2

        # get FWHM for peak and associated crossing indices for left and right sides of profile
        self.FWHM, self.crossing_x_left, self.crossing_x_right = self.get_FWHM()

        # get binarised version of peak based on whether profile is above thresh height.
        self.binary_profile = self.get_FWHM_binary_profile()

    def get_FWHM(self) -> tuple[float, int, int]:
        """Gets the FWHM of the peak using linear interpolation on left and right
        half heights.

        Returns:
            tuple[float, int, int]: FWHM of peak and left and right crossing indices.
        """

        # interpolate crossing index corresponding to left half height
        crossing_x_left = np.interp(
            self.half_height_left, self.y[: self.peak_idx], self.x[: self.peak_idx]
        )

        # interpolate crossing index corresponding to right half height
        crossing_x_right = np.interp(
            self.half_height_right,
            self.y[self.peak_idx :][::-1],
            self.x[self.peak_idx :][::-1],
        )

        # FWHM is crossing indices subtracted
        FWHM = crossing_x_right - crossing_x_left

        return FWHM, crossing_x_left, crossing_x_right

    def get_FWHM_binary_profile(self) -> XY:
        """Returns a binary profile, where each y-value is either set
        to background level or peak height, depending on position of
        x index relative to crossing indices

        Returns:
            XY: binary profile to represent FWHM calculations visually.
        """

        # get copy of self to be binary profile
        binary_profile = self.copy()

        # where x idx between crossing indices, set to peak value, else
        # set to relevant background level
        binary_profile.y = np.where(
            binary_profile.x <= self.crossing_x_left,
            self.y[0],
            np.where(
                (binary_profile.x > self.crossing_x_left)
                & (binary_profile.x <= self.crossing_x_right),
                self.peak_y,
                np.where(binary_profile.x > self.crossing_x_right, self.y[-1], 0),
            ),
        )

        return binary_profile
