"""
line_profile.py

Script defines the LineProfile class, which contains line profile calculations across
a particular image of gafchromic film. The line profile is determined, preprocessed and then
segmented into individual Peak objects, for further analysis within those objects.

Written by Nathan Crossley, 2025.

"""

import numpy as np

from scipy import signal
from skimage.measure import profile_line

from src.image_processing.entities.xy import XY
from src.image_processing.entities.peak import Peak
from src.image_processing.connected_peaks_model.connected_peaks_model import (
    ConnectedPeaksModel,
)


class LineProfile:
    """Class representing line profile calculations across a particular image.
    Line profile is expected to have distinct peaks corresponding to irradiated
    slices during CT quality assurance testing."""

    def __init__(
        self, start_point: list | tuple, end_point: list | tuple, ref_image: np.ndarray
    ):
        """Initialises LineProfile class.
        A line profile is calculated based on start and end coordinates, along
        with a reference image. A connnected peaks model is fitted to this profile
        for noise reduction and simplification. From this profile, a list of custom
        Peak objects is created, representing the detected peaks in the fitted profile.

        Args:
            start_point (list | tuple): Start point of line profile.
            end_point (list | tuple): End point of line profile.
            ref_image (np.ndarray): Image to take line profile across.
        """

        # store passed args as instance attributes for convenience
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.ref_image = ref_image

        # get line profile between start and end points, using reference image and store as XY object
        profile_y = profile_line(self.ref_image, start_point[::-1], end_point[::-1], 1)
        self.profile = XY(np.arange(len(profile_y)), profile_y)

        # apply median and savgol filters for smoothing purposes and invert to turn
        # irradiated slices from minima into maxima (as this is more intuitive)
        self.profile = self.filter_and_invert(self.profile)

        # fit connected peaks model to profile, for accurate smoothing.
        self.fitted_profile = ConnectedPeaksModel.run(self.profile)

        # split fitted profile into Peak objects, for further individual analysis.
        self.peaks = self.split_into_peaks(self.fitted_profile)

    @staticmethod
    def filter_and_invert(profile: XY) -> XY:
        """Applies median and savgol filters to profile and inverts it.
        This applies basic smoothing to the profile and turns irradiated slices
        from minima into maxima, as this is more intuitive.

        Args:
            profile (XY): Input profile to smooth and invert.

        Returns:
            XY: Smoothed and inverted profile.
        """

        # dynamically define kernel size (must be odd for convolution)
        k = len(profile.y) // 50
        k += (k + 1) % 2

        # apply median filter and savgol cubic fitting
        profile.y = signal.medfilt(
            profile.y,
            k,
        )
        profile.y = signal.savgol_filter(profile.y, int(len(profile.y) / 50), 3)

        # invert and zero profile
        profile.y -= np.max(profile.y)
        profile.y *= -1

        return profile

    @staticmethod
    def split_into_peaks(profile: XY) -> list["Peak"]:
        """Splits line profile into Peak objects for further analysis.
        This segmentation is based on trough locations in profile.

        Args:
            profile (XY): Input line profile to split into peaks.

        Returns:
            list[Peak]: List of Peak objects.
        """

        # get location of troughs within profile, with threshold prominence
        troughs, props = signal.find_peaks(
            -profile.y,
            height=float(np.min(-profile.y)),
            prominence=float(np.ptp(profile.y) / 4),
        )

        # create list of x indices for profile segmentation (start and end included)
        split_indices = sorted([0, len(profile.y)] + troughs.tolist())

        # segment profile based on pairs of indices.
        segments = [
            profile[:, start:end]
            for start, end in zip(split_indices, split_indices[1:])
        ]

        # create a list of Peak objects, containing an object for each segment
        peaks = [Peak(segment.x, segment.y) for segment in segments]

        return peaks
