"""
connected_peaks_model.py

This script defines the ConnectedPeaksModel class. This class is responsible for
fitting a custom blended sigmoid model to the passed profile, for the fitting and
smoothing purposes. Input profile is expected to be a line profile across a piece
of Gafchromic film in CT quality assurance testing.

Written by Nathan Crossley, 2025.

"""

import itertools
import numpy as np
from scipy import signal

from src.image_processing.utils import find_highest_peak
from src.image_processing.entities.xy import XY
from src.image_processing.connected_peaks_model.blended_profile import BlendedProfile
from src.image_processing.connected_peaks_model.sigmoid import Sigmoid
from src.image_processing.connected_peaks_model.blending_task import BlendingTask


class ConnectedPeaksModel:
    """Class for modelling a particular profile with a 'connected peaks model'.
    This is a model that is fitted to the raw profile, and is derived from blending
    analytical sigmoids fitted across individual parts of the profile."""

    @staticmethod
    def run(profile: XY) -> XY:
        """Models an XY object using a "connected peaks model". This fits individual
        sigmoids across different parts of the profile and blends them together,
        to form a modelled profile. Returns the fitted profile.

        Args:
            profile (XY): Input profile to fit connected peaks model to.

        Returns:
            XY: Fitted profile using connected peaks model.
        """

        # detect profile troughs using threshold height and prominence
        troughs, props = signal.find_peaks(
            -profile.y,
            height=float(np.min(-profile.y)),
            prominence=float(np.ptp(profile.y) / 4),
            distance=len(profile.y) // 30,
        )

        # define indices to split profile by troughs (including start and end)
        split_indices = sorted([0, len(profile.y)] + troughs.tolist())

        # segment profile by troughs by applying split indices
        segments = [
            profile[:, start:end]
            for start, end in zip(split_indices, split_indices[1:])
        ]

        # defines indices to split individual segments into two based on global peak location
        split_indices = [find_highest_peak(seg.y) for seg in segments]

        # split segments into sub-segments based on global peaks within segments
        sub_segments = [
            [seg[:, :idx], seg[:, idx:]] for idx, seg in zip(split_indices, segments)
        ]
        sub_segments = list(itertools.chain(*sub_segments))

        # fit an analytical sigmoid to each sub-segment
        sigmoids = [Sigmoid(sub_seg.x, sub_seg.y) for sub_seg in sub_segments]

        # for pairs of sub-segments, blend sigmoid profiles together to model a "peak"
        peaks_modelled = [
            BlendedProfile(sig1, sig2)
            for sig1, sig2 in zip(sigmoids[::2], sigmoids[1::2])
        ]

        # blend all modelled peaks together into one profile
        fitted_profile = BlendingTask(peaks_modelled).blended_profile.profile

        # return profile to caller
        return fitted_profile
