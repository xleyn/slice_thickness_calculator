"""
blending_task.py

Script defines BlendingTask class, which is reponsible for the task of blending multiple
peaks together (each of type BlendedProfile) using the sigmoids analyticaly defined at their edges.

Written by Nathan Crossley, 2025.

"""

import numpy as np

from src.image_processing.entities.xy import XY
from src.image_processing.connected_peaks_model.blended_profile import BlendedProfile


class BlendingTask:
    """Class responsible for the task of blending multiple peaks together
    (each of type BlendedProfile), using the sigmoids analytically defined at their edges.
    peaks are blended by defining a sigmoid weighting function."""

    def __init__(self, peaks_modelled: list[BlendedProfile]):
        """Initialises BlendingTask class and creates blended
        profile by systematically blending peaks together.

        Args:
            peaks_modelled (list[BlendedProfile]): Peaks to blend together.
        """

        # store peaks as instance attr for convenience
        self.peaks_modelled = peaks_modelled

        # if there's more than one detected peak, blend the first two together as the blended profile
        if len(self.peaks_modelled) > 1:
            self.blended_profile = self.blend_peaks(*self.peaks_modelled[:2])

        # else store the sigle peak as the blended profile
        else:
            self.blended_profile = self.peaks_modelled[0]

        # if there are more than two peaks, blend in the extras to blended profile
        if len(self.peaks_modelled) > 2:
            for peak_to_blend in self.peaks_modelled[2:]:
                self.blend_in_peak(peak_to_blend)

    def blend_peaks(
        self, peak1: BlendedProfile, peak2: BlendedProfile
    ) -> BlendedProfile:
        """Blends two peaks together that have adjoined x-ranges.

        Args:
            peak1 (BlendedProfile): First peak to blend.
            peak2 (BlendedProfile): Second peak to blend.

        Returns:
            BlendedProfile: Blended profile from two peaks.
        """

        # get combined x-range of peaks
        x_range = peak1.profile.get_combined_x_range(peak2.profile)

        # extrapolate both peaks within full x range
        peak1_extrap = peak1.extrapolate(x_range)
        peak2_extrap = peak2.extrapolate(x_range)

        # define transition x and transition width for sigmoid blending weight function
        transition_x = (peak1.profile.x[-1] + peak2.profile.x[0]) / 2
        transition_width = (
            1
            / 10
            * (
                abs(transition_x - peak1.sig_R.popt[2])
                + abs(transition_x - peak2.sig_L.popt[2])
            )
        )

        # create blending weight sigmoid function across whole x range
        blending_weight = 1 / (
            1 + np.exp([-(x - transition_x) / transition_width for x in x_range])
        )

        # blend peaks together using extrapolated peaks and blending weight func
        blended = XY(
            x_range,
            (1 - blending_weight) * peak1_extrap.y + blending_weight * peak2_extrap.y,
        )

        return BlendedProfile(peak1.sig_L, peak2.sig_R, blended)

    def blend_in_peak(self, peak_to_blend: BlendedProfile):
        """Blends in a peak to the existing blended profile.

        Args:
            peak_to_blend (BlendedProfile): Peak to blend into existing blended profile.
        """

        # blend current blended profile with passed peak and reassign to original attr
        self.blended_profile = self.blend_peaks(self.blended_profile, peak_to_blend)
