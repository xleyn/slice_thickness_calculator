"""
blended_profile.py

Script defining BlendedProfile, a class responsible for storing a profile that is derived
from two or more profiles blended together. Any profiles that have been used to form this profile
through blending should have been well defined by analytical sigmoids. As such, the resultant blended
profile should be well defined by analytical sigmoids at its left-most and right-most edges. This is
useful in case extrapolation is required for further blending.

Written by Nathan Crossley, 2025.

"""

import numpy as np

from src.image_processing.entities.xy import XY
from src.image_processing.connected_peaks_model.sigmoid import Sigmoid


class BlendedProfile:
    """Class storing a profile that is derived from two or more
    profiles blended together. The profiles used for blending should have adjoining
    x-ranges and should be well defined analytically by sigmoids at the left
    and right boundaries.
    """

    def __init__(self, sig_L: Sigmoid, sig_R: Sigmoid, profile: XY = None):
        """Initialises BlendedProfile class. Sigmoids should be passed
        to analytically define the left and right-most ends of the blended profile,
        for any required blending now or in the future. If a pre-existing profile
        is passed, this is stored as the profile instance attribute, otherwise this profile
        is derived by blending the left and right sigmoids within a shared x-range.

        Args:
            sig_L (Sigmoid): Functional form for the left-most end of the blended profile.
            sig_R (Sigmoid): Functional form for the right-most end of the blended profile.
            profile (XY, optional): XY object for if blended profile is more complex than two sigmoids. Defaults to None.
        """

        # store analytically defined simgoids for left and right-most ends of profile
        self.sig_L = sig_L
        self.sig_R = sig_R

        # if profile already defined, store it, else determine it by blending sigmoids together.
        self.profile = profile if profile is not None else self.blend_sigmoids()

    def blend_sigmoids(self) -> XY:
        """Blends two basic sigmoids together to form a blended profile within
        combined x-range. Necessary if profile not passed in initialiser.

        Returns:
            XY: Blended profile from two sigmoid objects.
        """

        # get total x-range spanned by sigmoids using XY data used for fitting
        x_range = self.sig_L.get_combined_x_range(self.sig_R)

        # evaluate both sigmoids across total combined x-range
        sig_L_extrap = self.sig_L.evaluate(x_range)
        sig_R_extrap = self.sig_R.evaluate(x_range)

        # calculate x value at which transition must occur between sigmoids
        transition_x = (self.sig_L.x[-1] + self.sig_R.x[0]) / 2

        # define approximate width across which transition occurs
        transition_width = (
            1
            / 10
            * (
                abs(transition_x - self.sig_L.popt[2])
                + abs(transition_x - self.sig_R.popt[2])
            )
        )

        # define a blending weight sigmoid across x range, used to blend sigmoids together
        blending_weight = 1 / (
            1 + np.exp([-(x - transition_x) / transition_width for x in x_range])
        )

        # blend sigmoids together using product of blending weights and sigmoids evaluated across combined x-range
        blended = XY(
            x_range,
            (1 - blending_weight) * sig_L_extrap.y + blending_weight * sig_R_extrap.y,
        )

        return blended

    def extrapolate(self, extrap_range: list[float]) -> XY:
        """Extrapolates the blended profile within a passed extrapolation range
        using sigmoids defined at edges of known x-range.

        Args:
            extrap_range (list[float]): X-range to extrapolate the blended profile to fit.

        Returns:
            XY: Extrapolated profile within extrap_range.
        """

        # get x range within which the blended profile currently spans.
        x_range = self.sig_L.get_combined_x_range(self.sig_R)

        # evaluate extrapolated region to left of known profile, using left sigmoid
        left_extrap = self.sig_L.evaluate(np.arange(extrap_range[0], x_range[0]))

        # evaluate extrapolated region to right of known profile, using right sigmoid
        right_extrap = self.sig_R.evaluate(
            np.arange(x_range[-1] + 1, extrap_range[-1] + 1)
        )

        # combine extrapolated regions with known profile to get combined extrapolated profile.
        x_combined = np.concatenate([left_extrap.x, self.profile.x, right_extrap.x])
        y_combined = np.concatenate([left_extrap.y, self.profile.y, right_extrap.y])

        return XY(x_combined, y_combined)
