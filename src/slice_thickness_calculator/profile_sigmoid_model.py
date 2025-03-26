import itertools
import warnings
import numpy as np

from scipy import signal
from scipy import optimize

from xy_tools import XYUtils, XY
from utils import find_highest_peak


class ProfileSigmoidModel:
    """Class for modelling a profile with blended sigmoids."""

    @staticmethod
    def run(profile: XY) -> XY:
        """Models an XY object using sigmoids and blends them together, returning the fitted profile.

        Args:
            profile (XY): Input profile fit piecewise sigmoid model.

        Returns:
            XY: Fitted profile using piecewise sigmoid model.
        """
        troughs, props = troughs, props = signal.find_peaks(
            -profile.y,
            height=float(np.min(-profile.y)),
            prominence=float(np.ptp(profile.y) / 4),
        )

        split_indices = sorted([0, len(profile.y)] + troughs.tolist())
        segments = [
            profile[:, start:end]
            for start, end in zip(split_indices, split_indices[1:])
        ]

        split_indices = [find_highest_peak(seg.y) for seg in segments]
        sub_segments = [
            [seg[:, :idx], seg[:, idx:]] for idx, seg in zip(split_indices, segments)
        ]
        sub_segments = list(itertools.chain(*sub_segments))

        sigmoids = [Sigmoid(sub_seg.x, sub_seg.y) for sub_seg in sub_segments]
        peaks_modelled = [
            BlendedProfile(sig1, sig2)
            for sig1, sig2 in zip(sigmoids[::2], sigmoids[1::2])
        ]
        fitted_profile = PeakBlender(peaks_modelled).blended_profile.profile

        return fitted_profile


class Sigmoid(XY):
    """Class for fitting a sigmoid to an XY object."""

    def __init__(self, *args: list[int | float]):
        """Initialises Sigmoid class."""
        self.popt = self.get_popt()

    def get_popt(self) -> list[float]:
        """Gets best fit parameters for sigmoid fitting.

        Returns:
            list[float]: List of best fit parameters for sigmoid fitting.
        """
        p0 = self.get_initial_guesses()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                popt, _ = optimize.curve_fit(
                    self.functional_form, self.x, self.y, p0=p0
                )
            except RuntimeError:
                print(
                    f"User warning: Runtime error in sigmoid fitting. Using initial guesses."
                )
                popt = p0
        return popt

    def get_initial_guesses(self) -> list[float]:
        """Gets initial guesses for sigmoid fitting.

        Returns:
            list[float]: List of initial guesses for sigmoid fitting.
        """
        A = np.ptp(self.y)
        b = np.min(self.y)
        abs_deriv = np.abs(np.diff(self.y) / np.diff(self.x))
        abs_grad_max = np.max(abs_deriv)
        k = np.sign(self.y[-1] - self.y[0]) * abs_grad_max * 0.5
        x0 = self.x[np.argmax(abs_deriv)]
        p0 = [A, k, x0, b]
        return p0

    @staticmethod
    def functional_form(
        x: list[float] | float, A: float, k: float, x0: float, b: float
    ) -> list[float] | float:
        """Evaluates the function form of the sigmoid for either an input x-array or a single x value.

        Args:
            x (list[float] | float): Input x-array or single x value.
            A (float): Amplitude parameter for sigmoid model.
            k (float): Steepness parameter for sigmoid model.
            x0 (float): X-centre parameter for sigmoid model.
            b (float): y-offset parameter for sigmoid model.

        Returns:
            list[float] | float: y-array or single y value for input x.
        """
        exp_term = np.exp(-k * (x - x0))
        return A / (1 + exp_term) + b

    def evaluate(self, x: list) -> XY:
        """Evaluates the sigmoid model for an input x-array.

        Args:
            x (list): Input x-array.

        Returns:
            XY: Output XY object for input x-array using fitted sigmoid model.
        """
        return XY(x, self.functional_form(x, *self.popt))


class BlendedProfile:
    """Class for blending two profiles together that have joined x-ranges."""

    def __init__(self, sig_L: Sigmoid, sig_R: Sigmoid, profile: XY = None):
        """Initialises BlendedProfile class

        Args:
            sig_L (Sigmoid): Functional form for the left-most end of the blended profile.
            sig_R (Sigmoid): Functional form for the right-most end of the blended profile.
            profile (XY, optional): XY object for if blended profile is more complex than two sigmoids. Defaults to None.
        """
        self.sig_L = sig_L
        self.sig_R = sig_R
        self.profile = profile if profile is not None else self.blend_sigmoids()

    def blend_sigmoids(self) -> XY:
        """Blends two basic sigmoids together to form a blended profile within total x-range.

        Returns:
            XY: Blended profile from two sigmoid objects.
        """
        x_range = XYUtils.get_x_range(self.sig_L, self.sig_R)

        sig_L_extrap = self.sig_L.evaluate(x_range)
        sig_R_extrap = self.sig_R.evaluate(x_range)

        transition_x = (self.sig_L.x[-1] + self.sig_R.x[0]) / 2
        transition_width = (
            1
            / 10
            * (
                abs(transition_x - self.sig_L.popt[2])
                + abs(transition_x - self.sig_R.popt[2])
            )
        )

        blending_weight = 1 / (
            1 + np.exp([-(x - transition_x) / transition_width for x in x_range])
        )
        blended = XY(
            x_range,
            (1 - blending_weight) * sig_L_extrap.y + blending_weight * sig_R_extrap.y,
        )

        return blended

    def extrapolate(self, extrap_range: list[float]) -> XY:
        """Extrapolates the blended profile within extrap_range using sigmoids defined at edges of known x-range.

        Args:
            extrap_range (list[float]): X-range to extrapolate the blended profile to fit.

        Returns:
            XY: Extrapolated profile within extrap_range.
        """
        x_range = XYUtils.get_x_range(self.sig_L, self.sig_R)

        left_extrap = self.sig_L.evaluate(np.arange(extrap_range[0], x_range[0]))
        right_extrap = self.sig_R.evaluate(
            np.arange(x_range[-1] + 1, extrap_range[-1] + 1)
        )

        x_combined = np.concatenate([left_extrap.x, self.profile.x, right_extrap.x])
        y_combined = np.concatenate([left_extrap.y, self.profile.y, right_extrap.y])

        return XY(x_combined, y_combined)


class PeakBlender:
    """Class for blending multiple peaks together."""

    def __init__(self, peaks_modelled: list[BlendedProfile]):
        """Initialises PeakBlender class and blends peaks together.

        Args:
            peaks_modelled (list[BlendedProfile]): Peaks to blend.
        """
        self.peaks_modelled = peaks_modelled

        if len(self.peaks_modelled) > 1:
            self.blended_profile = self.blend_peaks(*self.peaks_modelled[:2])
        else:
            self.blended_profile = self.peaks_modelled[0]

        if len(self.peaks_modelled) > 2:
            for peak_to_blend in self.peaks_modelled[2:]:
                self.blend_in_peak(peak_to_blend)

    def blend_peaks(
        self, peak1: BlendedProfile, peak2: BlendedProfile
    ) -> BlendedProfile:
        """Blends two peaks together that have joined x-ranges.

        Args:
            peak1 (BlendedProfile): First peak to blend.
            peak2 (BlendedProfile): Second peak to blend.

        Returns:
            BlendedProfile: Blended profile from two peaks.
        """
        x_range = XYUtils.get_x_range(peak1.profile, peak2.profile)
        peak1_extrap = peak1.extrapolate(x_range)
        peak2_extrap = peak2.extrapolate(x_range)

        transition_x = (peak1.profile.x[-1] + peak2.profile.x[0]) / 2
        transition_width = (
            1
            / 10
            * (
                abs(transition_x - peak1.sig_R.popt[2])
                + abs(transition_x - peak2.sig_L.popt[2])
            )
        )

        blending_weight = 1 / (
            1 + np.exp([-(x - transition_x) / transition_width for x in x_range])
        )
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

        self.blended_profile = self.blend_peaks(self.blended_profile, peak_to_blend)
