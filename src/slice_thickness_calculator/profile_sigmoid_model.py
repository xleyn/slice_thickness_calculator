import itertools
import warnings
import numpy as np

from scipy import signal
from scipy import optimize

from xy_tools import XYUtils, XY
from utils import find_highest_peak


class ProfileSigmoidModel:
    @staticmethod
    def run(profile: XY) -> XY:
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
    def __init__(self, *args: list[int | float]):
        self.popt = self.get_popt()

    def get_popt(self):
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

    def get_initial_guesses(self):
        A = np.ptp(self.y)
        b = np.min(self.y)
        abs_deriv = np.abs(np.diff(self.y) / np.diff(self.x))
        abs_grad_max = np.max(abs_deriv)
        k = np.sign(self.y[-1] - self.y[0]) * abs_grad_max * 0.5
        x0 = self.x[np.argmax(abs_deriv)]
        p0 = [A, k, x0, b]
        return p0

    @staticmethod
    def functional_form(x, A, k, x0, b):
        exp_term = np.exp(-k * (x - x0))
        return A / (1 + exp_term) + b

    def evaluate(self, x: list):
        return XY(x, self.functional_form(x, *self.popt))


class BlendedProfile:
    def __init__(self, sig_L: Sigmoid, sig_R: Sigmoid, profile: XY = None):
        self.sig_L = sig_L
        self.sig_R = sig_R
        self.profile = profile if profile is not None else self.blend_sigmoids()

    def blend_sigmoids(self):
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

    def extrapolate(self, extrap_range):
        x_range = XYUtils.get_x_range(self.sig_L, self.sig_R)

        left_extrap = self.sig_L.evaluate(np.arange(extrap_range[0], x_range[0]))
        right_extrap = self.sig_R.evaluate(
            np.arange(x_range[-1] + 1, extrap_range[-1] + 1)
        )

        x_combined = np.concatenate([left_extrap.x, self.profile.x, right_extrap.x])
        y_combined = np.concatenate([left_extrap.y, self.profile.y, right_extrap.y])

        return XY(x_combined, y_combined)


class PeakBlender:
    def __init__(self, peaks_modelled: list[BlendedProfile]):
        self.peaks_modelled = peaks_modelled

        if len(self.peaks_modelled) > 1:
            self.blended_profile = self.blend_peaks(*self.peaks_modelled[:2])
        else:
            self.blended_profile = self.peaks_modelled[0]

        if len(self.peaks_modelled) > 2:
            for peak_to_blend in self.peaks_modelled[2:]:
                self.blend_in_peak(peak_to_blend)

    def blend_peaks(self, peak1: BlendedProfile, peak2: BlendedProfile):
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

    def blend_in_peak(self, peak_to_blend):
        self.blended_profile = self.blend_peaks(self.blended_profile, peak_to_blend)
