from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image

from scipy import signal
from skimage.measure import profile_line

from io_manager import IO
from xy_tools import XY
from profile_sigmoid_model import ProfileSigmoidModel
from utils import find_highest_peak


class SliceThicknessImage:
    """Class for a single slice thickness image (displaying gafchromic film)."""

    def __init__(self, path: Path):
        """Initialises SliceThicknessImage class by reading image and cropping/rotating to gaf.

        Args:
            path (Path): Path to image.
        """
        self.path = path
        self.mm_per_pix = 25.4 / self.detect_dpi(path)

        image = cv2.imread(path)
        image = self.rotate_and_crop_to_gaf(image)

        self.image = {
            "RGB": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "GRAY": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        }
        print(f"Image loaded: {self.path.name}")

    @staticmethod
    def detect_dpi(path: Path) -> int:
        """Detects dpi of input image.

        Args:
            path (Path): Path to input image.

        Returns:
            int: dpi on input image.
        """
        dpi = Image.open(path).info.get("dpi")
        if dpi is None:
            print(
                f"Warning: Image {path.name} has no dpi information. Resorting to default dpi of 600. This may cause inaccurate results."
            )
            return 600

        elif dpi[0] != dpi[1]:
            print(
                f"Warning: Image {path.name} has asymmetric dpi: {dpi}. Resorting to default dpi of 600."
            )
            return 600

        else:
            return dpi[0]

    @classmethod
    def rotate_and_crop_to_gaf(cls, image: np.ndarray) -> np.ndarray:
        """Takes a BGR image and crops to the gafchromic film, rotating so the film is axis aligned.

        Args:
            image (np.ndarray): BGR image to crop and rotate.

        Returns:
            np.ndarray: cropped and rotated image.
        """
        image = cv2.copyMakeBorder(
            image,
            image.shape[0] // 10,
            image.shape[0] // 10,
            image.shape[1] // 10,
            image.shape[1] // 10,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rect = cls.locate_gafchromic_film(image_GRAY)
        mask = np.zeros_like(image_GRAY)
        cv2.fillPoly(mask, [np.intp(cv2.boxPoints(rect))], 255)

        image = cls.rotate_no_clip(image, rect[2])
        mask = cls.rotate_no_clip(mask, rect[2])

        image = cls.crop_to_gafchromic_film(image, mask)
        return image

    @staticmethod
    def locate_gafchromic_film(
        image: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        """Locates the gafchromic film in the image, returning rect params for a rect enclosing the film.

        Args:
            image (np.ndarray): Image to locate film within.

        Returns:
            tuple[tuple[float, float], tuple[float, float], float]: Rect enclosing film.
        """

        gauss_blur = cv2.GaussianBlur(image, (3, 3), 0)
        _, thresh = cv2.threshold(
            gauss_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(~thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=lambda c: len(c), reverse=True)[0]
        rect = cv2.minAreaRect(contour)
        if rect[1][0] > rect[1][1]:
            rect = (rect[0], rect[1][::-1], rect[2] + 90)
        return rect

    @staticmethod
    def rotate_no_clip(image: np.ndarray, theta: float | int) -> np.ndarray:
        """Rotates an image by angle theta. Ensures that gaf film does not clip out of image after rotation.

        Args:
            image (np.ndarray): Image to rotate.
            theta (float | int): Angle of rotation.

        Returns:
            np.ndarray: Rotated image.
        """

        (h, w) = image.shape[:2]
        # Calculate the center of the image
        center = (w // 2, h // 2)

        # Calculate the rotation matrix
        matrix = cv2.getRotationMatrix2D(center, theta, 1.0)

        # Get the new bounding box dimensions after rotation
        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])

        # Calculate the new width and height
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Rotate the image and resize it to the new size (without clipping)
        rotated_image = cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated_image

    @staticmethod
    def crop_to_gafchromic_film(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crops the image to the gafchromic film given a cropping mask.

        Args:
            image (np.ndarray): Image to crop to film.
            mask (np.ndarray): Mask to crop to (using bbox)

        Returns:
            np.ndarray: Image cropped to film.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y : y + h, x : x + w]
        return cropped_image

    def analyse_image(self):
        """Analyses individual image by initialising LineProfile object."""
        shape = self.image["RGB"].shape
        y_pad = min(10, shape[0] // 75)
        self.line_profile = LineProfile(
            [shape[1] // 2, y_pad],
            [shape[1] // 2, shape[0] - y_pad],
            self.image["GRAY"],
        )
        self.mm_cf = (
            self.mm_per_pix
            * np.linalg.norm(
                self.line_profile.end_point - self.line_profile.start_point
            )
            / len(self.line_profile.profile.y)
        )

    def save_results_graphics(self):
        """Saves results graphics to figures directory and displays figure."""
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(3, 8)
        fig_ax1 = fig.add_subplot(gs[:, :-1])
        fig_ax2 = fig.add_subplot(gs[:, -1])

        fig_ax1.plot(
            self.line_profile.profile.x * self.mm_cf,
            self.line_profile.profile.y,
            label="Raw profile",
            color=(0.75, 0, 0),
            alpha=0.4,
        )
        for i, peak in enumerate(self.line_profile.peaks):
            fig_ax1.plot(
                peak.x * self.mm_cf,
                peak.y,
                label="Blended sigmoid model",
                color=(0, 0.75, 0),
            )
            fig_ax1.plot(
                peak.binary_profile.x * self.mm_cf,
                peak.binary_profile.y,
                label="Binary profile",
                ls="--",
                color=(0, 0.5, 1),
            )
            fig_ax1.text(
                (peak.crossing_x_left + peak.crossing_x_right) / 2 * self.mm_cf,
                peak.peak_y * 1.05,
                f"{peak.FWHM * self.mm_cf:.1f} mm",
                ha="center",
                va="center",
            )

        fig_ax2.imshow(self.image["RGB"])
        fig_ax2.plot(
            [self.line_profile.start_point[0], self.line_profile.end_point[0]],
            [self.line_profile.start_point[1], self.line_profile.end_point[1]],
            color="red",
        )

        fig_ax1.set_xlabel("Distance along line profile (mm)")
        fig_ax1.set_ylabel("Pixel Value")
        fig_ax1.set_title(f"Slice Thickness Analysis: {self.path.stem}")
        handles, labels = fig_ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig_ax1.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower right",
            bbox_to_anchor=(1, -0.3),
        )
        fig_ax2.set_axis_off()

        plt.savefig(
            IO.paths_from_proj_dir["figures_dir"].joinpath(
                f"{self.path.stem}_results.png"
            ),
            dpi=600,
        )
        plt.show(block=True)
        plt.close()


class LineProfile:
    """Class for a single line profile across a gafchromic film image."""

    def __init__(
        self, start_point: list | tuple, end_point: list | tuple, ref_image: np.ndarray
    ):
        """Initialises LineProfile class.

        Args:
            start_point (list | tuple): Start point of line profile.
            end_point (list | tuple): End point of line profile.
            ref_image (np.ndarray): Image to take line profile across.
        """
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.ref_image = ref_image

        profile_y = profile_line(self.ref_image, start_point[::-1], end_point[::-1], 1)
        self.profile = XY(np.arange(len(profile_y)), profile_y)
        self.profile = self.filter_and_invert(self.profile)

        self.fitted_profile = ProfileSigmoidModel.run(self.profile)
        self.peaks = self.split_into_peaks(self.fitted_profile)

    @staticmethod
    def filter_and_invert(profile: XY) -> XY:
        """Applies median and savgol filters to profile and inverts it.

        Args:
            profile (XY): Input profile to smooth and invert.

        Returns:
            XY: Smoothed and inverted profile.
        """
        profile.y = signal.medfilt(
            profile.y,
            int(len(profile.y) / 200) + 0 if int(len(profile.y) / 50) % 2 == 0 else 1,
        )
        profile.y = signal.savgol_filter(profile.y, int(len(profile.y) / 50), 3)
        profile.y -= np.max(profile.y)
        profile.y *= -1
        return profile

    @staticmethod
    def split_into_peaks(profile: XY) -> list["Peak"]:
        """Splits line profile into Peak objects for further analysis.

        Args:
            profile (XY): Input line profile to split into peaks.

        Returns:
            list[Peak]: List of Peak objects.
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
        peaks = [Peak(segment.x, segment.y) for segment in segments]
        return peaks


class Peak(XY):
    """Class for processing of a singular peak."""

    def __init__(self, *args: list[int | float]):
        """Initialises Peak object and calculates FWHM."""
        self.peak_idx = find_highest_peak(self.y)
        self.peak_y = self.y[self.peak_idx]

        self.half_height_left = self.y[0] + (self.peak_y - self.y[0]) / 2
        self.half_height_right = self.y[-1] + (self.peak_y - self.y[-1]) / 2

        self.FWHM = self.get_FWHM()
        self.binary_profile = self.get_FWHM_binary_profile()

    def get_FWHM(self) -> float:
        """Gets the FWHM of the peak using half height crossing idxs and linear interpolation.

        Returns:
            float: FWHM of peak.
        """
        self.crossing_x_left = np.interp(
            self.half_height_left, self.y[: self.peak_idx], self.x[: self.peak_idx]
        )
        self.crossing_x_right = np.interp(
            self.half_height_right,
            self.y[self.peak_idx :][::-1],
            self.x[self.peak_idx :][::-1],
        )
        return self.crossing_x_right - self.crossing_x_left

    def get_FWHM_binary_profile(self) -> XY:
        """Returns a binary profile, where each y-value is either set to background level or peak height, depending on x-coord.

        Returns:
            XY: binary profile to represent FWHM calculations visually.
        """
        binary_profile = self.copy()
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
