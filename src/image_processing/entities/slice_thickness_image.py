"""
slice_thickness_image.py

This script defines the SliceThicknessImage class, which contains all slice thickness
analysis at the image level. Preprocesses the image when intialised. When analysis function
is called, a LineProfile object is instantiated and pipeline for image continues from there.

Written by Nathan Crossley, 2025.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image

from src.pipeline.io_manager import IOManager
from src.image_processing.entities.line_profile import LineProfile


class SliceThicknessImage:
    """Class for a image for the slice thickness task.
    This should be an image of gafchromic film, scanned in on the
    RRPS printer."""

    def __init__(self, path: Path):
        """Initialises SliceThicknessImage class by reading image and
        cropping/rotating to gaf. RGB and grayscale representations are stored.

        Args:
            path (Path): Path to image.
        """

        # store file path to image
        self.path = path

        # calculate mm per pixel factor from dpi metadata
        self.mm_per_pix = 25.4 / self._detect_dpi(path)

        # read image using OpenCV in BGR format
        image = cv2.imread(path)

        # rotate the image so gaf is axis aligned and then crop
        image = self._rotate_and_crop_to_gaf(image)

        # store RGB and grayscale representations of image
        self.image = {
            "RGB": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "GRAY": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        }

        # inform user that image has been loaded
        print(f"Image loaded: {self.path.name}")

    @staticmethod
    def _detect_dpi(path: Path) -> int:
        """Detects dpi of input image from metadata.

        Args:
            path (Path): Path to input image.

        Returns:
            int: dpi of input image.
        """

        # get dpi from image metadata
        dpi = Image.open(path).info.get("dpi")

        # if dpi is None, warn user and return default dpi of 600.
        if dpi is None:
            print(
                f"Warning: Image {path.name} has no dpi information. Resorting to default dpi of 600. This may cause inaccurate results."
            )
            return 600

        # if dpi is asymmetric, warn user and return default dpi of 600.
        elif dpi[0] != dpi[1]:
            print(
                f"Warning: Image {path.name} has asymmetric dpi: {dpi}. Resorting to default dpi of 600."
            )
            return 600

        # else return detected dpi
        else:
            return dpi[0]

    @classmethod
    def _rotate_and_crop_to_gaf(cls, image: np.ndarray) -> np.ndarray:
        """Takes a BGR image of gafchromic film and rotates so the film
        is axis aligned. Also crops the image to the film.

        Args:
            image (np.ndarray): BGR image to crop and rotate.

        Returns:
            np.ndarray: cropped and rotated image.
        """

        # pad the image with a border to help with contour detection
        image = cv2.copyMakeBorder(
            image,
            image.shape[0] // 10,
            image.shape[0] // 10,
            image.shape[1] // 10,
            image.shape[1] // 10,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        # get grayscale representation of image
        image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # construct mask of gaf film edge by detecting minAreaRect and filling its poly
        rect = cls._locate_gafchromic_film(image_GRAY)
        mask = np.zeros_like(image_GRAY)
        cv2.fillPoly(mask, [np.intp(cv2.boxPoints(rect))], 255)

        # rotate the image and mask so gaf film is axis aligned using angle from MinAreaRect
        image = cls._rotate_no_clip(image, rect[2])
        mask = cls._rotate_no_clip(mask, rect[2])

        # crop the image to the gaf film using the rotated mask
        image = cls._crop_to_gafchromic_film(image, mask)

        return image

    @staticmethod
    def _locate_gafchromic_film(
        image: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        """Locates the gafchromic film in the image, returning
        params for a MinAreaRect enclosing the film.

        Args:
            image (np.ndarray): Image to locate film within.

        Returns:
            tuple[tuple[float, float], tuple[float, float], float]: MinAreaRect enclosing film.
        """

        # apply Gaussian blur to remove noise from iamge
        gauss_blur = cv2.GaussianBlur(image, (3, 3), 0)

        # binarise image into mask using Otsu threshold (good enough to pick out film edge)
        _, thresh = cv2.threshold(
            gauss_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # find contours and sort by length to get the longest contour
        contours, _ = cv2.findContours(~thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=lambda c: len(c), reverse=True)[0]

        # get MinAreaRect enclosing that contour
        rect = cv2.minAreaRect(contour)

        # ensure that the width is always less than the height and adjust angle accordingly
        if rect[1][0] > rect[1][1]:
            rect = (rect[0], rect[1][::-1], rect[2] + 90)

        return rect

    @staticmethod
    def _rotate_no_clip(image: np.ndarray, theta: float | int) -> np.ndarray:
        """Rotates an image by angle theta.
        Ensures that gaf film does not clip out of image after rotation.

        Args:
            image (np.ndarray): Image to rotate.
            theta (float | int): Angle of rotation.

        Returns:
            np.ndarray: Rotated image.
        """

        # Store height, width and centre of image for convenience
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Calculate the rotation matrix for passed angle
        matrix = cv2.getRotationMatrix2D(center, theta, 1.0)

        # Get the new image bounding box dimensions after rotation
        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Apply affine matrix
        rotated_image = cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated_image

    @staticmethod
    def _crop_to_gafchromic_film(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crops the image to the gafchromic film given a mask
        of the film. Image cropped based on the on-axis bbox around the mask.

        Args:
            image (np.ndarray): Image to crop.
            mask (np.ndarray): Mask of film, used for cropping.

        Returns:
            np.ndarray: Image cropped to film.
        """

        # get contour with largest area from mask (film edge)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # crop image based on on-axis bbox about film edge contour
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y : y + h, x : x + w]

        return cropped_image

    def analyse_image(self):
        """Pipeline function to trigger the analysis process for
        an individual SliceThicknessImage object, through the instantiation
        of a LineProfile object.
        """

        # get image shape and dynamically define a y-padding for line profile placement
        shape = self.image["RGB"].shape
        y_pad = max(5, shape[0] // 100)

        # instantiate a LineProfile object to trigger line profile analysis for this image.
        self.line_profile = LineProfile(
            [shape[1] // 2, y_pad],
            [shape[1] // 2, shape[0] - y_pad],
            self.image["GRAY"],
        )

        # calculate mm per data point in line profile.
        self.mm_cf = (
            self.mm_per_pix
            * np.linalg.norm(
                self.line_profile.end_point - self.line_profile.start_point
            )
            / len(self.line_profile.profile.y)
        )

    def save_results_graphics(self):
        """Saves results graphics to figures directory and displays figure.
        Figure shows the line profile placement on the cropped and rotated image,
        along with the measured line profile and the calculated slice thicknesses."""

        # create figure and axes using gridspec
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(3, 8)
        fig_ax1 = fig.add_subplot(gs[:, :-1])
        fig_ax2 = fig.add_subplot(gs[:, -1])

        # on axis 1, plot raw line profile
        fig_ax1.plot(
            self.line_profile.profile.x * self.mm_cf,
            self.line_profile.profile.y,
            label="Raw profile",
            color=(0.75, 0, 0),
            alpha=0.4,
        )

        # for each peak in detected peaks
        for i, peak in enumerate(self.line_profile.peaks):

            # plot the connected peaks model
            fig_ax1.plot(
                peak.x * self.mm_cf,
                peak.y,
                label="Connected peaks model",
                color=(0, 0.75, 0),
            )

            # plot the binary profile (shows FWHM values visually)
            fig_ax1.plot(
                peak.binary_profile.x * self.mm_cf,
                peak.binary_profile.y,
                label="Binary profile",
                ls="--",
                color=(0, 0.5, 1),
            )

            # add FWHM floats above binary profile peaks
            fig_ax1.text(
                (peak.crossing_x_left + peak.crossing_x_right) / 2 * self.mm_cf,
                peak.peak_y * 1.05,
                f"{peak.FWHM * self.mm_cf:.1f} mm",
                ha="center",
                va="center",
            )

        # on axis 2, show RGB image
        fig_ax2.imshow(self.image["RGB"])

        # plot line to show position of line profile
        fig_ax2.plot(
            [self.line_profile.start_point[0], self.line_profile.end_point[0]],
            [self.line_profile.start_point[1], self.line_profile.end_point[1]],
            color="red",
        )

        # set axis labels, title and legends for both axes where required
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

        # save figure using path stored in IOManager
        plt.savefig(
            IOManager.paths_from_proj_dir["figures_dir"].joinpath(
                f"{self.path.stem}_results.png"
            ),
            dpi=600,
        )

        # show the user the plot before moving on
        plt.show(block=True)
        plt.close()
