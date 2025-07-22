"""
slice_thickness_task.py

Script to define pipeline class for managing slice thickness analysis process.
Ensures that I/O errors will not be raised and selects images to be analysed.
Each image is analysed, figures are saved and the log is updated.

Written by Nathan Crossley, 2025.
"""

import sys
import traceback
import time

from src.pipeline.io_manager import IOManager
from src.image_processing.entities.slice_thickness_image import (
    SliceThicknessImage,
)


class SliceThicknessTask:
    """Pipeline class for managing slice thickness analysis process.
    Ensures that I/O errors will not be raised and selects images to be analysed.
    Each image is analysed, figures are saved and the log is updated."""

    def __init__(self):
        """Initialises SliceThicknessTask class. Potential I/O errors are handled early.
        Images to analyse are loaded and SliceThicknessImage objects are instantiated for each.
        """

        # ensures that various I/O errors will not be raised at a later date.
        IOManager.creation_control()

        # informs user that images are being loaded
        print("Loading images from input folder...")

        # get images to analyse and instantiate SliceThicknessImage objects
        self.images = [SliceThicknessImage(path) for path in IOManager.pull_images()]

    def run(self):
        """Triggers the rest of the slice thickness analysis pipeline to run.
        Images are analysed, figures are saved and the log is updated."""

        # try to run pipeline, catching errors
        try:

            # inform user that images are being analysed
            print("Analysing images...")

            # process each image
            for image in self.images:

                # inform user that a particular image is being analysed
                print(f"Analysing image: {image.path.name}")

                # analyse image, save results graphics and update log
                image.analyse_image()
                image.save_results_graphics()
                IOManager.update_log(image)

            # inform user that all images have been analysed
            print("All images analysed! Quitting programme.")
            time.sleep(30)

        # if any errors are caught, inform user and exit after delay
        except Exception as e:
            traceback.print_exc()
            print(f"Error at runtime: {e}")
            time.sleep(30)
            sys.exit()
