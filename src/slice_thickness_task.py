import time
import traceback
import sys

from src.io_manager import IO
from src.image_objects import SliceThicknessImage


class SliceThicknessTask:
    """Class for running slice thickness analysis task."""

    def __init__(self):
        """Initialises SliceThicknessTask class by loading images."""
        IO.creation_control()
        print("Loading images from input folder...")
        self.images = [SliceThicknessImage(path) for path in IO.pull_images()]

    def run(self):
        try:
            """Runs slice thickness analysis on loaded images"""
            print("Analysing images...")
            for image in self.images:
                print(f"Analysing image: {image.path.name}")
                image.analyse_image()
                image.save_results_graphics()
                IO.update_log(image)
            print("All images analysed! Quitting programme.")
            time.sleep(30)
        except Exception as e:
            traceback.print_exc()
            print(f"Error at runtime: {e}")
            time.sleep(30)
            sys.exit()
