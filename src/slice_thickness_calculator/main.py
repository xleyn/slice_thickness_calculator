import time

from io_manager import IO
from image_objects import SliceThicknessImage


class SliceThicknessTask:
    """Class for running slice thickness analysis task."""

    def __init__(self):
        """Initialises SliceThicknessTask class by loading images."""
        IO.creation_control()
        print("Loading images from input folder...")
        self.images = [SliceThicknessImage(path) for path in IO.pull_images()]

    def run(self):
        """Runs slice thickness analysis on loaded images"""
        print("Analysing images...")
        for image in self.images:
            print(f"Analysing image: {image.path.name}")
            image.analyse_image()
            image.save_results_graphics()
            IO.update_log(image)
        print("All images analysed! Quitting programme.")
        time.sleep(5)


if __name__ == "__main__":
    slice_thickness_task = SliceThicknessTask()
    slice_thickness_task.run()
