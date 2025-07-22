"""
main.py

Entrypoint for Slice Thickness Calculator application. Imports the SliceThicknessTask class
from the pipeline package and runs the task to begin the slice thickness analysis process.

Written by Nathan Crossley, 2025.
"""

from src.pipeline.slice_thickness_task import SliceThicknessTask

if __name__ == "__main__":

    # instantiate and run SliceThicknessTask
    slice_thickness_task = SliceThicknessTask()
    slice_thickness_task.run()
