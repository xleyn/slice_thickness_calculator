<h1>The Slice Thickness Calculator Project</h1>
<h2>Scope of Work</h2>
Welcome to the Slice Thickness Calculator project. 
This tool was developed for the Regional Radiation Protection Service (RRPS) to help automate the slice thickness validation in CT commissionings.
The previous tools were inaccurate and inefficient, requiring slices to be analysed one at a time. 
The process was also less streamlined, requiring more manual user input.
This project offers an improvement in both automation and accuracy through the use of Python scripting.

<h2>Using the Slice Thickness Calculator</h2>
<p>
  This repository contains all of the required tools for the slice thickness analysis.
  A conda environment should be configured and activated for dependency management using the environment.yml file. A build script is also provided for easy compilation and distribution.
</p>
The subsequent instructions can be followed to use this tool:
<ul>
  <li>
    Prior to runtime, the user should scan the Gafchromic film image in at 600 dpi. 
    This image should be saved with an appropriate name in the runtime_io/image_input directory.
    If this directory is missing, run the script to rectify these issues. 
  </li>
  <li>
    Once the image is correctly saved, the script should be run (through main.py). 
    The image will be analysed and a figure displayed showing the line profile, connected peaks model and resulting slice thicknesses.
  </li>
  <li>
    The figure can then be closed. For convenience, it will be saved in runtime_io/results/figures.
    The measured slice thicknesses will also be recorded in your personal Excel log, saved at runtime_io/results/slice_thickness_log.xlsx.
  </li>
</ul>
