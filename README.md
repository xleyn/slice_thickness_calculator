<h1>The CT Slice Thickness Calculator Project</h1>
Welcome to the CT Slice Thickness Calculator Project! Developed for the Regional Radiation Protection Service as part of my Professional Training Year, this is a tool to automate aspects of CT imaging unit testing.
<h2>Background</h2>
<p> Part of the CT unit testing process is quality testing, namely evaluating whether the unit is functioning as required. During an axial CT scan, patients are imaged in discrete slices along the axial direction (head to toe). These slices are finite and are determined by the scanner, although one cannot assume that the true slice thicknesses are as expected. As the irradiated slice thickness is proportional to the patient dose, it is essential to ensure that the true irradiated slices are of the intended thickness for the patient's safety.</p>
<p>This test is an important aspect of CT quality testing and the Regional Radiation Protection Service carries out the procedure as follows:
<ul>
  <li>Prepare a long strip of Gafchromic Film, a material which darkens with exposure to radiation.</li>
  <li>Place the film in the isocentre of the CT scanner (the geometric centre of the gantry).</li>
  <li>Irradiate the film with a set slice thickness, yielding a black band on the film.</li>
  <li>Shift the film in the axial direction (along) and irradiate the film again with a slightly increased slice thickness.</li>
  <li>Repeat this process until one has a sample with many separate black bands of different thicknesses, such as below.</li>
  <img width="1053" height="163" alt="image" src="https://github.com/user-attachments/assets/b92139a7-3964-421e-a98d-eb89ef356b5f" />
  <li>Measure the slice thicknesses of all the bands with a ruler and calculate the deviation from their expected values. Compare these deviations against a set tolerance to ascertain whether the CT scanner passes this test or not.</li>
</ul></p>
<p>This methodology is flawed for multiple reasons, however the most significant factor is that repeat measurements (or measurements from different people) often yields widely different results! This is due to either the misalignment of the ruler or the inconsistencies associated with judging whether the edges of a band lie (note the penumbra effect). Due to tight acceptance criteria, any variation can even cause tests to pass or fail with repeat measurements. The inconsistency highlighted the need for a new solution with improved accuracy and repeatability - and this repository is the result! The Slice Thickness Calculator Project is a Python tool that utilises image processing methods to calculate the slice thicknesses of all bands from a scanned image of the Gafchromic Film sample.<\p>
<h2>Technical Implementation</h2>







<!-- <h2>Using the Slice Thickness Calculator</h2>
<p>
  This repository contains all of the required tools for the slice thickness analysis.
  A conda environment should be configured and activated for dependency management using the environment.yml file. A build script is also provided for easy compilation and distribution.
</p>
The subsequent instructions can be followed to use this tool:
<ul>
  <li>
    Prior to runtime, the user should scan the Gafchromic film image in at 600 dpi (any dpi should work but 600 will give the most precision). 
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
</ul> -->
