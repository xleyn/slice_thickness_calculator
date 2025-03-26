from scipy import signal
import numpy as np

def find_highest_peak(y):
    peaks, props = signal.find_peaks(y, height=0)
    peak_x = peaks[np.argmax(props["peak_heights"])]
    return peak_x