"""
xy.py

Script defining a subclass of np.ndarray, XY, for handling linked
1D-arrays of x and y values. This makes later code cleaner and more readable.

Written by Nathan Crossley, 2025.
"""

import numpy as np
from typing import Self


class XY(np.ndarray):
    """Subclass of np.ndarray for handling linked 1D-arrays of x and y values.
    Allows for easy manipulation and access of x and y values"""

    def __new__(cls, *args: list[int | float]):
        """Initialise class, checking that input arrays are valid."""

        # check that all input arrays have the same length
        if len(set(map(len, args))) != 1:
            raise ValueError("All input arrays must have the same length")

        # create array from args
        arr = np.array(args)

        # if dimensions are not correct, raise error
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("args of XY should be two 1d arrays")

        # return the array as an instance of the class
        return np.asarray(arr).view(cls)

    @property
    def x(self) -> Self:
        """Property for x-array"""
        return self[0].flatten()

    @property
    def y(self) -> Self:
        """Property for y-array"""
        return self[1].flatten()

    @y.setter
    def y(self, val: np.ndarray):
        """Setter for y property"""

        # check that input is either a list or numpy.ndarray
        if isinstance(val, (np.ndarray, list)):

            # check that input has the same length as current y as should not modify
            if len(val) != len(self.y):
                raise ValueError("Cannot modify shape of XY.y")

        # else raise error
        else:
            raise TypeError("Expected input to be either a list or numpy.ndarray")

        # set value of y if passed checks
        self[1] = val

    def get_combined_x_range(self, other: Self) -> np.ndarray:
        """Gets the combined x-range of this XY object and another
        XY object. Returns a numpy array of the combined x-range.

        Args:
            other (XY): Another XY object to get shared x-range with.

        Returns:
            np.ndarray: Combined x-range of this and other XY object.
        """

        return np.arange(
            min(np.min(self.x), np.min(other.x)),
            max(np.max(self.x), np.max(other.x)) + 1,
        )
