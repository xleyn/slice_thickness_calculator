import numpy as np
from typing import Self


class XY(np.ndarray):
    """Class for 2D numpy array for plotting"""

    def __new__(cls, *args: list[int | float]):
        """Initialise class"""
        if len(set(map(len, args))) != 1:
            raise ValueError("All input arrays must have the same length")
        arr = np.array(args)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("args of XY should be two 1d arrays")
        return np.asarray(arr).view(cls)

    @property
    def x(self) -> Self:
        """Property for x array of plotting series"""
        return self[0].flatten()

    @property
    def y(self) -> Self:
        """Property for y array of plotting series"""
        return self[1].flatten()

    @y.setter
    def y(self, val: np.ndarray):
        """Setter for y property"""
        if isinstance(val, (np.ndarray, list)):
            if len(val) != len(self.y):
                raise ValueError("Cannot modify shape of XY.y")
        else:
            raise TypeError("Expected input to be either a list or numpy.ndarray")
        self[1] = val


class XYUtils:
    @staticmethod
    def get_x_range(xy1: XY, xy2: XY):
        x_min = min(np.concatenate([xy1.x, xy2.x]))
        x_max = max(np.concatenate([xy1.x, xy2.x]))
        x_range = np.arange(x_min, x_max + 1)
        return x_range
