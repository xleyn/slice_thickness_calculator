"""
sigmoid.py

Script defines subclass of XY, Sigmoid, that is used to fit an alalytical sigmoid
to an XY profile. Best fit parameters are stored so the functional sigmoid can be
evaluated with any input x values.

Written by Nathan Crossley, 2025.

"""

import warnings

import numpy as np
from scipy import optimize

from src.image_processing.entities.xy import XY


class Sigmoid(XY):
    """Subclass of XY, utilised to fit an analytical sigmoid
    to an XY profile. This is perfomed using least squares curve
    fitting from scipy.optimize"""

    def __init__(self, *args: list[int | float]):
        """Initialises Sigmoid class.
        Gets optimal parameters for sigmoid fitting."""

        # get and store optimal parameters for sigmoid fitting
        self.popt = self.get_popt()

    def get_popt(self) -> list[float]:
        """Gets best fit parameters for analytical sigmoid fitting.

        Returns:
            list[float]: List of best fit parameters for sigmoid fitting.
        """

        # get initial guesses for each fitting parameter
        p0 = self.get_initial_guesses()

        # catch runtime warning in fitting for custom handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # try to get optimal fitting parameters by fitting analytical sigmoid model
            # using raw profile and initial guesses
            try:
                popt, _ = optimize.curve_fit(
                    self.functional_form, self.x, self.y, p0=p0
                )

            # in the case of runtime error in fitting, inform user and use intial guesses as may be good enough.
            except RuntimeError:
                print(
                    f"User warning: Runtime error in sigmoid fitting. Using initial guesses."
                )
                popt = p0

        return popt

    def get_initial_guesses(self) -> list[float]:
        """Returns suitable initial guesses for sigmoid fitting
        based on profile of self.

        Returns:
            list[float]: List of initial guesses for sigmoid fitting.
        """

        # amplitude parameter guess is peak-to-peak amplitude
        A = np.ptp(self.y)

        # y-offest parameter guess is profile minimum
        b = np.min(self.y)

        # calculate absolute derivate profile and get maximum
        abs_deriv = np.abs(np.diff(self.y) / np.diff(self.x))
        abs_grad_max = np.max(abs_deriv)

        # steepness parameter guess is half of maximum abs gradient, with appropriate sign applied
        k = np.sign(self.y[-1] - self.y[0]) * abs_grad_max * 0.5

        # x-centre parameter guess is x coord of maximum abs gradient
        x0 = self.x[np.argmax(abs_deriv)]

        # collate guesses and return
        p0 = [A, k, x0, b]

        return p0

    @staticmethod
    def functional_form(
        x: list[float] | float, A: float, k: float, x0: float, b: float
    ) -> list[float] | float:
        """Defines the functional (analytical) form of a sigmoid. Sigmoid
        shape defined by input parameters. Sigmoid function evaluated for input
        x-array or single x-value.

        Args:
            x (list[float] | float): Input x-array or single x value.
            A (float): Amplitude parameter for sigmoid model.
            k (float): Steepness parameter for sigmoid model.
            x0 (float): X-centre parameter for sigmoid model.
            b (float): y-offset parameter for sigmoid model.

        Returns:
            list[float] | float: y-array or single y value for input x.
        """

        exp_term = np.exp(-k * (x - x0))

        return A / (1 + exp_term) + b

    def evaluate(self, x: list) -> XY:
        """Evaluates the best-fit sigmoid model for an input x-array.
        Returns an XY object representing the fitted sigmoid.

        Args:
            x (list): Input x-array.

        Returns:
            XY: Output XY object for input x-array using fitted sigmoid model.
        """

        return XY(x, self.functional_form(x, *self.popt))
