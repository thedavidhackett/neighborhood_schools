from typing import List, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt # type: ignore

class TimeSeries:
    """A time series

    A time series using a numpy array and matplotlib.pyplot to display data, calculate a
    moving average and perform regressions.

    Parameters
    ----------
    filepath : str
        A csv file to parse into a time series
    header_row : bool
        Set to true if the csv file has a header row, false otherwise
    cols_to_use : Sequence[int]
        A sequence of ints indicating the columns from the csv file to use in
        the time series. The first column will be used as the time, the second
        column will be used as the data points.
    time_label : str, default='years'
        A label for the points in time
    data_label : str, default='x'
        A label for the data points

    Attributes
    ----------
    data : numpy.ndarray
        A numpy array representing the time series data.
    time_label : str
        A label for the points in time
    data_label : str
        A label for the data points
    """

    def __init__(self, filepath : str, header_row : bool, cols_to_use : Sequence[int], \
        time_label : str = "year", data_label : str = "x") -> None:
        self._data : np.ndarray = np.loadtxt(filepath, skiprows=int(header_row),\
             usecols= cols_to_use, delimiter=",")
        self.time_label : str = time_label
        self.data_label : str = data_label

    @property
    def data(self) -> np.ndarray:
        """Time series data

        Returns
        -------
        numpy.ndarray
            A numpy array representing the time series data
        """
        return self._data

    def plot(self) -> None:
        """Plots the time series on a line graph
        """
        _, axes = plt.subplots()
        axes.plot(self._data[:, 0], self._data[:, 1], linestyle="", marker=".", \
            markersize=5)
        axes.set(title = "{} over {}s".format(self.data_label, self.time_label),\
             ylabel=self.data_label, xlabel=self.time_label)

        plt.show()

    def calculate_moving_average(self, m : int) -> np.ndarray:
        """Calculates the moving average of the time series

        Parameters
        ----------
        m : int
            The amount of data points to include in the moving average, must be
            an odd integer

        Returns
        -------
        numpy.ndarray
            A numpy array including the moving average for each data point in
            the time series. Will return None if the data point doesn't have

        Raises
        ------
        ValueError
            If m is not an odd number
        """
        if m % 2 == 0:
            raise ValueError("M must be an odd number")
        k : int = int((m - 1)/2)

        i : int = k
        moving_average : List = []
        while i + k < len(self.data):
            moving_average.append([self.data[i][0], self.data[(i - k):(i + k + 1), 1].mean(axis=0)])
            i += 1

        return np.array(moving_average)

    def plot_moving_average(self, m : int) -> None:
        """Plots time series data with moving average

        Parameters
        ----------
        m : int
            The amount of data points to include in the moving average. Must be
            an odd number.
        """
        _, axes = plt.subplots()
        moving_average : np.ndarray = self.calculate_moving_average(m)
        axes.plot(self._data[:, 0], self._data[:, 1], linestyle="", marker=".",\
             markersize=5, label="{} for {}".format(self.data_label, self.time_label))
        axes.plot(moving_average[:, 0], moving_average[:, 1], label="moving average")
        axes.legend()
        axes.set(title = "{} over {}s with moving average by {} {}s".\
            format(self.data_label, self.time_label, m, self.time_label), \
                ylabel=self.data_label, xlabel=self.time_label)

        plt.show()

    def linear_regression(self, m : int, yintercept : bool, plot : bool = True)\
         -> Tuple[List[float], List[float]]:
        """Fits moving average to a linear model

        Plots the regression vs the moving average and returns the beta and r
        values

        Parameters
        ----------
        m : int
            The amount of data points to include in the moving average. Must be
            an odd number.
        yintercept : bool
            Whether to calculate the regression with a y intercept or not. True
            if a y intercept should be included, false otherwise.
        """
        moving_average = self.calculate_moving_average(m)

        i = 0
        A = []
        if yintercept:
            while i < len(moving_average):
                A.append([1, moving_average[i][0]])
                i += 1
        else:
            while i < len(moving_average):
                A.append([moving_average[i][0]])
                i += 1
        beta, R, _, _ = np.linalg.lstsq(A, moving_average[:, 1], rcond=None)

        if plot:
            _, axes = plt.subplots()

            axes.plot(moving_average[:, 0], moving_average[:, 1], linestyle="", \
                marker=".", markersize=5, label="moving average")

            if yintercept:
                axes.plot(moving_average[:, 0], moving_average[:, 0] * beta[1] + \
                    beta[0], label="regression line")
            else:
                axes.plot(moving_average[:, 0], moving_average[:, 0] * beta[0],\
                    label="regression line")

            axes.legend()

        plt.show()
        return (beta, R)

    def predict_point_n(self, m : int, yintercept : bool, n : int) -> float:
        """Predicts a given point based in the time series give a linear
        regression

        Parameters
        ----------
        m : int
            The amount of data points to include in the moving average. Must be
            an odd number.
        yintercept : bool
            Whether to calculate the regression with a y intercept or not. True
            if a y intercept should be included, false otherwise.
        n : int
            The point to predict in the moving average
        """
        beta, _ = self.linear_regression(m, yintercept, False)

        if yintercept:
            return n * beta[1] + beta[0]

        return n * beta[0]


    """TODOS
    So currently the predict point n uses an int for n and the linear
    regression uses the year so that needs to be reconciled. It also
    currently plots the linear regression again. There should be a separate
    method for plotting the regression.

    I don't like the set up for the yintercept boolean. Either do a separate
    function for each or figure out a way to standardize the return.

    Can we cache the moving average and regressions so it doesn't repeat that
    function call a bunch of times? Maybe look at what you did in that homework.
    """
