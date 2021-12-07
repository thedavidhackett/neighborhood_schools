from typing import Collection, List, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt # type: ignore

class PredictionData:
    """A prediction dataset

    A dataset utilizing a numpy array to make predictions.

    Parameters
    ----------
    filepath : str
        A csv file to parse into a time series
    cols_to_use : Sequence[int]
        A sequence of ints indicating the columns from the csv file to use in
        the dataset

    Attributes
    ----------
    _data : numpy.ndarray
        The data to use for predictions
    _map : Dict[str, int]
        A mapping for headers to the corresponding index in the data
    """
    def __init__(self, filepath : str, cols_to_use : Sequence[int]) -> None:
        with open(filepath) as file:
            headers = file.readline().strip().split(",")
            self._data : np.ndarray = np.loadtxt(file, delimiter=",", usecols =\
                 cols_to_use)
        mapping = {}
        i = 0
        for idx in cols_to_use:
            mapping[headers[idx]] = i
            i += 1
        self._map = mapping

    def __getitem__(self, key : str) -> np.ndarray:
        """Get a feature of the data set

        Parameters
        ----------
        key : str
            The column name of the feature

        Returns
        -------
        numpy.ndarray
            A numpy array of the values in that column
        """
        return self._data[:, self._map[key]]

    def __setitem__(self, key : str, value : np.ndarray) -> None:
        """Sets a feature of the data set

        Parameters
        ----------
        key : str
            The column name of the feature
        value : numpy.ndarray
            The value to set the feature as
        """
        self._data[:, self._map[key]] = value


    def plot(self, target : str, feature : str) -> None:
        """Plot a target feature against another feature

        Parameters
        ----------
        target : str
            The column name of the target
        feature : str
            The column name of the feature to plot against
        """
        _, axes = plt.subplots()
        axes.plot(self[feature], self[target], linestyle="", marker=".", \
            markersize=5)
        axes.set(title = "{} vs {}".format(target, feature),\
             ylabel=target, xlabel=feature)

        plt.show()

    def standardize(self, feature : str) -> None:
        """Standardizes a feature

        Centers the feature around a mean of 0 with a standard deviation of 1

        Parameters
        ----------
        feature : str
            The name of the feature to standardize
        """
        mean : np.number = self[feature].mean()
        std : np.number = self[feature].std()
        self[feature] = (self[feature] - mean)/std

    def standardize_multiple(self, features : Collection[str]) -> None:
        """Standardizes multiple features

        Centers each feature around a mean of 0 with a standard deviation of 1

        Parameters
        ----------
        features : Collection[str]
            A collection of feature names to standardize
        """
        for feature in features:
            self.standardize(feature)

    def linear_regression(self, target : str, feature : str, plot : bool = True)\
        -> Tuple[List[float], List[float]]:
        """Performs a linear regression between a target and a chosen feature.

        Parameters
        ----------
        target : str
            The column name of the target feature
        feature : str
            The column name of the feature to use in the model
        plot : bool, default=True
            Whether to plot the linear model or not
        """
        i = 0
        A = []
        while i < len(self[target]):
            A.append([1, self[feature][i]])
            i += 1
        beta, R, _, _ = np.linalg.lstsq(A, self[target], rcond=None)

        if plot:
            _, axes = plt.subplots()

            axes.plot(self[feature], self[target], linestyle="", marker=".", \
                markersize=5, label="data points")

            axes.plot(self[feature], self[feature] * beta[1] + beta[0],\
                label="regression line")

            axes.set(title = "{} vs {}".format(target, feature),\
                ylabel=target, xlabel=feature)

            axes.legend()

            plt.show()
        return (beta, R)

    def multi_variate_linear_regression(self, target : str, features : \
        List[str]) -> Tuple[List[float], List[float]]:
        """Performs a  multivariate linear regression between a target and a
        list of chosen features.

        Parameters
        ----------
        target : str
            The column name of the target feature
        feature : List[str]
            The column names of the feature to use in the model
        """
        i = 0
        A = []
        while i < len(self[target]):
            a = [1]
            for feature in features:
                a.append(self[feature][i])
            A.append(a)
            i += 1
        beta, R, _, _ = np.linalg.lstsq(A, self[target], rcond=None)

        return (beta, R)
