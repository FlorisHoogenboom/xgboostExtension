import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import check_X_y, check_array
import pandas as pd

def _preprare_data_in_groups(X, y=None, sample_weights=None, dtype='numeric'):
    """
    Takes the first column of the feature Matrix X given and
    transforms the data into groups accordingly.

    Parameters
    ----------
    X : (2d-array like) Feature matrix with the first column the group label

    y : (optional, 1d-array like) target values

    sample_weights : (optional, 1d-array like) sample weights

    dtype : (str or  None) dtype to be used for the matrix X, defaults to numeric

    Returns
    -------
    sizes: (1d-array) group sizes

    X_orig : (2d-array like) Same type as original X passed, but sorted

    X_features : (2d-array) features sorted per group

    y : (None or 1d-array) Target sorted per group

    sample_weights: (None or 1d-array) sample weights sorted per group
    """
    if isinstance(y, np.ndarray) or isinstance(y, pd.Series):
        X_arr, y = check_X_y(X, y, accept_sparse=True, y_numeric=True, dtype=dtype)
    else:
        X_arr = check_array(X, accept_sparse=True, dtype=dtype)

    if sparse.issparse(X_arr):
        group_labels = X_arr.getcol(0).toarray()[:,0]
    else:
        group_labels = X_arr[:,0]

    group_indices = group_labels.argsort()

    group_labels = group_labels[group_indices]
    _, sizes = np.unique(group_labels, return_counts=True)
    X_features = X_arr[group_indices, 1:]


    # To support DataFrames we need to use iloc
    # in some cases.
    if isinstance(X, pd.DataFrame):
        X_sorted = X.iloc[group_indices]
    else:
        X_sorted = X[group_indices]

    if y is not None:
        y = y[group_indices]

    if sample_weights is not None:
        sample_weights = sample_weights[group_indices]

    return sizes, X_sorted, X_features, y, sample_weights


class RankingEstimator(BaseEstimator):
    def __init__(self, estimator):
        """
        Transforms a simple sklearn estimator into an estimator
        that takes groups as an input. This method modifies the
        fit, predict, etc. methods of the provided estimator to
        take the group labels as an input in the first column.

        Parameters
        ----------
        estimator : (sklearn-estimator) The standard sklearn estimator
            that should be applied to a ranking problems.
        """
        self.estimator = estimator

    @if_delegate_has_method(delegate=('estimator'))
    def fit(self, X, y):
        _, _, X_features, y, _ = _preprare_data_in_groups(X, y)

        return self.estimator.fit(X_features, y)

    @if_delegate_has_method(delegate=('estimator'))
    def predict(self, X):
        """Call predict on the underlying estimator.

        Parameters
        -----------
        X : (ndarray, n_samples x (1 + n_features)), feature matrix with
            the first column containing the group label
        """

        # Remove groups column from the data
        _, _, X_features, _, _ = _preprare_data_in_groups(X)

        return self.estimator.predict(X_features)

    @if_delegate_has_method(delegate=('estimator'))
    def predict_proba(self, X):
        """Call predict_proba on the underlying estimator.

        Parameters
        -----------
        X : (ndarray, n_samples x (1 + n_features)), feature matrix with
            the first column containing the group label
        """

        # Remove groups column from the data
        _, _, X_features, _, _ = _preprare_data_in_groups(X)

        return self.estimator.predict_proba(X_features)

    @if_delegate_has_method(delegate=('estimator'))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the underlying estimator.

        Parameters
        -----------
        X : (ndarray, n_samples x (1 + n_features)), feature matrix with
            the first column containing the group label
        """

        # Remove groups column from the data
        _, _, X_features, _, _ = _preprare_data_in_groups(X)

        return self.estimator.predict_log_proba(X_features)

    @if_delegate_has_method(delegate=('estimator'))
    def decision_function(self, X):
        """Call descision_function on the underlying estimator.

        Parameters
        -----------
        X : (ndarray, n_samples x (1 + n_features)), feature matrix with
            the first column containing the group label
        """

        # Remove groups column from the data
        _, _, X_features, _, _ = _preprare_data_in_groups(X)

        return self.estimator.decision_function(X_features)

    # get_params is not overriden since this causes
    # cloning of the estimator to fail.

    def set_params(self, **params):
        # Do override set_params since this enables
        # arguments to the wrapped class to be passed
        # without __
        self.estimator.set_params(**params)
        return self
