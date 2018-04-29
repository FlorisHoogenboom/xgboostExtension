from sklearn.metrics.scorer import _BaseScorer
from xgboostextension.xgbranker import XGBRanker, _preprare_data_in_groups
from xgboostextension.scorer.util import _make_grouped_metric


class _RankingScorer(_BaseScorer):
    def __init__(self, score_func, sign=1):
        """
        Base class for applying scoring functions to ranking problems.
        This class transforms a ranking metric into a scoring function
        that can be applied to estimations that take a group indicator in
        their first column.

        Parameters
        ----------
        """
        if not score_func.__module__ == 'xgboostextension.scorer.metrics':
            raise ValueError(
                'Only score functions included with this package are supported'
            )

        super(_RankingScorer, self).__init__(
            _make_grouped_metric(score_func),
            sign,
            None
        )

    def __call__(self, estimator, X, y, sample_weight=None):
        if not isinstance(estimator, XGBRanker):
            raise NotImplementedError((
                'Currently only scoring for the'
                'XGBRanker model is supported.'
            ))

        sizes, _, y_sorted, _ = _preprare_data_in_groups(X, y)

        y_predicted = estimator.predict(X)

        return self._sign * self._score_func(sizes, y_sorted, y_predicted)