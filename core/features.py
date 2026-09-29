import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import (
    get_range_values_per_column,
    impute_dataframe_range,
)


class TrainingFeatureSelector(TransformerMixin, BaseEstimator):
    """Learn imputation values and relevant feature names from training rows only."""

    def __init__(self, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TrainingFeatureSelector':
        """Fit tsfresh relevance selection after training-only imputation."""
        self.max_, self.min_, self.median_ = get_range_values_per_column(X)
        clean = impute_dataframe_range(X.copy(), self.max_, self.min_, self.median_)
        self.columns_ = select_features(clean, y, n_jobs=self.n_jobs).columns.tolist()
        if not self.columns_:
            raise ValueError('No relevant features found in the inner training fold')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transform without learning from evaluation rows."""
        clean = impute_dataframe_range(X.copy(), self.max_, self.min_, self.median_)
        return clean[self.columns_]
