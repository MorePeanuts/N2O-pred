from dataclasses import dataclass
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from .data import (
    NUMERIC_STATIC_FEATURES,
    NUMERIC_DYNAMIC_FEATURES_RF,
    CATEGORICAL_STATIC_FEATURES,
    CATEGORICAL_DYNAMIC_FEATURES,
    LABELS,
)


@dataclass
class RandomForestConfig:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }


class RandomForestN2OPredictor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required for leaf nodes
            max_features: Number of features considered when searching for the best split
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.is_fitted = False
        self.features_names = None
        self.target_name = None

    def fit(self, df: pd.DataFrame) -> 'RandomForestN2OPredictor':
        features_cols = (
            NUMERIC_STATIC_FEATURES
            + NUMERIC_DYNAMIC_FEATURES_RF
            + CATEGORICAL_STATIC_FEATURES
            + CATEGORICAL_DYNAMIC_FEATURES
        )
        target_col = LABELS[0]
        self.features_names = features_cols
        self.target_name = target_col

        X = df[features_cols].values
        y = df[target_col].values
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                'The model has not been trained, so feature importance is meaningless.'
            )

        X = df[self.features_names].values
        return self.model.predict(X)

    def get_feature_importances(self) -> dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError(
                'The model has not been trained, so feature importance is meaningless.'
            )

        importances = self.model.feature_importances_
        return dict(zip(self.features_names, importances, strict=False))  # type: ignore

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> 'RandomForestN2OPredictor':
        assert path.exists(), f'The model path {path} does not exist.'
        with path.open('rb') as f:
            return pickle.load(f)

    def count_parameters(self) -> int:
        if not self.is_fitted:
            return 0

        total_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
        return total_nodes
