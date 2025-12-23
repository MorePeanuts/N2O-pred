import pickle
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from typing import Literal
from dataclasses import dataclass
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from .data import (
    NUMERIC_STATIC_FEATURES,
    NUMERIC_DYNAMIC_FEATURES_RF,
    CATEGORICAL_STATIC_FEATURES,
    CATEGORICAL_DYNAMIC_FEATURES,
    LABELS,
    NUMERIC_DYNAMIC_FEATURES_RNN,
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


@dataclass
class RNNConfig: ...


class N2OPredictorRF:
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

    def fit(self, df: pd.DataFrame) -> 'N2OPredictorRF':
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
    def load(path: Path) -> 'N2OPredictorRF':
        assert path.exists(), f'The model path {path} does not exist.'
        with path.open('rb') as f:
            return pickle.load(f)

    def count_parameters(self) -> int:
        if not self.is_fitted:
            return 0

        total_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
        return total_nodes


class RNNModel(nn.Module):
    def __init__(
        self,
        num_numeric_static: int,
        num_numeric_dynamic: int,
        categorical_static_cardinalities: list[int],
        categorical_dynamic_cardinalities: list[int],
        embedding_dim: int = 8,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
        mlp_hidden_dim: int = 64,
        rnn_type: Literal['GRU', 'LSTM'] = 'LSTM',
    ):
        """
        Args:
            num_numeric_static: 静态数值特征数量
            num_numeric_dynamic: 动态数值特征数量
            categorical_static_cardinalities: 静态分类特征的类别数列表
            categorical_dynamic_cardinalities: 动态分类特征的类别数列表
            embedding_dim: 嵌入维度
            hidden_size: RNN隐藏层大小
            num_layers: RNN层数
            rnn_type: RNN类型 ('GRU' 或 'LSTM')
            dropout: Dropout比例
        """
        super().__init__()

        self.num_numeric_static = num_numeric_static  # type: ignore
        self.num_numeric_dynamic = num_numeric_dynamic  # type: ignore
        self.hidden_size = hidden_size  # type: ignore
        self.num_layers = num_layers  # type: ignore
        self.rnn_type = rnn_type  # type: ignore

        # 静态分类特征的Embedding层
        self.static_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, embedding_dim)
                for cardinality in categorical_static_cardinalities
            ]
        )

        # 动态分类特征的Embedding层
        self.dynamic_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, embedding_dim)
                for cardinality in categorical_dynamic_cardinalities
            ]
        )

        # 计算静态特征的总维度
        static_dim = num_numeric_static + len(categorical_static_cardinalities) * embedding_dim

        # 静态特征MLP（生成RNN初始hidden）
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size * num_layers),
        )

        # 计算动态特征的总维度
        dynamic_dim = num_numeric_dynamic + len(categorical_dynamic_cardinalities) * embedding_dim

        # 动态特征投影层
        self.dynamic_projection = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_size),
        )

        # RNN层
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f'Unsupported RNN type: {rnn_type}')

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    if 'rnn' in name:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        numeric_static_features: torch.Tensor,
        categorical_static_features: torch.Tensor,
        numeric_dynamic_features: torch.Tensor,
        categorical_dynamic_features: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            numeric_static_features: [batch_size, num_static_numeric]
            categorical_static_features: [batch_size, num_static_categorical]
            numeric_dynamic_features: [batch_size, max_seq_len, num_dynamic_numeric]
            categorical_dynamic_features: [batch_size, max_seq_len, num_dynamic_categorical]
            seq_lengths: [batch_size]

        Returns:
            predictions: [batch_size, max_seq_len]
        """
        batch_size, max_seq_len = (
            numeric_dynamic_features.shape[0],
            numeric_dynamic_features.shape[1],
        )

        # 1. 处理静态特征
        # 嵌入静态分类特征
        static_cat_embedded = []
        for i, embedding in enumerate(self.static_embeddings):
            static_cat_embedded.append(embedding(categorical_static_features[:, i]))

        if static_cat_embedded:
            static_cat_embedded = torch.cat(static_cat_embedded, dim=1)
            static_features = torch.cat([numeric_static_features, static_cat_embedded], dim=1)
        else:
            static_features = numeric_static_features

        # 通过MLP生成初始hidden state
        h0 = self.static_mlp(static_features)  # [batch_size, hidden_size * num_layers]
        h0 = h0.view(
            batch_size, self.num_layers, self.hidden_size
        )  # [batch_size, num_layers, hidden_size]
        h0 = h0.transpose(0, 1).contiguous()  # [num_layers, batch_size, hidden_size]

        # 2. 处理动态特征
        # 嵌入动态分类特征
        dynamic_cat_embedded = []
        for i, embedding in enumerate(self.dynamic_embeddings):
            dynamic_cat_embedded.append(embedding(categorical_dynamic_features[:, :, i]))

        if dynamic_cat_embedded:
            dynamic_cat_embedded = torch.stack(
                dynamic_cat_embedded, dim=2
            )  # [batch_size, max_seq_len, num_cat, emb_dim]
            dynamic_cat_embedded = dynamic_cat_embedded.view(
                batch_size, max_seq_len, -1
            )  # [batch_size, max_seq_len, num_cat * emb_dim]
            dynamic_features = torch.cat([numeric_dynamic_features, dynamic_cat_embedded], dim=2)
        else:
            dynamic_features = numeric_dynamic_features

        # 投影动态特征
        dynamic_projected = self.dynamic_projection(
            dynamic_features
        )  # [batch_size, max_seq_len, hidden_size]

        # 3. 通过RNN
        # Pack padded sequence以处理变长序列
        packed_input = nn.utils.rnn.pack_padded_sequence(
            dynamic_projected, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        if self.rnn_type == 'LSTM':
            # LSTM需要cell state
            c0 = torch.zeros_like(h0)
            packed_output, _ = self.rnn(packed_input, (h0, c0))
        else:
            packed_output, _ = self.rnn(packed_input, h0)

        # Unpack
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_seq_len
        )  # [batch_size, max_seq_len, hidden_size]

        # 4. 输出层
        predictions = self.output_layer(rnn_output).squeeze(-1)  # [batch_size, max_seq_len]

        return predictions


class N2OPredictorRNN:
    def __init__(
        self,
    ):
        """
        Args:

        """
        super().__init__()

    def fit(self): ...

    def predict(self): ...

    def save(self): ...

    def load(self): ...

    def count_parameters(self): ...
