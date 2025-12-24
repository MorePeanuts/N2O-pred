import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass


# 定义字段分组
NUMERIC_STATIC_FEATURES = ['Clay', 'CEC', 'BD', 'pH', 'SOC', 'TN']
# 完整的动态数值特征列表（用于序列数据存储）
NUMERIC_DYNAMIC_FEATURES = [
    'Temp',
    'Prec',
    'ST',
    'WFPS',
    'Split N amount',
    'ferdur',
    'Total N amount',
]
CATEGORICAL_STATIC_FEATURES = ['crop_class']
CATEGORICAL_DYNAMIC_FEATURES = ['fertilization_class', 'appl_class']
GROUP_VARIABLES = ['No. of obs', 'Publication', 'control_group', 'sowdur']
DROP_VARIABLES = ['NH4+-N', 'NO3_-N', 'MN', 'C/N']
LABELS = ['Daily fluxes']

# 模型专用的特征列表
# RNN模型专用：使用前6个特征（不包括Total N amount）
NUMERIC_DYNAMIC_FEATURES_RNN = ['Temp', 'Prec', 'ST', 'WFPS', 'Split N amount', 'ferdur']
# RF模型专用：去掉Split N amount和ferdur，使用Total N amount
NUMERIC_DYNAMIC_FEATURES_RF = ['Temp', 'Prec', 'ST', 'WFPS', 'Total N amount']


@dataclass
class SequentialN2OData:
    seq_id: tuple[int, int]
    seq_length: int
    no_of_obs: list[int]
    sowdurs: list[int]
    numeric_static: pd.Series
    numeric_dynamic: pd.DataFrame
    categorical_static: pd.Series
    categorical_dynamic: pd.DataFrame
    targets: pd.DataFrame

    @classmethod
    def from_dict(cls, sequence: dict):
        return cls(
            seq_id=tuple(sequence['seq_id']),
            seq_length=sequence['seq_length'],
            no_of_obs=list(int(obs) for obs in sequence['No. of obs']),
            sowdurs=list(int(dur) for dur in sequence['sowdurs']),
            numeric_static=pd.Series(sequence['numeric_static'], index=NUMERIC_STATIC_FEATURES),
            numeric_dynamic=pd.DataFrame(
                sequence['numeric_dynamic'],
                columns=NUMERIC_DYNAMIC_FEATURES,  # type: ignore
            ),
            categorical_static=pd.Series(
                sequence['categorical_static'], index=CATEGORICAL_STATIC_FEATURES
            ),
            categorical_dynamic=pd.DataFrame(
                sequence['categorical_dynamic'],
                columns=CATEGORICAL_DYNAMIC_FEATURES,  # type: ignore
            ),
            targets=pd.DataFrame(sequence['targets'], columns=LABELS),  # type: ignore
        )

    def to_dict(self):
        return {
            'seq_id': list(self.seq_id),
            'seq_length': self.seq_length,
            'No. of obs': self.no_of_obs,
            'sowdurs': self.sowdurs,
            'numeric_static': self.numeric_static.tolist(),
            'numeric_dynamic': self.numeric_dynamic.to_numpy().tolist(),
            'categorical_static': self.categorical_static.tolist(),
            'categorical_dynamic': self.categorical_dynamic.to_numpy().tolist(),
            'targets': self.targets.to_numpy().tolist(),
        }

    def to_pd_rows(self):
        rows = []
        for i in range(self.seq_length):
            row = {
                'No. of obs': self.no_of_obs[i],
                'Publication': self.seq_id[0],
                'control_group': self.seq_id[1],
                'sowdur': self.sowdurs[i],
            }

            for j, name in enumerate(NUMERIC_STATIC_FEATURES):
                row[name] = self.numeric_static.iloc[j]
            for j, name in enumerate(NUMERIC_DYNAMIC_FEATURES):
                row[name] = self.numeric_dynamic.iloc[i, j]
            for j, name in enumerate(CATEGORICAL_STATIC_FEATURES):
                row[name] = self.categorical_static.iloc[j]
            for j, name in enumerate(CATEGORICAL_DYNAMIC_FEATURES):
                row[name] = self.categorical_dynamic.iloc[i, j]
            for j, name in enumerate(LABELS):
                row[name] = self.targets.iloc[i, j]

            rows.append(row)

        return rows


class SequentialN2ODataset(Dataset):
    def __init__(
        self,
        data_path: Path | None = None,
        sequences: list[SequentialN2OData] | None = None,
        encoders_path: Path | None = None,
    ):
        if sequences:
            self.sequences = sequences
        elif data_path:
            assert data_path.exists(), f'data path {data_path} doesnot exist.'
            with data_path.open('rb') as f:
                sequences = pickle.load(f)
                self.sequences = [SequentialN2OData.from_dict(seq_data) for seq_data in sequences]
        else:
            raise RuntimeError(
                'At least one of the parameters data_path and sequences must not be None.'
            )

        if encoders_path is None:
            encoders_path = Path(__file__).parents[2] / 'datasets/encoders.pkl'
        assert encoders_path.exists(), f'encoders path {encoders_path} doesnot exist.'
        with encoders_path.open('rb') as f:
            self.encoders = pickle.load(f)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):  # type: ignore
        if isinstance(idx, (list, np.ndarray)):
            sub_sequences = [self.sequences[i] for i in idx]
            return SequentialN2ODataset(sequences=sub_sequences)
        return self.sequences[idx]

    def flatten_to_dataframe(self) -> pd.DataFrame:
        rows = []

        for seq_data in self.sequences:
            row_data = seq_data.to_pd_rows()
            rows.extend(row_data)

        return pd.DataFrame(rows)

    def get_categorical_static_cardinalities(self):
        return [len(self.encoders[feature].classes_) for feature in CATEGORICAL_STATIC_FEATURES]

    def get_categorical_dynamic_cardinalities(self):
        return [len(self.encoders[feature].classes_) for feature in CATEGORICAL_DYNAMIC_FEATURES]

    def get_num_numeric_static(self):
        return len(NUMERIC_STATIC_FEATURES)

    def get_num_numeric_dynamic(self):
        return len(NUMERIC_DYNAMIC_FEATURES_RNN)


# TODO:为N2O模型实现数据集，考虑特征工程的位置
class N2ODatasetForLSTM(Dataset):
    def __init__(self, seq_dataset, train_set: bool, scalers: dict | None = None):
        pass

    def _expand_to_daily_sequences(self):
        pass
