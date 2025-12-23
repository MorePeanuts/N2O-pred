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
    no_of_obs: list
    sowdurs: list
    numeric_static: list
    numeric_dynamic: list
    categorical_static: list
    categorical_dynamic: list
    targets: list

    @classmethod
    def from_dict(cls, sequence: dict):
        return cls(
            seq_id=tuple(sequence['seq_id']),
            seq_length=sequence['seq_length'],
            no_of_obs=sequence['No. of obs'],
            sowdurs=sequence['sowdurs'],
            numeric_static=sequence['numeric_static'],
            numeric_dynamic=sequence['numeric_dynamic'],
            categorical_static=sequence['categorical_static'],
            categorical_dynamic=sequence['categorical_dynamic'],
            targets=sequence['targets'],
        )

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
                row[name] = self.numeric_static[j]
            for j, name in enumerate(NUMERIC_DYNAMIC_FEATURES):
                row[name] = self.numeric_dynamic[i][j]
            for j, name in enumerate(CATEGORICAL_STATIC_FEATURES):
                row[name] = self.categorical_static[j]
            for j, name in enumerate(CATEGORICAL_DYNAMIC_FEATURES):
                row[name] = self.categorical_dynamic[i][j]
            for _, name in enumerate(LABELS):
                row[name] = self.targets[i]

            rows.append(row)

        return rows


class SequentialN2ODataset(Dataset):
    def __init__(
        self, data_path: Path | None = None, sequences: list[SequentialN2OData] | None = None
    ):
        if sequences:
            self.sequences = sequences
        elif data_path:
            assert data_path.exists(), f'data path {data_path} not exist.'
            with data_path.open('rb') as f:
                sequences = pickle.load(f)
                self.sequences = [SequentialN2OData.from_dict(seq_data) for seq_data in sequences]
        else:
            raise RuntimeError(
                'At least one of the parameters data_path and sequences must not be None.'
            )

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
