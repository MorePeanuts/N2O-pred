import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass, field


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
    mask: list[bool] | None = field(default=None, init=False)

    @classmethod
    def from_dict(cls, sequence: dict):
        sowdurs = list(int(dur) for dur in sequence['sowdurs'])
        return cls(
            seq_id=tuple(sequence['seq_id']),
            seq_length=sequence['seq_length'],
            no_of_obs=list(int(obs) for obs in sequence['No. of obs']),
            sowdurs=sowdurs,
            numeric_static=pd.Series(sequence['numeric_static'], index=NUMERIC_STATIC_FEATURES),
            numeric_dynamic=pd.DataFrame(
                sequence['numeric_dynamic'],
                columns=NUMERIC_DYNAMIC_FEATURES,  # type: ignore
                index=sowdurs,  # type: ignore
            ),
            categorical_static=pd.Series(
                sequence['categorical_static'], index=CATEGORICAL_STATIC_FEATURES
            ),
            categorical_dynamic=pd.DataFrame(
                sequence['categorical_dynamic'],
                columns=CATEGORICAL_DYNAMIC_FEATURES,  # type: ignore
                index=sowdurs,  # type: ignore
            ),
            targets=pd.DataFrame(sequence['targets'], columns=LABELS, index=sowdurs),  # type: ignore
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

    def expand_to_daily_sequence(self):
        """
        Through interpolation, the interval between sample points in the sequence
         is adjusted to 1 day, and the mask field is used to distinguish between real
         sample points and interpolated sample points.

        [WARNING] This function performs an in-place operation; if you need to preserve
         the original sequence data, you should make a copy first.
        """
        obs_map = {self.sowdurs[i]: i for i in range(self.seq_length)}
        start_day = self.sowdurs[0]
        end_day = self.sowdurs[-1]
        no_of_obs = pd.Series(self.no_of_obs, index=self.sowdurs)
        self.seq_length = end_day - start_day + 1
        self.mask = list(True if i in obs_map else False for i in range(start_day, end_day + 1))
        self.sowdurs = list(range(start_day, end_day + 1))

        # 先将插值点全部设为缺失值
        self.numeric_dynamic = self.numeric_dynamic.reindex(self.sowdurs)
        self.categorical_dynamic = self.categorical_dynamic.reindex(self.sowdurs)
        self.targets = self.targets.reindex(self.sowdurs)
        no_of_obs = no_of_obs.reindex(self.sowdurs)

        # 对不同的列进行不同的插值操作
        self.no_of_obs = no_of_obs.fillna(-1).astype(int).tolist()
        self.categorical_dynamic = self.categorical_dynamic.ffill().bfill().astype(int)
        self.targets = self.targets.interpolate().ffill().bfill()
        self.numeric_dynamic['Prec'] = self.numeric_dynamic['Prec'].fillna(0)
        self.numeric_dynamic[['Temp', 'ST', 'WFPS']] = (
            self.numeric_dynamic[['Temp', 'ST', 'WFPS']].interpolate().ffill().bfill()
        )

        # 处理距离上次施肥的天数
        pos_ferdur = self.numeric_dynamic['ferdur'].mask(
            self.numeric_dynamic['ferdur'] <= 0, np.nan
        )
        fert_date = (self.numeric_dynamic.index - pos_ferdur).dropna().astype(int).unique().tolist()
        min_fert_date = min(fert_date) if len(fert_date) > 0 else 0
        last_fert_sowdur = pd.Series(np.nan, index=np.arange(min_fert_date, end_day + 1, dtype=int))
        last_fert_sowdur.loc[fert_date] = fert_date
        last_fert_sowdur = last_fert_sowdur.ffill().reindex(self.sowdurs)
        # last_fert_sowdur = last_fert_sowdur.fillna(last_fert_sowdur.index.to_series())
        last_fert_sowdur = last_fert_sowdur.fillna(-1)
        # self.numeric_dynamic['ferdur'] = (self.numeric_dynamic.index - last_fert_sowdur).astype(int)
        self.numeric_dynamic['ferdur'] = pd.Series(
            np.where(
                last_fert_sowdur == -1,
                0,
                (self.numeric_dynamic.index - last_fert_sowdur).astype(int),
            ),
            index=self.sowdurs,
        )

        # 根据插值后的ferdur处理施肥量
        self.numeric_dynamic[['Split N amount', 'Total N amount']] = (
            self.numeric_dynamic[['Split N amount', 'Total N amount']]
            .fillna(
                self.numeric_dynamic[['Split N amount', 'Total N amount']]
                .groupby(last_fert_sowdur)
                .transform('first')
            )
            .fillna(0)
        )

    def __repr__(self):
        def _fmt_list(lst, max_len=6):
            """辅助函数：格式化列表，过长则截断显示"""
            if lst is None:
                return 'None'
            n = len(lst)
            if n == 0:
                return '[]'
            if n <= max_len:
                return str(lst)
            return (
                f'[{", ".join(map(str, lst[:3]))}, ..., {", ".join(map(str, lst[-3:]))}] (len={n})'
            )

        def _fmt_df(df):
            """辅助函数：格式化DataFrame信息"""
            if df is None:
                return 'None'
            return f'Shape={df.shape}, Cols={list(df.columns)}'

        def _fmt_series(s):
            """辅助函数：格式化Series信息"""
            if s is None:
                return 'None'
            # 转换成字典形式显示的更紧凑，如果太长只显示Index
            d = s.to_dict()
            if len(d) > 5:
                return f'Shape={s.shape}, Index={list(s.index)}'
            return str(d)

        # 计算 mask 中 True 的比例（如果有的话）
        mask_info = 'None'
        if self.mask is not None:
            valid_count = sum(self.mask)
            total = len(self.mask)
            mask_info = f'{_fmt_list(self.mask)} (True Ratio: {valid_count}/{total})'

        # 构建输出字符串
        repr_str = [
            f'<SequentialN2OData | ID: {self.seq_id} | Length: {self.seq_length}>',
            '  [Metadata]',
            f'    - sowdurs:     {_fmt_list(self.sowdurs)}',
            f'    - no_of_obs:   {_fmt_list(self.no_of_obs)}',
            f'    - mask:        {mask_info}',
            '',
            '  [Static Features] (Invariant)',
            f'    - Numeric:     {_fmt_series(self.numeric_static)}',
            f'    - Categorical: {_fmt_series(self.categorical_static)}',
            '',
            '  [Dynamic Features] (Time Series)',
            f'    - Numeric:     {_fmt_df(self.numeric_dynamic)}',
            f'    - Categorical: {_fmt_df(self.categorical_dynamic)}',
            '',
            '  [Targets]',
            f'    - Data:        {_fmt_df(self.targets)}',
            '-' * 60,  # 分割线
        ]

        return '\n'.join(repr_str)

    def print(self, file=None):
        def _indent(text, prefix='    '):
            """给多行文本添加缩进，保持视觉层级"""
            if text is None:
                return '    None'
            return '\n'.join(prefix + line for line in str(text).splitlines())

        # 使用 option_context 强制显示所有行列，不折叠
        with pd.option_context(
            'display.max_rows',
            None,
            'display.max_columns',
            None,
            'display.width',
            1000,
            'display.precision',
            4,
        ):  # 您可以调整精度
            # 1. 基础元数据
            header = f'<SequentialN2OData | ID: {self.seq_id} | Length: {self.seq_length}>'

            # 2. 列表数据 (直接显示全部)
            meta_section = [
                '  [Metadata Lists]',
                f'    - sowdurs ({len(self.sowdurs)}):',
                f'{_indent(self.sowdurs)}',
                '',
                f'    - no_of_obs ({len(self.no_of_obs)}):',
                f'{_indent(self.no_of_obs)}',
                '',
                f'    - mask ({len(self.mask) if self.mask else 0}):',
                f'{_indent(self.mask)}',
            ]

            # 3. 静态特征 (Series)
            static_section = [
                '  [Static Features]',
                '    - Numeric:',
                f'{_indent(self.numeric_static)}',
                '',
                '    - Categorical:',
                f'{_indent(self.categorical_static)}',
            ]

            # 4. 动态特征 (DataFrame - 全量)
            dynamic_section = [
                '  [Dynamic Features]',
                '    - Numeric (Full DataFrame):',
                f'{_indent(self.numeric_dynamic)}',
                '',
                '    - Categorical (Full DataFrame):',
                f'{_indent(self.categorical_dynamic)}',
            ]

            # 5. 目标变量 (DataFrame - 全量)
            target_section = [
                '  [Targets]',
                '    - Data (Full DataFrame):',
                f'{_indent(self.targets)}',
            ]

            # 组合所有部分
            full_repr = (
                [header, '']
                + meta_section
                + ['']
                + static_section
                + ['']
                + dynamic_section
                + ['']
                + target_section
                + ['-' * 80]
            )

            print('\n'.join(full_repr), file=file)


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
            data_path = Path(__file__).parents[2] / 'datasets/data_EUR_processed.pkl'
            assert data_path.exists(), (
                'At least one of the parameters data_path and sequences must not be None.'
            )
            with data_path.open('rb') as f:
                sequences = pickle.load(f)
                self.sequences = [SequentialN2OData.from_dict(seq_data) for seq_data in sequences]

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
