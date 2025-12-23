"""
Data processing script
"""

import pickle
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from n2o_pred.data import (
    NUMERIC_DYNAMIC_FEATURES,
    NUMERIC_STATIC_FEATURES,
    CATEGORICAL_DYNAMIC_FEATURES,
    CATEGORICAL_STATIC_FEATURES,
    DROP_VARIABLES,
    LABELS,
)


def preprocessing_data(raw_data_path: Path, output_path: Path, encoders_path: Path):
    # 加载原始数据
    logger.info(f'Loadding raw data from {raw_data_path}...')
    df = pd.read_csv(raw_data_path, index_col=0)
    df = df.reset_index()

    logger.info(f'Drop field: {DROP_VARIABLES}')
    df = df.drop(columns=DROP_VARIABLES, errors='ignore')

    logger.info('Change ferdur=-1 to 0')
    df.loc[df['ferdur'] == -1, 'ferdur'] = 0

    logger.info('Sort by (Publication, control_group, sowdur)')
    df = df.sort_values(['Publication', 'control_group', 'sowdur']).reset_index(drop=True)

    # 编码分类变量并保存编码器
    logger.info('Encoding categorical variables')
    categorical_features = CATEGORICAL_STATIC_FEATURES + CATEGORICAL_DYNAMIC_FEATURES
    encoders = {}

    for col in categorical_features:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        logger.info(f'  {col}: {len(encoder.classes_)} classes')

    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    with encoders_path.open('wb') as f:
        pickle.dump(encoders, f)
    logger.info(f'Encoders have been saved into {encoders_path}')

    # 构建时间序列数据
    logger.info('Building time series data...')
    sequences = []
    grouped_df = df.groupby(['Publication', 'control_group'])

    for (publication, control_group), group in grouped_df:  # type: ignore
        group = group.sort_values('sowdur').reset_index(drop=True)
        # TN 字段前向填充（同序列内）
        group['TN'] = group['TN'].ffill()
        # 如果第一个值是NaN，则后向填充
        group['TN'] = group['TN'].bfill()
        # 最后均值填充
        if group['TN'].isna().any():
            mean_tn = df['TN'].mean()
            group['TN'] = group['TN'].fillna(mean_tn if not pd.isna(mean_tn) else 1.0)

        seq_data = {
            'seq_id': [int(publication), int(control_group)],
            'seq_length': len(group),
            'No. of obs': group['No. of obs'].tolist(),
            'sowdurs': group['sowdur'].tolist(),
            'numeric_static': group[NUMERIC_STATIC_FEATURES].iloc[0].values.tolist(),
            'numeric_dynamic': group[NUMERIC_DYNAMIC_FEATURES].values.tolist(),
            'categorical_static': group[CATEGORICAL_STATIC_FEATURES].iloc[0].values.tolist(),
            'categorical_dynamic': group[CATEGORICAL_DYNAMIC_FEATURES].values.tolist(),
            'targets': group[LABELS].values.flatten().tolist(),
        }
        sequences.append(seq_data)

    logger.info(f'Total sequences: {len(sequences)}')
    seq_lengths = [seq['seq_length'] for seq in sequences]
    logger.info(
        f'Sequence length: min={min(seq_lengths)}, max={max(seq_lengths)}. mean={sum(seq_lengths) / len(seq_lengths):.2f}. median={sorted(seq_lengths)[len(seq_lengths) // 2]}'
    )

    # 保存序列数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as f:
        pickle.dump(sequences, f)
    logger.info(f'Preprocessed data have been saved into {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess the raw data and convert it into sequential data.')
    parser.add_argument('--input-path', type=str, help='')
    parser.add_argument('--output-path', type=str, help='')
    parser.add_argument('--encoders-path', type=str, help='')
    args = parser.parse_args()

    if args.input_path is None:
        raw_data_path = Path(__file__).parents[1] / 'datasets/data_EUR_raw.csv'
    else:
        raw_data_path = Path(args.input_path)
    if args.output_path is None:
        output_path = Path(__file__).parents[1] / 'datasets/data_EUR_processed.pkl'
    else:
        output_path = Path(args.output_path)
    if args.encoders_path is None:
        encoders_path = Path(__file__).parents[1] / 'datasets/encoders.pkl'
    else:
        encoders_path = Path(args.encoders_path)

    preprocessing_data(raw_data_path, output_path, encoders_path)
