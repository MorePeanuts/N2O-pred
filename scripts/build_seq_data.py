import pandas as pd
import random
import numpy as np
import pickle
from pprint import pprint
from pathlib import Path

ordered_data_path = Path(__file__).parents[1] / 'datasets/data_EUR_reordered.csv'

if not ordered_data_path.exists():
    raise RuntimeError(
        f'Ordered datasets not found: {ordered_data_path}, run `gen_processed_data.py` first.'
    )

df = pd.read_csv(ordered_data_path)
print('Original data shape: ', df.shape)

# Remove features with high missing rates: NH4+-N, NO3_-N, MN, C/N
drop_features = ['NH4+-N', 'NO3_-N', 'MN', 'C/N']
df = df.drop(columns=drop_features)
print(f'After dropping high missing features: {df.shape}')
print(f'Remaining columns: {df.columns}')

print('Handle missing values for TN')
# First, forward-fill within each sequence, then fill any remaining missing values with the global median
print(f'TN missing ratio before: {df["TN"].isnull().mean():.2%}')
# Group by (Publication, control_group) and forward-fill within each sequence using .ffill()
df['TN'] = df.groupby(['Publication', 'control_group'])['TN'].transform(lambda x: x.ffill())
# Fill remaining missing values with the global median
df['TN'] = df['TN'].fillna(df['TN'].median())
print(f'TN missing ratio after: {df["TN"].isnull().mean():.2%}')

print('Group by (Publication, control_group) to build sequence...')
df = df.sort_values(['Publication', 'control_group', 'sowdur']).reset_index(drop=True)

# Count the number of sequences
seq_groups = df.groupby(['Publication', 'control_group'])
n_sequences = len(seq_groups)
print(f'\nTotal number of sequences: {n_sequences}')

# Count the length distribution of sequences
seq_lengths = seq_groups.size()
print('\nSequence length statistics:')
print(seq_lengths.describe())

print('Filter out sequences that are too short...')
MIN_SEQ_LENGTH = 8
valid_seq_ids = seq_lengths[seq_lengths >= MIN_SEQ_LENGTH].index  # type: ignore
df = df[df.set_index(['Publication', 'control_group']).index.isin(valid_seq_ids)].reset_index(
    drop=True
)

# Recount
seq_groups = df.groupby(['Publication', 'control_group'])
n_sequences_filtered = len(seq_groups)
print(f'\nAfter filtering sequences < {MIN_SEQ_LENGTH} steps:')
print(f'Number of sequences: {n_sequences_filtered} (removed {n_sequences - n_sequences_filtered})')
print(f'Total samples: {len(df)}')

seq_lengths_filtered = seq_groups.size()
print('\nFiltered sequence length statistics:')
print(seq_lengths_filtered.describe())

numeric_static_features = ['Clay', 'CEC', 'BD', 'pH', 'SOC', 'TN']
numeric_dynamic_features = ['Temp', 'Prec', 'ST', 'WFPS', 'Split N amount', 'ferdur']
classification_static_features = ['crop_class']
classification_dynamic_features = ['fertilization_class', 'appl_class']
group_variables = ['No. of obs', 'Publication', 'control_group', 'sowdur']
labels = ['Daily fluxes']

output_path = ordered_data_path.parent / 'data_EUR_sequential.pkl'
sequences = []
for (pub, group), seq_df in df.groupby(['Publication', 'control_group']):  # type: ignore
    sequence = {
        'seq_id': [int(pub), int(group)],
        'seq_length': len(seq_df),
        'No. of obs': seq_df['No. of obs'].values.astype(int),  # type: ignore
        'sowdurs': seq_df['sowdur'].values.astype(int),  # type: ignore
        'numeric_static': seq_df[numeric_static_features].iloc[0].values.astype(np.float32),  # type: ignore
        'numeric_dynamic': seq_df[numeric_dynamic_features].values.astype(np.float32),  # type: ignore
        'categorical_static': {
            k: seq_df[k].iloc[0]  # type: ignore
            for k in classification_static_features
        },
        'categorical_dynamic': {
            k: seq_df[k].values  # type: ignore
            for k in classification_dynamic_features
        },
        'targets': seq_df[labels].values.astype(np.float32),  # type: ignore
    }
    sequences.append(sequence)
with open(output_path, 'wb') as f:
    pickle.dump(sequences, f)
print(f'Sequential data has been written to {output_path}')
print('Sequential data example:')
pprint(random.choice(sequences))
