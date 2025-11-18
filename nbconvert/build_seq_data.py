#!/usr/bin/env python
# coding: utf-8

# # Sequence Data Construction
# 
# This notebook is used to construct the sequence data required for three model schemes:
# 
# - Scheme 1 (ObsStep): observation step, event-driven
# 
# - Scheme 2 (DailyStep): daily step, time-driven
# 
# - Scheme 3 (RF): random forest baseline

# In[1]:


import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Set a random seed to ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED}")


# ## 1. General Preprocessing

# In[2]:


# Load data
data_path = Path('../datasets/data_EUR_reordered.csv')
df = pd.read_csv(data_path)
print(f"Original data shape: {df.shape}")
print(f"Columns: {df.columns}")


# In[3]:


# Remove features with high missing rates: NH4+-N, NO3_-N, MN, C/N
high_missing_features = ['NH4+-N', 'NO3_-N', 'MN', 'C/N']
df = df.drop(columns=high_missing_features)
print(f"After dropping high missing features: {df.shape}")
print(f"Remaining columns: {df.columns}")


# In[4]:


# Handle missing values for TN (33% missing rate)
# First, forward-fill within each sequence, then fill any remaining missing values with the global median
print(f"TN missing ratio before: {df['TN'].isnull().mean():.2%}")

# Group by (Publication, control_group) and forward-fill within each sequence using .ffill()
df['TN'] = df.groupby(['Publication', 'control_group'])['TN'].transform(lambda x: x.ffill())

# Fill remaining missing values with the global median
df['TN'] = df['TN'].fillna(df['TN'].median())

print(f"TN missing ratio after: {df['TN'].isnull().mean():.2%}")


# In[5]:


# Group by (Publication, control_group) to build sequences
# Sort in ascending order by sowdur
df = df.sort_values(['Publication', 'control_group', 'sowdur']).reset_index(drop=True)

# Count the number of sequences
seq_groups = df.groupby(['Publication', 'control_group'])
n_sequences = len(seq_groups)
print(f"\nTotal number of sequences: {n_sequences}")

# Count the length distribution of sequences
seq_lengths = seq_groups.size()
print("\nSequence length statistics:")
print(seq_lengths.describe())


# In[6]:


# Filter out sequences that are too short (<8 steps)
MIN_SEQ_LENGTH = 8
valid_seq_ids = seq_lengths[seq_lengths >= MIN_SEQ_LENGTH].index
df = df[df.set_index(['Publication', 'control_group']).index.isin(valid_seq_ids)].reset_index(drop=True)

# Recount
seq_groups = df.groupby(['Publication', 'control_group'])
n_sequences_filtered = len(seq_groups)
print(f"\nAfter filtering sequences < {MIN_SEQ_LENGTH} steps:")
print(f"Number of sequences: {n_sequences_filtered} (removed {n_sequences - n_sequences_filtered})")
print(f"Total samples: {len(df)}")

seq_lengths_filtered = seq_groups.size()
print("\nFiltered sequence length statistics:")
print(seq_lengths_filtered.describe())


# In[7]:


# Data split: randomly split at the sequence level into 90% training set and 10% test set
all_seq_ids = list(seq_groups.groups.keys())
n_train = int(len(all_seq_ids) * 0.9)
n_test = len(all_seq_ids) - n_train

np.random.shuffle(all_seq_ids)
train_seq_ids = all_seq_ids[:n_train]
test_seq_ids = all_seq_ids[n_train:]

# Create training and test sets
df['split'] = 'test'
df.loc[df.set_index(['Publication', 'control_group']).index.isin(train_seq_ids), 'split'] = 'train'

print(f"\nData split (random_seed={RANDOM_SEED}):")
print(f"Train sequences: {n_train} ({n_train/len(all_seq_ids):.1%})")
print(f"Test sequences: {n_test} ({n_test/len(all_seq_ids):.1%})")
print(f"Train samples: {(df['split']=='train').sum()}")
print(f"Test samples: {(df['split']=='test').sum()}")


# ## 2. Scheme 1: Observation Step (ObsStep)
# 
# **Characteristics**: Event-driven; each step corresponds to one observation

# In[8]:


def build_obs_step_sequences(df_split: pd.DataFrame) -> list[dict]:
    """
    Construct observation step-length sequence data

    Features:
        - Each step corresponds to one observation
        - Compute the time interval time_delta
        - Handle ferdur: replace -1 with 0
        - Retain all features including ferdur, sowdur, time_delta
    """
    sequences = []

    # Define feature columns
    static_features = ['Clay', 'CEC', 'BD', 'pH', 'SOC', 'TN']
    classification_static_features = ['crop_class']
    dynamic_features = ['Temp', 'Prec', 'ST', 'WFPS']
    # fertilization_features = ['fertilization_class', 'Split N amount', 'appl_class', 'ferdur', 'sowdur']
    target = 'Daily fluxes'

    for (pub, group), seq_df in df_split.groupby(['Publication', 'control_group']):
        seq_df = seq_df.sort_values('sowdur').reset_index(drop=True)

        # Compute the time interval
        time_delta = np.zeros(len(seq_df))
        time_delta[1:] = seq_df['sowdur'].values[1:] - seq_df['sowdur'].values[:-1]

        # Handle ferdur: replace -1 with 0
        ferdur_processed = seq_df['ferdur'].values.copy()
        ferdur_processed[ferdur_processed < 0] = 0

        # Build the sequence dictionary
        sequence = {
            'seq_id': (pub, group),
            'seq_length': len(seq_df),

            # Static Features (Numerical)
            'static_numeric': seq_df[static_features].iloc[0].values.astype(np.float32),

            # Static Features (Classification)
            'static_categorical': {
                feat: seq_df[feat].iloc[0] for feat in classification_static_features
            },

            # Dynamic Features (Numerical)
            'dynamic_numeric': seq_df[dynamic_features].values.astype(np.float32),

            # Fertilization-related features
            'fertilization_categorical': {
                'fertilization_class': seq_df['fertilization_class'].values,
                'appl_class': seq_df['appl_class'].values
            },
            'fertilization_numeric': np.stack([
                seq_df['Split N amount'].values,
                ferdur_processed,
                seq_df['sowdur'].values,
                time_delta
            ], axis=1).astype(np.float32),

            # Target variable
            'target': seq_df[target].values.astype(np.float32)
        }

        sequences.append(sequence)

    return sequences

# Build Training and test Sets
print("Building ObsStep sequences...")
train_obs = build_obs_step_sequences(df[df['split'] == 'train'])
test_obs = build_obs_step_sequences(df[df['split'] == 'test'])

print(f"ObsStep Train sequences: {len(train_obs)}")
print(f"ObsStep Val sequences: {len(test_obs)}")
print(f"Example sequence keys: {train_obs[0].keys()}")
print(f"Example static_numeric shape: {train_obs[0]['static_numeric'].shape}")
print(f"Example dynamic_numeric shape: {train_obs[0]['dynamic_numeric'].shape}")
print(f"Example fertilization_numeric shape: {train_obs[0]['fertilization_numeric'].shape}")


# In[9]:


train_obs[0]


# In[10]:


# Save ObsStep sequence data
output_dir = Path('../datasets')
output_dir.mkdir(exist_ok=True)

with open(output_dir / 'sequences_obs_step_train.pkl', 'wb') as f:
    pickle.dump(train_obs, f)

with open(output_dir / 'sequences_obs_step_test.pkl', 'wb') as f:
    pickle.dump(test_obs, f)

print(f"\nObsStep sequences saved to {output_dir}")


# ## 3. Schema 2: DailyStep
# 
# **Characteristics**: Time-driven, each step corresponds to one day

# In[11]:


def build_daily_step_sequences(df_split: pd.DataFrame) -> list[dict]:
    """
    Construct daily step-sequence data

    Features:
        - Each step corresponds to one day
        - Daily data generated via interpolation
        - `Prec` filled with 0; other numeric features linearly interpolated
        - Fertilization feature reconstruction:
            * Split N amount > 0 on fertilization days, = 0 on other days
            * ferdur: days since last fertilization (increments by 1 each day), -1 if not fertilized yet
        - Remove `sowdur`
        - Create observation mask
    """
    sequences = []

    # Define feature columns
    static_features = ['Clay', 'CEC', 'BD', 'pH', 'SOC', 'TN']
    classification_static_features = ['crop_class']
    dynamic_numeric_features = ['Temp', 'ST', 'WFPS']  # Linear interpolation required
    prec_feature = 'Prec'  # Pad with 0
    fertilization_categorical = ['fertilization_class', 'appl_class']
    target = 'Daily fluxes'

    for (pub, group), seq_df in df_split.groupby(['Publication', 'control_group']):
        seq_df = seq_df.sort_values('sowdur').reset_index(drop=True)

        # Determine the range of daily step sizes
        min_day = int(seq_df['sowdur'].min())
        max_day = int(seq_df['sowdur'].max())
        n_days = max_day - min_day + 1

        # Create daily indices
        daily_index = np.arange(min_day, max_day + 1)

        # Create observation mask
        observed_mask = np.zeros(n_days, dtype=bool)
        obs_days = seq_df['sowdur'].values.astype(int)
        observed_mask[obs_days - min_day] = True

        # Initialize daily data
        daily_data = {
            'days': daily_index,
            'observed_mask': observed_mask
        }

        # Linear interpolation dynamic numerical features
        for feat in dynamic_numeric_features:
            values = seq_df[feat].values
            interpolated = np.interp(daily_index, obs_days, values)
            daily_data[feat] = interpolated.astype(np.float32)

        # Prec padded with 0
        daily_prec = np.zeros(n_days, dtype=np.float32)
        daily_prec[obs_days - min_day] = seq_df[prec_feature].values.astype(np.float32)
        daily_data[prec_feature] = daily_prec

        # Fertilization feature reconstruction
        # Calculate fertilization dates
        fertilization_days = []
        fertilization_amounts = []

        for _, row in seq_df.iterrows():
            if row['ferdur'] >= 0:  # Fertilization records exist
                fert_day = int(row['sowdur'] - row['ferdur'])
                if fert_day >= min_day and fert_day <= max_day:
                    fertilization_days.append(fert_day)
                    fertilization_amounts.append(row['Split N amount'])

        # Split N amount: Only > 0 on fertilization days
        daily_split_n = np.zeros(n_days, dtype=np.float32)
        for fert_day, amount in zip(fertilization_days, fertilization_amounts, strict=True):
            daily_split_n[fert_day - min_day] = amount
        daily_data['Split N amount'] = daily_split_n

        # ferdur: days since last fertilization
        # For each day, calculate how many days have passed since the last fertilization
        # If no fertilization yet, set to -1
        daily_ferdur = np.full(n_days, -1, dtype=np.float32)

        if fertilization_days:
            # Sort fertilization days
            sorted_fert_days = sorted(fertilization_days)

            for day_idx in range(n_days):
                current_day = daily_index[day_idx]
                # Find the most recent fertilization day before or on current day
                last_fert_day = None
                for fert_day in sorted_fert_days:
                    if fert_day <= current_day:
                        last_fert_day = fert_day
                    else:
                        break

                # If there was a fertilization before this day, calculate ferdur
                if last_fert_day is not None:
                    daily_ferdur[day_idx] = current_day - last_fert_day

        daily_data['ferdur'] = daily_ferdur

        # fertilization_class, appl_class: Forward fill
        for feat in fertilization_categorical:
            daily_values = np.empty(n_days, dtype=object)
            # Initialize with the first observation value
            daily_values[:] = seq_df[feat].iloc[0]
            # Forward fill
            for _, (day, value) in enumerate(zip(obs_days, seq_df[feat].values, strict=True)):
                day_idx = day - min_day
                daily_values[day_idx:] = value
            daily_data[feat] = daily_values

        # Target variable: Linear interpolation (only true values at observation points)
        target_values = seq_df[target].values
        daily_target = np.interp(daily_index, obs_days, target_values).astype(np.float32)
        daily_data[target] = daily_target

        # Build the sequence dictionary
        sequence = {
            'seq_id': (pub, group),
            'seq_length': n_days,
            'min_day': min_day,
            'max_day': max_day,

            # Static Features (Numerical)
            'static_numeric': seq_df[static_features].iloc[0].values.astype(np.float32),

            # Static Features (Classification)
            'static_categorical': {
                feat: seq_df[feat].iloc[0] for feat in classification_static_features
            },

            # Dynamic Features (Numerical) - Exclude time interval features
            'dynamic_numeric': np.stack([
                daily_data['Temp'],
                daily_data[prec_feature],
                daily_data['ST'],
                daily_data['WFPS']
            ], axis=1),

            # Fertilization-related features
            'fertilization_categorical': {
                feat: daily_data[feat] for feat in fertilization_categorical
            },
            'fertilization_numeric': np.stack([
                daily_data['Split N amount'],
                daily_data['ferdur']
            ], axis=1).astype(np.float32),

            # Observation mask
            'observed_mask': observed_mask,

            # Target variable
            'target': daily_target
        }

        sequences.append(sequence)

    return sequences

# Build Training and test Sets
print("Building DailyStep sequences...")
train_daily = build_daily_step_sequences(df[df['split'] == 'train'])
test_daily = build_daily_step_sequences(df[df['split'] == 'test'])

print(f"DailyStep Train sequences: {len(train_daily)}")
print(f"DailyStep Val sequences: {len(test_daily)}")
print(f"Example sequence keys: {train_daily[0].keys()}")
print(f"Example static_numeric shape: {train_daily[0]['static_numeric'].shape}")
print(f"Example dynamic_numeric shape: {train_daily[0]['dynamic_numeric'].shape}")
print(f"Example fertilization_numeric shape: {train_daily[0]['fertilization_numeric'].shape}")
print(f"Example observed_mask sum: {train_daily[0]['observed_mask'].sum()} / {train_daily[0]['seq_length']}")


# In[12]:


train_daily[2]


# check data:

# In[13]:


split_n_amount = train_daily[2]['fertilization_numeric']
print('calculated fertilization days: ', np.where(split_n_amount > 0)[0])
print('real fertilization days: ', np.array([20, 27, 34, 41, 48, 55, 62, 69, 76]))


# In[14]:


# Save DailyStep sequence data
with open(output_dir / 'sequences_daily_step_train.pkl', 'wb') as f:
    pickle.dump(train_daily, f)

with open(output_dir / 'sequences_daily_step_test.pkl', 'wb') as f:
    pickle.dump(test_daily, f)

print(f"\nDailyStep sequences saved to {output_dir}")


# ## 4. Schema 3: Random Forest Format
# 
# **Characteristics**: Treat each time point as an independent sample

# In[15]:


def build_rf_data(df_split: pd.DataFrame) -> pd.DataFrame:
    """
    Build Random Forest format data

    Characteristics:
        - Treat each time point as an independent sample
        - Features: All static features + current time step dynamic features
    """
    # Define id columns
    id_cols = ['No. of obs']
    # Define feature columns
    static_features = ['Clay', 'CEC', 'BD', 'pH', 'SOC', 'TN']
    classification_static_features = ['crop_class']
    dynamic_features = ['Temp', 'Prec', 'ST', 'WFPS']
    fertilization_features = ['fertilization_class', 'Split N amount', 'appl_class', 'ferdur']
    target = 'Daily fluxes'

    # Copy the data to avoid modifying the original data
    df_rf = df_split.copy()

    # Select all features
    all_features = (
        id_cols + static_features + classification_static_features + dynamic_features + fertilization_features
    )

    # Construct the final dataframe
    rf_data = df_rf[all_features + [target]].copy()

    return rf_data

# Building the Training and test Sets
print("Building Random Forest format data...")
train_rf = build_rf_data(df[df['split'] == 'train'])
test_rf = build_rf_data(df[df['split'] == 'test'])

print(f"RF Train samples: {len(train_rf)}")
print(f"RF Val samples: {len(test_rf)}")
features = train_rf.columns.drop('No. of obs')
print(f"RF features: {features.tolist()}")


# In[16]:


# Save RF format data
train_rf.to_csv(output_dir / 'rf_data_train.csv', index=False)
test_rf.to_csv(output_dir / 'rf_data_test.csv', index=False)

print(f"\nRF format data saved to {output_dir}")

