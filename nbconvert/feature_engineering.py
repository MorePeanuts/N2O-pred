#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering and Normalization
# 
# This notebook is used for feature engineering and normalization of sequential data

# In[1]:


import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ## 1. Load Sequence Data

# In[2]:


# Load data
data_dir = Path('../datasets')

with open(data_dir / 'sequences_obs_step_train.pkl', 'rb') as f:
    train_obs = pickle.load(f)
with open(data_dir / 'sequences_obs_step_test.pkl', 'rb') as f:
    test_obs = pickle.load(f)

with open(data_dir / 'sequences_daily_step_train.pkl', 'rb') as f:
    train_daily = pickle.load(f)
with open(data_dir / 'sequences_daily_step_test.pkl', 'rb') as f:
    test_daily = pickle.load(f)

print(f"ObsStep - Train: {len(train_obs)}, Val: {len(test_obs)}")
print(f"DailyStep - Train: {len(train_daily)}, Val: {len(test_daily)}")


# ## 2. Categorical Feature Encoding
# 
# Use `OneHotEncoder` to encode `crop_class`, `fertilization_class`, and `appl_class`.

# In[3]:


# Collect unique values of categorical features
def collect_categorical_values(sequences, feature_name, is_static=True):
    values = []
    for seq in sequences:
        if is_static:
            values.append(seq['static_categorical'][feature_name])
        else:
            values.extend(seq['fertilization_categorical'][feature_name])
    return np.unique(values)

crop_class_values = collect_categorical_values(train_obs, 'crop_class', is_static=True)
fertilization_class_values = collect_categorical_values(train_obs, 'fertilization_class', is_static=False)
appl_class_values = collect_categorical_values(train_obs, 'appl_class', is_static=False)

print(f"crop_class: {crop_class_values}")
print(f"fertilization_class: {fertilization_class_values}")
print(f"appl_class: {appl_class_values}")


# In[4]:


# Create OneHotEncoders
encoders = {}
encoders['crop_class'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoders['crop_class'].fit(crop_class_values.reshape(-1, 1))
encoders['fertilization_class'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoders['fertilization_class'].fit(fertilization_class_values.reshape(-1, 1))
encoders['appl_class'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoders['appl_class'].fit(appl_class_values.reshape(-1, 1))

encoders['crop_class'].transform(crop_class_values.reshape(-1, 1))


# ## 3. Numerical Feature Transformation and Normalization

# In[5]:


# Define conversion function
def symlog_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))

def inverse_symlog_transform(y):
    return np.sign(y) * (np.exp(np.abs(y)) - 1)

def log1p_transform(x):
    return np.log1p(x)

# Feature Configuration
feature_config = {
    'transformations': {
        'Daily fluxes': 'symlog',
        'Prec': 'log1p',
        'Split N amount': 'log1p',
        'ferdur': 'log1p',
        'sowdur': 'log1p',
        'others': 'standard'
    }
}


# In[6]:


# Collecting Numerical Features of the Training Set
def collect_features(sequences, key):
    return np.vstack([seq[key] if seq[key].ndim > 1 else seq[key].reshape(-1, 1) for seq in sequences])

# ObsStep Features
train_target_obs = collect_features(train_obs, 'target')
train_dynamic_obs = collect_features(train_obs, 'dynamic_numeric')
train_fert_obs = collect_features(train_obs, 'fertilization_numeric')
train_static_obs = np.vstack([seq['static_numeric'] for seq in train_obs])

def collect_masked_features(sequences, key):
    lst = []
    for seq in sequences:
        mask = seq['observed_mask']
        lst.append(seq[key][mask] if seq[key].ndim > 1 else seq[key].reshape(-1, 1)[mask])
    return np.vstack(lst)

# DailyStep Features
train_target_daily = collect_masked_features(train_daily, 'target')
train_dynamic_daily = collect_masked_features(train_daily, 'dynamic_numeric')
train_fert_daily = collect_masked_features(train_daily, 'fertilization_numeric')
train_static_daily = np.vstack([seq['static_numeric'] for seq in train_daily])


# In[7]:


train_target_obs.shape, train_dynamic_obs.shape, train_fert_obs.shape, train_static_obs.shape


# In[8]:


train_target_daily.shape, train_dynamic_daily.shape, train_fert_daily.shape, train_static_daily.shape


# In[9]:


# Create Scalers
scalers = {}

# static features
scalers['static_numeric_obs'] = StandardScaler()
scalers['static_numeric_obs'].fit(train_static_obs)
scalers['static_numeric_daily'] = StandardScaler()
scalers['static_numeric_daily'].fit(train_static_daily)

# ObsStep dynamic features (Prec needs log1p)
train_dynamic_obs_t = train_dynamic_obs.copy()
train_dynamic_obs_t[:, 1] = log1p_transform(train_dynamic_obs[:, 1])
scalers['dynamic_numeric_obs'] = StandardScaler()
scalers['dynamic_numeric_obs'].fit(train_dynamic_obs_t)

# ObsStep fertilization features (Split N amount requires log1p)
train_fert_obs_t = train_fert_obs.copy()
train_fert_obs_t[:, :3] = log1p_transform(train_fert_obs[:, :3])
scalers['fertilization_numeric_obs'] = StandardScaler()
scalers['fertilization_numeric_obs'].fit(train_fert_obs_t)

# DailyStep dynamic features (Prec needs log1p)
train_dynamic_daily_t = train_dynamic_daily.copy()
train_dynamic_daily_t[:, 1] = log1p_transform(train_dynamic_daily[:, 1])
scalers['dynamic_numeric_daily'] = StandardScaler()
scalers['dynamic_numeric_daily'].fit(train_dynamic_daily_t)

# DailyStep fertilization features (Split N amount, ferdur)
# For ferdur: need to handle -1 specially (indicates no fertilization yet)
train_fert_daily_t = train_fert_daily.copy()
# Split N amount (column 0): log1p transform
train_fert_daily_t[:, 0] = log1p_transform(train_fert_daily[:, 0])
# ferdur (column 1): log1p only for non-negative values, keep -1 as is
mask_fert = train_fert_daily[:, 1] >= 0
train_fert_daily_t[mask_fert, 1] = log1p_transform(train_fert_daily[mask_fert, 1])
scalers['fertilization_numeric_daily'] = StandardScaler()
scalers['fertilization_numeric_daily'].fit(train_fert_daily_t)

# target variable (symlog)
train_target_t = symlog_transform(train_target_obs)
scalers['target_obs'] = StandardScaler()
scalers['target_obs'].fit(train_target_t)

train_target_t = symlog_transform(train_target_daily)
scalers['target_daily'] = StandardScaler()
scalers['target_daily'].fit(train_target_t)


# In[10]:


seq = train_obs[0]
scalers['static_numeric_obs'].transform(seq['static_numeric'][None, :]).flatten()


# ## 4. Apply to RNN dataset

# In[11]:


def transform_obs_sequences(seqs):
    result = []
    for seq in seqs:
        t_seq = {'seq_id': seq['seq_id'], 'seq_length': seq['seq_length']}
        t_seq['static_numeric'] = scalers['static_numeric_obs'].transform(seq['static_numeric'][None, :]).flatten().astype(np.float32)
        t_seq['static_categorical_encoded'] = encoders['crop_class'].transform([[seq['static_categorical']['crop_class']]]).flatten().astype(np.float32)

        dyn = seq['dynamic_numeric'].copy()
        dyn[:, 1] = log1p_transform(dyn[:, 1])
        t_seq['dynamic_numeric'] = scalers['dynamic_numeric_obs'].transform(dyn).astype(np.float32)

        fert = seq['fertilization_numeric'].copy()
        fert[:, :3] = log1p_transform(fert[:, :3])
        t_seq['fertilization_numeric'] = scalers['fertilization_numeric_obs'].transform(fert).astype(np.float32)

        t_seq['fertilization_categorical_encoded'] = {
            'fertilization_class': encoders['fertilization_class'].transform(seq['fertilization_categorical']['fertilization_class'].reshape(-1,1)).astype(np.float32),
            'appl_class': encoders['appl_class'].transform(seq['fertilization_categorical']['appl_class'].reshape(-1,1)).astype(np.float32)
        }

        tgt = symlog_transform(seq['target'])
        t_seq['target'] = scalers['target_obs'].transform(tgt.reshape(-1,1)).flatten().astype(np.float32)
        t_seq['target_original'] = seq['target']
        result.append(t_seq)
    return result

print("Transforming ObsStep...")
train_obs_t = transform_obs_sequences(train_obs)
test_obs_t = transform_obs_sequences(test_obs)
print(f"Done: {len(train_obs_t)} train, {len(test_obs_t)} test")


# In[12]:


train_obs_t[0]


# In[13]:


def transform_daily_sequences(seqs):
    result = []
    for seq in seqs:
        t_seq = {
            'seq_id': seq['seq_id'], 'seq_length': seq['seq_length'],
            'min_day': seq['min_day'], 'max_day': seq['max_day'],
            'observed_mask': seq['observed_mask']
        }
        t_seq['static_numeric'] = scalers['static_numeric_daily'].transform(seq['static_numeric'][None, :]).flatten().astype(np.float32)
        t_seq['static_categorical_encoded'] = encoders['crop_class'].transform([[seq['static_categorical']['crop_class']]]).flatten().astype(np.float32)

        dyn = seq['dynamic_numeric'].copy()
        dyn[:, 1] = log1p_transform(dyn[:, 1])
        t_seq['dynamic_numeric'] = scalers['dynamic_numeric_daily'].transform(dyn).astype(np.float32)

        fert = seq['fertilization_numeric'].copy()
        # Split N amount (column 0): log1p transform
        fert[:, 0] = log1p_transform(fert[:, 0])
        # ferdur (column 1): log1p only for non-negative values
        mask_fert = fert[:, 1] >= 0
        fert[mask_fert, 1] = log1p_transform(fert[mask_fert, 1])
        t_seq['fertilization_numeric'] = scalers['fertilization_numeric_daily'].transform(fert).astype(np.float32)

        t_seq['fertilization_categorical_encoded'] = {
            'fertilization_class': encoders['fertilization_class'].transform(seq['fertilization_categorical']['fertilization_class'].reshape(-1,1)).astype(np.float32),
            'appl_class': encoders['appl_class'].transform(seq['fertilization_categorical']['appl_class'].reshape(-1,1)).astype(np.float32)
        }

        tgt = symlog_transform(seq['target'])
        t_seq['target'] = scalers['target_daily'].transform(tgt.reshape(-1,1)).flatten().astype(np.float32)
        t_seq['target_original'] = seq['target']
        result.append(t_seq)
    return result

print("Transforming DailyStep...")
train_daily_t = transform_daily_sequences(train_daily)
test_daily_t = transform_daily_sequences(test_daily)
print(f"Done: {len(train_daily_t)} train, {len(test_daily_t)} test")


# ## 5. Save

# In[14]:


processed_dir = data_dir / 'processed'
processed_dir.mkdir(exist_ok=True)

with open(processed_dir / 'sequences_obs_step_train_processed.pkl', 'wb') as f:
    pickle.dump(train_obs_t, f)
with open(processed_dir / 'sequences_obs_step_test_processed.pkl', 'wb') as f:
    pickle.dump(test_obs_t, f)
with open(processed_dir / 'sequences_daily_step_train_processed.pkl', 'wb') as f:
    pickle.dump(train_daily_t, f)
with open(processed_dir / 'sequences_daily_step_test_processed.pkl', 'wb') as f:
    pickle.dump(test_daily_t, f)

print(f"Saved to {processed_dir}")


# In[15]:


preprocessor_dir = Path('../preprocessor')
preprocessor_dir.mkdir(exist_ok=True)

with open(preprocessor_dir / 'encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
with open(preprocessor_dir / 'scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
with open(preprocessor_dir / 'feature_config.json', 'w') as f:
    json.dump(feature_config, f, indent=2)

print(f"\nPreprocessors saved to {preprocessor_dir.absolute()}")

