#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

raw_data = pd.read_csv('../datasets/data_EUR_raw.csv', na_values=['-'])


# Check the data type of each column in the raw data.

# In[15]:


raw_data.dtypes


# Divide the variables into different groups

# In[16]:


numeric_static_features = {
    'Clay': '粘土含量',
    'CEC': '阳离子交换容量',
    'BD': '土壤容重',
    'pH': 'pH',
    'SOC': '有机碳含量',
    'TN': '总氮含量',
    'C/N': '碳氮比',
}

numeric_dynamic_features = {
    'Temp': '温度',
    'Prec': '降水量',
    'ST': '土壤温度',
    'WFPS': '土壤含水量',
    'NH4+-N': '铵态氮',
    'NO3_-N': '硝态氮',
    'MN': '矿质氮含量（铵态+硝态氮）',
}

classification_static_features = {
    'crop_class': '作物类型',
}

fertilization_features = {
    'fertilization_class': '上次施肥类型',
    'Split N amount': '上次施肥量',
    'appl_class': '上次施肥方式',
    'ferdur': '该次测量距上次施肥的天数',
}

optional_features = {'NH4+-N': '铵态氮', 'NO3_-N': '硝态氮', 'MN': '矿质氮含量（铵态+硝态氮）'}

auxiliary_variables = {
    'Sowing date': '播种日期',
    'Harvest date': '收获日期',
    'Fertilization date': '施肥日期',
    'Emission date': '排放日期(观测日期)',
}

group_variables = {
    'No. of obs': 'ID',
    'Publication': '文献编号',
    'control_group': '组号',
    'sowdur': '该次测量到播种之间的天数',
}

drop_variables = {
    'Crop type',
    'Irrigation (date)',
    'Pesticides/Herbicides (date)',
    'Fertilizer N type',
    'fertilization_code',
    'inhibitor',
    'Total N amount',
    'Appl.code',
    'SE',
    'Duration',
}

labels = {
    'Daily fluxes': 'N2O排放通量',
}


# Make sure our group can cover all the columns.

# In[17]:


len(
    set(raw_data.columns)
    - numeric_dynamic_features.keys()
    - numeric_static_features.keys()
    - fertilization_features.keys()
    - classification_static_features.keys()
    - group_variables.keys()
    - drop_variables
    - auxiliary_variables.keys()
    - labels.keys()
) == 0


# Remove redundant columns, adjust the column order and delete all samples from the fallow period.

# In[18]:


processed_data = raw_data.drop(columns=drop_variables)
processed_data = processed_data[
    [
        *group_variables.keys(),
        *labels.keys(),
        *classification_static_features.keys(),
        *numeric_static_features.keys(),
        *fertilization_features.keys(),
        *numeric_dynamic_features.keys(),
    ]
]

print('Before drop all samples of fallow period, total sample: ', len(processed_data))
fallow_mask = (processed_data['sowdur'] < 0) | (processed_data['control_group'] < 0)
print('Fallow samples: ', fallow_mask.sum())
processed_data = processed_data[~fallow_mask]
print('After delete all samples of fallow period, processed sample: ', len(processed_data))


# For samples where `Split N amount` is greater than 0 but `ferdur` is less than zero, reset `ferdur` to 0 to facilitate model processing.

# In[19]:


processed_data.loc[
    (~processed_data['Split N amount'].isnull()) & (processed_data['ferdur'] < 0), 'ferdur'
] = 0


# Fill the missing parts of `Split N amount` with zeros.

# In[20]:


processed_data['Split N amount'] = processed_data['Split N amount'].fillna(0)


# Check the statistical characteristics and missing proportion of numeric types.

# In[21]:


numeric_columns = (
    labels.keys()
    | numeric_static_features.keys()
    | numeric_dynamic_features.keys()
    | {'ferdur', 'Split N amount'}
)

numeric_df = processed_data[list(numeric_columns)]
stats = numeric_df.describe().transpose()

missing_ratio = numeric_df.isnull().sum() / len(numeric_df) * 100
missing_df = missing_ratio.to_frame(name='missing_ratio (%)')

numeric_summary = stats.join(missing_df).round(3)
numeric_summary


# Ensure that the `Split N amount` column only has missing values when `ferdur < 0`.

# In[22]:


processed_data[(processed_data['Split N amount'].isnull()) & (processed_data['ferdur'] >= 0)]


# In[23]:


processed_data[(~processed_data['Split N amount'].isnull()) & (processed_data['ferdur'] < 0)]


# Check the statistical characteristics and missing proportion of classification variables.

# In[24]:


classification_columns = classification_static_features.keys() | (
    fertilization_features.keys() - {'ferdur', 'Split N amount'}
)

classification_df = processed_data[list(classification_columns)]
stats = classification_df.describe().transpose()

missing_ratio = classification_df.isnull().sum() / len(classification_df) * 100
missing_df = missing_ratio.to_frame(name='missing_ratio (%)')

classification_summary = stats.join(missing_df).round(3)
classification_summary


# Sort according to the variables in `group_variables` and save the processed variables to `../datasets/data_EUR_reordered.csv`.

# In[25]:


reordered_data = processed_data.sort_values(
    by=['Publication', 'control_group', 'sowdur'], ascending=True
)
reordered_data.to_csv('../datasets/data_EUR_reordered.csv', index=False)
reordered_data


# In[26]:


reordered_data.dtypes
