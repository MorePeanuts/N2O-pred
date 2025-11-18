import pandas as pd
from pathlib import Path

raw_data_path = Path(__file__).parents[1] / 'datasets/data_EUR_raw.csv'

if not raw_data_path.exists():
    raise RuntimeError(f'Raw dataset not found: {raw_data_path}')

raw_data = pd.read_csv(raw_data_path, na_values=['-'])

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
    'Split N amount': '上次施肥量',
    'ferdur': '该次测量距上次施肥的天数',
}

classification_static_features = {
    'crop_class': '作物类型',
}

classification_dynamic_features = {
    'fertilization_class': '上次施肥类型',
    'appl_class': '上次施肥方式',
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
    'Sowing date',
    'Harvest date',
    'Fertilization date',
    'Emission date',
}

labels = {
    'Daily fluxes': 'N2O排放通量',
}

columns_covered = (
    len(
        set(raw_data.columns)
        - numeric_dynamic_features.keys()
        - numeric_static_features.keys()
        - classification_dynamic_features.keys()
        - classification_static_features.keys()
        - group_variables.keys()
        - drop_variables
        - labels.keys()
    )
    == 0
)

if not columns_covered:
    raise RuntimeError('Columns not coverd.')

print('Remove redundant columns...')
processed_data = raw_data.drop(columns=drop_variables)  # type: ignore
processed_data = processed_data[
    [
        *group_variables.keys(),
        *numeric_static_features.keys(),
        *numeric_dynamic_features.keys(),
        *classification_static_features.keys(),
        *classification_dynamic_features.keys(),
        *labels.keys(),
    ]
]
print('✅done.\n')

print('Remove all fallow period samples...')
print('Before drop all samples of fallow period, total sample: ', len(processed_data))
fallow_mask = (processed_data['sowdur'] < 0) | (processed_data['control_group'] < 0)
print('Fallow samples: ', fallow_mask.sum())
processed_data = processed_data[~fallow_mask]
print('After delete all samples of fallow period, processed sample: ', len(processed_data))
print('✅done.\n')

print(
    'For samples where `Split N amount` is greater than 0 but `ferdur` is less than zero, '
    'reset `ferdur` to 0 to facilitate model processing.'
)
processed_data.loc[
    (~processed_data['Split N amount'].isnull()) & (processed_data['ferdur'] < 0), 'ferdur'
] = 0
print('✅done.\n')

print('Fill the missing parts of `Split N amount` with zeros.')
processed_data['Split N amount'] = processed_data['Split N amount'].fillna(0)
print('✅done.\n')

print('Check the statistical characteristics and missing proportion of numeric types.')
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
print(numeric_summary)
print('✅done.\n')

print('Check the statistical characteristics and missing proportion of classification variables.')
classification_columns = (
    classification_static_features.keys() | classification_dynamic_features.keys()
)

classification_df = processed_data[list(classification_columns)]
stats = classification_df.describe().transpose()

missing_ratio = classification_df.isnull().sum() / len(classification_df) * 100
missing_df = missing_ratio.to_frame(name='missing_ratio (%)')

classification_summary = stats.join(missing_df).round(3)
print(classification_summary)
print('✅done.\n')

print('Sort according to the variables in `group_variables`')
processed_data = processed_data.sort_values(
    by=list(group_variables.keys() - {'No. of obs'}),
    ascending=True,
)
print(processed_data.dtypes)
processed_data.to_csv(raw_data_path.parent / 'data_EUR_reordered.csv')
print('Data saved.\n✅done.\n')
