# 数据处理流程

## 数据探索和重排序

```bash
uv run scripts/gen_processed_data.py
```

- 去除冗余列，包括：

```python
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
```

- 按照 `(Publication, control_group, sowdur)` 的顺序升序排列。

- 数据保存位置：`datasets/data_EUR_reordered.csv`

## 构建序列数据

```bash
uv run scripts/build_seq_data.py
```

- 删除高缺失列：

```python
drop_features = ['NH4+-N', 'NO3_-N', 'MN', 'C/N']
```

- 序列内前向填充 `TN`

- 按照 `(Publication, control_group, sowdur)` 分组构建序列

- 数据保存位置：`datasets/data_EUR_sequential.pkl`

- 序列数据结构：

```json
{
    "seq_id": ["Publication", "control_group"],
    "seq_length": "The length of sequence.",
    "No. of obs": ["All No. of obs of samples in the sequence"],
    "sowdurs": ["All sowdur of samples in the sequence"],
    "numeric_static": ["Clay", "CEC", "BD", "pH", "SOC", "TN"],
    "numeric_dynamic": [["Temp", "Prec", "ST", "WFPS", "Split N amount", "ferdur"], "..."],
    "categorical_static": {
        "crop_class": "crop_class"
    },
    "categorical_dynamic": {
        "fertilization_class": "fertilization_class",
        "appl_class": "appl_class"
    },
    "targets": ["Daily fluxes"]
}
```