# RNN 模型预测数据处理

## 构建序列数据

RNN-ObsStep 模型的数据：

- `../datasets/sequences_obs_step_train.pkl`
- `../datasets/sequences_obs_step_val.pkl`

数据结构：

- seq_id: (Publication, control_group)
- seq_length: sequence length
- static_numeric: (Clay, CEC, BD, pH, SOC, TN, C/N)
- static_categorical: dict(crop_class)
- dynamic_numeric: (Temp, Prec, ST, WFPS) * seq_length
- fertilization_categorical: dict(fertilization_class, appl_class) * seq_length
- fertilization_numeric: (Split N amount, ferdur, sowdur, time_delta) * seq_length
- target: (Daily fluxes) * seq_length

其中 time_delta 为序列中相邻两点间隔的天数

RNN-DailyStep 模型的数据：

- `../datasets/sequences_daily_step_train.pkl`
- `../datasets/sequences_daily_step_val.pkl`

数据结构：

- seq_id: (Publication, control_group)
- seq_length: sequence length (with padding, = max_day + 1)
- min_day: min sowdur
- max_day: max sowdur
- static_numeric: (Clay, CEC, BD, pH, SOC, TN, C/N)
- static_categorical: dict(crop_class)
- dynamic_numeric: (Temp, Prec, ST, WFPS) * seq_length
- fertilization_categorical: dict(fertilization_class, appl_class) * seq_length
- fertilization_numeric: (Split N amount) * seq_length
- observed_mask: only True at real observation
- target: (Daily fluxes) * seq_length

