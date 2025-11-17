# RNN 模型预测土壤 N2O 排放量方案

## 模型方案对比

实现三种方案并对比：

- **方案1 (RNN-ObsStep)**：以观测时间为步长的 RNN，时间间隔作为输入特征
- **方案2 (RNN-DailyStep)**：以每日为步长的 RNN，通过插值生成每日数据，使用观测掩码损失
- **方案3 (RF-Baseline)**：随机森林模型作为 baseline 进行对比

## 1. 序列数据构建

**目标文件**: `notebooks/build_seq_data.ipynb`

### 通用预处理

- 从 `datasets/data_EUR_reordered.csv` 加载数据
- 删除高缺失率特征：NH4+-N, NO3_-N, MN（缺失率>60%）
- 处理 C/N 和 TN 的缺失值（33%缺失率）：序列内前向填充 + 全局中位数填充
- 按 (Publication, control_group) 分组构建序列
- 按 sowdur 升序排列每个序列
- 统计序列长度分布，过滤过短序列（如<10步）
- 数据划分：按序列级别随机划分为 90% 训练集，10% 验证集（记录随机种子=42）

### 方案1数据：观测步长（ObsStep）

**特点**：事件驱动，每个 step 对应一次观测

- 每个序列的步长对应一次观测
- 计算时间间隔：`time_delta = sowdur[t] - sowdur[t-1]`（首个时间步为0）
- 保存为 `datasets/sequences_obs_step_train.pkl` 和 `val.pkl`

### 方案2数据：每日步长（DailyStep）

**特点**：时间驱动，每个 step 对应一天

对每个序列，从第一天（min sowdur）到最后一天（max sowdur），生成连续的每日时间步：

**数值特征插值策略**：

- Prec（日降水量）：填充为 0（非观测日无降水记录）
- 其他数值特征（Temp, ST, WFPS等）：线性插值

**施肥相关特征重构**：

- 通过原始 ferdur 计算施肥日期，具体而言，如果 ferdur = -1，则表示还没有施肥，如果ferdur>= 0，则可以计算出上一次施肥的具体天数（RNN中的第几步）：`fertilization_day = sowdur - ferdur`
- Split N amount：只在施肥日>0，其他日=0（意义变为"当天施肥量"）
- fertilization_class, appl_class：前向填充（保持上次施肥类型和方式）
- 移除 ferdur, sowdur：不再需要时间间隔特征（由RNN步长隐式表达）

**分类特征**：

- crop_class：所有日期相同（静态特征）

**观测掩码**：

- `observed_mask[i] = 1` 如果第i天有真实观测，否则为0

保存为 `datasets/sequences_daily_step_train.pkl` 和 `val.pkl`

### 方案3数据：随机森林格式

- 将每个时间点作为独立样本（展平所有序列）
- 特征：所有静态特征 + 当前时间步的动态特征
- 保存为 `datasets/rf_data_train.csv` 和 `val.csv`

## 2. 特征工程和归一化

**目标文件**: `notebooks/feature_engineering.ipynb`

### 分类特征编码

- 使用 OneHotEncoder 编码：crop_class, fertilization_class, appl_class

### 数值特征归一化（基于训练集统计）

根据数据探索结果，采用不同的转换策略：

需要注意的是，仅在训练集上进行，且仅在真实样本上进行（排除掉方案2中的插值样本）

**Daily fluxes（目标变量）**：

- 先使用 Symlog 转换（处理极端突变和负值）
- 再使用 StandardScaler

**零膨胀和长尾分布特征**：

- Prec, Split N amount: log(x+1) 转换 + StandardScaler
- ferdur, sowdur: log(x+1) 转换 + StandardScaler（仅方案1）

**其他数值特征**：

- 静态特征（Clay, CEC, BD, pH, SOC, TN, C/N）: StandardScaler
- 动态特征（Temp, ST, WFPS）: StandardScaler
- 时间间隔（time_delta）: StandardScaler（仅方案1）

### 应用到RNN的2种方案的数据

- 加载方案1、2的数据
- 应用预处理器进行转换
- 保存转换后的数据（另存在datasets的子目录中）
- 需要注意，随机森林方案不需要进行特征工程

### 保存预处理器

- `preprocessor/scalers.pkl`
- `preprocessor/encoders.pkl`
- `preprocessor/feature_config.json`（记录每个特征的转换方法）

## 3. 模型架构实现

**目标文件**: `src/n2o_pred/models.py`

### 方案1和2的 RNN 模型

两种方案的模型代码是通用的，仅数据集和输入维度存在区别。

```python
class N2OPredictorRNN(nn.Module):
```

**通用架构**：

- 静态特征通过 MLP 编码，在每个时间步与动态特征拼接
- LSTM/GRU 层（2层，hidden_size=128）
- 全连接输出层映射到标量预测值

### 方案3的随机森林

- sklearn.ensemble.RandomForestRegressor
- 超参数搜索：n_estimators, max_depth, min_samples_split

## 4. 数据加载器

**目标文件**: `src/n2o_pred/dataset.py`

- `SequenceDataset_ObsStep`：方案1的 Dataset，处理变长序列
- `SequenceDataset_DailyStep`：方案2的 Dataset，包含 observed_mask，也要处理变长序列
- 实现相应的 collate_fn（填充到batch中的最长序列，并使用padding_mask）

## 5. 训练流程

### 任务管理

- 每次训练生成唯一任务编号（train_{model}_{datetime}等）
- 所有输出保存到 `outputs/train_{model}_{datetime}/` 
- 训练配置和随机种子记录到 `outputs/train_{model}_{datetime}/config.json`

### 训练 Notebooks（三个独立文件）

**`notebooks/train_rf_baseline.ipynb`**（方案3）

- 加载随机森林格式数据
- RandomForestRegressor 训练和超参数搜索
- 评估：RMSE, MAE, R²
- 保存到 `outputs/train_rf_{datetime}/`

**`notebooks/train_rnn_obs_step.ipynb`**（方案1）

- 加载观测步长序列数据
- 训练 N2ORNN_ObsStep 模型
- 损失函数：MSELoss with padding mask
- 优化器：Adam，学习率调度：ReduceLROnPlateau
- 早停策略
- 保存到 `outputs/train_rnn_obs_{datetime}/`

**`notebooks/train_rnn_daily_step.ipynb`**（方案2）

- 加载每日步长序列数据
- 训练 N2ORNN_DailyStep 模型
- 损失函数：MSELoss with observed mask（只在真实观测点计算）
- 优化器：Adam，学习率调度：ReduceLROnPlateau
- 早停策略
- 保存到 `outputs/train_rnn_daily_{datetime}/`

### 训练输出

每个任务目录包含：

- `best_model.pth` 或 `best_model.pkl`：最佳模型
- `config.json`：训练配置、随机种子、超参数
- `training_log.csv`：训练过程记录（loss, metrics）
- `predictions.pkl`：验证集预测结果
- `figures/`：训练曲线等可视化

## 6. 模型对比分析

**目标文件**: `notebooks/model_comparison.ipynb`

### 评估指标对比

- 加载三个最佳模型的验证集预测结果
- 计算并对比 RMSE, MAE, R² 指标
- 生成对比表格

### 推理可视化

- 从验证集随机采样若干序列（如10-20个）
- 使用三个模型分别进行推理
- 对每个样本序列，绘制预测 vs 真实曲线（三个模型的预测在同一张图）
- 展示不同序列长度、不同施肥条件下的效果

### 误差分析

- 分析哪些情况下各模型预测效果较差
- 不同施肥类型、作物类型下的误差分布
- 序列长度对预测效果的影响

### 输出

- 生成对比报告保存到 `outputs/model_comparison_{datetime}/models-comparation.md`
- 保存可视化图表到 `outputs/model_comparison_{datetime}/figures/`
- 评测指标对比和使用的模型路径和配置到 `outputs/model_comparison_{datetime}/config.json`

## 关键技术点

1. **特征转换**：根据数据分布特性选择合适的转换（Symlog、log(x+1)、StandardScaler）

2. **方案1（观测步）vs 方案2（每日步）的核心区别**：

   - 方案1：事件驱动，保留时间间隔特征（time_delta, ferdur, sowdur）
   - 方案2：时间驱动，移除时间间隔特征，施肥特征重构为"当天施肥量"

3. **方案2的特殊处理**：

   - Prec 填充为0，其他数值特征线性插值
   - 通过 ferdur 计算施肥日期，Split N amount 只在施肥日>0
   - 损失只在 observed_mask=1 处计算

4. **任务管理**：每次训练生成唯一任务编号，输出按任务组织到 outputs/

5. **可复现性**：记录随机种子、配置到文件