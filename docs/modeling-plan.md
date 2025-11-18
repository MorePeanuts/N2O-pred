## 土壤 N2O 排放量随时间变化的建模方案

### 数据特征

静态数值特征

- `Clay`：粘土含量
- `CEC`：阳离子交换容量
- `BD`：土壤容重
- `pH`：土壤 pH 值
- `SOC`：有机碳含量
- `TN`：总氮含量

动态数值特征（随时间变化）

- `Temp`：温度
- `Prec`：降水量
- `ST`：土壤温度
- `WFPS`：土壤含水量
- `Split N amount`：上次施肥量
- `ferdur`：距离上次施肥的天数

静态分类特征：

- `crop_class`：作物类型

动态分类特征（随时间变化）

- `fertilization_class`：上次施肥类型
- `appl_class`：上次施肥方式

标签：

- `Daily fluxes`：N2O排放通量

辅助变量：

- `sowdur`：距离作物播种的时间
- `No. of obs`：样本点的 ID
- `Publication`：文献编号
- `control_group`：组别编号

> RNN 模型需要使用序列数据，使用 `(Publication, control)` 来定位一个序列，按照 `sowdur` 升序排列

### 随机森林（Baseline）

- 将每个时间点作为独立的样本（等价为展平后的序列）
- 使用所有的特征进行训练和预测
- 分类变量使用 LabelEncoder 编码后输入模型

### RNN-ObsStep

- 观测事件驱动：每个 step 对应一次观测事件
- 分类特征使用 LabelEncoder 后通过 Embedding 层进行嵌入，静态分类变量和动态分类变量分别使用一个 Embedding 层
- 计算相邻两个 step 之间的时间间隔：`time_delta = sowdur[t] - sowdur[t-1]`（首个时间步为0）

### RNN-DailyStep

- 时间驱动：每个 step 对应一天
- 对于每个序列，从第一天（min sowdur）到最后一天（max sowdur），生成连续的每日时间步
- 鉴于序列中间隔不均匀，插入样本点时需要执行以下规则：
  - 静态数值特征插值策略：前向填充（同一序列保持一致）
  - 动态数值特征插值策略：
    - `Prec` 降水量采用 0 进行填充
    - `Split N amount` 采用前向填充
    - `ferdur` 通过计算得到，如果 ferdur = -1，则表示还没有施肥，使用 -1 填充；如果 ferdur >= 0，则进行 +1 递增填充
    - `Temp`, `ST`, `WFPS` 使用线性插值
  - 动态分类特征：前向填充
  - 标签：线性填充，但是在训练时使用掩码损失函数，不参与模型更新
- 分类特征使用 LabelEncoder 后通过 Embedding 层进行嵌入，静态分类变量和动态分类变量分别使用一个 Embedding 层
- 观测掩码：`observed_mask[i] = 1` 表示第 i 天是真实观测点，0 则表示是填充的点
