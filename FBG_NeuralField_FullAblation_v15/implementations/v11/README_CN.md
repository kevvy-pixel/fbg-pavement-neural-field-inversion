# FBG 共享主干多层神经场 v11

本包验证一个严格控制变量的多层假设：在 Step19 风格 FiLM 神经场上，
土层辅助监督能否在不增加独立大网络的情况下改善沥青层零样本预测。

比较两个模型：

1. `Step19-SOTA-FiLM`：仅使用沥青层训练；
2. `SharedMultilayer-FiLM`：使用同一 FiLM 主干，在土层样本上仅增加一个
   线性位移残差头。

两者使用完全相同的空间特征、条件编码、主干宽度、FiLM 深度、随机种子、
划分和训练预算。多层模型没有 `eta` 应变比例约束。

## 1. 相对 v10 的关键变化

- 不再把双点加载压缩成几何中心，完整保留 `p1/p2/hasP2`；
- 恢复 Step19 风格的边界距离、测点—加载点相对坐标和多频 Fourier 特征；
- 使用两层条件编码器和 5 层、96 宽 FiLM 主干；
- 在物理坐标尺度上对位移代理场求二阶导，输出微应变；
- 多层模型采用 `共享位移场 + 土层线性残差头`；
- 沥青和土层采用独立标准差归一化与可学习任务不确定性权重；
- 两种模型均使用沥青验证 RMSE 选择最佳权重，保证主任务早停标准一致。

多层位移代理写作：

```text
w_asphalt = w_shared
w_soil    = w_shared + r_soil
```

其中 `r_soil` 只是共享隐藏表示上的线性输出，不是独立神经网络。

## 2. 环境与运行

要求 MATLAB R2024a+ 和 Deep Learning Toolbox。

```matlab
clear functions
rehash toolboxcache
CHECK_ENVIRONMENT
RUN_DEMO
```

真实数据诊断：

```matlab
RUN_DATA_DIAGNOSTICS( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

600 次迭代快速对照：

```matlab
RUN_QUICK( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

完整 5 折、每折最多 1800 次迭代：

```matlab
RUN_FULL( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

## 3. 原生数据兼容

项目原生 `Response_Matrix_Table.csv` 可直接读取：

- `x_cm/y_cm`：传感器坐标；
- `p1_x/p1_y`：第一加载点；
- `p2_x/p2_y`：第二加载点，缺失时使用 p1 填充特征并令 `hasP2=0`；
- `load_kg`：按标准重力换算为 kN；
- `delta_lambda_resp_nm`：按 `1.2e-3 nm/με` 换算为微应变；
- `Asphalt/Soil`：分别规范化为沥青层和土层；
- 温度补偿记录被排除。

已验证真实数据包含 1100 条有效记录、22 个工况，其中 8 个双点工况、
400 条双点记录被完整保留。

## 4. 输出

每次实验写入：

```text
工作区/outputs/时间戳_模式_v11/
```

主要输出：

- `predictions_all.csv`；
- `global_asphalt_summary.csv`；
- `target_95cm_summary.csv`；
- `soil_auxiliary_summary.csv`；
- `paired_improvement_summary.csv`；
- `learned_task_weights.csv`；
- `fold_XX/模型名/best_model.mat`；
- 各模型训练历史、逐点预测和逐工况指标。

## 5. 已完成的真实数据快速结果

配置：固定种子 2026，95 cm 工况测试，600 次最大迭代，两模型共享相同主干
与训练预算。

### 沥青层 95 cm

| 模型 | R² | RMSE/με | MAE/με | Bias/με |
|---|---:|---:|---:|---:|
| Step19-SOTA-FiLM | 0.4618 | 187.57 | 135.11 | -60.96 |
| SharedMultilayer-FiLM | 0.2866 | 215.96 | 171.73 | -93.93 |

多层模型相对基线：

- RMSE 增加 `28.39 με`，退化 `15.13%`；
- MAE 增加 `36.63 με`；
- 因此总体结果为负向消融。

### 分方向结果

| 方向 | Step19 RMSE/με | 多层 RMSE/με | 变化 |
|---|---:|---:|---:|
| 纵向 x | 220.09 | 203.79 | 改善 16.30 |
| 横向 y | 132.32 | 231.19 | 退化 98.87 |

共享多层监督对纵向存在正信号，但横向任务冲突抵消并反转了总体收益。

### 训练诊断

- Step19 最佳沥青验证 RMSE：`64.00 με`，第 280 次；
- 多层最佳沥青验证 RMSE：`67.93 με`，第 600 次；
- 学习到的沥青/土层权重约为 `1.286 / 1.318`；
- 土层权重没有自动降低，说明简单不确定性加权不能解决方向冲突；
- 土层测试 RMSE 为 `196.76 με`，`R²=-1.683`。

## 6. 结论

本 v11 不替换 Step19-SOTA，应保留为结构消融。结果支持下一步只测试更小的
方向特异辅助机制，例如分别学习纵向/横向土层权重，或仅在纵向分支启用土层
辅助监督；不支持直接扩大共享多层网络或重新加入未经辨识的 `eta` 强约束。

当前结果来自一个 95 cm 留出工况，不等同于完整 5 折结论。只有候选结构在快速
实验中不损害主任务后，才值得投入完整 5 折训练。
