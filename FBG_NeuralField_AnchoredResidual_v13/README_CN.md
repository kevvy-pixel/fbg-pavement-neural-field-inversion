# FBG 最近工况锚定残差神经场 v13

v13 针对 Step19 在 95 cm 工况的载荷位置外推崩溃。模型不再从零重建整个应变
场，而是先从训练数据库选择同加载类型的最近实测工况，再学习受距离门控的小残差。

本版本不使用测试工况的真实应变作为输入。95 cm 测试时自动选择训练集中的
90 cm 工况，使用两者共有的传感器坐标作为锚点。

## 1. 无泄漏锚点规则

- 训练样本：只能选择另一个训练工况，采用留一工况锚定；
- 验证样本：只能从训练工况选择锚点；
- 测试样本：只能从训练工况选择锚点；
- 优先选择相同 `load_type` 和相同单点/双点状态的工况；
- 工况距离同时考虑载荷位置、第二载荷点和载荷大小；
- 传感器优先同方向、同坐标匹配，否则只允许小范围最近点匹配；
- 所有选择结果写入 `anchor_audit.csv`，便于检查数据泄漏。

工况距离为：

```text
d = sqrt(
    delta_p1_x^2 + delta_p1_y^2
    + has_p2 * (delta_p2_x^2 + delta_p2_y^2)
    + (0.35 * delta_load)^2
)
```

位置、载荷均使用训练集尺度归一化。

## 2. 对照模型

1. `Step19-SOTA-FiLM`
   - 原始绝对应变场基线。
2. `NearestAnchor-Persistence`
   - 直接使用最近训练工况的同坐标实测应变；
   - 不训练神经网络，是必须超过的简单基线。
3. `AnchoredResidual-FiLM`
   - 纵向和横向均允许学习锚点修正。
4. `AnchoredResidual-XOnly-FiLM`
   - 只学习纵向修正；
   - 横向严格保持锚点值，防止方向任务冲突。

残差预测形式为：

```text
gate = 1 - exp(-condition_distance / 0.18)
prediction = anchor_strain + gate * neural_residual
```

`XOnly` 模型在横向令 `gate=0`。训练损失额外加入权重 `0.01` 的修正量平方惩罚，
使近邻工况保持保守修正。

## 3. 运行

要求 MATLAB R2024a 或更高版本，以及 Deep Learning Toolbox。

```matlab
clear functions
rehash toolboxcache
CHECK_ENVIRONMENT
RUN_DEMO
```

真实数据快速实验：

```matlab
RUN_QUICK( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

完整五折实验：

```matlab
RUN_FULL( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

## 4. 95 cm 真实数据结果

配置：随机种子 2026，95 cm 工况留出，最大 400 次迭代。95 cm 的 28 个沥青
测点全部匹配到 90 cm 训练工况：

- 锚点覆盖率：`100%`；
- 工况归一化距离：`0.04167 = 5/120`；
- 最大传感器匹配距离：`0`，即全部为同坐标匹配。

### 4.1 总体结果

| 模型 | R² | RMSE/μɛ | MAE/μɛ | Bias/μɛ |
|---|---:|---:|---:|---:|
| Step19-SOTA-FiLM | 0.4618 | 187.57 | 135.11 | -60.96 |
| NearestAnchor-Persistence | 0.8479 | 99.73 | 67.04 | -25.86 |
| AnchoredResidual-FiLM | 0.8400 | 102.27 | 67.91 | -15.49 |
| AnchoredResidual-XOnly-FiLM | **0.8574** | **96.55** | **60.87** | **-16.95** |

`AnchoredResidual-XOnly-FiLM` 相对 Step19：

- RMSE 降低 `91.02 μɛ`；
- RMSE 改善 `48.53%`；
- MAE 降低 `74.23 μɛ`；
- R² 提高 `0.3956`。

相对纯锚点，它又将 RMSE 降低 `3.18 μɛ`、MAE 降低 `6.17 μɛ`。提升不大，
说明本数据上主要收益来自正确使用 90 cm 实测锚点，而不是复杂网络。

### 4.2 分方向结果

| 模型 | x RMSE/μɛ | y RMSE/μɛ |
|---|---:|---:|
| Step19-SOTA-FiLM | 220.09 | 132.32 |
| NearestAnchor-Persistence | 125.06 | 48.51 |
| AnchoredResidual-FiLM | 124.75 | 60.47 |
| AnchoredResidual-XOnly-FiLM | **120.62** | **48.51** |

XY 残差会破坏原本很强的横向锚点，因此最终候选只修正纵向。

### 4.3 训练诊断

| 模型 | 最佳迭代 | 最佳验证 RMSE/μɛ | 停止迭代 |
|---|---:|---:|---:|
| Step19-SOTA-FiLM | 280 | 64.00 | 400 |
| AnchoredResidual-FiLM | 120 | 69.26 | 360 |
| AnchoredResidual-XOnly-FiLM | 40 | 74.79 | 280 |

验证集仍集中在 60 cm，而不是边界载荷位置，所以验证 RMSE 排名与 95 cm 测试
排名不一致。必须保留锚点基线和边界专用验证，不能只依赖当前早停指标。

## 5. 当前结论

对于 95 cm：

- 推荐模型：`AnchoredResidual-XOnly-FiLM`；
- 强制保留的安全基线：`NearestAnchor-Persistence`；
- 不建议继续使用 Step19 直接对载荷位置做零样本外推；
- v12 已证明土层辅助不能修复这一问题，因此 v13 不再把土层加入主损失。

该方案需要部署端拥有一个带实测响应的历史工况库。如果没有兼容锚点，应回退到
绝对场模型并报告低置信度，不能把锚定模型的结果解释为无参考零样本预测。

目前结果来自一个 95 cm 留出工况。投入最终使用前仍应运行五折，并重点增加
80、85、90、92.5、95 cm 等边界位置的重复试验。

## 6. 输出

每次实验写入 `工作区/outputs/时间戳_模式_v13/`：

- `predictions_all.csv`
- `global_asphalt_summary.csv`
- `target_95cm_summary.csv`
- `direction_asphalt_summary.csv`
- `anchor_summary.csv`
- `paired_improvement_summary.csv`
- `fold_XX/模型名/anchor_audit.csv`
- `fold_XX/模型名/test_predictions.csv`
- 神经网络模型的 `best_model.mat` 和 `training_history.csv`
