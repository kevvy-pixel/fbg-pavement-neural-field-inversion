# FBG 神经动力学场 v5：最终少样本校准实验

v5 用于验证 95 cm 极端边界工况的最终校准方案。它保留 Residual-FiLM
作为主模型，对比：

1. 不校准；
2. 标量偏置校准；
3. 正则化仿射校准；
4. 仅微调输出层。

空间边界专家仍保留在 `RUN_ALL` 的主模型消融中，但不再重复参与少样本微调。
v4 已证明其校准收益接近零。

## 1. 推荐运行方式：复用旧模型

你已经完成主模型训练，解压后直接运行：

```matlab
RUN_FIXED_FEWSHOT( ...
    "D:\path\to\FBG_NeuralField_Workspace\outputs\20260727_203124_quick", ...
    "D:\path\to\Response_Matrix_Table.csv", ...
    "quick", ...
    "D:\path\to\FBG_NeuralField_Workspace")
```

旧运行目录只需包含：

```text
02_residual_film/model.mat
```

输入 CSV 必须与旧模型训练时使用的数据一致。程序会检查空间范围、条件维数和
归一化尺度。

## 2. 首次检查

如需先验证代码：

```matlab
CHECK_ENVIRONMENT
RUN_DEMO
```

## 3. 配置

| profile | 唯一空间划分数 | 输出层迭代数 | 用途 |
| --- | ---: | ---: | --- |
| `demo` | 2 | 8 | 流程检查 |
| `quick` | 10 | 120 | 推荐的最终快速实验 |
| `research` | 20 | 500 | 正式论文实验 |

建议先运行 quick。结果合理后，仍可复用相同旧模型运行 research，无需重训主干。

## 4. 公平评测协议

每个重复：

- 从 28 个目标坐标中选择 8 个校准候选；
- 剩余 20 个坐标作为固定测试集；
- `k=0,2,4,6,8` 使用嵌套校准点；
- 同一重复中的所有方法和 k 使用相同测试坐标；
- 每个重复的 8 点校准池必须与其他重复不同。

`unique_split_check.csv` 保存校准池签名，用于检查不存在重复划分。

## 5. 四种方法

| method | 公式或操作 |
| --- | --- |
| `residual_zero` | 直接使用 Residual-FiLM |
| `bias_only` | \(\hat y^*=\hat y+b\) |
| `ridge_affine` | \(\hat y^*=a\hat y+b\)，并正则约束 \(a\to1,b\to0\) |
| `output_head` | 冻结神经场，仅更新输出层 |

岭仿射默认设置：

```matlab
cfg.fewshot.affineSlopePenalty = 1.0;
cfg.fewshot.affineOffsetPenalty = 0.05;
cfg.fewshot.affineScaleBounds = [0.5 1.5];
```

这些值在运行前固定，不使用测试集调参。

## 6. 输出文件

```text
04_final_calibration_95cm/
├─ unique_split_check.csv
├─ fixed_holdout_repeat_metrics.csv
├─ fixed_holdout_summary.csv
├─ paired_improvement_summary.csv
├─ method_decision_table.csv
├─ best_method_by_k.csv
├─ fixed_holdout_predictions.csv
├─ fixed_holdout_performance.png
└─ repeat_01/
   ├─ calibration_pool.csv
   ├─ fixed_test_points.csv
   ├─ k_00/
   ├─ k_02/
   └─ ...
```

重点文件：

- `fixed_holdout_summary.csv`：均值和标准差；
- `paired_improvement_summary.csv`：相对零样本的配对提升和 RMSE 胜率；
- `method_decision_table.csv`：复杂校准是否同时优于偏置和零样本；
- `best_method_by_k.csv`：每个 k 的最低平均 RMSE 方法；
- `unique_split_check.csv`：证明所有空间划分唯一。

## 7. 修正后的复杂度判定

`ridge_affine` 和 `output_head` 必须同时满足：

- 相对 `bias_only`：R² 至少提高 0.02，RMSE 至少降低 5%；
- 相对 `residual_zero`：R² 至少提高 0.02，RMSE 至少降低 5%。

只有四项全部满足，`passes_complexity_threshold` 才为 true。这样不会再因为
某个较差的两点偏置结果而错误判定复杂模型有效。

## 8. 完整重训

只有无法使用旧模型时才运行：

```matlab
RUN_ALL("quick", ...
    "D:\data\Response_Matrix_Table.csv", ...
    "D:\FBG_runs")
```

`RUN_ALL` 会重新训练 Plain-FiLM、Residual-FiLM 和边界专家，因此耗时明显更长。

## 9. 运行后上传

请上传：

```text
fixed_holdout_summary.csv
paired_improvement_summary.csv
method_decision_table.csv
best_method_by_k.csv
unique_split_check.csv
run.log
```
