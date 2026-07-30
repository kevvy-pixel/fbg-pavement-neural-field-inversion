# 新增实验与消融实验索引

本文档依据《FBG 路面响应反演模型完整实验演进报告》整理新增实验的目的、
证据等级和最终去留。早期快速筛选与后期严格对照采用不同训练预算和评估口径，
不能直接混合统计。

## 最终模型框架

1. 全局主干：`Step19_BoundaryAware_FiLM_AD_DNF.m`。
2. 无新工况测点：直接输出 Step19-SOTA 零样本预测。
3. 有 6 个稀疏测点且精度优先：使用 Ridge-6 仿射校准。
4. 有 6 个稀疏测点且风险优先：使用 Gated Ridge-6；校准证据不足时回退 SOTA。
5. Residual-FiLM、空间边界专家、Ridge-4 和 4 点门控仅作为消融证据。

## 版本与目录映射

| 阶段 | 目录 | 主要入口 | 目的 | 报告结论 | 定位 |
|---|---|---|---|---|---|
| 初版 | `experiments/ablations/ResidualMoE_initial` | `RUN_ALL.m` | Plain、Residual-FiLM、空间边界专家与 95 cm few-shot 初测 | 快速实验仅显示轻微收益；专家成本更高，证据不足 | 结构消融 |
| v4 | `experiments/ablations/FixedFewshot_v4` | `RUN_FIXED_FEWSHOT.m` | 固定测点下比较不同少样本校准方案 | 仅用于固定测点公平比较，不能证明空间选择稳健性 | 少样本消融 |
| v5 | `experiments/ablations/FinalCalibration_v5` | `RUN_FIXED_FEWSHOT.m` | 整理校准流程并补充固定 few-shot 运行 | 属于正式稳健性实验前的流程版本 | 历史/流程消融 |
| v6 | `experiments/ablations/GlobalBenchmark_v6` | `RUN_GLOBAL_BENCHMARK.m` | 22 工况快速全局基准、时长和复杂度比较 | 小网络短训练下 Residual/MoE 略好，只能作为继续实验的信号 | 快速筛选消融 |
| v7 | `experiments/ablations/SOTAFullComparison_v7` | `RUN_SOTA_FULL_COMPARISON.m` | 恢复 Step19 同规格做完整训练公平对照 | SOTA `R2=0.8467, RMSE=79.51`；Residual `R2=0.7106, RMSE=109.26` | 严格负向消融 |
| v8 | `experiments/ablations/SOTACalibrationGate_v8` | `RUN_SOTA_CALIBRATION_GATE.m` | SOTA + Ridge-4 与 4 点 LOOCV 门控 | Ridge-4 单次有收益；4 点门控仅放行 4/22，平均收益为负 | 校准/门控消融 |
| v9 | `experiments/calibration/SOTARidgeRobustness_v9` | `RUN_SOTA_RIDGE_ROBUSTNESS.m` 或 `RUN_NEXT_EXPERIMENT.m` | 20 次嵌套空间划分比较 Ridge-4/6，并审计 6 点门控 | Ridge-6 平均精度最佳；6 点门控降低结果波动 | 保留的校准方案 |

## 关键定量结论

| 方法 | 总体结果 | 95 cm 结果 | 最终决策 |
|---|---|---|---|
| Step19-SOTA | 严格完整零样本 `R2=0.8467, RMSE=79.51 με` | 原始报告 `R2=0.3703, RMSE=202.898 με` | 保留为全局主干 |
| Residual-FiLM | 严格完整零样本 `R2=0.7106, RMSE=109.26 με` | 最差工况 RMSE 高于 SOTA | 不作为主干 |
| 空间边界专家 | 快速筛选仅比 Residual 改善约 `0.4 με`，参数和训练成本更高 | 样本不足，易混淆边界规律与单工况噪声 | 不进入严格全训练 |
| Ridge-4 | 重复划分均值优于 SOTA，但方差较大 | 平均改善约 `11.10 με`，胜率 85% | 低传感器消融 |
| Ridge-6 | 总体 RMSE `86.63 -> 80.44 με`，平均下降 7.15% | 平均改善 `15.78 με`，胜率 85% | 精度优先推荐 |
| Gated Ridge-6 | 总体波动标准差降至 `2.80 με`，仅在证据充分时放行 | 平均改善较小但可回退 SOTA | 风险优先推荐 |
| 4 点 LOOCV Gate | 仅 3 点拟合 2 个仿射参数，验证不稳定 | 判别与真实收益负相关 | 淘汰 |

## 如何运行

各版本是独立 MATLAB 包。先将 MATLAB 当前目录切换到对应包，再运行该包
`README_CN.md` 中列出的入口。v9 默认读取已经完成的 v7 全训练输出，不重新训练
神经网络。

私有数据文件可放在对应包根目录，或通过入口函数传入完整路径：

```matlab
RUN_ALL("quick", "D:\path\to\Response_Matrix_Table.csv")
```

不要提交 `Response_Matrix_Table.csv`、模型 `.mat`、`outputs/` 或其他生成结果。
用于报告复核的汇总 CSV 位于 `docs/model-evolution-report/data/`，不包含原始私有
响应矩阵。

## 解释边界

- `R2≈0.95` 主要对应训练/验证或排除极边界后的统计，不等于严格未见工况测试。
- 20 次重复空间划分检验的是测点选择敏感性，不是 20 次独立物理加载。
- Ridge 仿射校准只修正整体比例和零偏，不能修复峰值位置或局部形状错误。
- v9 的门控阈值仍需独立外部工况验证。
