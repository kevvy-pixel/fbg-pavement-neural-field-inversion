# 修正数据集下机器学习到神经网络实验总结（v2，产物对齐版）

> 本文档以当前 D 盘实验产物为准。早期 `EXPERIMENT_SUMMARY_ML_TO_NN.md` 中存在时间窗口、FiLM 运行版本和汇总指标漂移，不能作为最终记录；本 v2 用于替代早期手工汇总。

## 1. 研究目标与当前边界

目标是从正确层位映射的稀疏 FBG 数据中重建加载过程的分层实时微应变响应场，重点是跨加载工况的场建模和泛化，不是单纯追求最低 RMSE。

当前不做：

- 毫米级绝对沉降标定；
- 预警阈值；
- 固定测试集调参；
- 将微应变结果直接解释为实测位移。

## 2. 数据版本和时间窗口

早期重组数据目录（历史记录）：

```text
C:\Users\27356\Documents\Codex\2026-10-07\c-users-27356-documents-codex-2026\outputs\reorganized_dataset_excel_mapping
```

当前修正版协议的权威数据目录见第 14 节：

```text
C:\Users\27356\Documents\Codex\2026-10-09\c-users-27356-documents-codex-2026\outputs\guochuang_fbg_pavement_dataset\reorganized_dataset
```

正确结构：28 个沥青层结构传感器、22 个土层结构传感器、D2-CH4 温度参考通道（排除）。

原始动态缓存：

```text
C:\Users\27356\Documents\Codex\2026-08-01\kevvy-pixel-fbg-pavement-neural-field\outputs\reorganized_mapping_rebuild\dynamic_dataset_1s.npz
```

实际缓存形状为：

```text
response_ue: (22, 50, 599)
time_s: 0 ... 598 s
```

因此必须区分：

```text
原始动态缓存：0–598 s
当前模型训练窗口：0–280 s，1 s 分辨率
```

FiLM 训练表：

```text
film_dynamic_training_table.csv
```

共 309100 行，对应 22 个案例、50 个结构传感器和 281 个相对时间点。

## 3. 数据划分与泄漏检查

固定划分文件：

```text
fixed_case_split_reorganized_mapping.csv
```

```text
development：17 个案例
fixed_test：5 个案例
```

当前模型训练、标准化和 OOF 评估均只使用 development cases。当前所有 manifest 均记录：

```text
fixed_test_opened: false
```

没有将 `case_id`、文件名或传感器 ID embedding 输入神经网络。标准化参数按训练 fold 计算。

## 4. 无工况质量权重

质量权重只由以下目录的 `Peaks.*.txt` 计算：

```text
D:\matlab\guochuang\demo1无工况结果
D:\matlab\guochuang\demo2无工况结果
```

指标包括无工况噪声、漂移、峰跟踪完整性和跨文件稳定性。

当前权重输出：

```text
sensor_quality_weights_from_no_load_peaks.csv
```

D2-CH1 的 9 峰/8 传感器身份匹配仍需进一步审计。因此质量加权指标目前应视为诊断性指标，所有结果必须同时报告普通 RMSE。

训练和评价权重不是同一个公式：

```text
训练：max(q, 0.05) × squared error
评价：原始 q × squared error 的加权 RMSE
```

后续报告必须明确这一点，不能把两者称作同一个优化目标。

## 5. RBF/Prony 机器学习基线

当前连续时间基线使用冻结 Prony 时间基：

```text
τ1 = 40 s
τ2 = 160 s
常数项 + 线性漂移项
```

development-only 结果：

| 方案 | 全时段 RMSE | 后期 RMSE | 沥青层后期 | 土层后期 |
|---|---:|---:|---:|---:|
| 无图 | 61.00 | 67.38 | 89.94 | 4.85 |
| 统一传感器串图 | 60.07 | 66.28 | 88.46 | 4.99 |
| 土层图约束 | 59.64 | 65.62 | 87.56 | 5.33 |

这里的“统一图”是多个传感器串链式子图的联合，不是全路面二维空间图。

## 6. 质量加权和弱曲率实验

质量加权/图约束结果目录：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\LayeredViscoelasticSTNF_v4\reorganized_mapping_v5_quality_loss_global_graph
```

结果显示统一传感器串图在普通和加权指标上较稳；强烈按质量权重改变图方程反而变差。弱曲率先验收益很小，不能称为已验证的完整物理约束。

## 7. 坐标 MLP

坐标 MLP 结构为：

```text
加载条件 + x,y,z + t + 层位 + 方向 → 64 → 64 → 微应变
```

结果目录：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\LayeredViscoelasticSTNF_v4\coordinate_mlp_baseline
```

其 development OOF 结果为：

```text
全时段：40.77 με
加载早期：48.86 με
加载后期：36.16 με
沥青层后期：47.29 με
土层后期：9.79 με
```

这说明坐标神经网络值得继续研究，但仍不是 fixed-test 结论。

## 8. FiLM 和分层 FiLM

正确的 FiLM 输入分组为：

```text
工况分支：荷载大小、荷载位置、加载类型、方向及荷载几何特征
空间时间分支：x,y,z,t
层位：选择沥青/土层专属输出头
```

正式 FiLM 目录：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\LayeredViscoelasticSTNF_v4\film_dynamic_oof
```

当前目录中 `layered_summary.json` 的实际结果为：

| 指标 | 当前正式产物 |
|---|---:|
| 全时段普通 RMSE | 30.735 με |
| 加载早期 | 38.220 με |
| 加载后期 | 26.263 με |
| 沥青层后期 | 35.075 με |
| 土层后期 | 1.235 με |
| 全时段质量加权 | 12.592 με |
| 后期质量加权 | 9.192 με |

正则化目录：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\LayeredViscoelasticSTNF_v4\film_dynamic_oof_regularized
```

当前 `layered_summary.json` 实际结果为：

```text
全时段普通 RMSE：29.354 με
全时段质量加权 RMSE：11.003 με
```

由于这两个目录对应不同运行配置，不能再使用“29.55 优于 30.71”这样的未绑定产物的结论。正式报告应同时给出运行目录、脚本、时间窗口和汇总文件。

## 9. 统计口径

当前脚本主要使用：

```text
先计算每个 fold 的 RMSE，再对 fold RMSE 求均值
```

因此当前结果属于 fold macro-average，不一定等于所有 OOF 点拼接后的 pooled RMSE。

后续正式报告必须同时提供：

```text
pooled OOF RMSE
fold macro mean ± standard deviation
最差 fold
最差 case
```

不能把未注明口径的单一均值直接称为“全部 17 个 development cases 的 pooled OOF RMSE”。

## 10. 泛化审计

当前逐案例审计显示误差主要集中在：

```text
双点加载
低荷载双点加载
部分纵向线加载
沥青层空间响应
```

土层响应整体更稳定。这个现象说明模型存在跨工况差异，但不能据此断言问题一定来自网络容量不足，也不能单独证明收益来自 FiLM 调制或分层输出头。

留一荷载类型实验目前已经生成 development-only 文件：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\LayeredViscoelasticSTNF_v4\film_dynamic_leave_type_out\layered_film_metrics.csv
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\LayeredViscoelasticSTNF_v4\film_dynamic_leave_type_out\layered_summary.json
```

该实验包含 5 个 fold，`fixed_test_opened` 仍为 `false`。正式结论前还需要在结果中明确每个 fold 对应的被留出荷载类型。

## 11. 当前可以说和不能说的结论

可以说：

1. 正确层位映射后的数据流程已建立；
2. RBF-Prony 提供了可复现的机器学习基线；
3. 坐标 MLP 和 FiLM 值得继续作为连续场模型研究；
4. 分层 FiLM 对土层响应建模较稳定；
5. 沥青层和双点加载仍是主要泛化难点。

不能说：

1. FiLM 已经在固定测试集上优于所有模型；
2. 质量加权 RMSE 等价于整体预测精度；
3. 弱曲率先验已经证明了力学物理约束有效；
4. 分层输出或 FiLM 调制的因果收益已经被完全隔离；
5. 模型已经具备道路级泛化能力。

## 12. 后续实验纪律

```text
1. 先冻结数据版本、时间窗口、质量权重和 case split
2. 自动从 summary/metrics 文件生成报告表格
3. 同时报告 pooled 和 fold macro 统计
4. 同时报告普通和质量加权指标
5. 保留失败案例、失败 fold 和负消融
6. 完成 D2-CH1 峰身份审计
7. 模型结构冻结后才打开 fixed-test 一次
```

最终目标仍然是可解释、跨工况的实时分层微应变响应场，而不是一个只在当前 development cases 上指标漂亮的模型。

## 13. 2026-10-08 协议化重算与当前状态

本节记录统一协议 `protocol_20261008` 的最新结果，优先级高于此前手工整理的旧结果。原始动态缓存范围为 0–598 s；本轮模型实验固定使用 0–280 s。训练每 5 s 取样，评价使用 0–280 s 的全部 1 s 时间点。development 使用 17 个工况和 4 个 case-level folds，fixed-test 的 5 个工况仍未打开。

### 13.1 B 阶段三种子复核

| 模型 | pooled overall RMSE 均值 ± std |
|---|---:|
| B0 坐标 MLP | 37.938 ± 1.092 με |
| B3 分层 FiLM | 35.170 ± 1.299 με |
| B5 正则化分层 FiLM | 39.645 ± 7.397 με |

B3 在三个随机种子上均优于 B0，因此进入后续外推实验。B5 在一个种子上明显退化，暂不作为主模型。B1 分层坐标 MLP 和 B4 大容量分层 FiLM 的首轮结果也没有显示出稳定收益。

### 13.2 A0 Prony/RBF 统一重算

| 候选 | pooled overall | pooled early | pooled late | fold mean ± std |
|---|---:|---:|---:|---:|
| A0-1：τ=40 s | 32.650 | 41.989 | 27.059 | 30.199 ± 12.157 |
| A0-2：τ=40,160 s | 32.616 | 42.016 | 26.979 | 30.151 ± 12.187 |
| A0-3：τ=40,160 s+线性漂移 | 32.615 | 41.999 | 26.989 | 30.150 ± 12.185 |
| A0-4：A0-3+统一串图约束 | 32.615 | 41.999 | 26.989 | 30.150 ± 12.185 |
| A0-5：A0-3+土层图约束 | 32.615 | 41.999 | 26.989 | 30.150 ± 12.185 |

A0-3～A0-5 的随机 OOF 误差低于 B3，但 A0-4/A0-5 与 A0-3 几乎没有差异，因此目前不能声称图约束有效。A0 的线性时空基函数需要通过外推实验继续检验，不能仅凭随机 OOF 选作最终模型。

### 13.3 B0/B3 留一荷载类型外推

| 留出类型 | B0 overall | B3 overall | B0 late | B3 late |
|---|---:|---:|---:|---:|
| DualAsymmetric | 49.400 | 70.688 | 35.089 | 58.180 |
| DualSymmetric | 73.554 | 66.966 | 63.868 | 55.537 |
| LongitudinalLine | 41.231 | 42.527 | 29.205 | 37.321 |
| SingleCenter | 20.614 | 20.787 | 13.721 | 13.972 |
| SingleEccentric | 23.418 | 22.903 | 15.444 | 18.661 |

B3 在 DualSymmetric 和 SingleEccentric 上略有优势，在 SingleCenter 上接近 B0，但在 DualAsymmetric 上明显失败。因此当前不能写成“B3 已完成跨荷载类型泛化验证”；更准确的表述是：B3 在随机 development OOF 上稳定优于 B0，但双点荷载，尤其双点不对称荷载，仍是主要泛化风险。

### 13.4 当前冻结状态

```text
A0：统一 Prony/RBF 基线，待完成留一荷载类型外推
B0：坐标 MLP 结构基线
B3：development 阶段主候选，尚未通过全部泛化验证
B5：正则化敏感性对照，不作为主模型
fixed-test：继续关闭
```

最新协议化明细见：
`D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261008\PHASE_A_B_REPORT.md`

## 14. 2026-10-11 修正版协议最终结果与模型冻结

本节优先级高于第 13 节及其之前的历史记录。第 13 节保留用于追踪实验演化，但不能再作为当前模型选择依据。

最终权威实验目录：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected
```

该协议使用重新整理后的数据包，并统一采用：

```text
训练数据：development cases
时间窗口：0–280 s
训练采样：每 5 s
评价采样：每 1 s
外层划分：case-level development folds
主损失：普通 MSE
fixed-test：仅在最终阶段打开一次
```

修正版数据入口为：

```text
C:\Users\27356\Documents\Codex\2026-10-09\c-users-27356-documents-codex-2026\outputs\guochuang_fbg_pavement_dataset\reorganized_dataset
```

荷载几何使用动态训练表中按 `case_id` 绑定的 `p1/p2`。`LongitudinalLine` 在物理解释上统一视为 `SinglePointAxisScan`，即沿轴线改变单个加载点的位置，不构造虚假的线荷载长度、角度或端点特征。`spatial_model/load_points_coordinates.csv` 仅用于几何审计，不作为当前模型的训练荷载输入。

### 14.1 审计和外推阶段状态

修正版协议已完成：

```text
F0       SAL2/SAT2 传感器与响应审计
F1       充分收敛 development OOF
F1.5     指标口径与预测一致性复核
F2.5     DualAsymmetric 失败分解
F3.5     p1/p2 交换敏感性审计
G        B0/B3/G2/G3/G2+G3 最小模型消融
F2       G2 留一荷载类型外推
F3       G2 传感器串留出
F4       G2 双重外推
F7       最终 fixed-test
```

F1.5 已确认旧阶段的 A0 指标差异来自 fold macro 与 pooled OOF 的混用，以及旧 A0 训练时排除内部验证 fold；不是数据损坏。后续正式指标统一为：每个 seed 独立计算 pooled OOF，再报告 seed mean ± std；fold macro 仅作为稳定性指标。

### 14.2 最终 fixed-test 结果

fixed-test 固定案例为：

```text
35kg双点对称加载
40kg纵中心线42.5cm加载
40kg纵中心线95cm加载
5kg双点不对称加载
80kg单点中心加载
```

最终 overall RMSE：

| 模型 | fixed-test overall RMSE |
|---|---:|
| A0-3 确定性 Prony/RBF | 34.360 με |
| B0 坐标 MLP | 34.033 ± 0.018 με |
| G2 交换不变分层 FiLM | 34.317 ± 0.150 με |

G2 比 B0 高约 0.284 με，因此没有全面超过 B0。但 G2 的分层结果具有明确结构性优势：

- late soil：1.634 με，优于 B0 的 4.045 με；
- SSL1、SSL2、SST 等土层串表现更好；
- late asphalt：29.352 με，接近 B0 的 28.858 με；
- SAL2：43.657 με，与 B0 的 43.260 με 基本相当；
- SAT2：65.122 με，高于 B0 的 64.278 με，仍是沥青层风险点。

fixed-test 未用于调参、模型结构修改或 fixed-test 案例重新选择。完成后不再使用该集合进行模型选择。

### 14.3 最终模型角色

```text
A0-3：确定性时序基线
B0：最终 fixed-test 整体最佳基线
G2：交换不变、layer-aware、土层空间泛化更强的主候选
B3：历史分层 FiLM 对照，存在双点荷载顺序敏感性
G3：共享主干消融，未通过交换不变性要求
G2+G3：组合消融，未显示出相对于 G2 的明确收益
```

因此当前不应将 G2 表述为“全结构统一最优模型”。准确表述是：

> G2 是一种使用交换不变双点荷载表示的分层 FiLM 神经场模型。它的整体 fixed-test 精度与 B0 接近，但没有超过 B0；其主要优势体现在土层响应重建、部分空间外推和双点荷载输入表示的物理合理性。

### 14.4 B0 和 G2 的本地模型位置

最终 fixed-test 使用的 B0 checkpoint：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\checkpoints\B0_20261008.pt
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\checkpoints\B0_20261009.pt
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\checkpoints\B0_20261010.pt
```

最终 fixed-test 使用的 G2 checkpoint：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\checkpoints\G2_20261008.pt
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\checkpoints\G2_20261009.pt
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\checkpoints\G2_20261010.pt
```

对应的最终 fixed-test 预测、训练历史和汇总结果位于：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\predictions
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\history
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\metrics_summary.csv
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\metrics_by_file.csv
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\metrics_by_case_sensor.csv
```

模型结构和推理实现位于：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\g_ablation_runner.py
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F7_fixed_test\fixed_test_runner.py
```

在 `g_ablation_runner.py` 中：

```text
B0：Core(invariant=False, residual=False)
G2：Core(invariant=True, residual=False)
```

G2 的 `invariant=True` 表示两个荷载点先分别编码，再通过无序聚合和不变量几何特征形成荷载条件，因此满足：

```text
f(p1, p2) = f(p2, p1)
```

### 14.5 当前实验边界和结论

当前实验到此冻结，不再继续扩展 PINN、强 PDE residual、GNN、Neural Operator 或新的结构搜索。原因是当前研究问题已经完成了以下闭环：

```text
数据审计
→ 统一时间和评价协议
→ 传统时序基线
→ 坐标 MLP 与分层 FiLM
→ 跨荷载类型外推
→ 传感器串空间外推
→ 双重外推
→ 最终 fixed-test
```

可以得出的最终结论是：

1. A0-3、B0、G2 均已在统一修正版协议下完成最终评估；
2. B0 是 fixed-test 整体 RMSE 最佳模型；
3. G2 没有全面超过 B0，但在土层空间泛化和荷载点交换不变性方面具有明确优势；
4. SAT2 是当前沥青层空间泛化的主要风险点；
5. 不能将质量加权 RMSE、development OOF 或单一传感器串优势直接解释为全结构统一泛化能力；
6. fixed-test 结果已冻结，后续如继续研究残差模型，需要新的未见测试工况，不能再把当前 fixed-test 当作独立新测试集。

因此，本项目当前推荐的主框架为：

```text
主基线：B0 坐标 MLP
结构候选：G2 交换不变、分层 FiLM 神经场
传统基线：A0-3 Prony/RBF
```

## 15. 模型消融总表与最终测试表

为便于论文、汇报和后续复核，已单独生成模型消融总表：

```text
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\ABLATION_MASTER_TABLE.md
D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\ABLATION_MASTER_TABLE.csv
```

表格分为两类：

1. 所有已完成模型的 development OOF、C1、D1、F4 和结构消融结果；
2. 仅包含 A0-3、B0、G2 的正式 fixed-test 结果。

B3、G3 和 G2+G3 只作为结构消融或 development/外推对照，不追加到正式 fixed-test 表中。这样可以避免在已经打开 fixed-test 后，用事后 test 结果重新选择模型。

最终呈现时，建议将 B0 作为整体精度主基线，将 G2 作为交换不变和土层空间泛化候选，将 A0-3 作为确定性时序基线。
