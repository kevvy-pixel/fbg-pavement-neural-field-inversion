# FBG Pavement Neural-Field Inversion

基于 FBG 实测应变、坐标神经场、二阶自动微分、FiLM 条件调制和历史工况锚定的
路面响应反演项目。

English summary: this MATLAB repository reconstructs continuous pavement
response fields from FBG measurements. The current model uses a compatible
historical measured condition as a leakage-free anchor, predicts only the
longitudinal residual, reduces high-frequency Fourier features, and optionally
regularizes sensitivity to load-position perturbations.

本项目延续
[`fbg-multilayer-pavement-modeling`](https://github.com/kevvy-pixel/fbg-multilayer-pavement-modeling)
的数据处理与多层路面分析工作。

## 当前模型结论

完整 v15 消融包含 12 个模型、5 个工况级折，共完成 60 个折-模型评估：

- 55 次完整神经网络训练；
- 5 次最近锚点持久性基线评估；
- 每个神经模型最高 1800 次迭代，保留原始验证与早停过程；
- 616 条沥青记录每条恰好作为未见测试样本出现一次；
- 95 cm 工况的 28 条记录全部参与对应测试折；
- 没有裁剪工况、传感器记录或训练步骤。

当前完整五折点估计最佳模型为：

`AnchoredResidual-XOnly-LowFreqSmooth`

| 指标 | 全部沥青测试记录 | 95 cm 工况 |
|---|---:|---:|
| RMSE | **61.04 μɛ** | **82.66 μɛ** |
| MAE | **30.67 μɛ** | **52.95 μɛ** |
| R² | **0.9097** | **0.8955** |

不含显式位置平滑的
`AnchoredResidual-XOnly-LowFreq` 全局 RMSE 为 61.28 μɛ、R² 为 0.9089，
只比平滑版本高 0.25 μɛ RMSE。因此：

- 当前五折点估计最佳：`LowFreqSmooth`；
- 更简洁的默认候选：`LowFreq`；
- 两者的最终选择仍需多随机种子重复验证。

## 使用条件与部署边界

模型选择取决于部署时可用的信息：

| 部署条件 | 建议模型 | 解释 |
|---|---|---|
| 有兼容历史实测工况库 | v14 LowFreqSmooth | 当前五折点估计最佳 |
| 有兼容历史库，优先简洁性 | v14 LowFreq | 与最佳模型仅差 0.25 μɛ RMSE |
| 没有任何兼容历史参考 | Step19 绝对场 | 旧的无参考回退方案，应报告低置信度 |
| 目标工况已有少量实测点 | Step19 + Ridge-6 | v9 稀疏校准场景，不是纯预测 |

锚定模型在预测目标工况时不使用该工况的真实应变，但需要另一个兼容历史工况的
实测响应作为参考。因此它属于“目标工况无测量、历史参考可用”的预测，不属于
“无历史参考的零样本预测”，也不能默认泛化到没有兼容锚点的其他领域。

## 为什么旧报告的 R²=0.9164 高于当前 R²=0.9097

这两个数字采用不同统计口径，不能直接比较。

旧 Step19 参考表
[`LEGACY_STEP19_REFERENCE.csv`](experiments/ablations/SOTAFullComparison_v7/LEGACY_STEP19_REFERENCE.csv)
给出的指标为：

| 旧指标范围 | R² | RMSE/μɛ | 是否为主要泛化指标 |
|---|---:|---:|---|
| 训练集 | 0.9535 | 39.40 | 否 |
| 验证集 | 0.9659 | 43.01 | 否 |
| 测试集 | **0.7810** | 123.08 | **是** |
| train + validation + test 合并 | 0.9164 | 58.71 | 否 |
| 合并后再排除极端边界 | 0.9586 | 40.63 | 否 |

旧 README 中的 “all-data R²=0.9164” 把训练、验证和测试预测放在一起计算。
其中训练和验证记录参与了拟合或模型选择，误差显著低于真正未见测试工况，因此
合并后的 R² 会被容易样本抬高。它描述整体拟合情况，但不是纯泛化能力。

当前 v15 的 R²=0.9097 是严格的五折 out-of-fold 指标：每条沥青记录的预测都来自
一个没有使用该记录所属测试工况训练的模型。它相当于把五个互斥测试折拼接后统一
计算 R²，616 条记录全部是测试预测。

因此：

- `0.9164 > 0.9097` 不表示旧 Step19 比当前模型更好；
- 合理的旧泛化参考是 Step19 测试 R²=0.7810，而不是合并 R²=0.9164；
- 当前 R²=0.9097 的评估更严格、覆盖更完整，也更适合作为主结果；
- 由于旧 Step19 和 v15 的训练实现、折分及实验阶段并非完全一致，
  `0.7810` 与 `0.9097` 仍应视为不同实验阶段的证据，而非唯一控制变量的严格配对。

README 后续统一使用 out-of-fold/test 指标描述泛化；训练、验证和合并指标只作为
诊断值，并显式标注范围。

## 完整五折消融排名

| 排名 | 模型 | 全局 RMSE/μɛ | MAE/μɛ | R² | 95 cm RMSE/μɛ |
|---:|---|---:|---:|---:|---:|
| 1 | v14 LowFreqSmooth | **61.04** | **30.67** | **0.9097** | **82.66** |
| 2 | v14 LowFreq | 61.28 | 30.95 | 0.9089 | 83.03 |
| 3 | v13 AnchoredResidual-XOnly | 61.86 | 31.31 | 0.9072 | 96.27 |
| 4 | v13 AnchoredResidual-XY | 63.03 | 34.17 | 0.9037 | 116.18 |
| 5 | v13 NearestAnchor-Persistence | 66.67 | 31.80 | 0.8922 | 99.73 |
| 6 | v10 Step19 | 92.40 | 58.77 | 0.7930 | 98.35 |
| 7 | v10 BoundaryRelative | 102.81 | 64.88 | 0.7437 | 96.06 |
| 8 | v10 Multilayer-Transfer | 119.23 | 74.14 | 0.6553 | 137.25 |
| 9 | v12 SoilMeanAux-XY | 148.48 | 64.56 | 0.4654 | 215.71 |
| 10 | v14 Step19 | 150.21 | 66.18 | 0.4529 | 222.10 |
| 11 | v12 SoilMeanAux-X | 151.30 | 68.13 | 0.4449 | 219.83 |
| 12 | v11 SharedMultilayer | 162.79 | 82.22 | 0.3574 | 247.45 |

统一结果来自
[`FBG_NeuralField_FullAblation_v15/aggregate/`](FBG_NeuralField_FullAblation_v15/aggregate/)。

## 最新模型机制

### 1. 无泄漏历史工况锚定

- 训练样本只能选择另一个训练工况作为锚点；
- 验证和测试只能从训练工况库选择锚点；
- 目标测试工况的真实应变不会进入模型输入或锚点库；
- v15 五折测试锚点覆盖率为 100%。

### 2. XOnly 残差

模型只学习纵向 X 分量相对历史锚点的修正，横向 Y 分量保持锚点值。

完整五折中，XOnly 系列的 Y 方向结果为：

- RMSE：22.33 μɛ；
- MAE：8.71 μɛ；
- R²：0.9064。

允许同时修改 X/Y 的 v13 XY 残差虽然降低了 X 误差，却把 Y RMSE 从
22.33 μɛ 增加到 47.75 μɛ。因此 XOnly 的作用是保护已经可靠的横向参考。

### 3. 降低 Fourier 频率

| 特征 | v13 | v14 |
|---|---|---|
| 绝对坐标 `x/y` | `[1,2,4,8]` | `[1,2,4]` |
| 相对坐标 `x-p/y-p` | `[1,2,4]` | `[1,2]` |

位移代理场通过二阶自动微分得到应变，高频特征的二阶导数按频率平方放大。删除
最高频率可降低载荷位置小扰动引起的曲率振荡。

### 4. 载荷位置平滑

LowFreqSmooth 对载荷位置施加小幅正负扰动，并约束扰动前后的预测变化。该约束
不平滑原始应变数据，因此不会把模型结构收益与数据去噪收益混在一起。

## 土层数据结论

数据审计表明沥青层与土层测点没有可靠的一一空间配准。完整五折结果进一步显示：

- v11 强共享多层场：RMSE 162.79 μɛ；
- v12 土层 X 条件摘要：RMSE 151.30 μɛ；
- v12 土层 XY 条件摘要：RMSE 148.48 μɛ；
- v10 显式多层传递：RMSE 119.23 μɛ；
- v14 历史锚定低频平滑：RMSE 61.04 μɛ。

因此土层数据当前不适合直接作为强共享局部空间监督。下一步更合理的用途是：

- 估计历史锚点可靠性；
- 对多个历史锚点进行软选择；
- 判断目标工况是否超出历史覆盖范围；
- 输出预测置信度或触发无锚点回退模型。

## 目录结构

```text
.
├─ Step13_...m ... Step21_...m
│  └─ 原始模型演进；Step19 现作为无历史参考回退基线
├─ run_main.m
│  └─ 旧 Step19 入口，不是当前锚定模型入口
├─ experiments/
│  ├─ ablations/
│  └─ calibration/
├─ FBG_NeuralField_MultilayerBoundary_v10/
├─ FBG_NeuralField_SharedMultilayer_v11/
├─ FBG_NeuralField_SoilConditionAux_v12/
├─ FBG_NeuralField_AnchoredResidual_v13/
├─ FBG_NeuralField_LowFreqSmooth_v14/
├─ FBG_NeuralField_FullAblation_v15/
│  ├─ aggregate/
│  ├─ best_model_v14_lowfreqsmooth/
│  ├─ figures/
│  ├─ full_runs/
│  └─ implementations/
├─ EXPERIMENT_EVOLUTION_v10_v15.md
└─ docs/
```

各版本辅助函数和配置不同。运行实验时必须进入对应版本目录，不能混用不同版本的
`src/` 文件。

## 环境要求

- MATLAB R2024a 或更高版本；
- Deep Learning Toolbox；
- Statistics and Machine Learning Toolbox（仅部分校准实验需要）；
- 足够内存用于二阶自动微分。

完整实验已在 MATLAB R2024a、CPU 环境运行。各折使用确定性种子
`2026 + fold - 1`。

## 数据准备

真实数据不随仓库发布。需要获得授权的：

```text
Response_Matrix_Table.csv
```

数据字段和约定见 [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md)。

旧根脚本可先检查数据结构：

```matlab
run("check_data_schema.m")
```

所有目录深度下的 `Response_Matrix_Table.csv` 都由 `.gitignore` 排除。

## 运行最新模型

在 MATLAB 中进入 v14 目录：

```matlab
cd("FBG_NeuralField_LowFreqSmooth_v14")
clear functions
rehash toolboxcache
CHECK_ENVIRONMENT
```

快速单折检查：

```matlab
RUN_QUICK( ...
    "D:\path\to\Response_Matrix_Table.csv", ...
    "D:\path\to\FBG_NeuralField_Workspace")
```

完整五折：

```matlab
RUN_FULL( ...
    "D:\path\to\Response_Matrix_Table.csv", ...
    "D:\path\to\FBG_NeuralField_Workspace")
```

`RUN_QUICK` 只用于流程检查，不能代替完整五折结论。

## 最佳模型归档

五个 LowFreqSmooth 折模型保存在：

[`FBG_NeuralField_FullAblation_v15/best_model_v14_lowfreqsmooth/`](FBG_NeuralField_FullAblation_v15/best_model_v14_lowfreqsmooth/)

该目录包含：

- 5 个 `best_model.mat`；
- 每折训练历史；
- 每折测试预测；
- 工况指标和锚点审计；
- SHA-256 清单；
- 交叉验证模型的使用边界说明。

五折结果不存在天然的唯一部署权重。不能任取某一折模型并称为“全数据最终模型”。
若需要单模型部署，应冻结结构和超参数后增加全数据训练入口，并使用新的外部测试集
评估。

## 实验演进与报告

- v10–v15 逐版改动：
  [`EXPERIMENT_EVOLUTION_v10_v15.md`](EXPERIMENT_EVOLUTION_v10_v15.md)
- v15 完整中文报告：
  [`FBG_NeuralField_FullAblation_v15/完整对比消融实验报告.md`](FBG_NeuralField_FullAblation_v15/完整对比消融实验报告.md)
- 统一覆盖审计：
  [`FBG_NeuralField_FullAblation_v15/aggregate/coverage_audit.csv`](FBG_NeuralField_FullAblation_v15/aggregate/coverage_audit.csv)
- 旧 Step13–Step21 演进：
  [`docs/MODEL_EVOLUTION.md`](docs/MODEL_EVOLUTION.md)
- 旧 v4–v9 实验索引：
  [`docs/EXPERIMENT_INDEX_CN.md`](docs/EXPERIMENT_INDEX_CN.md)

## 方法限制

- 锚定模型依赖兼容历史实测工况，历史域覆盖不足时可能失效；
- 目前历史锚点为最近邻硬选择，尚未学习多锚点软权重；
- LowFreqSmooth 相对 LowFreq 的优势只有 0.25 μɛ RMSE，需多随机种子确认；
- 位移输出是用于应变二阶导数的位移代理场，不是经独立位移传感器校准的绝对位移；
- 当前结论来自同一实验数据域，不能自动推广到不同道路结构、材料、传感器布置或
  载荷制度。

## Data and licensing

The source code and experimental dataset are separate works. The dataset is
maintained privately and is not distributed through this repository. Derived
metrics, figures, and model weights in the v15 package do not include the raw
response table.

No open-source license has been selected for this code snapshot. Public
visibility alone does not grant permission to reuse or redistribute the code.
