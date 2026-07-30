# FBG 神经动力学场：Residual-FiLM、空间边界专家与 Few-shot 实验包

本实验包是一个独立的 MATLAB 扩展，不会覆盖原仓库中的 Step01-Step06。
它读取原工程生成的 `Response_Matrix_Table.csv`，依次完成：

1. 数据检查与 95 cm 残差诊断；
2. 普通 FiLM 位移神经场基线；
3. Residual-FiLM-AD-DNF；
4. 空间感知边界残差专家；
5. 95 cm few-shot 校准；
6. 汇总对比。

每一步都会在独立文件夹中保存配置、数据划分、训练历史、模型、预测表、
评价指标和图片。运行不会自动修改或删除已有结果。

## 1. 环境要求

- MATLAB R2022b 或更高版本（推荐 R2024a/R2024b）；
- Deep Learning Toolbox；
- Statistics and Machine Learning Toolbox；
- GPU 可选。没有 GPU 时会自动使用 CPU。

## 2. 最快验证：运行合成数据演示

在 MATLAB 中把当前目录切换到本实验包根目录，然后运行：

```matlab
CHECK_ENVIRONMENT
RUN_DEMO
```

演示会生成：

```text
demo_data/Response_Matrix_Table_demo.csv
outputs/<时间戳>_demo/
```

演示的目的仅是检查代码、自动微分、训练循环和输出目录是否能完整运行，
不代表真实实验精度。

## 3. 运行真实数据

将原工程生成的 `Response_Matrix_Table.csv` 放到本实验包根目录，或者直接传入
它的完整路径：

```matlab
RUN_ALL("quick", "D:\your_project\Response_Matrix_Table.csv")
```

推荐先运行 quick：

```matlab
RUN_ALL("quick")
```

确认没有数据格式问题后，再运行完整研究配置：

```matlab
RUN_ALL("research")
```

可用配置：

- `demo`：极少迭代，仅做流程检查；
- `quick`：快速消融，适合首次使用；
- `research`：正式训练，迭代次数更多。

## 4. 输出结构

每次运行都会生成新的时间戳目录：

```text
FBG_NeuralField_Workspace/
└─ outputs/20260727_153000_quick/
├─ 00_data_validation/
│  ├─ config.json
│  ├─ config.mat
│  ├─ dataset_summary.csv
│  ├─ case_summary.csv
│  ├─ split_assignments.csv
│  └─ target_95cm_raw.csv
├─ 01_plain_film/
│  ├─ model.mat
│  ├─ training_history.csv
│  ├─ metrics.csv
│  ├─ case_metrics.csv
│  ├─ predictions.csv
│  └─ figures/
├─ 02_residual_film/
│  └─ 与上一步相同
├─ 03_spatial_edge_expert/
│  └─ 与上一步相同，并额外输出 gate 和 edge residual
├─ 04_fewshot_95cm/
│  ├─ fewshot_summary.csv
│  ├─ k_00/
│  ├─ k_02/
│  ├─ k_04/
│  └─ ...
├─ 05_comparison/
│  ├─ model_comparison.csv
│  └─ figures/
├─ run_manifest.json
└─ run.log
```

程序会自动寻找可写位置：优先使用当前目录，其次使用 MATLAB `userpath`，
最后使用系统临时目录。实际位置会在运行开始时显示为 `Writable workspace`。

也可以显式指定输出位置：

```matlab
RUN_ALL("research", "D:\data\Response_Matrix_Table.csv", "D:\FBG_runs")
```

## 5. 输入表要求

最少需要以下列：

```text
case_id
sensor_id
x_cm
y_cm
layer
direction
delta_lambda_resp_nm
load_type
load_kg
p1_x
p1_y
p2_x
p2_y
```

程序默认：

- 仅训练 `Asphalt` 层；
- `纵`/`Longitudinal` 使用 `d²w/dx²`；
- `横`/`Transverse` 使用 `d²w/dy²`；
- FBG 系数为 `0.0012 nm/με`；
- 中性轴距离默认 `4 cm`，可在 `fbg_config.m` 中修改；
- 包含 `95cm` 的工况作为极边界测试工况；
- 包含 `90cm` 的工况强制保留在训练集中。

如果真实字段不同，程序会在 `00_data_validation` 阶段停止，并明确列出缺失列。

## 6. 关键模型

Residual-FiLM 主干：

```text
h(l+1) = h(l) + alpha * F(FiLM(h(l), condition))
```

空间边界专家：

```text
w = w_base + g(x,y,c) * delta_w_edge
```

门控同时使用：

- 查询点到边界距离；
- 加载点到边界距离；
- 查询点相对加载点坐标；
- 荷载大小、位置和类型；
- 全局主干隐特征。

这避免了旧边界专家“同一工况全场使用同一个 gate”的问题。

## 7. 如何解释结果

重点比较：

- `02_residual_film` 是否优于 `01_plain_film`；
- `03_spatial_edge_expert` 是否提高 95 cm，同时不降低普通工况；
- `04_fewshot_95cm/fewshot_summary.csv` 中需要多少标定点才能稳定提高 95 cm；
- 95 cm 的 Bias、峰值误差和中心线形状是否改善。

在分析边界专家前，必须先确认 `02_residual_film/metrics.csv` 中训练集
`R2 > 0`。如果训练集仍为负值，说明主干尚未学会已知工况，此时不能把测试误差
解释为95 cm外推问题，也不应继续采用边界专家结论。

正式报告应使用 `research` 配置，并至少重复 3-5 个随机种子。本包默认输出单个
种子的完整流程，便于先确认方法有效。

## 8. 常见问题

### 找不到数据

使用完整路径：

```matlab
RUN_ALL("quick", "D:\data\Response_Matrix_Table.csv")
```

### GPU 内存不足

在 `fbg_config.m` 中减小：

```matlab
cfg.training.batchSize
cfg.model.width
```

或者强制 CPU：

```matlab
cfg.runtime.useGPU = false;
```

### 95 cm 工况名称不同

修改：

```matlab
cfg.data.targetCasePattern = "95cm";
```

### 二阶导数训练较慢

这是 AD-DNF 的正常计算成本。先用 `quick`，确认无误后再用 `research`。
