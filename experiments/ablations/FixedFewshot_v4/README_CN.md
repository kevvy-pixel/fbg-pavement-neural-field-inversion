# FBG 神经动力学场 v4：固定测试集 Few-shot 公平评测

本实验包读取原工程生成的 `Response_Matrix_Table.csv`，依次完成：

1. 数据检查与 95 cm 残差诊断；
2. 普通 FiLM 位移神经场基线；
3. Residual-FiLM-AD-DNF；
4. 空间感知边界残差专家；
5. 95 cm 固定留出集、多随机种子 few-shot 校准；
6. 自动汇总和复杂度判定。

每一步都在独立文件夹中保存配置、划分、历史、模型、预测表、指标和图片。
运行不会删除已有结果。

## 1. 环境要求

- MATLAB R2022b 或更高版本（推荐 R2024a/R2024b）；
- Deep Learning Toolbox；
- Statistics and Machine Learning Toolbox；
- GPU 可选。没有 GPU 时自动使用 CPU。

## 2. 首次检查

把 MATLAB 当前目录切换到本实验包根目录，然后运行：

```matlab
CHECK_ENVIRONMENT
RUN_DEMO
```

演示只验证代码和输出流程，不代表真实实验精度。

## 3. 运行全部真实数据实验

传入真实 CSV 的完整路径：

```matlab
RUN_ALL("quick", "D:\your_project\Response_Matrix_Table.csv", "D:\FBG_runs")
```

确认 quick 完成后再运行：

```matlab
RUN_ALL("research", "D:\your_project\Response_Matrix_Table.csv", "D:\FBG_runs")
```

- `demo`：极少迭代、2 次 few-shot 重复；
- `quick`：快速消融、5 次 few-shot 重复；
- `research`：正式训练、10 次 few-shot 重复。

## 4. 复用已经训练完成的旧模型

如果上一版 quick 已成功跑完，不必重新训练三个主模型：

```matlab
RUN_FIXED_FEWSHOT( ...
    "D:\FBG_runs\outputs\20260727_120000_quick", ...
    "D:\data\Response_Matrix_Table.csv", ...
    "quick", ...
    "D:\FBG_runs")
```

第一个参数是旧运行目录，里面必须存在：

```text
02_residual_film/model.mat
03_spatial_edge_expert/model.mat
```

第二个参数必须是旧模型训练时使用的同一份 CSV。程序会检查空间范围、条件
维数和归一化尺度，防止误用不同数据。

## 5. 固定留出集实验设计

每次重复先选择 8 个校准候选坐标，其余坐标固定为测试集。随后
`k=0,2,4,6,8` 均在完全相同的测试坐标上评价。校准点是嵌套的：
`k=2` 的点包含在 `k=4` 中，以此类推。不同重复采用不同空间覆盖起点。

四种方法如下：

| method | 含义 |
| --- | --- |
| `residual_zero` | Residual-FiLM 不做目标工况校准 |
| `bias_only` | 只用校准点拟合一个标量偏置 |
| `output_head` | 冻结主干，只更新 Residual-FiLM 输出层 |
| `edge_expert` | 冻结主干，只更新 gate 与局部边界专家 |

`method_decision_table.csv` 使用以下复杂度门槛：

- 相对 `bias_only`，平均 R² 至少提高 0.02；
- 相对 `bias_only`，平均 RMSE 至少下降 5%。

两项同时满足时，`passes_complexity_threshold` 才为 true。

## 6. 主要输出

```text
outputs/<时间戳>_quick/
├─ 00_data_validation/
├─ 01_plain_film/
├─ 02_residual_film/
├─ 03_spatial_edge_expert/
├─ 04_fixed_holdout_fewshot_95cm/
│  ├─ fixed_holdout_repeat_metrics.csv
│  ├─ fixed_holdout_summary.csv
│  ├─ fixed_holdout_predictions.csv
│  ├─ method_decision_table.csv
│  ├─ fixed_holdout_performance.png
│  ├─ repeat_01/
│  │  ├─ calibration_pool.csv
│  │  ├─ fixed_test_points.csv
│  │  ├─ k_00/
│  │  ├─ k_02/
│  │  └─ ...
│  └─ repeat_02/...
├─ 05_comparison/
├─ run_manifest.json
└─ run.log
```

其中：

- `fixed_holdout_repeat_metrics.csv`：每个重复、方法和 k 的指标；
- `fixed_holdout_summary.csv`：均值与标准差；
- `fixed_holdout_predictions.csv`：全部固定测试点预测；
- `method_decision_table.csv`：输出层/专家相对偏置校准的收益；
- 每个 k 文件夹保存实际校准点和可训练适配器的训练历史。

## 7. 输入表要求

至少包含：

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

默认仅训练 `Asphalt` 层，包含 `95cm` 的工况作为目标测试工况，包含 `90cm`
的工况强制留在训练集。FBG 系数默认 `0.0012 nm/με`，中性轴距离默认
`4 cm`，均可在 `fbg_config.m` 中修改。

## 8. 如何解释结果

先检查 `02_residual_film/metrics.csv` 中训练集 `R2 > 0`，再分析目标工况。

重点看：

- Residual-FiLM 是否稳定优于 Plain-FiLM；
- `fixed_holdout_summary.csv` 的均值是否改善且标准差是否可接受；
- 简单 `bias_only` 是否已经解决主要误差；
- `method_decision_table.csv` 中专家是否达到复杂度门槛；
- Bias、峰值误差和 95 cm 纵中心线形状是否同时改善。

固定测试集内的各 k 可以直接比较；不同实验包或不同数据划分的指标不能直接
当作同一基准。

## 9. 常见问题

### 拒绝访问

显式指定可写输出目录：

```matlab
RUN_ALL("quick", "D:\data\Response_Matrix_Table.csv", "D:\FBG_runs")
```

### GPU 内存不足

在 `fbg_config.m` 中减小 `cfg.training.batchSize` 和 `cfg.model.width`，
或设置：

```matlab
cfg.runtime.useGPU = false;
```

### 二阶导数训练较慢

先使用 quick。主模型已经完成时，优先用 `RUN_FIXED_FEWSHOT`，只执行新版校准
评测。
