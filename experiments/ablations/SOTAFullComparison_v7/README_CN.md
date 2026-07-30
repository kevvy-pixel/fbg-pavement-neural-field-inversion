# FBG神经动力学场v7：原始Step19-SOTA完全训练对照

本版本只回答一个问题：

> 在相同容量、相同工况折分、相同初始化、相同训练批次和相同训练预算下，
> Residual-FiLM是否优于原始Step19 FiLM？

## 1. 公平对照协议

| 项目 | Step19-SOTA | Step19-Residual |
| --- | --- | --- |
| 空间隐藏维度 | 96 | 96 |
| 条件隐藏维度 | 64 | 64 |
| FiLM块 | 5 | 5 |
| 最大迭代/折 | 2400 | 2400 |
| 工况折分 | 相同5折 | 相同5折 |
| 初始化张量 | 完全相同 | 完全相同 |
| minibatch随机序列 | 完全相同 | 完全相同 |
| x镜像增强 | 开启 | 开启 |
| y镜像增强 | 关闭 | 关闭 |
| 边界距离特征长度 | 0.180 | 0.180 |
| 边界加权 | 开启 | 开启 |
| 自动微分 | 位移场二阶导 | 位移场二阶导 |

两种主干唯一有意差异是：

- SOTA：`h = tanh(u)`
- Residual：`h = h + 0.15*u`

Ridge-4不参与主干训练，只在每个未见工况的4个稀疏校准坐标上求解尺度和偏置。

说明：原始Step19源码未随报告提供。本版本严格复现报告中明确记录的宽度、层数、
镜像增强、边界特征长度、边界加权和AD-DNF路线，因此属于
“Step19报告兼容复现”。如果以后提供原始Step19 `.m`文件，可以再做逐行源码级核对。

## 2. 首次检查

将MATLAB当前目录切换到解压后的文件夹：

```matlab
clear functions
rehash toolboxcache
CHECK_ENVIRONMENT
RUN_DEMO
```

演示会生成合成数据并运行极短的2折测试，只用于检查程序能否执行。

## 3. 完全训练

```matlab
RUN_FULL( ...
    "D:\path\to\Response_Matrix_Table.csv", ...
    "D:\path\to\FBG_NeuralField_Workspace")
```

也可以直接调用：

```matlab
RUN_SOTA_FULL_COMPARISON( ...
    "full", ...
    "D:\你的路径\Response_Matrix_Table.csv", ...
    "D:\path\to\FBG_NeuralField_Workspace")
```

`full`会训练：

- 5个fold；
- 每个fold训练1个Step19-SOTA和1个Step19-Residual；
- 每个模型最多2400次迭代；
- 合计10次完整主干训练。

CPU运行时间可能较长。GPU可用时程序会自动使用GPU。

## 4. 中断后继续

每个已完成模型都保存在对应fold文件夹。假设中断目录为：

```text
D:\path\to\FBG_NeuralField_Workspace\outputs\
20260728_120000_full_sota_comparison
```

继续运行：

```matlab
RUN_SOTA_FULL_COMPARISON( ...
    "full", ...
    "D:\你的路径\Response_Matrix_Table.csv", ...
    "", ...
    "D:\path\to\FBG_NeuralField_Workspace\outputs\20260728_120000_full_sota_comparison")
```

已经存在`model.mat`和`training_history.csv`的阶段会直接载入，不再重新训练。

## 5. 评价口径

### 全零样本测试

`full_zero_shot_summary.csv`将每个工况作为未见工况测试一次，主要用于比较两个主干。

### 公共固定坐标

每个测试工况选择4个空间覆盖校准坐标，其余坐标作为公共测试集。以下三种方法使用
完全相同的固定测试坐标：

- `Step19-SOTA-FiLM`
- `Step19-Residual-FiLM`
- `Step19-Residual+Ridge-4`

### 95 cm

`target_95cm_summary.csv`单独输出95 cm工况在三种方法下的结果。

## 6. 主要输出

```text
outputs/<时间戳>_full_sota_comparison/
├─ comparison_protocol.csv
├─ dataset_summary.csv
├─ case_fold_assignments.csv
├─ full_zero_shot_summary.csv
├─ full_zero_shot_case_metrics.csv
├─ full_zero_shot_predictions.csv
├─ common_fixed_test_summary.csv
├─ common_fixed_test_case_metrics.csv
├─ common_fixed_test_predictions.csv
├─ calibration_point_assignments.csv
├─ paired_improvement_summary.csv
├─ target_95cm_summary.csv
├─ training_timing_by_fold.csv
├─ runtime_summary.csv
├─ model_complexity.csv
├─ sota_full_comparison.png
├─ fold_01/
│  ├─ case_split.csv
│  ├─ fold_configuration.mat
│  ├─ 01_step19_sota_film/
│  └─ 02_step19_residual_film/
└─ run.log
```

## 7. 完成后请上传

```text
full_zero_shot_summary.csv
common_fixed_test_summary.csv
paired_improvement_summary.csv
target_95cm_summary.csv
runtime_summary.csv
training_timing_by_fold.csv
case_fold_assignments.csv
run.log
```

不要只上传某一折的结果。
