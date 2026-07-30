# FBG 神经动力学场 v6：全工况交叉验证与时间基准

v6 用于回答两个最终问题：

1. Residual-FiLM 和岭仿射校准在全部工况上是否仍然有效？
2. Residual-FiLM、边界专家及岭仿射的训练和适配时间分别是多少？

## 1. 运行顺序

解压后将 MATLAB 当前目录切换到本文件夹，先运行：

```matlab
CHECK_ENVIRONMENT
RUN_DEMO
```

演示成功后，运行真实数据 quick：

```matlab
RUN_GLOBAL_BENCHMARK( ...
    "quick", ...
    "D:\path\to\Response_Matrix_Table.csv", ...
    "D:\path\to\FBG_NeuralField_Workspace")
```

quick 会重新训练每一折的模型，不读取之前的单个95 cm模型。

## 2. 为什么必须重新训练

全工况评价采用按完整工况分折：

- 同一工况的所有坐标只能属于一个折；
- 测试工况的任何测点都不会进入训练；
- 每个工况恰好测试一次；
- 每一折的响应尺度和荷载尺度只从训练工况计算。

因此不能直接复用之前“固定95 cm为测试集”的模型。

## 3. quick与research

| profile | 工况折数 | Plain迭代 | Residual迭代 | Edge迭代 |
| --- | ---: | ---: | ---: | ---: |
| `demo` | 2 | 12 | 18 | 12 |
| `quick` | 3 | 400 | 600 | 300 |
| `research` | 5 | 1600 | 2400 | 1200 |

quick 的3折已经覆盖全部工况，适合先判断模型路线。只有结果合理后才运行
research。

## 4. 比较模型

公共固定测试坐标上比较：

| model | 说明 |
| --- | --- |
| `Plain-FiLM` | 普通FiLM基线 |
| `Residual-FiLM` | 残差主干 |
| `Spatial-edge-expert` | 边界专家消融 |
| `Residual+Ridge-4` | 每个测试工况使用4个稀疏测点 |
| `Residual+Ridge-6` | 每个测试工况使用6个稀疏测点 |

对每个测试工况，程序先选择最多6个空间覆盖校准点，并将其余坐标固定为公共测试
集。三个零样本模型和两个校准模型都在相同坐标上评价。

程序还会单独输出三个神经网络在全部留出坐标上的零样本结果，避免公共固定测试集
排除校准候选点后丢失信息。

## 5. 整体性能指标

`global_model_summary.csv` 同时输出：

- Micro R²、RMSE、MAE、Bias：合并全部测试点计算；
- Macro R²、RMSE、MAE、Bias：先按工况计算，再对工况平均；
- 工况间标准差；
- 最差工况RMSE；
- 实际参与评价的工况数与测点数。

Macro结果是论文中的主要整体指标，Micro结果作为补充。

## 6. 时间与复杂度

程序记录：

- 每折模型训练时间；
- Edge专家单独训练时间与“Residual主干+专家”的端到端时间；
- 达到最佳验证RMSE的时间；
- Plain和Residual达到同一验证RMSE门槛的时间；
- 测试集推理时间与每个测点推理毫秒数；
- 每个测试工况求解岭仿射参数的平均时间；
- 神经网络参数数量；
- 岭仿射新增参数数量，固定为2。

推理计时前会先预热一次。岭仿射会重复求解后取平均，避免单次计时分辨率不足。

## 7. 输出文件

```text
outputs/<时间戳>_quick_global_benchmark/
├─ case_fold_assignments.csv
├─ global_model_summary.csv
├─ full_zero_shot_summary.csv
├─ full_zero_shot_case_metrics.csv
├─ common_fixed_test_case_metrics.csv
├─ common_fixed_test_predictions.csv
├─ calibration_point_assignments.csv
├─ training_timing_by_fold.csv
├─ convergence_by_fold.csv
├─ runtime_summary.csv
├─ model_complexity.csv
├─ global_model_comparison.png
├─ runtime_comparison.png
├─ fold_01/
│  ├─ case_split.csv
│  ├─ fold_configuration.mat
│  ├─ 01_plain_film/
│  ├─ 02_residual_film/
│  └─ 03_spatial_edge_expert/
├─ fold_02/
└─ run.log
```

## 8. 如何解释训练时间

`stage_training_seconds`：

- Plain和Residual为各自主干训练时间；
- Edge为仅训练专家适配器的时间。

`end_to_end_training_seconds`：

- Plain和Residual与stage相同；
- Edge为Residual主干时间加专家时间。

对于 `Residual+Ridge-4/6`：

- 主干训练时间与Residual-FiLM相同；
- `mean_calibration_seconds` 是新工况的额外适配时间；
- 校准只求解两个参数，不进行反向传播。

## 9. 运行完成后上传

请上传：

```text
global_model_summary.csv
full_zero_shot_summary.csv
runtime_summary.csv
convergence_by_fold.csv
model_complexity.csv
case_fold_assignments.csv
common_fixed_test_case_metrics.csv
run.log
```
