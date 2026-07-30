# FBG 神经动力学场：真实边界与多层传递 v10

本包用于比较三种严格同划分模型：

1. `Step19-SOTA-FiLM`：仅使用沥青层数据的 FiLM 连续场基线；
2. `BoundaryRelative-FiLM`：显式加入试件真实边界距离和“测点—加载点”相对坐标；
3. `Multilayer-Transfer-FiLM`：共享工况编码、沥青/土层双解码器，并学习随工况变化的传递系数 `eta`。

本版的目标是零样本预测，不要求 95 cm 工况额外提供 4～6 个校准点。所有模型仍以平滑位移代理场为输出，并通过二阶自动微分得到应变。

## 1. 环境

- MATLAB R2024a 或更新版本；
- Deep Learning Toolbox；
- 建议使用可写的 MATLAB 工作区目录；
- GPU 可选，CPU 也能运行，但完整 5 折训练会较慢。

解压后，把 MATLAB 当前目录切换到本包根目录，首先运行：

```matlab
clear functions
rehash toolboxcache
CHECK_ENVIRONMENT
RUN_DEMO
```

`RUN_DEMO` 会生成合成的沥青层和土层数据，并以很少的迭代数检查完整流程。Demo 指标只用于确认程序可运行，不能用于比较模型性能。

## 2. 真实数据运行

先运行不训练网络的数据诊断：

```matlab
RUN_DATA_DIAGNOSTICS( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

然后运行快速实验：

```matlab
RUN_QUICK( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

快速版正常后，再运行完整 5 折实验：

```matlab
RUN_FULL( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

第二个参数可省略。省略时，程序会自动选择一个可写目录，并打印实际路径。

## 3. 输入表

程序会自动识别常见英文列名，最终统一为以下字段：

| 规范字段 | 含义 | 常见可识别别名 |
|---|---|---|
| `case_id` | 工况编号 | `CaseID`, `case`, `LoadCase` |
| `sensor_x_cm` | 测点 x 坐标/cm | `Sensor_X_cm`, `x_cm`, `sensor_x` |
| `sensor_y_cm` | 测点 y 坐标/cm | `Sensor_Y_cm`, `y_cm`, `sensor_y` |
| `load_x_cm` | 加载点 x 坐标/cm | `Load_X_cm`, `load_x`, `px_cm`, `p1_x` |
| `load_y_cm` | 加载点 y 坐标/cm | `Load_Y_cm`, `load_y`, `py_cm`, `p1_y` |
| `load_kN` | 荷载大小 | `Load_kN`, `load`, `load_level`, `load_kg` |
| `direction` | 应变方向 | `Direction`, `axis`, `orientation` |
| `layer` | 传感器层位 | `Layer`, `material`, `sensor_layer` |
| `strain_ue` | 实测应变/με | `Measured_Strain_ue`, `strain`, `response`, `delta_lambda_resp_nm` |
| `load_type` | 荷载类型，可选 | `Load_Type`, `type` |

项目现有的原生 `Response_Matrix_Table.csv` 也可直接读取。程序会作以下显式转换：

- `x_cm`、`y_cm` → 传感器坐标；
- `p1_x`、`p1_y` → 第一加载点坐标；
- 双点工况存在有效 `p2_x`、`p2_y` 时，取两个加载点的几何中心作为本版单一等效加载位置；
- `load_kg` 按标准重力加速度换算为 `load_kN`；
- `delta_lambda_resp_nm` 按 Step19 使用的 `1.2e-3 nm/με` 换算为 `strain_ue`。

`direction` 应能够区分 x/纵向与 y/横向。`layer` 应能够区分 asphalt/沥青层与 soil/基层/土层。若缺少 `layer` 列，程序会把全部数据视为沥青层；此时可以运行前两种模型，但多层模型不具备有效土层监督。
不属于沥青层或土层的记录（例如温度补偿记录）会被排除并给出警告。

试件真实边界默认设置为：

```text
x = 0～120 cm
y = 0～50 cm
```

如实际尺寸不同，请修改 `src/fbg_config.m` 中的 `specimenLengthCm` 和 `specimenWidthCm`。

## 4. 输出

每次实验都会新建：

```text
工作区/outputs/时间戳_模式_v10/
```

主要文件包括：

- `run.log`：完整运行日志；
- `00_data_diagnostics/shift_90_95_summary.csv`；
- `00_data_diagnostics/cross_layer_clamp_profiles.csv`；
- `00_data_diagnostics/measured_transfer_by_case.csv`；
- `predictions_all.csv`：全部逐点预测；
- `global_asphalt_summary.csv`：沥青层整体指标；
- `target_95cm_summary.csv`：95 cm 专项指标；
- `paired_improvement_summary.csv`：相对 Step19 的配对改善；
- `soil_auxiliary_summary.csv`：土层预测指标；
- `transfer_eta_by_case.csv`：学习到的工况传递系数；
- `fold_XX/模型名/`：各折模型、训练历史、预测表和指标。

`split` 是所有汇总表统一使用的划分字段名。程序内部所有预测索引均强制转换为列向量，避免 MATLAB 建表时出现行列方向不一致。

## 5. 实验解释

- `BoundaryRelative-FiLM` 相对 Step19 的差异，主要检验真实边界与相对加载坐标是否改善有限边界工况；
- `Multilayer-Transfer-FiLM` 相对边界模型的差异，主要检验土层辅助监督和可学习层间传递是否提供额外信息；
- `eta` 是模型中的等效响应传递参数，不应直接解释为固定材料界面常数；
- 95 cm 指标应结合 90/95 平移一致性、上下层夹持剖面和实测传递比共同判断。

## 6. 已验证范围

本包已在 MATLAB R2024a、Deep Learning Toolbox、CPU 环境完成以下验证：

- `CHECK_ENVIRONMENT` 的二阶自动微分冒烟测试；
- 合成双层数据的三模型端到端 `RUN_DEMO`；
- 项目原生 22 工况响应矩阵的数据规范化与 `RUN_DATA_DIAGNOSTICS`；
- 项目原生响应矩阵的三模型 180 迭代 `RUN_QUICK`；
- 逐折模型保存、逐点预测、总体/95 cm/土层指标及 `eta` 汇总文件生成。

完整 5 折 `RUN_FULL` 的运行时间明显更长，建议确认快速实验输出后再启动。如出现错误，请保留完整报错堆栈和 `run.log`。
