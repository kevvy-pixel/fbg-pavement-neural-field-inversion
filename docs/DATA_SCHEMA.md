# CorrectedData_v2输入结构

当前最优模型读取数据仓库中的：

```text
CorrectedData_v2/Corrected_Response_Matrix_Table_v2.csv
```

## 必需字段

| 字段 | 含义 |
|---|---|
| `case_id` | 加载工况唯一标识 |
| `sensor_id` | FBG测点标识 |
| `x_cm`,`y_cm`,`z_cm` | 传感器坐标，单位cm |
| `layer` | `Asphalt`、`Soil`或`Temperature` |
| `direction` | `Longitudinal`、`Transverse`或`Temperature` |
| `channel` | 解调通道，如D1_CH1 |
| `interrogator` | Demod1或Demod2 |
| `zero_load_baseline_nm` | 同次试验加载前基线 |
| `delta_lambda_initial_nm` | 加载初期相对基线变化 |
| `delta_lambda_final_nm` | 持载末期相对基线变化；当前模型目标 |
| `delta_lambda_peak_signed_nm` | 加载阶段绝对值最大带符号响应 |
| `loaded_creep_change_nm` | 持载末期减加载初期 |
| `preload_noise_std_nm` | 加载前噪声标准差 |
| `load_kg` | 加载质量，单位kg |
| `load_type` | 加载类型 |
| `p1_x`,`p1_y` | 第一个加载点坐标 |
| `p2_x`,`p2_y` | 第二个加载点坐标；单点工况允许为空 |
| `load_onset_s` | 检测到的加载切换时刻 |
| `analysis_quality` | 数据质量标记 |

## 正确通道映射

| 通道 | 层位 | 方向 | 数量 |
|---|---|---|---:|
| D1_CH1 | 沥青层 | 纵向 | 8 |
| D1_CH2 | 沥青层 | 横向 | 6 |
| D1_CH3 | 沥青层 | 纵向 | 8 |
| D1_CH4 | 沥青层 | 横向 | 6 |
| D2_CH1 | 土层 | 纵向 | 8 |
| D2_CH2 | 土层 | 横向 | 6 |
| D2_CH3 | 土层 | 纵向 | 8 |
| D2_CH4 | 温度参考 | 温度 | 1 |

## 规模检查

标准版本应包含：

- 22个工况；
- 每工况28个沥青FBG；
- 每工况22个土层FBG；
- 每工况1个温度参考；
- 总计1122行。

在MATLAB中运行：

```matlab
run('check_data_schema.m')
```

## 质量策略

- 沥青目标当前全部保留；
- `EXCLUDE_HIGH_PRELOAD_NOISE`土层行默认不作为辅助输入；
- `REVIEW_HISTORICAL_DRIFT`建议降权；
- `TEMPERATURE_REFERENCE`不作为应变标签。

## 数据过程限制

原始记录只包含零荷载到加载并持载，不包含卸载。`delta_lambda_final_nm`表示持载末期响应，不等价于卸载后残余沉降。
