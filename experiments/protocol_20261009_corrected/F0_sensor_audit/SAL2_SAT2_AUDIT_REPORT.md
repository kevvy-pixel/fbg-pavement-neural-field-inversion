# SAL2/SAT2 专项审计（F0）

## 审计范围

- 主数据：新数据包 `reorganized_dataset/film_dynamic_training_table.csv`。
- 当前动态训练表覆盖 0–280 s；原始 Demod 文本已保留在 `raw_data/`，但其仪器导出格式不直接作为模型输入。
- 主指标继续保留全部传感器；本报告只提供 SAL2/SAT2 专项诊断，不删除异常传感器。

## 映射和空间关系

sensor_id channel sensor_string   layer    direction  x_cm  y_cm  z_cm unit same_string_neighbors            nearest_same_layer  nearest_dist_cm
D2_CH1_S1  D2_CH1          SAL2 Asphalt Longitudinal  37.0  18.0  -4.0   cm   D2_CH1_S2;D2_CH1_S3 D2_CH1_S2;D2_CH2_S2;D2_CH2_S3   5.00;5.52;7.11
D2_CH1_S2  D2_CH1          SAL2 Asphalt Longitudinal  42.0  18.0  -4.0   cm   D2_CH1_S1;D2_CH1_S3 D2_CH2_S2;D2_CH2_S3;D2_CH1_S1   0.71;4.53;5.00
D2_CH1_S3  D2_CH1          SAL2 Asphalt Longitudinal  47.0  18.0  -4.0   cm   D2_CH1_S2;D2_CH1_S4 D2_CH2_S2;D2_CH1_S2;D2_CH1_S4   4.53;5.00;5.00
D2_CH1_S4  D2_CH1          SAL2 Asphalt Longitudinal  52.0  18.0  -4.0   cm   D2_CH1_S3;D2_CH1_S5 D2_CH1_S3;D2_CH1_S5;D1_CH1_S1   5.00;5.00;7.28
D2_CH1_S5  D2_CH1          SAL2 Asphalt Longitudinal  57.0  18.0  -4.0   cm   D2_CH1_S4;D2_CH1_S6 D2_CH1_S4;D2_CH1_S6;D1_CH1_S2   5.00;5.00;7.28
D2_CH1_S6  D2_CH1          SAL2 Asphalt Longitudinal  62.0  18.0  -4.0   cm   D2_CH1_S5;D2_CH1_S7 D2_CH1_S5;D2_CH1_S7;D1_CH1_S3   5.00;5.00;7.28
D2_CH1_S7  D2_CH1          SAL2 Asphalt Longitudinal  67.0  18.0  -4.0   cm   D2_CH1_S6;D2_CH1_S8 D2_CH1_S6;D2_CH1_S8;D1_CH1_S4   5.00;5.00;7.28
D2_CH1_S8  D2_CH1          SAL2 Asphalt Longitudinal  72.0  18.0  -4.0   cm   D2_CH1_S7;D2_CH1_S6 D2_CH1_S7;D1_CH1_S5;D1_CH1_S6   5.00;7.28;7.62
D2_CH2_S1  D2_CH2          SAT2 Asphalt   Transverse  42.5  12.5  -4.0   cm   D2_CH2_S2;D2_CH2_S3 D2_CH2_S2;D2_CH1_S2;D2_CH1_S3   5.00;5.52;7.11
D2_CH2_S2  D2_CH2          SAT2 Asphalt   Transverse  42.5  17.5  -4.0   cm   D2_CH2_S1;D2_CH2_S3 D2_CH1_S2;D2_CH1_S3;D2_CH2_S1   0.71;4.53;5.00
D2_CH2_S3  D2_CH2          SAT2 Asphalt   Transverse  42.5  22.5  -4.0   cm   D2_CH2_S2;D2_CH2_S4 D2_CH1_S2;D2_CH2_S2;D2_CH2_S4   4.53;5.00;5.00
D2_CH2_S4  D2_CH2          SAT2 Asphalt   Transverse  42.5  27.5  -4.0   cm   D2_CH2_S3;D2_CH2_S5 D2_CH2_S3;D2_CH2_S5;D1_CH1_S1   5.00;5.00;7.91
D2_CH2_S5  D2_CH2          SAT2 Asphalt   Transverse  42.5  32.5  -4.0   cm   D2_CH2_S4;D2_CH2_S6 D2_CH2_S4;D2_CH2_S6;D2_CH2_S3  5.00;5.00;10.00
D2_CH2_S6  D2_CH2          SAT2 Asphalt   Transverse  42.5  37.5  -4.0   cm   D2_CH2_S5;D2_CH2_S4 D2_CH2_S5;D2_CH2_S4;D1_CH1_S1 5.00;10.00;14.58

## 响应异常摘要

       load_type sensor_string  max_abs_ue_mean  late_abs_mean_ue  max_abs_ue_max  sign_change_mean
  DualAsymmetric          SAL2        65.355208         10.043719      610.891667          4.583333
  DualAsymmetric          SAT2        75.644444         21.651602      419.200000          5.833333
   DualSymmetric          SAL2        66.325625         29.115854      610.975000          6.525000
   DualSymmetric          SAT2        17.349028          5.506723       61.391667          3.200000
LongitudinalLine          SAL2        63.024349         12.571425      616.866667          7.593750
LongitudinalLine          SAT2        80.355035         20.433169      375.141667          4.625000
    SingleCenter          SAL2        44.698090          6.320229      614.375000          8.138889
    SingleCenter          SAT2        18.336265          4.246358       79.925000         10.740741
 SingleEccentric          SAL2        95.351042         24.747821      182.225000          0.000000
 SingleEccentric          SAT2        37.586806         12.234890       73.150000          0.333333

## D2-CH1/质量权重

质量权重文件列：sensor_id, channel, sensor_index, n_files, median_std_nm, max_std_nm, median_abs_slope_nm_s, max_range_nm, max_missing_rate, matched_omit_observed_peak, noise_ratio, weight_noise, weight_drift, weight_tracking, quality_weight, quality_rule
匹配到 SAL2/SAT2/D2_CH1 的行数：8

## 当前结论

1. SAL2 位于 D2_CH1，8 个点，沥青层，纵向方向，坐标 x=37–72 cm、y=18 cm。
2. SAT2 位于 D2_CH2，6 个点，沥青层，横向方向，坐标 x=42.5 cm、y=12.5–37.5 cm。
3. 曲线按荷载类型分别保存，虚线 90 s 用于 early/late 分段。
4. SAL2/SAT2 不因误差较大而从主指标删除；后续同时报告全传感器与专项诊断。
5. 需要结合质量权重原始峰文件进一步确认 D2_CH1 峰身份；在此之前 raw weighted RMSE 仅作诊断。
