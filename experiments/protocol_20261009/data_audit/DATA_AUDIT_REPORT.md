# 数据阶段 0 审计（Protocol 2026-10-09 修正版）

统一数据根目录：`C:\Users\27356\Documents\Codex\2026-10-09\c-users-27356-documents-codex-2026\outputs\guochuang_fbg_pavement_dataset`

## 完整性

```json
{
  "training_rows": 309100,
  "unique_cases": 22,
  "unique_structural_sensors": 50,
  "time_min": 0,
  "time_max": 280,
  "time_points_per_case_sensor": 281,
  "missing_response": 0,
  "temperature_rows": 0,
  "development_cases": 17,
  "fixed_test_cases": 5
}
```

## 荷载几何核对

```text
         case_id        load_type  training_p1_x_cm  training_p1_y_cm  training_p2_x_cm  training_p2_y_cm  spatial_p1_x_cm  spatial_p1_y_cm  spatial_p2_x_cm  spatial_p2_y_cm geometry_status                   chosen_source  geometry_semantics
      10kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
     10kg双点不对称加载   DualAsymmetric              40.0              25.0              90.0              25.0             40.0               15             80.0             35.0        mismatch film_dynamic_training_table.csv           DualPoint
      10kg双点对称加载    DualSymmetric              40.0              25.0              80.0              25.0             60.0               15             60.0             35.0        mismatch film_dynamic_training_table.csv           DualPoint
      20kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
     25kg双点不对称加载   DualAsymmetric              40.0              25.0              90.0              25.0             40.0               15             80.0             35.0        mismatch film_dynamic_training_table.csv           DualPoint
      25kg双点对称加载    DualSymmetric              40.0              25.0              80.0              25.0             60.0               15             60.0             35.0        mismatch film_dynamic_training_table.csv           DualPoint
      30kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
      35kg双点对称加载    DualSymmetric              40.0              25.0              80.0              25.0             60.0               15             60.0             35.0        mismatch film_dynamic_training_table.csv           DualPoint
      40kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
      40kg单点偏心加载  SingleEccentric              60.0              20.0               NaN               NaN             60.0               15              NaN              NaN        mismatch film_dynamic_training_table.csv         SinglePoint
      40kg双点对称加载    DualSymmetric              40.0              25.0              80.0              25.0             60.0               15             60.0             35.0        mismatch film_dynamic_training_table.csv           DualPoint
  40kg纵中心线30cm加载 LongitudinalLine              30.0              25.0               NaN               NaN             30.0               25              NaN              NaN           match film_dynamic_training_table.csv SinglePointAxisScan
40kg纵中心线42.5cm加载 LongitudinalLine              42.5              25.0               NaN               NaN             42.5               25              NaN              NaN           match film_dynamic_training_table.csv SinglePointAxisScan
  40kg纵中心线90cm加载 LongitudinalLine              90.0              25.0               NaN               NaN             90.0               25              NaN              NaN           match film_dynamic_training_table.csv SinglePointAxisScan
  40kg纵中心线95cm加载 LongitudinalLine              95.0              25.0               NaN               NaN             95.0               25              NaN              NaN           match film_dynamic_training_table.csv SinglePointAxisScan
      50kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
       5kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
      5kg双点不对称加载   DualAsymmetric              40.0              25.0              90.0              25.0             40.0               15             80.0             35.0        mismatch film_dynamic_training_table.csv           DualPoint
       5kg双点对称加载    DualSymmetric              40.0              25.0              80.0              25.0             60.0               15             60.0             35.0        mismatch film_dynamic_training_table.csv           DualPoint
      60kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
      70kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
      80kg单点中心加载     SingleCenter              60.0              25.0               NaN               NaN             60.0               25              NaN              NaN           match film_dynamic_training_table.csv         SinglePoint
```

## 结论

- 当前正式模型输入继续采用 `reorganized_dataset/film_dynamic_training_table.csv` 中与 case_id 逐行绑定的 p1/p2。
- `LongitudinalLine` 统一解释为 `SinglePointAxisScan`，不构造 line_start/line_end/line_length/line_angle。
- `spatial_model/load_points_coordinates.csv` 仅作为空间示意和几何审计来源。
- 所有 mismatch 必须在物理坐标含义确认后再解释；当前不把 mismatch 自动归因于模型错误。
- fixed-test 仍保持关闭。