# 当前最优模型：零目标观测Direct RBF场预测器

本目录保存基于 `CorrectedData_v2` 的当前最优模型。任务定义为：测试工况不提供任何沥青层目标响应，仅根据荷载大小、加载位置和加载类型预测28个沥青层FBG的持载末期响应。

主程序为 `direct_rbf_zero_observation.py`，支持固定五折验证、全数据训练保存和新工况预测。

## 五折验证

```powershell
python best_model/direct_rbf_zero_observation.py cross-validate `
  ../fbg-multilayer-pavement-data/CorrectedData_v2/Corrected_Response_Matrix_Table_v2.csv `
  --folds best_model/fixed_case_folds.csv `
  --output results/direct_rbf_cv
```

## 保存部署模型

```powershell
python best_model/direct_rbf_zero_observation.py fit `
  ../fbg-multilayer-pavement-data/CorrectedData_v2/Corrected_Response_Matrix_Table_v2.csv `
  --model models/direct_rbf_corrected_v2.npz
```

## 新工况预测

新工况CSV至少包含：`case_id, load_kg, load_type, p1_x, p1_y, p2_x, p2_y`。

```powershell
python best_model/direct_rbf_zero_observation.py predict `
  models/direct_rbf_corrected_v2.npz new_cases.csv `
  --output predictions.csv
```

输出为每个新工况28行，字段包括 `case_id`、`sensor_id` 和 `predicted_ue`。

## 适用边界

- 输出是持载末期FBG应变响应，不是绝对沉降；
- 当前仅适用于训练荷载类型和位置覆盖范围内的插值或弱外推；
- 不使用测试工况土层或沥青层响应；
- 当前数据不包含卸载，模型不能预测卸载后残余沉降。
