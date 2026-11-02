# Protocol 2026-10-09：新数据包修正版阶段 A

## 数据与语义冻结

- 主数据入口：`guochuang_fbg_pavement_dataset/reorganized_dataset/film_dynamic_training_table.csv`
- 坐标单位：cm
- 动态训练窗口：0–280 s，281 个时间点
- 22 个工况、50 个结构传感器、309100 行
- 17 个 development cases；5 个 fixed-test cases
- `LongitudinalLine` 统一解释为 `SinglePointAxisScan`
- 模型荷载坐标使用动态训练表中的 case-bound `p1/p2`
- `spatial_model/load_points_coordinates.csv` 仅用于几何审计
- 主训练损失为普通 MSE；质量权重未用于主模型训练
- fixed-test 未打开

## 阶段 0 审计

详见：

`D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009\data_audit\DATA_AUDIT_REPORT.md`

几何核对结果：13 个工况 match，9 个工况 mismatch。mismatch 主要为双点和单点偏心工况，不能在来源确认前作物理方向解释。

## 阶段 A 首轮筛选结果

本轮只保留 A0-3、B0、B3，四折 case-level OOF，B0/B3 使用 3 个随机种子。为先完成修正版协议筛选，本轮训练预算为 5 epochs；结果不作为充分收敛最终排名。

| 模型 | overall RMSE 均值 | early RMSE | late RMSE | late asphalt | late soil |
|---|---:|---:|---:|---:|---:|
| A0-3 | 30.076 | 38.895 | 24.485 | 32.294 | 5.537 |
| B0 | 30.090 | 39.291 | 24.179 | 32.150 | 3.283 |
| B3 | 30.400 | 39.780 | 24.357 | 32.533 | 0.966 |

## 当前判断

1. 在新数据包、普通 MSE、G1 几何输入和短训练预算下，A0-3 与 B0 接近，B3 的土层误差更低，但整体 RMSE 暂未优于 A0-3/B0。
2. 这不是最终模型排序，因为当前仍是筛选预算，且 B0/B3 的跨种子差异需要在充分收敛后重新报告。
3. 旧版使用质量加权损失、虚拟线段特征和旧数据路径的结果，不与本报告混合。
4. 下一步先按修正版数据入口充分重跑阶段 A，再做双点荷载外推和天然传感器串留出。
5. 在荷载几何来源未进一步确认、模型结构未冻结前，不打开 fixed-test，也不进入残差混合模型。
