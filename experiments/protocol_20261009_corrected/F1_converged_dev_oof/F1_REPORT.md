# F1：A0/B0/B3 充分收敛 development OOF

- 数据：新数据包 `guochuang_fbg_pavement_dataset/reorganized_dataset`
- 训练采样：5 s；评价采样：1 s；loss：ordinary MSE
- 训练上限：150 epochs；至少 50 epochs；patience=20
- fixed-test：closed

## Seed mean ± std（每个 seed 先独立计算 pooled OOF，再汇总）

| model | overall mean | std |
|---|---:|---:|
| A0-3 | 32.699 | nan |
| B0 | 32.559 | 0.012 |
| B3 | 32.946 | 0.025 |

## 最差案例和传感器串

```text
model   case_id   rmse_ue
 A0-3 5kg双点对称加载 74.904187
   B0 5kg双点对称加载 75.387715
   B3 5kg双点对称加载 75.247334

model sensor_string   layer   rmse_ue
 A0-3          SAL2 Asphalt 59.234972
   B0          SAL2 Asphalt 59.717901
   B3          SAL2 Asphalt 59.892083

SAL2:
model   rmse_ue
 A0-3 59.234972
   B0 59.717901
   B3 59.892083
```

## 解释边界

1. 所有传感器仍保留在主指标；SAL2 仅作专项诊断。
2. B3 的土层改善不能自动等同于全结构最优。
3. F2/F3 正式外推应使用本轮充分收敛 checkpoint 和同一评价脚本。