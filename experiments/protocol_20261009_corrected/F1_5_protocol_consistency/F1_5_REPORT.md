# F1.5：协议一致性与指标复算

## 结论

- 数据 SHA256 一致：True
- development fold/case 一致：True
- A0 OOF 合并键完全匹配：True（238850 行）
- truth 最大差值：0

## A0 统一复算

- 旧 Stage A：fold macro=30.076119，pooled=32.551142
- F1：fold macro=30.272772，pooled=32.698686
- A0 逐行预测平均绝对差：2.616433 με
- A0 逐行预测最大绝对差：25.289154 με

## 原因判定

1. `30.076` 与 `32.699` 首先来自 fold-macro 与 pooled OOF 两种聚合口径混用。
2. A0 逐行预测仍有差异，因为旧 Stage A 错误地从外层训练集再排除了一个内部验证 fold；F1 修正版使用完整外层训练集。
3. 数据、fold、评价时间、传感器数量、truth 和合并键没有变化。
4. 后续统一以每个 seed 独立计算 pooled OOF，再报告 seed mean±std；fold macro 仅作为稳定性指标。
5. `LongitudinalLine` 报告名称统一为 `SinglePointAxisScan`，原字段保留。
