# F2.5 / F3.5：DualAsymmetric、SAL2 与双点交换敏感性审计

## F2.5：DualAsymmetric 分解

| 统计范围 | B0 RMSE | B3 RMSE | B3−B0 |
|---|---:|---:|---:|
| 全部传感器 | 39.029 | 41.548 | +2.519 |
| 排除 SAL2 的诊断统计 | 37.337 | 38.715 | +1.378 |

因此 B3 的退化不能完全归因于 SAL2；即使排除 SAL2，B3 仍比 B0 高约 1.38 με。

按层位：

| 层位 | B0 | B3 | B3−B0 |
|---|---:|---:|---:|
| Asphalt | 52.094 | 55.515 | +3.421 |
| Soil | 2.845 | 0.896 | −1.949 |

B3 的 DualAsymmetric 退化主要来自沥青层，而土层明显改善。

按传感器串，退化较明显的包括：

- SAL2：+7.115 με；
- SAT1：+5.909 με；
- SAL1：+3.503 με；
- SAT2：+2.464 με。

这表明问题是“沥青层双点外推 + 多个空间串”的组合，不是单一 SAL2 传感器异常。

## F3.5：DualSymmetric 与 SAL2

DualSymmetric 全部传感器：

- B0：52.835 με；
- B3：52.796 με；
- 差异：−0.038 με。

排除 SAL2 的诊断统计中，B3 反而高于 B0 约 0.713 με；因此 B3 的 DualSymmetric 总体接近，不代表沥青层普遍改善。

## 双点交换敏感性

对已保存 checkpoint 做了无需重新训练的 `p1,p2` 交换测试。

DualAsymmetric：

- B0 交换后 RMSE 平均上升约 2.43 με；
- B3 交换后 RMSE 平均下降约 2.65 με；
- B3 的原始输入与交换输入预测差异约 7.62–10.75 με。

这说明 B3 对双点顺序存在明显敏感性，且原始点顺序可能恰好落在其较差的表示区域。由于两个荷载点没有独立力分配记录，当前不能把 p1/p2 顺序解释为物理受力顺序；后续应优先测试交换不变荷载编码。

DualSymmetric 的交换影响较小，但 B3 仍存在小幅变化。

## 当前结论

1. B3 的 DualAsymmetric 失败主要发生在沥青层。
2. SAL2 会放大误差，但不是唯一原因。
3. B3 的土层优势是真实存在的，但不足以抵消沥青层退化。
4. 双点输入顺序敏感性是 G2 交换不变编码最直接的动机。
5. 在 F1.5、F2.5、F3.5 结果基础上，暂不进入残差模型和物理约束。

详细文件包括：

- `DualAsymmetric_layer_decomposition.csv`
- `DualAsymmetric_sensor_string_decomposition.csv`
- `DualAsymmetric_distance_decomposition.csv`
- `DualAsymmetric_sal2_diagnostic.csv`
- `DualSymmetric_sal2_diagnostic.csv`
- `swap_sensitivity.csv`
- `curves/`
