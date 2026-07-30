# FBG 低频锚定与载荷位置平滑神经场 v14

v14 在 v13 最近工况锚定框架上只研究两项改动：

1. 降低坐标 Fourier 特征的最高频率；
2. 对载荷位置施加 ±1 cm 扰动一致性约束。

本版本不修改、不平滑原始应变数据，避免把模型结构收益和数据去噪收益混在一起。

## 1. 严格消融模型

1. `Step19-SOTA-FiLM`
   - 原始绝对应变场基线。
2. `NearestAnchor-Persistence`
   - 直接使用最近训练工况的同坐标实测应变。
3. `AnchoredResidual-XOnly-FiLM`
   - v13 当前频率参考。
4. `AnchoredResidual-XOnly-LowFreq`
   - 只降低 Fourier 频率。
5. `AnchoredResidual-XOnly-LowFreqSmooth`
   - 降低 Fourier 频率并加入载荷位置平滑损失。

所有锚定模型沿用 v13 的无泄漏规则：训练采用留一工况锚定，验证和测试只能从
训练工况库选择锚点。95 cm 的 28 个测点全部匹配到 90 cm 同坐标测点。

## 2. Fourier 降频

| 特征 | v13 当前频率 | v14 低频 |
|---|---|---|
| 绝对坐标 `x/y` | `[1,2,4,8]` | `[1,2,4]` |
| 相对坐标 `x-p/y-p` | `[1,2,4]` | `[1,2]` |

位移场通过二阶自动微分得到应变，高频特征的曲率贡献按频率平方放大。删除绝对
频率 8 和相对频率 4，目标是降低载荷位置小扰动产生的曲率振荡。

## 3. 载荷位置平滑约束

对于单加载点训练样本，构造：

```text
p_plus  = p + 1 cm
p_minus = p - 1 cm
```

加载点坐标、锚点位置差和锚点工况距离同步更新。平滑损失为：

```text
L_position =
    0.5 * MSE((prediction(p_plus)  - prediction(p)) / strain_scale)
  + 0.5 * MSE((prediction(p_minus) - prediction(p)) / strain_scale)
```

总损失：

```text
L = L_asphalt
  + 0.01 * L_correction
  + 0.05 * L_position
  + 1e-7 * L_regularization
```

平滑约束使用有限扰动，不增加对载荷位置的高阶自动微分。

## 4. 运行

要求 MATLAB R2024a 或更高版本，以及 Deep Learning Toolbox。

```matlab
clear functions
rehash toolboxcache
CHECK_ENVIRONMENT
RUN_DEMO
```

真实数据快速实验：

```matlab
RUN_QUICK( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

完整五折：

```matlab
RUN_FULL( ...
    "D:\你的数据\Response_Matrix_Table.csv", ...
    "C:\Users\你的用户名\Documents\MATLAB\FBG_NeuralField_Workspace")
```

## 5. 95 cm 真实数据结果

### 5.1 总体精度

| 模型 | R² | RMSE/μɛ | MAE/μɛ | Bias/μɛ |
|---|---:|---:|---:|---:|
| Step19-SOTA-FiLM | 0.4618 | 187.57 | 135.11 | -60.96 |
| NearestAnchor-Persistence | 0.8479 | 99.73 | 67.04 | -25.86 |
| AnchoredResidual-XOnly-FiLM | 0.8574 | 96.55 | **60.87** | -16.95 |
| AnchoredResidual-XOnly-LowFreq | **0.8731** | **91.07** | 62.35 | **-16.88** |
| AnchoredResidual-XOnly-LowFreqSmooth | 0.8724 | 91.32 | 62.50 | -17.04 |

低频模型相对 Step19：

- RMSE 降低 `96.51 μɛ`，改善 `51.45%`；
- R² 提高 `0.4113`。

相对 v13 XOnly：

- RMSE 降低 `5.48 μɛ`；
- R² 提高 `0.0157`；
- MAE 增加 `1.48 μɛ`，说明低频模型改善了大误差点，但部分普通点绝对误差略增。

### 5.2 分方向 RMSE

| 模型 | x/μɛ | y/μɛ |
|---|---:|---:|
| Step19-SOTA-FiLM | 220.09 | 132.32 |
| NearestAnchor-Persistence | 125.06 | 48.51 |
| AnchoredResidual-XOnly-FiLM | 120.62 | 48.51 |
| AnchoredResidual-XOnly-LowFreq | **112.91** | 48.51 |
| AnchoredResidual-XOnly-LowFreqSmooth | 113.26 | 48.51 |

XOnly 规则保证横向结果始终等于锚点，所有变化来自纵向残差。

### 5.3 ±1 cm 位置敏感度

指标是95 cm预测在载荷位置分别移动 ±1 cm 后，逐点平均绝对变化的 RMS。

| 模型 | 全部测点 RMS/μɛ | x方向 RMS/μɛ | 最大变化/μɛ |
|---|---:|---:|---:|
| Step19-SOTA-FiLM | 26.42 | 26.45 | 76.42 |
| NearestAnchor-Persistence | 0 | 0 | 0 |
| AnchoredResidual-XOnly-FiLM | 9.61 | 12.71 | 39.84 |
| AnchoredResidual-XOnly-LowFreq | 4.29 | 5.68 | 12.57 |
| AnchoredResidual-XOnly-LowFreqSmooth | **4.25** | **5.62** | **12.48** |

相对 v13 XOnly，低频模型将纵向位置敏感度降低约 `55.3%`，最大变化降低约
`68.5%`。额外平滑约束只带来约 `1%` 的边际稳定性改善，同时 RMSE 增加
`0.25 μɛ`。

### 5.4 训练诊断

| 模型 | 最佳迭代 | 最佳验证 RMSE/μɛ | 停止迭代 |
|---|---:|---:|---:|
| Step19-SOTA-FiLM | 280 | 64.00 | 400 |
| AnchoredResidual-XOnly-FiLM | 40 | 74.79 | 280 |
| AnchoredResidual-XOnly-LowFreq | 20 | 96.61 | 260 |
| AnchoredResidual-XOnly-LowFreqSmooth | 20 | 95.92 | 260 |

验证集仍然全部集中在 60 cm，验证排名不能代表95 cm边界泛化排名。低频版本虽然
验证 RMSE 较高，但95 cm测试更好，进一步证明后续必须建立边界位置验证集。

## 6. 结论

- 降低 Fourier 频率有效：同时提高95 cm精度并显著降低位置敏感度；
- 当前 `0.05` 权重的位置平滑约束有效但边际收益很小；
- 推荐保留 `AnchoredResidual-XOnly-LowFreq` 作为 v14 精度候选；
- `LowFreqSmooth` 保留为稳定性消融，不替代低频无平滑模型；
- 暂不对原始数据去噪。下一步只有在重复试验能够估计测量噪声后，才比较平滑
  样条、小波或高斯过程去噪，避免删除真实边界峰值。

当前结论仍来自单个95 cm留出工况，最终使用前需要完整五折和边界位置重复试验。

## 7. 主要输出

每次实验写入 `工作区/outputs/时间戳_模式_v14/`：

- `global_asphalt_summary.csv`
- `direction_asphalt_summary.csv`
- `position_stability_summary.csv`
- `anchor_summary.csv`
- `paired_improvement_summary.csv`
- `predictions_all.csv`
- 各模型的锚点审计、训练历史和逐点预测
