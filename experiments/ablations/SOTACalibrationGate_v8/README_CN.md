# FBG神经动力学场v8：SOTA稀疏校准与无泄漏门控

v8建立在已完成的v7全训练结果上。对于现有结果目录，**不重新训练任何神经网络**。

## 1. 最快运行方法

解压后将MATLAB当前目录切换到本文件夹，执行：

```matlab
clear functions
rehash toolboxcache
CHECK_ENVIRONMENT

RUN_POSTPROCESS_EXISTING( ...
    "D:\path\to\FBG_NeuralField_Workspace\outputs\20260728_103759_full_sota_comparison")
```

请将路径替换为你本地实际的`*_full_sota_comparison`目录。

程序读取每折已经保存的：

```text
fold_XX/01_step19_sota_film/predictions.csv
fold_XX/fold_configuration.mat
calibration_point_assignments.csv
```

因此不需要再次运行5折、10个主干的训练。

## 2. 比较方法

在完全相同的固定测试坐标上比较：

| 方法 | 说明 |
| --- | --- |
| `Step19-SOTA-FiLM` | 不校准的全局SOTA主干 |
| `Step19-SOTA+Ridge-4` | 所有工况强制使用4点岭校准 |
| `Step19-SOTA+Gated-Ridge-4` | 由校准点LOOCV决定是否使用岭校准 |

## 3. 门控规则

4个校准坐标分别轮流作为内部验证坐标：

1. 使用其余3个坐标拟合尺度`a`与偏置`b`；
2. 预测被留出的坐标；
3. 汇总4次留一结果；
4. 将LOOCV岭校准RMSE与原始SOTA RMSE比较。

只有同时满足以下条件才启用最终Ridge-4：

- LOOCV相对RMSE提升不低于5%；
- LOOCV绝对RMSE提升不低于2 με；
- 最终尺度没有靠近`[0.5,1.5]`边界；
- 所有拟合结果有限。

固定测试坐标只用于最后评价，不参与门控判断。

## 4. 输出位置

结果保存在原v7输出目录下：

```text
03_sota_ridge_gate/
├─ method_summary.csv
├─ case_metrics.csv
├─ predictions.csv
├─ gate_decision_by_case.csv
├─ calibration_loocv_predictions.csv
├─ paired_improvement_summary.csv
├─ target_95cm_summary.csv
├─ gate_audit_summary.csv
├─ gate_configuration.csv
├─ method_comparison.png
├─ gate_diagnostic.png
└─ run.log
```

## 5. 完成后请上传

```text
method_summary.csv
gate_decision_by_case.csv
paired_improvement_summary.csv
target_95cm_summary.csv
gate_audit_summary.csv
run.log
```

## 6. 从头完整运行

如果以后需要从原始响应矩阵重新训练并自动执行门控，可以运行：

```matlab
RUN_FULL( ...
    "D:\你的路径\Response_Matrix_Table.csv", ...
    "D:\path\to\FBG_NeuralField_Workspace")
```

已有v7结果时不要使用该命令，使用`RUN_POSTPROCESS_EXISTING`即可。
