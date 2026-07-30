# FBG神经动力学场v9：SOTA Ridge-4/6空间划分稳健性实验

v9直接读取已经完成的v7全训练结果，不重新训练神经网络。

## 1. 运行命令

解压后将MATLAB当前目录切换到本文件夹：

```matlab
clear functions
rehash toolboxcache
CHECK_ENVIRONMENT

RUN_NEXT_EXPERIMENT( ...
    "D:\path\to\FBG_NeuralField_Workspace\outputs\20260728_103759_full_sota_comparison")
```

将路径替换为本地实际的`*_full_sota_comparison`目录。

## 2. 为什么进行重复空间划分

之前的Ridge-4结果只对应一组校准坐标，无法判断改善是否依赖这一次测点选择。v9对
每个工况执行20个互不重复的空间覆盖划分。

每次划分：

1. 选择6个空间覆盖校准坐标；
2. Ridge-4使用第1至4个坐标；
3. Ridge-6使用第1至6个坐标；
4. 第7个及其后的坐标构成公共固定测试集；
5. SOTA、Ridge-4、Ridge-6和门控Ridge-6在相同测试集上评价。

Ridge-4和Ridge-6为嵌套测点设计，因此差异不会来自不同的测试坐标。

## 3. 比较方法

```text
Step19-SOTA-FiLM
Step19-SOTA+Ridge-4
Step19-SOTA+Ridge-6
Step19-SOTA+Gated-Ridge-6
```

6点门控只使用6个校准坐标的留一坐标交叉验证，不读取固定测试集。

## 4. 输出

结果保存在原全训练目录下：

```text
04_sota_ridge_robustness/
├─ method_summary.csv
├─ global_metrics_by_repeat.csv
├─ repeat_case_metrics.csv
├─ case_robustness_summary.csv
├─ paired_improvement_summary.csv
├─ target_95cm_repeat_metrics.csv
├─ target_95cm_summary.csv
├─ ridge6_gate_decisions.csv
├─ ridge6_gate_audit.csv
├─ calibration_point_assignments.csv
├─ predictions.csv
├─ experiment_configuration.csv
├─ method_robustness.png
├─ case_win_rates.png
└─ run.log
```

## 5. 主要判断

- `method_summary.csv`：20次全工况Micro指标的均值和标准差；
- `paired_improvement_summary.csv`：所有工况×重复的配对收益和胜率；
- `case_robustness_summary.csv`：每个工况的平均收益、标准差、胜率及5%–95%区间；
- `target_95cm_summary.csv`：95 cm在Ridge-4/6下的稳健性；
- `ridge6_gate_audit.csv`：6点门控是否比4点门控可靠。

## 6. 完成后请上传

```text
method_summary.csv
paired_improvement_summary.csv
case_robustness_summary.csv
target_95cm_summary.csv
ridge6_gate_audit.csv
run.log
```

此步骤只读取已保存的SOTA预测并进行闭式岭回归，通常很快完成。
