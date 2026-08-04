# 基于FBG稀疏观测的路面连续场反演

本仓库研究如何根据加载工况和稀疏FBG传感数据重建多层路面响应场。当前推荐模型已经从旧版Step19边界感知神经场更新为基于 `CorrectedData_v2` 的**零目标观测Direct RBF多输出场模型**。

中文项目名称：**基于物理启发场表示的FBG路面响应反演与结构异常预警**。

## 当前研究任务

对一个训练阶段未见的新工况，不输入该工况的任何沥青层目标响应，只使用：

- 荷载大小；
- 第一个加载点坐标；
- 第二个加载点坐标及是否存在；
- 单点、双点对称、双点不对称、偏心或纵向位置加载类型；
- 加载点到试件边界的距离及交互特征；

预测持载末期28个沥青层FBG响应。

当前数据过程为：

`零荷载 → 加载瞬间 → 持载蠕变 → 持载末期`

没有卸载和恢复数据，因此当前模型不能预测卸载后的残余沉降。

## 推荐入口

Python直接运行：

```powershell
python best_model/direct_rbf_zero_observation.py cross-validate `
  ../fbg-multilayer-pavement-data/CorrectedData_v2/Corrected_Response_Matrix_Table_v2.csv `
  --output results/direct_rbf_cv
```

也可以在MATLAB仓库根目录运行：

```matlab
run('run_main.m')
```

MATLAB入口会定位同级数据仓库、检查修正数据结构，然后调用Python执行固定五折验证和全数据模型保存。

## 当前最优模型框架

### 1. 输入特征

工况元数据编码为：

\[
\mathbf{x}=[P,x_1,y_1,x_2,y_2,I_2,d_{12},d_{e1},d_{e2},
Px_1,Py_1,PI_2,\mathrm{onehot}(type)]
\]

其中：

- \(P\) 为归一化荷载；
- \((x_1,y_1),(x_2,y_2)\) 为归一化加载点坐标；
- \(I_2\) 表示是否存在第二加载点；
- \(d_{12}\) 为双点间距；
- \(d_{e1},d_{e2}\) 为加载点到最近边界的距离；
- `onehot(type)` 为加载类型编码。

### 2. 多输出RBF场回归

训练工况之间的核函数为：

\[
K_{ij}=\exp(-\gamma\|\tilde{\mathbf{x}}_i-\tilde{\mathbf{x}}_j\|^2)
\]

模型一次预测全部28个沥青测点：

\[
\mathbf{W}=(\mathbf{K}+\alpha\mathbf{I})^{-1}\mathbf{Y},\qquad
\hat{\mathbf{Y}}_*=\mathbf{K}_*\mathbf{W}
\]

其中 \(\mathbf{Y}\in\mathbb{R}^{N\times28}\) 是持载末期沥青响应场。与逐传感器训练不同，该形式共享同一工况相似度，并保留28通道联合结构。

### 3. 无泄漏模型选择

- 外层：固定工况五折；
- 测试工况28个沥青目标全部不可见；
- 内层：只在外层训练工况中执行留一工况验证；
- 内层联合选择 \(\gamma\in\{0.01,0.03,0.1,0.3,1,3\}\) 和 \(\alpha\in\{10^{-4},...,100\}\)；
- 所有特征标准化参数仅使用对应训练折估计。

## 修正数据上的主要结果

共22个独立工况、616个沥青目标，测试工况零沥青观测。

| 模型 | 总体RMSE（με） | MAE | 纵向RMSE | 横向RMSE | 95 cm RMSE |
|---|---:|---:|---:|---:|---:|
| **Direct RBF，仅元数据** | **32.152** | **13.069** | **39.879** | 17.079 | 57.809 |
| Mindlin/Kirchhoff残差RBF | 32.989 | 14.497 | 41.055 | 17.089 | 58.796 |
| Direct RBF，元数据+土层 | 33.087 | 14.059 | 41.062 | 17.500 | 55.446 |
| POD–RBF，仅元数据 | 35.286 | 14.949 | 44.319 | **16.918** | **54.828** |
| v19 Single锚定神经场 | 39.882 | 16.744 | 44.250 | 33.175 | 107.447 |
| v23可回退联级 | 40.262 | 16.998 | 44.848 | 33.176 | 107.447 |
| v27 Dynamic Winkler PINN | 40.608 | 17.901 | 45.153 | 33.606 | 108.326 |
| v19 Top3Soil | 40.998 | 18.601 | 45.764 | 33.609 | 108.325 |

Direct RBF相对v27的RMSE降低8.456 με；工况级50,000次配对Bootstrap区间为 `[0.635,17.227]` με，Direct RBF更优概率为98.67%。相对POD–RBF降低3.133 με，区间为 `[0.474,6.540]` με。

## 消融实验结论

1. **数据修正是主要增益来源。** 同一Direct RBF结构从旧数据53.854降到32.152 με。
2. **POD压缩不再是必要条件。** 数据映射修正后，直接28维输出优于POD–RBF。
3. **土层静态汇总没有平均增益。** Direct RBF加入土层后为33.087 με，未超过仅元数据版本。
4. **Top3Soil专家排序反转。** 修正后Single为39.882，Top3Soil为40.998 με。
5. **联级门控没有兑现专家互补。** 21个工况选择Single，1个切换Top3Soil且该切换增加误差。
6. **强物理残差跨折不稳定。** v27为40.608 με；Mindlin模型在五折中均退化到Kirchhoff极限。
7. **弱物理锚定可以保留。** Mindlin残差RBF与主模型差0.837 με，但区间跨零，不足以替换主模型。

详细数据见：

- `docs/ABLATION_RESULTS.md`；
- `results/corrected_v2_ablation/model_summary.csv`；
- `results/corrected_v2_ablation/before_after.csv`；
- `results/corrected_v2_ablation/paired_bootstrap.csv`。

## 代码结构

| 路径 | 作用 | 状态 |
|---|---|---|
| `best_model/direct_rbf_zero_observation.py` | 当前最优Direct RBF训练、验证、保存和预测 | **推荐** |
| `best_model/fixed_case_folds.csv` | 冻结的22工况五折划分 | **推荐** |
| `run_main.m` | MATLAB统一入口 | **推荐** |
| `check_data_schema.m` | CorrectedData_v2结构检查 | **推荐** |
| `Step13_*.m` | 直接响应MLP | 历史基线 |
| `Step14_*.m` | 位移代理神经场与自动微分应变 | 探索模型 |
| `Step15_*.m` | 沥青层与荷载均衡 | 消融 |
| `Step16_*.m` | 弱边界锚定 | 消融 |
| `Step17_*.m` | FiLM条件神经场 | 历史主干 |
| `Step18_*.m` | 点级空间插值 | 任务诊断 |
| `Step19_*.m` | 边界感知FiLM神经场 | 历史主模型 |
| `Step21_*.m` | 边界残差专家 | 负消融 |

## 运行环境

当前最优模型：

- Python 3.10或更高版本；
- NumPy；
- pandas。

历史MATLAB神经场：

- MATLAB；
- Deep Learning Toolbox；
- 支持二阶自动微分的足够内存。

安装Python依赖：

```powershell
python -m pip install -r best_model/requirements.txt
```

## 数据位置

推荐把三个仓库放在同一父目录：

```text
fbg-multilayer-pavement-data/
fbg-multilayer-pavement-modeling/
fbg-pavement-neural-field-inversion-repo/
```

也可以设置环境变量 `FBG_CORRECTED_DATA`，直接指向修正CSV。

## 物理与工程解释边界

- 模型预测的是持载末期FBG应变响应场；
- 由应变反演出的位移是相对挠度或位移代理，不是独立测量的绝对沉降；
- 未加入LVDT、DIC或激光位移标定前，不报告绝对毫米沉降；
- 当前没有真实塌陷标签，只能构建正常响应基线和异常评分，不能声称输出塌陷概率；
- 当前数据不含卸载过程，不能验证残余沉降和恢复率。

## 后续时空模型

当前Direct RBF作为持载末期静态端点先验，下一阶段建议构建：

`Direct RBF端点场 + 连续坐标神经场 + Burgers/广义Maxwell时间状态 + 弱Kirchhoff/Winkler约束 + 小型残差头`

目标输出为从0荷载到持载稳定全过程的 `(x,y,t)` 相对应变与相对挠度热力图。

## 许可说明

源代码与实验数据是独立成果。未附带开源许可证时，仓库公开可见不代表自动授予复制、修改或再分发权利。数据访问和公开须遵守试验授权与论文数据政策。
