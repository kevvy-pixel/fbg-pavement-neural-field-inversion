# 时空微应变场实验代码与产物索引

本目录保存从显式时间建模到修正版最终评估的实际本地实验文件。与仓库 `main` 的 Step13–21 静态/位移代理神经场不同，这里以 `relative_time_s` 为输入、直接预测分层微应变 `response_ue`。数据源为独立仓库 [`fbg-multilayer-pavement-data`](https://github.com/kevvy-pixel/fbg-multilayer-pavement-data) 的修正版数据集；训练与汇总指标的权威解释见 [`../docs/时空微应变场_完整实验记录.md`](../docs/时空微应变场_完整实验记录.md)。

## 演化与入口

| 路径 | 目的 | 主要入口 |
|---|---|---|
| `LayeredViscoelasticSTNF_v4/` | Prony/RBF、坐标 MLP、早期 FiLM 与分层模型探索 | `run_v4a.py`、`film_dynamic_oof/run_layered_film_dynamic.py` |
| `protocol_20261008/` | 统一 0–280 s、case-level OOF、A0/B0/B3 多种子及留型预试验 | `run_phase_ab.py`、`run_a0_prony.py`、`run_leave_type.py` |
| `protocol_20261009/` | 留一荷载类型与传感器串协议、几何审计；保留历史纠错轨迹 | `dataset_geometry_audit.py`、`protocol_20261009_runner.py` |
| `protocol_20261009_corrected/` | 正确映射数据上的 F0–F4、G 消融、F7 冻结测试 | `protocol_20261009_corrected_runner.py` 及各阶段 `*_runner.py` |

修正版中，`F0_sensor_audit` 检查 SAL2/SAT2；`F1_converged_dev_oof` 进行充分收敛 OOF；`F1_5_protocol_consistency` 校对统计口径；`F2_leave_type_out`、`F3_sensor_string_holdout` 与 `F4_double_holdout` 检验工况、空间及双重外推；`G_ablation` 比较 B0/B3/G2/G3；`F7_fixed_test` 是一次性冻结评估。分割定义在 `splits/`，各阶段 `RUN_MANIFEST.json` 和汇总 CSV 保留实际设置与结果。

## 数据、运行与解释边界

- 修正版训练输入是数据仓库 `reorganized_dataset/film_dynamic_training_table.csv`。原始缓存为 0–598 s，正式训练/评价窗口为 0–280 s；开发集 17 个工况，固定测试 5 个工况。
- 当前本地脚本含原机器的 `C:\Users\27356\...` 和 `D:\matlab\guochuang\...` 绝对路径。异机复现时应先将这些路径指向数据仓库和本目录，核对 `RUN_MANIFEST.json` 后再运行。
- 环境主要为 Python、NumPy、pandas、PyTorch；顶层 `requirements.txt` 列出直接依赖，V4 的 `requirements.txt` 保留其原始依赖。运行前应核对数据映射和 `split_role`，不能重新用已打开的 fixed-test 选择模型。
- 本次纳入源码、参数配置、划分文件、审计报告和汇总指标；没有复制 2.5 GB 的逐点预测、训练历史、checkpoint 和可从数据仓库取得的重复训练表。它们仍在原 D 盘实验目录。所复制本地文件的原始 SHA-256 见 `SOURCE_FILE_MANIFEST.csv`；Git 的换行符处理可能使检出后的字节哈希不同。
- 早期 `protocol_20261008`/`protocol_20261009` 和 V4 的指标只用于追溯方法演化；最终模型结论以 `protocol_20261009_corrected`、`ABLATION_MASTER_TABLE.md` 与 `F7_fixed_test/FIXED_TEST_REPORT.md` 为准。
