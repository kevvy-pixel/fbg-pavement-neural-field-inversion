# FBG FiLM 模型实验演进 LaTeX 报告

本文件夹包含一份可编辑、可重新编译的中文 LaTeX 报告，系统整理了：

- 原始 Step19 边界感知 FiLM 条件调制 AD-DNF；
- 95 cm 极边界问题及原始 `R²≈0.95` 的指标口径解释；
- Residual-FiLM 与空间边界专家的快速筛选；
- Step19-SOTA 与 Residual-FiLM 的同规格完整训练对照；
- 4 个稀疏测点的岭仿射校准及 4 点门控失败原因；
- 20 次嵌套空间划分下 Ridge-4、Ridge-6、Gated Ridge-6 的稳健性；
- 最终敲定的 SOTA + Ridge-6 / Gated Ridge-6 框架。

## 文件说明

- `FBG_FiLM_Model_Evolution_Report_CN.tex`：LaTeX 正文。
- `FBG_FiLM_Model_Evolution_Report_CN.pdf`：已编译报告。
- `BUILD_REPORT.bat`：Windows 双击编译。
- `BUILD_REPORT.sh`：Linux/macOS 编译。
- `make_figures.py`：由关键 CSV 重新生成图表。
- `data/`：正文数值所用的关键实验输出。
- `figures/`：PDF 矢量图和 PNG 预览图。
- `fonts/`：本地可放置 Noto Sans CJK SC 字体。两份 OTF 文件体积约 33 MB，
  默认不提交 Git；已编译 PDF 不受影响。

## Windows 编译

1. 安装 TeX Live 或 MiKTeX，并确保包含 XeLaTeX。
2. 将 `NotoSansCJKsc-Regular.otf` 和 `NotoSansCJKsc-Bold.otf` 放入
   `fonts/`；也可按本机字体设置修改 LaTeX 导言区。
3. 可选：安装 Python、pandas、matplotlib，以便重新生成图表。
4. 双击 `BUILD_REPORT.bat`。

如果只修改正文、不改数据图，直接运行两次：

```text
xelatex FBG_FiLM_Model_Evolution_Report_CN.tex
xelatex FBG_FiLM_Model_Evolution_Report_CN.tex
```

必须使用 XeLaTeX；不要使用 pdfLaTeX。

## 修改数据图

将新的 CSV 按现有列名放入 `data/`，然后运行：

```text
python make_figures.py
```

再重新编译 LaTeX。
