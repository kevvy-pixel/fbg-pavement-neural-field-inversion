# V4-A：固定测试集下的黏弹时间头验证

本实验只使用17个开发工况，不读取5个固定测试工况的原始FBG响应。目标是验证单指数Prony时间头是否优于自由相位输出和简单线性时间外推。

比较模型：

- `direct_full`：RBF根据荷载条件直接预测全部传感器和全部时间点；
- `prony1_full`：单指数Prony基展开，使用全部开发时间点；
- `prony2_full`：双指数Prony基展开，使用全部开发时间点；
- `linear_early`：只使用5–90秒拟合线性时间头，外推120–210秒；
- `prony1_early`：只使用5–90秒拟合单指数Prony时间头；
- `prony2_early`：只使用5–90秒拟合双指数Prony时间头。

开发集沿用原第2–5折作为四折交叉验证。每个外层折内再次使用其余开发折选择RBF核宽、岭参数和Prony时间常数。测试集在本阶段完全封存。

正式运行：

```powershell
D:\anaconda3\python.exe run_v4a.py --data-root D:\matlab\guochuang --output artifacts
```

`--smoke`只运行一个外层折并缩小参数网格，用于代码和数值稳定性检查；不能作为正式结果。
