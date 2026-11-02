# Protocol 2026-10-09：阶段 C+D 报告

## 执行边界

- 输入版本：G1 几何输入；荷载类型标签未作为主输入。
- development-only；fixed-test 5 个工况始终未打开。
- C1：5 类荷载类型留出；D1：7 条天然传感器串留出。
- 训练轮数为计算预算版：C1=5 epochs，D1=3 epochs；因此本报告用于协议筛选，不替代最终充分收敛训练。
- 所有预测文件均保留，按完整案例/传感器串留出，未按时间点随机拆分。
- `pooled_metrics.csv` 是跨种子拼接后的 pooled 指标；`metrics_by_holdout.csv` 是文件级 seed 统计。

## C1 汇总

| 模型 | 留出对象 | overall RMSE 均值 | std | 最差 case |
|---|---|---:|---:|---:|
| A0-3 | DualAsymmetric | 38.267 | 0.000 | 50.022 |
| A0-3 | DualSymmetric | 52.776 | 0.000 | 75.623 |
| A0-3 | LongitudinalLine | 19.141 | 0.000 | 20.956 |
| A0-3 | SingleCenter | 20.834 | 0.000 | 31.222 |
| A0-3 | SingleEccentric | 23.412 | 0.000 | 23.412 |
| B0 | DualAsymmetric | 38.479 | 0.000 | 50.803 |
| B0 | DualSymmetric | 52.775 | 0.000 | 75.516 |
| B0 | LongitudinalLine | 17.225 | 0.000 | 19.047 |
| B0 | SingleCenter | 20.295 | 0.000 | 33.414 |
| B0 | SingleEccentric | 24.418 | 0.000 | 24.418 |
| B3 | DualAsymmetric | 40.568 | 0.000 | 53.435 |
| B3 | DualSymmetric | 52.985 | 0.000 | 75.675 |
| B3 | LongitudinalLine | 15.398 | 0.000 | 17.163 |
| B3 | SingleCenter | 20.088 | 0.000 | 33.050 |
| B3 | SingleEccentric | 24.554 | 0.000 | 24.554 |

### 按阶段/层位均值

```text
 index split model         held_out        phase      mean      std  count
     0    C1  A0-3   DualAsymmetric        early 56.732017      NaN      1
     1    C1  A0-3   DualAsymmetric         late 24.983262      NaN      1
     2    C1  A0-3   DualAsymmetric late_asphalt 33.130006      NaN      1
     3    C1  A0-3   DualAsymmetric    late_soil  4.648779      NaN      1
     4    C1  A0-3   DualAsymmetric      overall 38.266555      NaN      1
     5    C1  A0-3    DualSymmetric        early 60.385313      NaN      1
     6    C1  A0-3    DualSymmetric         late 48.712352      NaN      1
     7    C1  A0-3    DualSymmetric late_asphalt 64.914756      NaN      1
     8    C1  A0-3    DualSymmetric    late_soil  5.455378      NaN      1
     9    C1  A0-3    DualSymmetric      overall 52.775994      NaN      1
    10    C1  A0-3 LongitudinalLine        early 27.010770      NaN      1
    11    C1  A0-3 LongitudinalLine         late 13.872238      NaN      1
    12    C1  A0-3 LongitudinalLine late_asphalt 17.896302      NaN      1
    13    C1  A0-3 LongitudinalLine    late_soil  5.453005      NaN      1
    14    C1  A0-3 LongitudinalLine      overall 19.141294      NaN      1
    15    C1  A0-3     SingleCenter        early 28.274812      NaN      1
    16    C1  A0-3     SingleCenter         late 16.095062      NaN      1
    17    C1  A0-3     SingleCenter late_asphalt 19.001388      NaN      1
    18    C1  A0-3     SingleCenter    late_soil 11.367967      NaN      1
    19    C1  A0-3     SingleCenter      overall 20.834119      NaN      1
    20    C1  A0-3  SingleEccentric        early 35.418184      NaN      1
    21    C1  A0-3  SingleEccentric         late 14.485584      NaN      1
    22    C1  A0-3  SingleEccentric late_asphalt 19.035965      NaN      1
    23    C1  A0-3  SingleEccentric    late_soil  3.961773      NaN      1
    24    C1  A0-3  SingleEccentric      overall 23.412049      NaN      1
    25    C1    B0   DualAsymmetric        early 57.231554 0.242957      3
    26    C1    B0   DualAsymmetric         late 24.921083 0.225533      3
    27    C1    B0   DualAsymmetric late_asphalt 33.280407 0.304895      3
    28    C1    B0   DualAsymmetric    late_soil  1.353647 0.136796      3
    29    C1    B0   DualAsymmetric      overall 38.479499 0.215801      3
    30    C1    B0    DualSymmetric        early 60.657305 0.196255      3
    31    C1    B0    DualSymmetric         late 48.547979 0.077690      3
    32    C1    B0    DualSymmetric late_asphalt 64.841922 0.095502      3
    33    C1    B0    DualSymmetric    late_soil  2.306661 0.451615      3
    34    C1    B0    DualSymmetric      overall 52.774605 0.120163      3
    35    C1    B0 LongitudinalLine        early 24.600975 0.176240      3
    36    C1    B0 LongitudinalLine         late 12.203582 0.216431      3
    37    C1    B0 LongitudinalLine late_asphalt 16.066395 0.310545      3
    38    C1    B0 LongitudinalLine    late_soil  3.149869 0.128341      3
    39    C1    B0 LongitudinalLine      overall 17.224952 0.171907      3
    40    C1    B0     SingleCenter        early 29.892908 0.035463      3
    41    C1    B0     SingleCenter         late 13.461172 0.038199      3
    42    C1    B0     SingleCenter late_asphalt 17.918937 0.049232      3
    43    C1    B0     SingleCenter    late_soil  1.779657 0.025879      3
    44    C1    B0     SingleCenter      overall 20.295413 0.033864      3
    45    C1    B0  SingleEccentric        early 36.992769 0.047386      3
    46    C1    B0  SingleEccentric         late 15.045450 0.079466      3
    47    C1    B0  SingleEccentric late_asphalt 20.045130 0.101977      3
    48    C1    B0  SingleEccentric    late_soil  1.747331 0.188724      3
    49    C1    B0  SingleEccentric      overall 24.417792 0.050955      3
    50    C1    B3   DualAsymmetric        early 59.956332 0.297923      3
    51    C1    B3   DualAsymmetric         late 26.688589 0.341852      3
    52    C1    B3   DualAsymmetric late_asphalt 35.659589 0.456518      3
    53    C1    B3   DualAsymmetric    late_soil  0.640451 0.053976      3
    54    C1    B3   DualAsymmetric      overall 40.568057 0.294496      3
    55    C1    B3    DualSymmetric        early 61.015890 0.544074      3
    56    C1    B3    DualSymmetric         late 48.671155 0.307164      3
    57    C1    B3    DualSymmetric late_asphalt 65.037164 0.409530      3
    58    C1    B3    DualSymmetric    late_soil  0.620549 0.141451      3
    59    C1    B3    DualSymmetric      overall 52.984786 0.392700      3
    60    C1    B3 LongitudinalLine        early 22.726694 0.162345      3
    61    C1    B3 LongitudinalLine         late 10.161400 0.160477      3
    62    C1    B3 LongitudinalLine late_asphalt 13.547572 0.214911      3
    63    C1    B3 LongitudinalLine    late_soil  1.037283 0.005576      3
    64    C1    B3 LongitudinalLine      overall 15.397544 0.149093      3
    65    C1    B3     SingleCenter        early 29.606184 0.292816      3
    66    C1    B3     SingleCenter         late 13.303178 0.179543      3
    67    C1    B3     SingleCenter late_asphalt 17.739812 0.240008      3
    68    C1    B3     SingleCenter    late_soil  1.298585 0.010115      3
    69    C1    B3     SingleCenter      overall 20.087820 0.220009      3
    70    C1    B3  SingleEccentric        early 37.118830 0.154048      3
    71    C1    B3  SingleEccentric         late 15.223958 0.190849      3
    72    C1    B3  SingleEccentric late_asphalt 20.339415 0.255123      3
    73    C1    B3  SingleEccentric    late_soil  0.480231 0.003039      3
    74    C1    B3  SingleEccentric      overall 24.554255 0.155335      3
```

## D1 汇总

| 模型 | 留出对象 | overall RMSE 均值 | std | 最差 case |
|---|---|---:|---:|---:|
| A0-3 | SAL1 | 31.959 | 0.000 | 72.916 |
| A0-3 | SAL2 | 68.655 | 0.000 | 187.762 |
| A0-3 | SAT1 | 13.003 | 0.000 | 20.425 |
| A0-3 | SAT2 | 34.965 | 0.000 | 123.424 |
| A0-3 | SSL1 | 11.750 | 0.000 | 26.744 |
| A0-3 | SSL2 | 6.960 | 0.000 | 9.956 |
| A0-3 | SST | 10.227 | 0.000 | 17.573 |
| B0 | SAL1 | 31.836 | 0.000 | 75.157 |
| B0 | SAL2 | 68.959 | 0.000 | 187.868 |
| B0 | SAT1 | 7.093 | 0.000 | 14.606 |
| B0 | SAT2 | 34.650 | 0.000 | 128.329 |
| B0 | SSL1 | 4.117 | 0.000 | 6.457 |
| B0 | SSL2 | 3.812 | 0.000 | 7.749 |
| B0 | SST | 4.683 | 0.000 | 7.349 |
| B3 | SAL1 | 32.174 | 0.000 | 75.822 |
| B3 | SAL2 | 68.738 | 0.000 | 187.425 |
| B3 | SAT1 | 4.597 | 0.000 | 14.984 |
| B3 | SAT2 | 34.681 | 0.000 | 127.354 |
| B3 | SSL1 | 2.350 | 0.000 | 5.113 |
| B3 | SSL2 | 1.015 | 0.000 | 2.792 |
| B3 | SST | 0.968 | 0.000 | 3.645 |

### 按阶段/层位均值

```text
 index split model held_out        phase      mean      std  count
    75    D1  A0-3     SAL1        early 48.721409      NaN      1
    76    D1  A0-3     SAL1         late 19.330305      NaN      1
    77    D1  A0-3     SAL1 late_asphalt 19.330305      NaN      1
    78    D1  A0-3     SAL1      overall 31.959111      NaN      1
    79    D1  A0-3     SAL2        early 80.692449      NaN      1
    80    D1  A0-3     SAL2         late 62.068361      NaN      1
    81    D1  A0-3     SAL2 late_asphalt 62.068361      NaN      1
    82    D1  A0-3     SAL2      overall 68.655029      NaN      1
    83    D1  A0-3     SAT1        early 15.880528      NaN      1
    84    D1  A0-3     SAT1         late 11.369638      NaN      1
    85    D1  A0-3     SAT1 late_asphalt 11.369638      NaN      1
    86    D1  A0-3     SAT1      overall 13.002935      NaN      1
    87    D1  A0-3     SAT2        early 51.509276      NaN      1
    88    D1  A0-3     SAT2         late 23.181378      NaN      1
    89    D1  A0-3     SAT2 late_asphalt 23.181378      NaN      1
    90    D1  A0-3     SAT2      overall 34.965321      NaN      1
    91    D1  A0-3     SSL1        early 11.471786      NaN      1
    92    D1  A0-3     SSL1         late 11.881608      NaN      1
    93    D1  A0-3     SSL1    late_soil 11.881608      NaN      1
    94    D1  A0-3     SSL1      overall 11.750455      NaN      1
    95    D1  A0-3     SSL2        early  9.210441      NaN      1
    96    D1  A0-3     SSL2         late  5.568580      NaN      1
    97    D1  A0-3     SSL2    late_soil  5.568580      NaN      1
    98    D1  A0-3     SSL2      overall  6.959839      NaN      1
    99    D1  A0-3      SST        early  7.177639      NaN      1
   100    D1  A0-3      SST         late 11.401632      NaN      1
   101    D1  A0-3      SST    late_soil 11.401632      NaN      1
   102    D1  A0-3      SST      overall 10.226554      NaN      1
   103    D1    B0     SAL1        early 49.265989 0.027782      3
   104    D1    B0     SAL1         late 18.343852 0.032972      3
   105    D1    B0     SAL1 late_asphalt 18.343852 0.032972      3
   106    D1    B0     SAL1      overall 31.836101 0.026636      3
   107    D1    B0     SAL2        early 81.499764 0.189117      3
   108    D1    B0     SAL2         late 62.060807 0.071532      3
   109    D1    B0     SAL2 late_asphalt 62.060807 0.071532      3
   110    D1    B0     SAL2      overall 68.958569 0.115834      3
   111    D1    B0     SAT1        early  8.333184 0.111333      3
   112    D1    B0     SAT1         late  6.415193 0.179584      3
   113    D1    B0     SAT1 late_asphalt  6.415193 0.179584      3
   114    D1    B0     SAT1      overall  7.093459 0.151970      3
   115    D1    B0     SAT2        early 51.903054 0.092101      3
   116    D1    B0     SAT2         late 22.031876 0.132145      3
   117    D1    B0     SAT2 late_asphalt 22.031876 0.132145      3
   118    D1    B0     SAT2      overall 34.649995 0.101520      3
   119    D1    B0     SSL1        early  5.085330 0.026242      3
   120    D1    B0     SSL1         late  3.560675 0.037684      3
   121    D1    B0     SSL1    late_soil  3.560675 0.037684      3
   122    D1    B0     SSL1      overall  4.116728 0.032513      3
   123    D1    B0     SSL2        early  4.406216 0.110291      3
   124    D1    B0     SSL2         late  3.492041 0.108377      3
   125    D1    B0     SSL2    late_soil  3.492041 0.108377      3
   126    D1    B0     SSL2      overall  3.812222 0.106154      3
   127    D1    B0      SST        early  4.198826 0.173682      3
   128    D1    B0      SST         late  4.897208 0.132682      3
   129    D1    B0      SST    late_soil  4.897208 0.132682      3
   130    D1    B0      SST      overall  4.682604 0.140169      3
   131    D1    B3     SAL1        early 49.722683 0.080493      3
   132    D1    B3     SAL1         late 18.624461 0.033514      3
   133    D1    B3     SAL1 late_asphalt 18.624461 0.033514      3
   134    D1    B3     SAL1      overall 32.174393 0.053103      3
   135    D1    B3     SAL2        early 81.250189 0.009673      3
   136    D1    B3     SAL2         late 61.855021 0.011225      3
   137    D1    B3     SAL2 late_asphalt 61.855021 0.011225      3
   138    D1    B3     SAL2      overall 68.737815 0.010511      3
   139    D1    B3     SAT1        early  6.289130 0.112819      3
   140    D1    B3     SAT1         late  3.504973 0.297368      3
   141    D1    B3     SAT1 late_asphalt  3.504973 0.297368      3
   142    D1    B3     SAT1      overall  4.596893 0.200067      3
   143    D1    B3     SAT2        early 51.830103 0.183533      3
   144    D1    B3     SAT2         late 22.186594 0.184616      3
   145    D1    B3     SAT2 late_asphalt 22.186594 0.184616      3
   146    D1    B3     SAT2      overall 34.681390 0.166011      3
   147    D1    B3     SSL1        early  3.580700 0.001186      3
   148    D1    B3     SSL1         late  1.422989 0.004922      3
   149    D1    B3     SSL1    late_soil  1.422989 0.004922      3
   150    D1    B3     SSL1      overall  2.349742 0.001973      3
   151    D1    B3     SSL2        early  1.533694 0.008521      3
   152    D1    B3     SSL2         late  0.630008 0.003807      3
   153    D1    B3     SSL2    late_soil  0.630008 0.003807      3
   154    D1    B3     SSL2      overall  1.014950 0.005758      3
   155    D1    B3      SST        early  1.383208 0.010099      3
   156    D1    B3      SST         late  0.685446 0.006182      3
   157    D1    B3      SST    late_soil  0.685446 0.006182      3
   158    D1    B3      SST      overall  0.968135 0.007141      3
```

## 当前判断

1. C1 和 D1 的冻结划分及完整预测已生成。
2. G1 版本下，DualAsymmetric 和 SAL2 等对象仍是高风险区域。
3. D1 中 B3 在多数土层传感器串上较好，但 SAL2 的沥青层外推仍困难。
4. 本轮使用短训练预算，下一步应对 C1 中排名靠前的模型/荷载类型补充充分收敛训练和多种子复核。
5. 在 C1、D1 充分收敛结果确认前，不进入阶段 E/F，也不打开 fixed-test。
