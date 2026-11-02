# F2/F3：正式荷载类型与传感器串外推

- 普通 MSE；充分收敛设置；fixed-test closed。
- A0-3 deterministic；B0/B3 三个 seed 分开计算。

## 模型平均

 index split model      mean       std
     0    C1  A0-3 31.434600 14.321992
     1    C1    B0 30.309395 15.359707
     2    C1    B3 30.923759 15.682538
     3    D1  A0-3 24.744240 22.306579
     4    D1    B0 22.173133 24.682129
     5    D1    B3 20.785093 26.038001

## F2 各留出荷载类型 overall pooled RMSE

split model         held_out   phase          sse      n  pooled_rmse
   C1  A0-3   DualAsymmetric overall 4.078892e+07  28100    38.099383
   C1  A0-3    DualSymmetric overall 1.612139e+08  56200    53.559077
   C1  A0-3 LongitudinalLine overall 1.441079e+07  28100    22.645963
   C1  A0-3     SingleCenter overall 4.176532e+07 112400    19.276347
   C1  A0-3  SingleEccentric overall 7.820135e+06  14050    23.592229
   C1    B0   DualAsymmetric overall 1.284134e+08  84300    39.029352
   C1    B0    DualSymmetric overall 4.706451e+08 168600    52.834545
   C1    B0 LongitudinalLine overall 2.061197e+07  84300    15.636731
   C1    B0     SingleCenter overall 1.357866e+08 337200    20.067104
   C1    B0  SingleEccentric overall 2.423643e+07  42150    23.979245
   C1    B3   DualAsymmetric overall 1.455207e+08  84300    41.547859
   C1    B3    DualSymmetric overall 4.699615e+08 168600    52.796163
   C1    B3 LongitudinalLine overall 2.062974e+07  84300    15.643470
   C1    B3     SingleCenter overall 1.347583e+08 337200    19.990978
   C1    B3  SingleEccentric overall 2.559119e+07  42150    24.640327

## F3 各传感器串 breakdown

model held_out  rmse_mean      n
 A0-3     SAL1  31.442750  38216
 A0-3     SAL2  68.341828  38216
 A0-3     SAT1  11.698125  28662
 A0-3     SAT2  34.958143  28662
 A0-3     SSL1  10.768371  38216
 A0-3     SSL2   5.842971  38216
 A0-3      SST  10.157489  28662
   B0     SAL1  32.020751 114648
   B0     SAL2  69.280576 114648
   B0     SAT1   5.397209  85986
   B0     SAT2  34.234477  85986
   B0     SSL1   4.445788 114648
   B0     SSL2   3.822756 114648
   B0      SST   5.993182  85986
   B3     SAL1  32.083582 114648
   B3     SAL2  69.378316 114648
   B3     SAT1   4.801292  85986
   B3     SAT2  34.904773  85986
   B3     SSL1   2.348525 114648
   B3     SSL2   1.002960 114648
   B3      SST   0.966479  85986

## 当前判断

1. B3 的 F2 整体外推约 30.923，低于 A0-3 但高于 B0；因此不能选为无条件主模型。
2. B3 的 F3 整体空间外推约 20.784，优于 B0 和 A0-3；主要收益来自土层，沥青层需要单独看。
3. SAL2 仍是最差传感器串之一，主指标没有删除 SAL2。
4. 阶段 E 双重外推暂不启动，先完成 F2/F3 的 DualAsymmetric、SAL2/SAT2 专项分解和协议冻结。
