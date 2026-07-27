%% =========================================================================
%  Step15_Asphalt_LoadBalanced_Displacement_Neural_Field.m
%
%  目标：
%    建立“位移神经场 + 自动微分应变监督”模型：
%
%      w_theta(x, y, condition, layer)  --->  displacement proxy field
%
%    训练时不直接拟合位移，而是通过自动微分得到曲率和应变：
%
%      epsilon_x_pred = -z * d2w/dx2
%      epsilon_y_pred = -z * d2w/dy2
%
%    再根据 FBG 方向选择：
%
%      纵向 FBG: epsilon_pred = epsilon_x_pred
%      横向 FBG: epsilon_pred = epsilon_y_pred
%
%    最终用 epsilon_pred 拟合实测 FBG 应变。
%
%  输入：
%    Response_Matrix_Table.csv
%
%  输出：
%    Step15_Asphalt_LoadBalanced_DNF_Results/
%      ├── Figs/      图片，全部英文标注
%      ├── Tables/    训练历史、点预测、工况指标、连续场 CSV
%      ├── Models/    训练好的模型和归一化参数
%      └── Logs/      中文训练日志和中文分析报告
%
%  重要说明：
%    1. 网络输出的 w 是 displacement proxy，不是直接实测绝对位移。
%    2. 由于 FBG 应变只约束 w 的二阶导数，w 本身存在刚体平移和整体倾斜自由度。
%       本脚本通过 gauge loss 和 slope loss 固定位移场的零点与平均倾斜。
%    3. 如果有真实 LVDT / DIC / 位移计数据，后续可进一步标定 w 的绝对量纲。
% =========================================================================

clear; clc; close all;

%% ========================= 0. 用户可调参数 ==============================
cfg = struct();

% ---------- 输入输出 ----------
cfg.inputCSV = 'Response_Matrix_Table.csv';
cfg.outBase  = 'Step15_Asphalt_LoadBalanced_DNF_Results';

% ---------- 几何参数 ----------
cfg.plateLength_cm = 120;
cfg.plateWidth_cm  = 50;

% ---------- FBG 与位移代理参数 ----------
cfg.k_eps_nm_per_ue  = 1.2e-3;     % 1.2 pm/ue = 0.0012 nm/ue
cfg.z_asphalt_m      = 0.020;      % 沥青层有效中性轴距离，仅用于位移代理
cfg.z_soil_m         = 0.020;      % 土层有效中性轴距离，未知时先设为同值
cfg.w_scale_m        = 1.0e-3;     % 网络输出 w_norm 后乘该尺度得到 w_m

% ---------- 数据选择 ----------
% 可选：
%   ["Asphalt"]       只使用沥青层，应变-曲率关系更合理，Step15 默认
%   ["Asphalt","Soil"] 同时使用沥青层与土层，适合探索，但物理解释更弱
cfg.trainLayers = ["Asphalt"];

% ---------- 数据划分 ----------
% case  = 按工况划分，检验跨工况泛化，推荐
% point = 按测点随机划分，检验插值重构能力
cfg.splitMode = 'case';
cfg.valRatio  = 0.15;
cfg.testRatio = 0.15;
cfg.randomSeed = 2026;

% ---------- 训练集数据增强 ----------
% 注意：增强只作用于训练集，验证/测试集不增强，避免数据泄漏。
cfg.useMirrorAugmentation = true;
cfg.augmentMirrorX = true;
cfg.augmentMirrorY = true;

% ---------- 网络结构 ----------
% Step15 适度扩容，但不直接上 128，避免在小样本 case 泛化下过拟合。
cfg.hiddenUnits     = 96;
cfg.numHiddenLayers = 5;
cfg.fourierFreqsXY  = [1 2 4 8];
cfg.fourierFreqsRel = [1 2 4];

% ---------- 训练超参数 ----------
cfg.maxEpochs        = 1800;
cfg.miniBatchSize    = 64;
cfg.initialLearnRate = 8.0e-4;
cfg.learnRateDropPeriod = 400;
cfg.learnRateDropFactor = 0.5;
cfg.gradientDecayFactor = 0.9;
cfg.squaredGradientDecayFactor = 0.999;
cfg.earlyStopPatience = 220;
cfg.printEveryEpoch = 20;

% ---------- loss 权重 ----------
cfg.lambdaL2      = 1.0e-6;    % 权重衰减
cfg.lambdaGauge   = 1.0e-5;    % 抑制 w 的整体平移
cfg.lambdaSlope   = 1.0e-6;    % 抑制 w 的平均倾斜
cfg.lambdaCurvReg = 5.0e-8;    % 轻微曲率能量正则，防止局部震荡

% ---------- 荷载均衡 loss ----------
% 目的：避免大荷载工况在全局 MSE 中占主导，使 5kg/10kg 小响应工况也有足够权重。
% 权重形式：
%   weight = 1 / (loadN + offset)^power
% 并归一化到平均约为 1，避免整体改变 loss 尺度。
cfg.useLoadWeightedLoss = true;
cfg.loadWeightOffset = 0.15;
cfg.loadWeightPower  = 0.5;
cfg.maxLoadWeight    = 3.0;

% ---------- 连续场输出 ----------
cfg.gridNX = 121;
cfg.gridNY = 61;

cfg.fieldCaseList = [ ...
    "40kg单点中心加载", ...
    "40kg单点偏心加载", ...
    "40kg纵中心线30cm加载", ...
    "40kg纵中心线42.5cm加载", ...
    "40kg纵中心线90cm加载", ...
    "40kg纵中心线95cm加载", ...
    "40kg双点对称加载", ...
    "25kg双点不对称加载" ...
];

cfg.fieldLayers = ["Asphalt"];

%% ========================= 1. 创建输出文件夹 =============================
dirs = struct();
dirs.base   = cfg.outBase;
dirs.figs   = fullfile(cfg.outBase, 'Figs');
dirs.tables = fullfile(cfg.outBase, 'Tables');
dirs.models = fullfile(cfg.outBase, 'Models');
dirs.logs   = fullfile(cfg.outBase, 'Logs');

makeDir(dirs.base);
makeDir(dirs.figs);
makeDir(dirs.tables);
makeDir(dirs.models);
makeDir(dirs.logs);

logFile = fullfile(dirs.logs, 'Training_Log_CN.txt');
fidLog = fopen(logFile, 'w');
assert(fidLog > 0, '无法创建训练日志文件。');

logBoth(fidLog, '============================================================');
logBoth(fidLog, 'Step15 沥青层荷载均衡位移神经场 + 自动微分应变监督');
logBoth(fidLog, '============================================================');
logBoth(fidLog, '输出文件夹：%s', cfg.outBase);
logBoth(fidLog, '数据划分方式：%s', cfg.splitMode);
logBoth(fidLog, '训练层：%s', strjoin(cfg.trainLayers, ', '));
logBoth(fidLog, '荷载均衡 loss：%d，offset=%.3g，power=%.3g，maxWeight=%.3g。', cfg.useLoadWeightedLoss, cfg.loadWeightOffset, cfg.loadWeightPower, cfg.maxLoadWeight);

%% ========================= 2. 环境检查 ===================================
if ~license('test','Neural_Network_Toolbox')
    fclose(fidLog);
    error(['未检测到 Deep Learning Toolbox。', newline, ...
           '本脚本需要 MATLAB Deep Learning Toolbox 才能训练 dlnetwork。']);
end

rng(cfg.randomSeed);

%% ========================= 3. 读取并清洗数据 =============================
assert(isfile(cfg.inputCSV), '找不到输入文件：%s。请把脚本放在 Response_Matrix_Table.csv 同目录下运行。', cfg.inputCSV);

T0 = readtable(cfg.inputCSV, 'TextType', 'string');

requiredCols = ["case_id","sensor_id","x_cm","y_cm","layer","direction", ...
    "delta_lambda_resp_nm","load_type","load_kg","p1_x","p1_y","p2_x","p2_y"];
missingCols = setdiff(requiredCols, string(T0.Properties.VariableNames));
assert(isempty(missingCols), '输入表缺少必要列：%s', strjoin(missingCols, ', '));

T0.case_id   = string(T0.case_id);
T0.sensor_id = string(T0.sensor_id);
T0.layer     = string(T0.layer);
T0.direction = string(T0.direction);
T0.load_type = string(T0.load_type);

T0.x_cm = double(T0.x_cm);
T0.y_cm = double(T0.y_cm);
T0.load_kg = double(T0.load_kg);
T0.p1_x = double(T0.p1_x);
T0.p1_y = double(T0.p1_y);
T0.p2_x = double(T0.p2_x);
T0.p2_y = double(T0.p2_y);
T0.delta_lambda_resp_nm = double(T0.delta_lambda_resp_nm);

% 兼容已有脚本中的修正
mask425 = T0.case_id == "40kg纵中心线42.5cm加载";
if any(mask425)
    T0.p1_x(mask425) = 42.5;
end

valid = isfinite(T0.x_cm) & isfinite(T0.y_cm) & isfinite(T0.load_kg) & ...
        isfinite(T0.p1_x) & isfinite(T0.p1_y) & isfinite(T0.delta_lambda_resp_nm) & ...
        ismember(T0.layer, cfg.trainLayers);
T0 = T0(valid, :);

T0.strain_ue = T0.delta_lambda_resp_nm ./ cfg.k_eps_nm_per_ue;

logBoth(fidLog, '数据读取完成：%d 行，%d 个原始工况。', height(T0), numel(unique(T0.case_id)));
logBoth(fidLog, '层信息：%s。', strjoin(unique(T0.layer), ', '));
logBoth(fidLog, '荷载类型：%s。', strjoin(unique(T0.load_type), ', '));

%% ========================= 4. 划分 train/val/test ========================
[trainIdx0, valIdx0, testIdx0, splitInfo] = splitDataByMode(T0, cfg);

Ttrain = T0(trainIdx0,:);
Tval   = T0(valIdx0,:);
Ttest  = T0(testIdx0,:);

logBoth(fidLog, '原始训练/验证/测试样本数：%d / %d / %d。', height(Ttrain), height(Tval), height(Ttest));
logBoth(fidLog, '训练工况数：%d；验证工况数：%d；测试工况数：%d。', ...
    numel(splitInfo.trainCases), numel(splitInfo.valCases), numel(splitInfo.testCases));

% 只增强训练集，防止数据泄漏
if cfg.useMirrorAugmentation
    TtrainAug = augmentTrainingTable(Ttrain, cfg);
    logBoth(fidLog, '启用镜像增强：训练集由 %d 行扩展到 %d 行。', height(Ttrain), height(TtrainAug));
    Ttrain = TtrainAug;
else
    logBoth(fidLog, '未启用镜像增强。');
end

%% ========================= 5. 准备训练张量 ===============================
featureInfo = buildFeatureInfo([Ttrain; Tval; Ttest], cfg);

trainData = tableToDataStruct(Ttrain, cfg, featureInfo, []);
valDataRaw = tableToDataStruct(Tval, cfg, featureInfo, []);
testDataRaw = tableToDataStruct(Ttest, cfg, featureInfo, []);
allDataRaw = tableToDataStruct(T0, cfg, featureInfo, []);

% 用训练集目标应变统计量做归一化
epsMu  = mean(trainData.epsMeasured_ue, 'omitnan');
epsStd = std(trainData.epsMeasured_ue, 'omitnan');
if epsStd < 1e-12
    epsStd = 1;
end

trainData.epsNorm = (trainData.epsMeasured_ue - epsMu) ./ epsStd;
valDataRaw.epsNorm = (valDataRaw.epsMeasured_ue - epsMu) ./ epsStd;
testDataRaw.epsNorm = (testDataRaw.epsMeasured_ue - epsMu) ./ epsStd;
allDataRaw.epsNorm = (allDataRaw.epsMeasured_ue - epsMu) ./ epsStd;

logBoth(fidLog, '应变归一化参数：均值 %.6g ue，标准差 %.6g ue。', epsMu, epsStd);
logBoth(fidLog, '网络输入特征维度：%d。', featureInfo.numFeatures);

featureTable = table((1:numel(featureInfo.featureNames))', featureInfo.featureNames(:), ...
    'VariableNames', {'feature_index','feature_name'});
writetable(featureTable, fullfile(dirs.tables, 'Feature_Names.csv'));

%% ========================= 6. 建立位移神经场网络 ==========================
net = buildMLP(featureInfo.numFeatures, cfg.hiddenUnits, cfg.numHiddenLayers);

logBoth(fidLog, '网络建立完成：%d 层隐藏层，每层 %d 个神经元。', ...
    cfg.numHiddenLayers, cfg.hiddenUnits);
logBoth(fidLog, '网络直接输出 w_norm，物理位移代理为 w_m = w_norm * %.3g m。', cfg.w_scale_m);

%% ========================= 7. 训练网络 ===================================
numTrain = numel(trainData.epsMeasured_ue);
numIterationsPerEpoch = max(1, ceil(numTrain / cfg.miniBatchSize));

trailingAvg = [];
trailingAvgSq = [];

bestValLoss = inf;
bestNet = net;
bestEpoch = 0;
badEpochs = 0;

history = table('Size',[cfg.maxEpochs, 12], ...
    'VariableTypes', repmat("double",1,12), ...
    'VariableNames', {'epoch','learn_rate','train_loss_total','train_loss_data', ...
                      'train_loss_gauge','train_loss_slope','train_loss_curv', ...
                      'val_loss_data','train_R2','val_R2','val_RMSE_ue','elapsed_sec'});

tStart = tic;

logBoth(fidLog, '开始训练。');
logBoth(fidLog, 'loss_data 是自动微分应变与实测应变之间的归一化 MSE。');

for epoch = 1:cfg.maxEpochs
    lr = cfg.initialLearnRate * cfg.learnRateDropFactor ^ floor((epoch-1)/cfg.learnRateDropPeriod);
    order = randperm(numTrain);

    epochTotal = 0;
    epochData  = 0;
    epochGauge = 0;
    epochSlope = 0;
    epochCurv  = 0;

    for it = 1:numIterationsPerEpoch
        b1 = (it-1)*cfg.miniBatchSize + 1;
        b2 = min(it*cfg.miniBatchSize, numTrain);
        batchIds = order(b1:b2);

        batch = makeDLBatch(sliceDataStruct(trainData, batchIds));

        [loss, gradients, parts] = dlfeval(@modelLoss, net, batch, cfg, epsMu, epsStd);

        globalIter = (epoch-1)*numIterationsPerEpoch + it;
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
            trailingAvg, trailingAvgSq, globalIter, lr, ...
            cfg.gradientDecayFactor, cfg.squaredGradientDecayFactor);

        nb = numel(batchIds);
        epochTotal = epochTotal + double(gather(extractdata(loss))) * nb;
        epochData  = epochData  + double(gather(extractdata(parts.dataLoss))) * nb;
        epochGauge = epochGauge + double(gather(extractdata(parts.gaugeLoss))) * nb;
        epochSlope = epochSlope + double(gather(extractdata(parts.slopeLoss))) * nb;
        epochCurv  = epochCurv  + double(gather(extractdata(parts.curvLoss))) * nb;
    end

    epochTotal = epochTotal / numTrain;
    epochData  = epochData  / numTrain;
    epochGauge = epochGauge / numTrain;
    epochSlope = epochSlope / numTrain;
    epochCurv  = epochCurv  / numTrain;

    % 评估：为了节省显存，按批评估
    predTrain = predictDatasetStrain(net, trainData, cfg, 512);
    predVal   = predictDatasetStrain(net, valDataRaw, cfg, 512);

    trainR2 = calcR2(trainData.epsMeasured_ue(:), predTrain.epsPred_ue(:));
    valR2   = calcR2(valDataRaw.epsMeasured_ue(:), predVal.epsPred_ue(:));
    valRMSE = sqrt(mean((predVal.epsPred_ue(:) - valDataRaw.epsMeasured_ue(:)).^2, 'omitnan'));
    valLossData = mean(((predVal.epsPred_ue(:) - epsMu)./epsStd - valDataRaw.epsNorm(:)).^2, 'omitnan');

    history.epoch(epoch) = epoch;
    history.learn_rate(epoch) = lr;
    history.train_loss_total(epoch) = epochTotal;
    history.train_loss_data(epoch) = epochData;
    history.train_loss_gauge(epoch) = epochGauge;
    history.train_loss_slope(epoch) = epochSlope;
    history.train_loss_curv(epoch) = epochCurv;
    history.val_loss_data(epoch) = valLossData;
    history.train_R2(epoch) = trainR2;
    history.val_R2(epoch) = valR2;
    history.val_RMSE_ue(epoch) = valRMSE;
    history.elapsed_sec(epoch) = toc(tStart);

    if valLossData < bestValLoss
        bestValLoss = valLossData;
        bestNet = net;
        bestEpoch = epoch;
        badEpochs = 0;
    else
        badEpochs = badEpochs + 1;
    end

    if mod(epoch, cfg.printEveryEpoch) == 0 || epoch == 1
        logBoth(fidLog, ['Epoch %4d | lr %.3g | total %.6g | data %.6g | ', ...
                         'gauge %.3g | slope %.3g | curv %.3g | Train R2 %.4f | Val R2 %.4f | Val RMSE %.4g ue | Best %d'], ...
            epoch, lr, epochTotal, epochData, epochGauge, epochSlope, epochCurv, ...
            trainR2, valR2, valRMSE, bestEpoch);
    end

    if badEpochs >= cfg.earlyStopPatience
        logBoth(fidLog, '触发早停：连续 %d 个 epoch 验证集未改善。', cfg.earlyStopPatience);
        break;
    end
end

history = history(1:epoch, :);
net = bestNet;

logBoth(fidLog, '训练结束。最优 Epoch = %d，最优验证 data loss = %.6g。', bestEpoch, bestValLoss);
writetable(history, fullfile(dirs.tables, 'Training_History.csv'));

%% ========================= 8. 总体评估 ===================================
predTrain = predictDatasetStrain(net, tableToDataStruct(T0(trainIdx0,:), cfg, featureInfo, epsMu, epsStd), cfg, 512);
predVal   = predictDatasetStrain(net, valDataRaw, cfg, 512);
predTest  = predictDatasetStrain(net, testDataRaw, cfg, 512);
predAll   = predictDatasetStrain(net, allDataRaw, cfg, 512);

metricsTrain = calcMetrics(T0.strain_ue(trainIdx0), predTrain.epsPred_ue);
metricsVal   = calcMetrics(T0.strain_ue(valIdx0),   predVal.epsPred_ue);
metricsTest  = calcMetrics(T0.strain_ue(testIdx0),  predTest.epsPred_ue);
metricsAll   = calcMetrics(T0.strain_ue,            predAll.epsPred_ue);

MetricsOverall = table( ...
    ["train";"validation";"test";"all"], ...
    [metricsTrain.R2; metricsVal.R2; metricsTest.R2; metricsAll.R2], ...
    [metricsTrain.RMSE; metricsVal.RMSE; metricsTest.RMSE; metricsAll.RMSE], ...
    [metricsTrain.MAE; metricsVal.MAE; metricsTest.MAE; metricsAll.MAE], ...
    [metricsTrain.Bias; metricsVal.Bias; metricsTest.Bias; metricsAll.Bias], ...
    'VariableNames', {'split','R2','RMSE_ue','MAE_ue','Bias_ue'});
writetable(MetricsOverall, fullfile(dirs.tables, 'Overall_Metrics.csv'));

logBoth(fidLog, '总体指标：');
logBoth(fidLog, '  Train      R2=%.4f, RMSE=%.6g ue', metricsTrain.R2, metricsTrain.RMSE);
logBoth(fidLog, '  Validation R2=%.4f, RMSE=%.6g ue', metricsVal.R2, metricsVal.RMSE);
logBoth(fidLog, '  Test       R2=%.4f, RMSE=%.6g ue', metricsTest.R2, metricsTest.RMSE);
logBoth(fidLog, '  All        R2=%.4f, RMSE=%.6g ue', metricsAll.R2, metricsAll.RMSE);

% 点预测表
PointPred = T0(:, ["case_id","sensor_id","x_cm","y_cm","layer","direction", ...
    "load_type","load_kg","p1_x","p1_y","p2_x","p2_y", ...
    "delta_lambda_resp_nm","strain_ue"]);
PointPred.pred_strain_ue = predAll.epsPred_ue(:);
PointPred.pred_dlambda_nm = PointPred.pred_strain_ue .* cfg.k_eps_nm_per_ue;
PointPred.pred_w_um = predAll.wPred_m(:) .* 1e6;
PointPred.pred_eps_x_ue = predAll.epsX_ue(:);
PointPred.pred_eps_y_ue = predAll.epsY_ue(:);
PointPred.residual_strain_ue = PointPred.pred_strain_ue - PointPred.strain_ue;
PointPred.split = strings(height(T0),1);
PointPred.split(trainIdx0) = "train";
PointPred.split(valIdx0)   = "validation";
PointPred.split(testIdx0)  = "test";
writetable(PointPred, fullfile(dirs.tables, 'Pointwise_Predictions.csv'));

% 分工况指标
caseList = unique(T0.case_id, 'stable');
CaseMetrics = table();
for i = 1:numel(caseList)
    cid = caseList(i);
    mask = T0.case_id == cid;
    mt = calcMetrics(T0.strain_ue(mask), predAll.epsPred_ue(mask));
    one = table(cid, unique(T0.load_type(mask),'stable'), unique(T0.load_kg(mask),'stable'), ...
        sum(mask), mt.R2, mt.RMSE, mt.MAE, mt.Bias, ...
        'VariableNames', {'case_id','load_type','load_kg','n_points','R2','RMSE_ue','MAE_ue','Bias_ue'});
    CaseMetrics = [CaseMetrics; one]; %#ok<AGROW>
end
writetable(CaseMetrics, fullfile(dirs.tables, 'Case_Metrics.csv'));

%% ========================= 9. 绘图：训练与评估 ===========================
plotTrainingHistory(history, dirs.figs);
plotParity(PointPred, dirs.figs);
plotResiduals(PointPred, dirs.figs);
plotCaseMetrics(CaseMetrics, dirs.figs);

%% ========================= 10. 连续场预测 ================================
logBoth(fidLog, '开始输出连续位移场和自动微分应变场。');

FieldSummary = table();
fieldCounter = 0;

for ic = 1:numel(cfg.fieldCaseList)
    cid = cfg.fieldCaseList(ic);
    if ~any(T0.case_id == cid)
        logBoth(fidLog, '跳过连续场输出：工况不存在：%s', cid);
        continue;
    end

    refRows = T0(T0.case_id == cid, :);
    refRow = refRows(1,:);

    for il = 1:numel(cfg.fieldLayers)
        lay = cfg.fieldLayers(il);
        if ~any(T0.case_id == cid & T0.layer == lay)
            logBoth(fidLog, '跳过连续场输出：工况 %s 无层 %s 数据。', cid, lay);
            continue;
        end

        fieldCounter = fieldCounter + 1;

        G = makeGridTable(refRow, lay, cfg);
        Gdata = tableToDataStruct(G, cfg, featureInfo, epsMu, epsStd);
        Pgrid = predictDatasetStrain(net, Gdata, cfg, 512);

        nx = cfg.gridNX;
        ny = cfg.gridNY;
        Xg = reshape(G.x_cm, [ny, nx]);
        Yg = reshape(G.y_cm, [ny, nx]);
        W_um = reshape(Pgrid.wPred_m .* 1e6, [ny, nx]);
        epsX = reshape(Pgrid.epsX_ue, [ny, nx]);
        epsY = reshape(Pgrid.epsY_ue, [ny, nx]);
        dLambdaX = epsX .* cfg.k_eps_nm_per_ue;
        dLambdaY = epsY .* cfg.k_eps_nm_per_ue;

        FieldTable = table(G.x_cm, G.y_cm, repmat(cid,height(G),1), repmat(lay,height(G),1), ...
            Pgrid.wPred_m(:).*1e6, Pgrid.epsX_ue(:), Pgrid.epsY_ue(:), ...
            dLambdaX(:), dLambdaY(:), ...
            'VariableNames', {'x_cm','y_cm','case_id','layer','w_proxy_um', ...
                              'eps_x_ue','eps_y_ue','dlambda_x_nm','dlambda_y_nm'});

        safeName = sprintf('Field_%02d_%s_%s', fieldCounter, englishCaseFileName(refRow), char(lay));
        csvName = fullfile(dirs.tables, safeName + ".csv");
        writetable(FieldTable, csvName);

        englishTitle = englishCaseTitle(refRow, lay);

        plotFieldContour(Xg, Yg, W_um, ...
            'Displacement Neural Field', 'X (cm)', 'Y (cm)', ...
            'Displacement Proxy w (um)', englishTitle, ...
            fullfile(dirs.figs, safeName + "_W_DisplacementProxy.png"));

        plotFieldContour(Xg, Yg, epsX, ...
            'Autodiff Longitudinal Strain Field', 'X (cm)', 'Y (cm)', ...
            'Longitudinal Strain epsilon_x (microstrain)', englishTitle, ...
            fullfile(dirs.figs, safeName + "_EpsX_LongitudinalStrain.png"));

        plotFieldContour(Xg, Yg, epsY, ...
            'Autodiff Transverse Strain Field', 'X (cm)', 'Y (cm)', ...
            'Transverse Strain epsilon_y (microstrain)', englishTitle, ...
            fullfile(dirs.figs, safeName + "_EpsY_TransverseStrain.png"));

        plotFieldContour(Xg, Yg, dLambdaX, ...
            'Predicted Longitudinal Wavelength Shift Field', 'X (cm)', 'Y (cm)', ...
            'Longitudinal Wavelength Shift (nm)', englishTitle, ...
            fullfile(dirs.figs, safeName + "_DlambdaX.png"));

        one = table(cid, lay, max(abs(W_um(:))), max(abs(epsX(:))), max(abs(epsY(:))), ...
            max(abs(dLambdaX(:))), max(abs(dLambdaY(:))), csvName, ...
            'VariableNames', {'case_id','layer','max_abs_w_um','max_abs_eps_x_ue', ...
                              'max_abs_eps_y_ue','max_abs_dlambda_x_nm','max_abs_dlambda_y_nm','csv_path'});
        FieldSummary = [FieldSummary; one]; %#ok<AGROW>

        logBoth(fidLog, '已输出连续场：%s | %s', cid, lay);
    end
end

writetable(FieldSummary, fullfile(dirs.tables, 'Field_Summary.csv'));

%% ========================= 11. 保存模型与报告 =============================
modelFile = fullfile(dirs.models, 'Step15_Asphalt_LoadBalanced_DNF_Model.mat');
save(modelFile, 'net', 'bestNet', 'cfg', 'featureInfo', ...
    'epsMu', 'epsStd', 'MetricsOverall', 'CaseMetrics', 'history', 'splitInfo');

reportFile = fullfile(dirs.logs, 'Analysis_Report_CN.txt');
writeChineseReport(reportFile, cfg, MetricsOverall, CaseMetrics, FieldSummary, ...
    modelFile, bestEpoch, bestValLoss);

logBoth(fidLog, '模型已保存：%s', modelFile);
logBoth(fidLog, '中文分析报告已保存：%s', reportFile);
logBoth(fidLog, '全部完成。');
fclose(fidLog);

disp(' ');
disp('==================== Step15 完成 ====================');
disp(['输出文件夹：', cfg.outBase]);
disp(['训练日志：', logFile]);
disp(['分析报告：', reportFile]);
disp(['模型文件：', modelFile]);

%% =========================================================================
%                              本地函数
% =========================================================================

function makeDir(p)
    if ~exist(p, 'dir')
        mkdir(p);
    end
end

function logBoth(fid, fmt, varargin)
    msg = sprintf(fmt, varargin{:});
    fprintf('%s\n', msg);
    fprintf(fid, '%s\n', msg);
end

function [trainIdx, valIdx, testIdx, splitInfo] = splitDataByMode(T, cfg)
    n = height(T);
    trainIdx = false(n,1);
    valIdx = false(n,1);
    testIdx = false(n,1);

    switch lower(cfg.splitMode)
        case 'case'
            cases = unique(T.case_id, 'stable');
            nc = numel(cases);
            order = randperm(nc);

            nTest = max(1, round(cfg.testRatio * nc));
            nVal  = max(1, round(cfg.valRatio * nc));
            nTrain = nc - nVal - nTest;

            if nTrain < 1
                nTrain = max(1, nc-2);
                nVal = 1;
                nTest = nc - nTrain - nVal;
            end

            testCases = cases(order(1:nTest));
            valCases  = cases(order(nTest+1:nTest+nVal));
            trainCases = setdiff(cases, [testCases; valCases], 'stable');

            trainIdx = ismember(T.case_id, trainCases);
            valIdx   = ismember(T.case_id, valCases);
            testIdx  = ismember(T.case_id, testCases);

        case 'point'
            order = randperm(n);
            nTest = max(1, round(cfg.testRatio * n));
            nVal  = max(1, round(cfg.valRatio * n));

            testIdx(order(1:nTest)) = true;
            valIdx(order(nTest+1:nTest+nVal)) = true;
            trainIdx = ~(valIdx | testIdx);

            trainCases = unique(T.case_id(trainIdx), 'stable');
            valCases   = unique(T.case_id(valIdx), 'stable');
            testCases  = unique(T.case_id(testIdx), 'stable');

        otherwise
            error('cfg.splitMode 只能是 case 或 point。');
    end

    splitInfo = struct();
    splitInfo.trainCases = trainCases;
    splitInfo.valCases = valCases;
    splitInfo.testCases = testCases;
end

function Tout = augmentTrainingTable(Tin, cfg)
    Tout = Tin;
    L = cfg.plateLength_cm;
    W = cfg.plateWidth_cm;

    augList = {};

    if cfg.augmentMirrorX
        Tx = Tin;
        Tx.x_cm = L - Tx.x_cm;
        Tx.p1_x = L - Tx.p1_x;
        m2 = isfinite(Tx.p2_x);
        Tx.p2_x(m2) = L - Tx.p2_x(m2);
        Tx.case_id = Tx.case_id + "_aug_mirrorX";
        augList{end+1} = Tx; %#ok<AGROW>
    end

    if cfg.augmentMirrorY
        Ty = Tin;
        Ty.y_cm = W - Ty.y_cm;
        Ty.p1_y = W - Ty.p1_y;
        m2 = isfinite(Ty.p2_y);
        Ty.p2_y(m2) = W - Ty.p2_y(m2);
        Ty.case_id = Ty.case_id + "_aug_mirrorY";
        augList{end+1} = Ty; %#ok<AGROW>
    end

    if cfg.augmentMirrorX && cfg.augmentMirrorY
        Txy = Tin;
        Txy.x_cm = L - Txy.x_cm;
        Txy.y_cm = W - Txy.y_cm;
        Txy.p1_x = L - Txy.p1_x;
        Txy.p1_y = W - Txy.p1_y;
        m2x = isfinite(Txy.p2_x);
        m2y = isfinite(Txy.p2_y);
        Txy.p2_x(m2x) = L - Txy.p2_x(m2x);
        Txy.p2_y(m2y) = W - Txy.p2_y(m2y);
        Txy.case_id = Txy.case_id + "_aug_mirrorXY";
        augList{end+1} = Txy; %#ok<AGROW>
    end

    for i = 1:numel(augList)
        Tout = [Tout; augList{i}]; %#ok<AGROW>
    end
end

function featureInfo = buildFeatureInfo(T, cfg)
    featureInfo = struct();
    featureInfo.loadTypes = unique(T.load_type, 'stable');
    featureInfo.fourierFreqsXY = cfg.fourierFreqsXY;
    featureInfo.fourierFreqsRel = cfg.fourierFreqsRel;

    % 构造一次虚拟数据以获得特征名和维度
    dummy = tableToDataStruct(T(1:min(2,height(T)),:), cfg, featureInfo, []);
    featureInfo.featureNames = dummy.featureNames;
    featureInfo.numFeatures = numel(dummy.featureNames);
end

function data = tableToDataStruct(T, cfg, featureInfo, epsMu, epsStd)
    L = cfg.plateLength_cm;
    W = cfg.plateWidth_cm;
    diagLen = sqrt(L^2 + W^2);

    x = double(T.x_cm);
    y = double(T.y_cm);

    p1x = double(T.p1_x);
    p1y = double(T.p1_y);
    p2x = double(T.p2_x);
    p2y = double(T.p2_y);

    hasP2 = isfinite(p2x) & isfinite(p2y);
    p2xFill = p2x;
    p2yFill = p2y;
    p2xFill(~hasP2) = p1x(~hasP2);
    p2yFill(~hasP2) = p1y(~hasP2);

    maxLoad = 80;  % 本实验最大中心加载约 80kg，固定尺度避免数据增强后变化
    loadN = double(T.load_kg) ./ maxLoad;

    xN = x ./ L;
    yN = y ./ W;
    p1xN = p1x ./ L;
    p1yN = p1y ./ W;
    p2xN = p2xFill ./ L;
    p2yN = p2yFill ./ W;

    layerSoil = double(T.layer == "Soil");
    dirLong = double(T.direction == "纵");

    epsMeasured = double(T.strain_ue);

    if nargin >= 5 && ~isempty(epsMu)
        epsNorm = (epsMeasured - epsMu) ./ epsStd;
    else
        epsNorm = nan(size(epsMeasured));
    end

    onehotLoadType = zeros(numel(featureInfo.loadTypes), height(T));
    for k = 1:numel(featureInfo.loadTypes)
        onehotLoadType(k,:) = double(T.load_type == featureInfo.loadTypes(k))';
    end

    data = struct();
    data.xN = single(xN(:)');
    data.yN = single(yN(:)');
    data.loadN = single(loadN(:)');
    data.p1xN = single(p1xN(:)');
    data.p1yN = single(p1yN(:)');
    data.p2xN = single(p2xN(:)');
    data.p2yN = single(p2yN(:)');
    data.hasP2 = single(hasP2(:)');
    data.layerSoil = single(layerSoil(:)');
    data.dirLong = single(dirLong(:)');
    data.epsMeasured_ue = single(epsMeasured(:)');
    data.epsNorm = single(epsNorm(:)');

    if cfg.useLoadWeightedLoss
        lw = 1 ./ ((loadN(:)' + cfg.loadWeightOffset) .^ cfg.loadWeightPower);
        lw = min(lw, cfg.maxLoadWeight);
        lw = lw ./ mean(lw, 'omitnan');  % 保持平均权重约为 1，避免改变 loss 尺度
    else
        lw = ones(1, height(T));
    end
    data.loadWeight = single(lw);

    data.onehotLoadType = single(onehotLoadType);

    % 常量
    data.L_cm = L;
    data.W_cm = W;
    data.diag_cm = diagLen;

    % 特征名
    data.featureNames = makeFeatureNames(featureInfo);
end

function names = makeFeatureNames(featureInfo)
    names = strings(0,1);

    base = ["x_norm","y_norm","load_norm", ...
            "p1_x_norm","p1_y_norm","p2_x_norm","p2_y_norm","has_second_load", ...
            "layer_soil", ...
            "dx1_norm","dy1_norm","dx2_norm","dy2_norm", ...
            "dist_load1_norm","dist_load2_norm", ...
            "dist_left_norm","dist_right_norm","dist_front_norm","dist_back_norm", ...
            "load_x_dx1","load_x_dy1","load_x_dist1","load_x_layer"];
    names = [names; base(:)];

    for f = featureInfo.fourierFreqsXY
        names = [names; ...
            sprintf("sin_x_f%d", f); sprintf("cos_x_f%d", f); ...
            sprintf("sin_y_f%d", f); sprintf("cos_y_f%d", f)]; %#ok<AGROW>
    end

    for f = featureInfo.fourierFreqsRel
        names = [names; ...
            sprintf("sin_dx1_f%d", f); sprintf("cos_dx1_f%d", f); ...
            sprintf("sin_dy1_f%d", f); sprintf("cos_dy1_f%d", f); ...
            sprintf("sin_dx2_f%d", f); sprintf("cos_dx2_f%d", f); ...
            sprintf("sin_dy2_f%d", f); sprintf("cos_dy2_f%d", f)]; %#ok<AGROW>
    end

    for k = 1:numel(featureInfo.loadTypes)
        nm = "load_type_" + string(matlab.lang.makeValidName(char(featureInfo.loadTypes(k))));
        names = [names; nm]; %#ok<AGROW>
    end
end

function batch = sliceDataStruct(data, idx)
    batch = data;
    batch.xN = data.xN(idx);
    batch.yN = data.yN(idx);
    batch.loadN = data.loadN(idx);
    batch.p1xN = data.p1xN(idx);
    batch.p1yN = data.p1yN(idx);
    batch.p2xN = data.p2xN(idx);
    batch.p2yN = data.p2yN(idx);
    batch.hasP2 = data.hasP2(idx);
    batch.layerSoil = data.layerSoil(idx);
    batch.dirLong = data.dirLong(idx);
    batch.epsMeasured_ue = data.epsMeasured_ue(idx);
    batch.epsNorm = data.epsNorm(idx);
    batch.loadWeight = data.loadWeight(idx);
    batch.onehotLoadType = data.onehotLoadType(:, idx);
end

function batchDL = makeDLBatch(batch)
    % 关键修复：
    % MATLAB 要对输入坐标 xN/yN 求导时，xN/yN 必须在进入 dlfeval 之前
    % 就已经是 dlarray。若在 dlfeval 内部由普通数值重新包装，部分版本
    % 会认为它不是 tracked dlarray，从而导致 dlgradient 报错。
    batchDL = batch;
    batchDL.xN = dlarray(single(batch.xN), 'CB');
    batchDL.yN = dlarray(single(batch.yN), 'CB');
    batchDL.loadN = dlarray(single(batch.loadN), 'CB');
    batchDL.p1xN = dlarray(single(batch.p1xN), 'CB');
    batchDL.p1yN = dlarray(single(batch.p1yN), 'CB');
    batchDL.p2xN = dlarray(single(batch.p2xN), 'CB');
    batchDL.p2yN = dlarray(single(batch.p2yN), 'CB');
    batchDL.hasP2 = dlarray(single(batch.hasP2), 'CB');
    batchDL.layerSoil = dlarray(single(batch.layerSoil), 'CB');
    batchDL.dirLong = dlarray(single(batch.dirLong), 'CB');
    batchDL.epsNorm = dlarray(single(batch.epsNorm), 'CB');
    batchDL.loadWeight = dlarray(single(batch.loadWeight), 'CB');
    batchDL.onehotLoadType = dlarray(single(batch.onehotLoadType), 'CB');
end

function net = buildMLP(numFeatures, hiddenUnits, numHiddenLayers)
    layers = [
        featureInputLayer(numFeatures, 'Normalization','none', 'Name','input')
    ];

    for i = 1:numHiddenLayers
        layers = [
            layers
            fullyConnectedLayer(hiddenUnits, 'Name', sprintf('fc%d', i))
            tanhLayer('Name', sprintf('tanh%d', i))
        ]; %#ok<AGROW>
    end

    layers = [
        layers
        fullyConnectedLayer(1, 'Name','w_norm_output')
    ];

    net = dlnetwork(layerGraph(layers));
end

function [loss, gradients, parts] = modelLoss(net, batch, cfg, epsMu, epsStd)
    pred = forwardWithAutodiffStrain(net, batch, cfg);

    epsTrueNorm = batch.epsNorm;
    epsPredNorm = (pred.epsPred_ue - epsMu) ./ epsStd;

    err2 = (epsPredNorm - epsTrueNorm).^2;
    dataLoss = mean(batch.loadWeight .* err2, 'all');

    % gauge loss：固定 w 的整体零点，避免应变监督下的刚体平移不唯一
    gaugeLoss = mean(pred.w_norm, 'all').^2;

    % slope loss：抑制整体线性倾斜不唯一
    slopeLoss = mean(pred.wxN, 'all').^2 + mean(pred.wyN, 'all').^2;

    % 曲率能量正则：非常弱，只用于抑制局部振荡
    curvLoss = mean(pred.wxx_m.^2 + pred.wyy_m.^2, 'all');

    l2Loss = l2Regularization(net);

    loss = dataLoss ...
        + cfg.lambdaGauge   * gaugeLoss ...
        + cfg.lambdaSlope   * slopeLoss ...
        + cfg.lambdaCurvReg * curvLoss ...
        + cfg.lambdaL2      * l2Loss;

    gradients = dlgradient(loss, net.Learnables);

    parts = struct();
    parts.dataLoss = dataLoss;
    parts.gaugeLoss = gaugeLoss;
    parts.slopeLoss = slopeLoss;
    parts.curvLoss = curvLoss;
    parts.l2Loss = l2Loss;
end

function pred = forwardWithAutodiffStrain(net, batch, cfg)
    xN = batch.xN;
    yN = batch.yN;

    loadN = batch.loadN;
    p1xN = batch.p1xN;
    p1yN = batch.p1yN;
    p2xN = batch.p2xN;
    p2yN = batch.p2yN;
    hasP2 = batch.hasP2;
    layerSoil = batch.layerSoil;
    dirLong = batch.dirLong;
    onehotLoadType = batch.onehotLoadType;

    Xfeat = constructFeaturesDL(xN, yN, loadN, p1xN, p1yN, p2xN, p2yN, ...
        hasP2, layerSoil, onehotLoadType, cfg);

    w_norm = forward(net, Xfeat);
    w_m = cfg.w_scale_m .* w_norm;

    % 自动微分：对归一化坐标求导
    wSum = sum(w_m, 'all');
    wxN = dlgradient(wSum, xN, 'EnableHigherDerivatives', true, 'RetainData', true);
    wyN = dlgradient(wSum, yN, 'EnableHigherDerivatives', true, 'RetainData', true);

    wxxN = dlgradient(sum(wxN, 'all'), xN, 'EnableHigherDerivatives', true, 'RetainData', true);
    wyyN = dlgradient(sum(wyN, 'all'), yN, 'EnableHigherDerivatives', true);

    Lm = cfg.plateLength_cm / 100;
    Wm = cfg.plateWidth_cm / 100;

    % 由归一化坐标导数换算到物理坐标导数
    wxx_m = wxxN ./ (Lm^2);
    wyy_m = wyyN ./ (Wm^2);

    zEff = cfg.z_asphalt_m .* (1 - layerSoil) + cfg.z_soil_m .* layerSoil;

    epsX_ue = -zEff .* wxx_m .* 1e6;
    epsY_ue = -zEff .* wyy_m .* 1e6;

    epsPred_ue = dirLong .* epsX_ue + (1 - dirLong) .* epsY_ue;

    pred = struct();
    pred.w_norm = w_norm;
    pred.w_m = w_m;
    pred.wxN = wxN;
    pred.wyN = wyN;
    pred.wxx_m = wxx_m;
    pred.wyy_m = wyy_m;
    pred.epsX_ue = epsX_ue;
    pred.epsY_ue = epsY_ue;
    pred.epsPred_ue = epsPred_ue;
end

function Xfeat = constructFeaturesDL(xN, yN, loadN, p1xN, p1yN, p2xN, p2yN, ...
    hasP2, layerSoil, onehotLoadType, cfg)

    dx1 = xN - p1xN;
    dy1 = yN - p1yN;
    dx2 = xN - p2xN;
    dy2 = yN - p2yN;

    d1 = sqrt(dx1.^2 + dy1.^2 + 1e-8);
    d2 = sqrt(dx2.^2 + dy2.^2 + 1e-8);

    dLeft  = xN;
    dRight = 1 - xN;
    dFront = yN;
    dBack  = 1 - yN;

    Xfeat = [
        xN
        yN
        loadN
        p1xN
        p1yN
        p2xN
        p2yN
        hasP2
        layerSoil
        dx1
        dy1
        dx2
        dy2
        d1
        d2
        dLeft
        dRight
        dFront
        dBack
        loadN .* dx1
        loadN .* dy1
        loadN .* d1
        loadN .* layerSoil
    ];

    for f = cfg.fourierFreqsXY
        Xfeat = [
            Xfeat
            sin(2*pi*f*xN)
            cos(2*pi*f*xN)
            sin(2*pi*f*yN)
            cos(2*pi*f*yN)
        ]; %#ok<AGROW>
    end

    for f = cfg.fourierFreqsRel
        Xfeat = [
            Xfeat
            sin(2*pi*f*dx1)
            cos(2*pi*f*dx1)
            sin(2*pi*f*dy1)
            cos(2*pi*f*dy1)
            sin(2*pi*f*dx2)
            cos(2*pi*f*dx2)
            sin(2*pi*f*dy2)
            cos(2*pi*f*dy2)
        ]; %#ok<AGROW>
    end

    Xfeat = [Xfeat; onehotLoadType];
end

function l2 = l2Regularization(net)
    l2 = dlarray(0);
    L = net.Learnables;
    for i = 1:size(L,1)
        pname = string(L.Parameter(i));
        if contains(pname, "Weights")
            W = L.Value{i};
            l2 = l2 + sum(W.^2, 'all');
        end
    end
end

function predOut = predictDatasetStrain(net, data, cfg, batchSize)
    n = numel(data.epsMeasured_ue);
    epsPred = nan(n,1);
    epsX = nan(n,1);
    epsY = nan(n,1);
    wPred = nan(n,1);

    for s = 1:batchSize:n
        e = min(s+batchSize-1, n);
        idx = s:e;
        batch = makeDLBatch(sliceDataStruct(data, idx));
        p = dlfeval(@predictBatchDL, net, batch, cfg);

        epsPred(idx) = double(gather(extractdata(p.epsPred_ue(:))));
        epsX(idx) = double(gather(extractdata(p.epsX_ue(:))));
        epsY(idx) = double(gather(extractdata(p.epsY_ue(:))));
        wPred(idx) = double(gather(extractdata(p.w_m(:))));
    end

    predOut = struct();
    predOut.epsPred_ue = epsPred;
    predOut.epsX_ue = epsX;
    predOut.epsY_ue = epsY;
    predOut.wPred_m = wPred;
end

function p = predictBatchDL(net, batch, cfg)
    p = forwardWithAutodiffStrain(net, batch, cfg);
end

function r2 = calcR2(yTrue, yPred)
    yTrue = yTrue(:);
    yPred = yPred(:);
    mask = isfinite(yTrue) & isfinite(yPred);
    yTrue = yTrue(mask);
    yPred = yPred(mask);
    denom = sum((yTrue - mean(yTrue)).^2);
    if numel(yTrue) < 2 || denom < 1e-12
        r2 = NaN;
    else
        r2 = 1 - sum((yTrue - yPred).^2) / denom;
    end
end

function m = calcMetrics(yTrue, yPred)
    yTrue = yTrue(:);
    yPred = yPred(:);
    mask = isfinite(yTrue) & isfinite(yPred);
    yTrue = yTrue(mask);
    yPred = yPred(mask);
    res = yPred - yTrue;

    m = struct();
    m.R2 = calcR2(yTrue, yPred);
    m.RMSE = sqrt(mean(res.^2, 'omitnan'));
    m.MAE = mean(abs(res), 'omitnan');
    m.Bias = mean(res, 'omitnan');
end

function plotTrainingHistory(history, figDir)
    fig = figure('Visible','off','Position',[100 100 900 520]);
    semilogy(history.epoch, history.train_loss_data, 'LineWidth', 1.8); hold on;
    semilogy(history.epoch, history.val_loss_data, 'LineWidth', 1.8);
    xlabel('Epoch');
    ylabel('Normalized Strain MSE');
    title('Training and Validation Strain Loss');
    legend('Train Data Loss','Validation Data Loss','Location','northeast');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Training_Validation_Loss.png'));
    close(fig);

    fig = figure('Visible','off','Position',[100 100 900 520]);
    plot(history.epoch, history.train_R2, 'LineWidth', 1.8); hold on;
    plot(history.epoch, history.val_R2, 'LineWidth', 1.8);
    xlabel('Epoch');
    ylabel('R^2');
    title('Training and Validation R^2');
    legend('Train','Validation','Location','southeast');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Training_Validation_R2.png'));
    close(fig);

    fig = figure('Visible','off','Position',[100 100 900 520]);
    semilogy(history.epoch, history.train_loss_gauge, 'LineWidth', 1.5); hold on;
    semilogy(history.epoch, history.train_loss_slope, 'LineWidth', 1.5);
    semilogy(history.epoch, history.train_loss_curv, 'LineWidth', 1.5);
    xlabel('Epoch');
    ylabel('Regularization Loss');
    title('Regularization Terms');
    legend('Gauge','Slope','Curvature','Location','best');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Regularization_Terms.png'));
    close(fig);
end

function plotParity(PointPred, figDir)
    fig = figure('Visible','off','Position',[100 100 720 640]);
    masks = {PointPred.split=="train", PointPred.split=="validation", PointPred.split=="test"};
    labels = {'Train','Validation','Test'};

    hold on;
    for i = 1:3
        m = masks{i};
        scatter(PointPred.strain_ue(m), PointPred.pred_strain_ue(m), 30, 'filled', 'MarkerFaceAlpha', 0.60);
    end

    allV = [PointPred.strain_ue; PointPred.pred_strain_ue];
    mn = min(allV, [], 'omitnan');
    mx = max(allV, [], 'omitnan');
    plot([mn mx], [mn mx], 'k--', 'LineWidth', 1.4);

    xlabel('Measured Strain (microstrain)');
    ylabel('Predicted Strain from Autodiff (microstrain)');
    title('Measured vs Predicted Strain');
    legend([labels, {'1:1 Line'}], 'Location','best');
    grid on; axis equal;
    xlim([mn mx]); ylim([mn mx]);
    saveas(fig, fullfile(figDir, 'Fig_Parity_Measured_vs_Predicted_Strain.png'));
    close(fig);
end

function plotResiduals(PointPred, figDir)
    res = PointPred.residual_strain_ue;

    fig = figure('Visible','off','Position',[100 100 850 520]);
    histogram(res, 30);
    xlabel('Residual Strain (microstrain)');
    ylabel('Count');
    title('Residual Distribution');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Residual_Histogram.png'));
    close(fig);

    fig = figure('Visible','off','Position',[100 100 850 520]);
    scatter(PointPred.pred_strain_ue, res, 28, 'filled', 'MarkerFaceAlpha', 0.60);
    yline(0, 'k--', 'LineWidth', 1.2);
    xlabel('Predicted Strain (microstrain)');
    ylabel('Residual Strain (microstrain)');
    title('Residuals vs Predicted Strain');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Residual_vs_Predicted.png'));
    close(fig);
end

function plotCaseMetrics(CaseMetrics, figDir)
    [~, order] = sort(CaseMetrics.R2, 'ascend', 'MissingPlacement','last');
    C = CaseMetrics(order, :);

    labels = strings(height(C),1);
    for i = 1:height(C)
        labels(i) = sprintf('%02d', i);
    end

    fig = figure('Visible','off','Position',[100 100 1050 520]);
    bar(C.R2);
    xlabel('Case Index');
    ylabel('R^2');
    title('Case-wise R^2');
    xticks(1:height(C));
    xticklabels(labels);
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Casewise_R2.png'));
    close(fig);

    fig = figure('Visible','off','Position',[100 100 1050 520]);
    bar(C.RMSE_ue);
    xlabel('Case Index');
    ylabel('RMSE (microstrain)');
    title('Case-wise RMSE');
    xticks(1:height(C));
    xticklabels(labels);
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Casewise_RMSE.png'));
    close(fig);

    mapTable = table((1:height(C))', C.case_id, C.load_type, C.load_kg, C.R2, C.RMSE_ue, ...
        'VariableNames', {'case_index_in_fig','case_id','load_type','load_kg','R2','RMSE_ue'});
    writetable(mapTable, fullfile(fileparts(figDir), 'Tables', 'Case_Index_Map_For_Figures.csv'));
end

function G = makeGridTable(refRow, layerName, cfg)
    xVec = linspace(0, cfg.plateLength_cm, cfg.gridNX);
    yVec = linspace(0, cfg.plateWidth_cm, cfg.gridNY);
    [Xg, Yg] = meshgrid(xVec, yVec);
    n = numel(Xg);

    G = table();
    G.case_id = repmat(refRow.case_id, n, 1);
    G.sensor_id = repmat("grid", n, 1);
    G.x_cm = Xg(:);
    G.y_cm = Yg(:);
    G.layer = repmat(string(layerName), n, 1);
    G.direction = repmat("纵", n, 1);
    G.channel = repmat("grid", n, 1);
    G.interrogator = repmat("grid", n, 1);
    G.delta_lambda_resp_nm = zeros(n,1);
    G.delta_lambda_peak_nm = zeros(n,1);
    G.raw_steady_nm = zeros(n,1);
    G.raw_peak_nm = zeros(n,1);
    G.baseline_nm = zeros(n,1);
    G.load_type = repmat(refRow.load_type, n, 1);
    G.load_kg = repmat(refRow.load_kg, n, 1);
    G.p1_x = repmat(refRow.p1_x, n, 1);
    G.p1_y = repmat(refRow.p1_y, n, 1);
    G.p2_x = repmat(refRow.p2_x, n, 1);
    G.p2_y = repmat(refRow.p2_y, n, 1);
    G.strain_ue = zeros(n,1);
end

function plotFieldContour(Xg, Yg, Zg, figTitle, xLabel, yLabel, cbarLabel, subTitle, outPath)
    fig = figure('Visible','off','Position',[100 100 900 650]);
    contourf(Xg, Yg, Zg, 30, 'LineColor','none');
    cb = colorbar;
    ylabel(cb, cbarLabel);
    xlabel(xLabel);
    ylabel(yLabel);
    title({figTitle; subTitle}, 'Interpreter','none');
    axis equal tight;
    grid on;
    saveas(fig, outPath);
    close(fig);
end

function s = englishCaseTitle(refRow, layerName)
    lt = englishLoadType(refRow.load_type);
    if isfinite(refRow.p2_x) && isfinite(refRow.p2_y)
        s = sprintf('P = %.0f kg | %s | Layer = %s | P1 = (%.1f, %.1f) cm | P2 = (%.1f, %.1f) cm', ...
            refRow.load_kg, lt, layerName, refRow.p1_x, refRow.p1_y, refRow.p2_x, refRow.p2_y);
    else
        s = sprintf('P = %.0f kg | %s | Layer = %s | P1 = (%.1f, %.1f) cm', ...
            refRow.load_kg, lt, layerName, refRow.p1_x, refRow.p1_y);
    end
end

function s = englishCaseFileName(refRow)
    lt = regexprep(englishLoadType(refRow.load_type), '\s+', '');
    s = sprintf('P%.0f_%s_X%.1f_Y%.1f', refRow.load_kg, lt, refRow.p1_x, refRow.p1_y);
    s = matlab.lang.makeValidName(s);
end

function s = englishLoadType(loadType)
    loadType = string(loadType);
    if loadType == "单点中心加载"
        s = 'Single Center Loading';
    elseif loadType == "双点对称加载"
        s = 'Dual Symmetric Loading';
    elseif loadType == "双点不对称加载"
        s = 'Dual Asymmetric Loading';
    elseif loadType == "单点偏心加载"
        s = 'Single Eccentric Loading';
    elseif loadType == "纵中心线加载"
        s = 'Longitudinal Shifted Loading';
    else
        s = char(loadType);
    end
end

function writeChineseReport(reportFile, cfg, MetricsOverall, CaseMetrics, FieldSummary, ...
    modelFile, bestEpoch, bestValLoss)

    fid = fopen(reportFile, 'w');
    if fid < 0
        warning('无法写入中文分析报告。');
        return;
    end

    fprintf(fid, 'Step15 沥青层荷载均衡位移神经场 + 自动微分应变监督 报告\n');
    fprintf(fid, '============================================================\n\n');

    fprintf(fid, '一、方法说明\n');
    fprintf(fid, '本脚本聚焦沥青层 Asphalt，不再混合土层伪弯曲约束；网络不直接拟合应变，而是直接输出连续位移代理场 w(x,y,c)。\n');
    fprintf(fid, '训练时利用 MATLAB 自动微分计算 w 对空间坐标的二阶导数，并根据 FBG 方向得到预测应变：\n');
    fprintf(fid, '  纵向 FBG: epsilon_x = -z * d2w/dx2\n');
    fprintf(fid, '  横向 FBG: epsilon_y = -z * d2w/dy2\n');
    fprintf(fid, '然后用该预测应变拟合实测 FBG 应变。\n\n');

    fprintf(fid, '二、关键设置\n');
    fprintf(fid, '数据划分方式：%s\n', cfg.splitMode);
    fprintf(fid, '训练层：%s\n', strjoin(cfg.trainLayers, ', '));
    fprintf(fid, '镜像增强：%d\n', cfg.useMirrorAugmentation);
    fprintf(fid, '荷载均衡 loss：%d, offset=%.3g, power=%.3g, maxWeight=%.3g\n', cfg.useLoadWeightedLoss, cfg.loadWeightOffset, cfg.loadWeightPower, cfg.maxLoadWeight);
    fprintf(fid, '隐藏层数：%d\n', cfg.numHiddenLayers);
    fprintf(fid, '每层神经元数：%d\n', cfg.hiddenUnits);
    fprintf(fid, 'w_scale_m：%.6g m\n', cfg.w_scale_m);
    fprintf(fid, 'z_asphalt_m：%.6g m\n', cfg.z_asphalt_m);
    fprintf(fid, 'z_soil_m：%.6g m\n', cfg.z_soil_m);
    fprintf(fid, '最优 Epoch：%d\n', bestEpoch);
    fprintf(fid, '最优验证 data loss：%.6g\n\n', bestValLoss);

    fprintf(fid, '三、总体性能\n');
    for i = 1:height(MetricsOverall)
        fprintf(fid, '%s: R2 = %.4f, RMSE = %.6g ue, MAE = %.6g ue, Bias = %.6g ue\n', ...
            MetricsOverall.split(i), MetricsOverall.R2(i), MetricsOverall.RMSE_ue(i), ...
            MetricsOverall.MAE_ue(i), MetricsOverall.Bias_ue(i));
    end
    fprintf(fid, '\n');

    fprintf(fid, '四、分工况薄弱项\n');
    [~, worstOrder] = sort(CaseMetrics.R2, 'ascend', 'MissingPlacement','last');
    nShow = min(6, height(CaseMetrics));
    fprintf(fid, 'R2 最低的 %d 个工况如下：\n', nShow);
    for k = 1:nShow
        row = CaseMetrics(worstOrder(k), :);
        fprintf(fid, '  %s | %s | %.0f kg | R2=%.4f | RMSE=%.6g ue\n', ...
            row.case_id, row.load_type, row.load_kg, row.R2, row.RMSE_ue);
    end
    fprintf(fid, '\n');

    fprintf(fid, '五、位移代理场解释\n');
    fprintf(fid, '由于训练数据本身是 FBG 应变，而不是位移，因此 w 的绝对零点和整体倾斜并不能由应变唯一确定。\n');
    fprintf(fid, '本脚本通过 gauge loss 与 slope loss 固定该自由度。\n');
    fprintf(fid, '因此 w 更适合解释为满足应变观测约束的相对位移代理场，而不是直接实测沉降。\n\n');

    fprintf(fid, '六、连续场输出\n');
    fprintf(fid, '连续场 CSV 数量：%d。\n', height(FieldSummary));
    fprintf(fid, '每个连续场包含：w_proxy_um, eps_x_ue, eps_y_ue, dlambda_x_nm, dlambda_y_nm。\n\n');

    fprintf(fid, '七、下一步建议\n');
    fprintf(fid, '1. 与 Step13/Step14 对比 R2、RMSE 和边界工况图像，重点观察 Asphalt-only 是否解除 Soil 物理冲突。\n');
    fprintf(fid, '2. 如果验证集性能下降明显，优先尝试 cfg.splitMode = point，确认模型是否具备场内插值能力。\n');
    fprintf(fid, '3. 如果训练不稳定，可减小 cfg.w_scale_m 或降低 cfg.initialLearnRate。\n');
    fprintf(fid, '4. 如果边界工况仍然差，可继续引入 FiLM 条件调制或中心线 PDE 残差项。\n\n');

    fprintf(fid, '模型文件：%s\n', modelFile);

    fclose(fid);
end
