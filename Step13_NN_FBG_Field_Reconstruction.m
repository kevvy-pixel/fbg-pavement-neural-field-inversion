%% =========================================================================
%  Step13_NN_FBG_Field_Reconstruction.m
%
%  目标：
%    用坐标型神经网络拟合连续 FBG 响应场 / 应变场，
%    并通过 FBG 标定 + 曲率积分生成 displacement proxy field。
%
%  输入：
%    Response_Matrix_Table.csv
%
%  输出：
%    Step13_NN_FBG_Field_Results/
%      ├── Figs/      图片，全部英文标注
%      ├── Tables/    训练、验证、分工况指标和网格预测 CSV
%      ├── Models/    训练好的网络和归一化参数
%      └── Logs/      中文训练日志和中文分析报告
%
%  说明：
%    1. 图中的标题、坐标轴、图例、色标均使用英文。
%    2. 命令行日志、文本报告、代码注释使用中文。
%    3. displacement proxy 不是直接实测位移，而是基于：
%          delta_lambda -> strain -> curvature -> 1D integration
%       得到的位移代理场。
% =========================================================================

clear; clc; close all;

%% ========================= 0. 用户可调参数 ==============================
cfg = struct();

% ---------- 输入输出 ----------
cfg.inputCSV = 'Response_Matrix_Table.csv';
cfg.outBase  = 'Step13_NN_FBG_Field_Results';

% ---------- 物理/标定参数 ----------
cfg.k_eps_nm_per_ue = 1.2e-3;      % FBG 灵敏系数，1.2 pm/ue = 0.0012 nm/ue
cfg.z_depth_m       = 0.020;       % 传感器到中性层距离，用于 strain -> curvature
cfg.plateLength_cm  = 120;         % 模型长度
cfg.plateWidth_cm   = 50;          % 模型宽度

% ---------- 训练目标 ----------
% 可选：
%   'strain'  : 训练目标为 microstrain，推荐
%   'dlambda' : 训练目标为 wavelength shift, nm
cfg.targetMode = 'strain';

% ---------- 数据划分 ----------
% 可选：
%   'case'  : 按工况划分训练/验证/测试，更严格，推荐
%   'point' : 按测点随机划分，拟合效果通常更好，但泛化评估偏乐观
cfg.splitMode = 'case';
cfg.valRatio  = 0.15;
cfg.testRatio = 0.15;
cfg.randomSeed = 2026;

% ---------- 神经网络结构 ----------
cfg.hiddenUnits = 64;
cfg.numHiddenLayers = 4;
cfg.fourierFreqs = [1 2 4 8];      % 坐标 Fourier 特征频率
cfg.activation = 'tanh';           % 目前脚本使用 tanh 激活

% ---------- 训练超参数 ----------
cfg.maxEpochs        = 1200;
cfg.miniBatchSize    = 64;
cfg.initialLearnRate = 2e-3;
cfg.learnRateDropPeriod = 300;
cfg.learnRateDropFactor = 0.5;
cfg.gradientDecayFactor = 0.9;
cfg.squaredGradientDecayFactor = 0.999;
cfg.l2Lambda         = 1e-5;       % L2 正则
cfg.earlyStopPatience = 180;
cfg.printEveryEpoch = 20;

% ---------- 输出连续场 ----------
cfg.gridNX = 121;
cfg.gridNY = 61;

% 指定需要输出连续场的工况。若某工况不存在，会自动跳过。
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

% 输出连续场的层
cfg.fieldLayers = ["Asphalt", "Soil"];

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
logBoth(fidLog, 'Step13 神经网络 FBG 连续响应场 / 应变场拟合');
logBoth(fidLog, '============================================================');
logBoth(fidLog, '输出文件夹：%s', cfg.outBase);
logBoth(fidLog, '训练目标：%s', cfg.targetMode);
logBoth(fidLog, '数据划分方式：%s', cfg.splitMode);

%% ========================= 2. 环境检查 ===================================
if ~license('test','Neural_Network_Toolbox')
    fclose(fidLog);
    error(['未检测到 Deep Learning Toolbox。', newline, ...
           '本脚本需要 MATLAB Deep Learning Toolbox 才能训练 dlnetwork。']);
end

rng(cfg.randomSeed);

%% ========================= 3. 读取并清洗数据 =============================
assert(isfile(cfg.inputCSV), '找不到输入文件：%s。请把脚本放在 Response_Matrix_Table.csv 同目录下运行。', cfg.inputCSV);

T = readtable(cfg.inputCSV, 'TextType', 'string');

requiredCols = ["case_id","sensor_id","x_cm","y_cm","layer","direction", ...
    "delta_lambda_resp_nm","load_type","load_kg","p1_x","p1_y","p2_x","p2_y"];
missingCols = setdiff(requiredCols, string(T.Properties.VariableNames));
assert(isempty(missingCols), '输入表缺少必要列：%s', strjoin(missingCols, ', '));

% 基本类型
T.case_id   = string(T.case_id);
T.sensor_id = string(T.sensor_id);
T.layer     = string(T.layer);
T.direction = string(T.direction);
T.load_type = string(T.load_type);

T.x_cm = double(T.x_cm);
T.y_cm = double(T.y_cm);
T.load_kg = double(T.load_kg);
T.p1_x = double(T.p1_x);
T.p1_y = double(T.p1_y);
T.p2_x = double(T.p2_x);
T.p2_y = double(T.p2_y);
T.delta_lambda_resp_nm = double(T.delta_lambda_resp_nm);

% 兼容已有脚本中的修正
mask425 = T.case_id == "40kg纵中心线42.5cm加载";
if any(mask425)
    T.p1_x(mask425) = 42.5;
end

% 删除无效响应
valid = isfinite(T.x_cm) & isfinite(T.y_cm) & isfinite(T.load_kg) & ...
        isfinite(T.p1_x) & isfinite(T.p1_y) & isfinite(T.delta_lambda_resp_nm);
T = T(valid, :);

% 目标变量
T.strain_ue = T.delta_lambda_resp_nm ./ cfg.k_eps_nm_per_ue;

switch lower(cfg.targetMode)
    case 'strain'
        yRaw = T.strain_ue;
        targetName = 'Strain';
        targetUnit = 'microstrain';
    case 'dlambda'
        yRaw = T.delta_lambda_resp_nm;
        targetName = 'Wavelength Shift';
        targetUnit = 'nm';
    otherwise
        fclose(fidLog);
        error('cfg.targetMode 只能是 strain 或 dlambda。');
end

logBoth(fidLog, '数据读取完成：%d 行，%d 个工况。', height(T), numel(unique(T.case_id)));
logBoth(fidLog, '层信息：%s。', strjoin(unique(T.layer), ', '));
logBoth(fidLog, '荷载类型：%s。', strjoin(unique(T.load_type), ', '));

%% ========================= 4. 构造特征 ===================================
featureInfo = buildFeatureInfo(T, cfg);
[Xraw, featureNames] = buildFeatureMatrix(T, cfg, featureInfo);

% 按训练集统计归一化参数，先划分索引
[trainIdx, valIdx, testIdx] = splitData(T, cfg);

if sum(trainIdx) < 10
    fclose(fidLog);
    error('训练样本过少，请检查数据划分方式。');
end

% 目标归一化
yMu = mean(yRaw(trainIdx), 'omitnan');
yStd = std(yRaw(trainIdx), 'omitnan');
if yStd < 1e-12
    yStd = 1;
end
Y = (yRaw - yMu) ./ yStd;

% 特征归一化
xMu = mean(Xraw(:, trainIdx), 2, 'omitnan');
xStd = std(Xraw(:, trainIdx), 0, 2, 'omitnan');
xStd(xStd < 1e-12) = 1;
X = (Xraw - xMu) ./ xStd;

% 转成网络输入格式：features x observations
Xtrain = single(X(:, trainIdx));
Ytrain = single(Y(trainIdx)');
Xval   = single(X(:, valIdx));
Yval   = single(Y(valIdx)');
Xtest  = single(X(:, testIdx));
Ytest  = single(Y(testIdx)');

logBoth(fidLog, '特征构造完成：%d 个输入特征。', size(X,1));
logBoth(fidLog, '训练/验证/测试样本数：%d / %d / %d。', sum(trainIdx), sum(valIdx), sum(testIdx));

% 保存特征名
featureTable = table((1:numel(featureNames))', featureNames(:), ...
    'VariableNames', {'feature_index','feature_name'});
writetable(featureTable, fullfile(dirs.tables, 'Feature_Names.csv'));

%% ========================= 5. 建立网络 ===================================
net = buildMLP(size(Xtrain,1), cfg.hiddenUnits, cfg.numHiddenLayers);

logBoth(fidLog, '网络建立完成：%d 层隐藏层，每层 %d 个神经元。', ...
    cfg.numHiddenLayers, cfg.hiddenUnits);

%% ========================= 6. 训练网络 ===================================
numTrain = size(Xtrain, 2);
numIterationsPerEpoch = max(1, ceil(numTrain / cfg.miniBatchSize));

trailingAvg = [];
trailingAvgSq = [];

bestValLoss = inf;
bestNet = net;
bestEpoch = 0;
badEpochs = 0;

history = table('Size',[cfg.maxEpochs, 7], ...
    'VariableTypes', {'double','double','double','double','double','double','double'}, ...
    'VariableNames', {'epoch','learn_rate','train_loss_norm','val_loss_norm','train_R2_raw','val_R2_raw','elapsed_sec'});

tStart = tic;

logBoth(fidLog, '开始训练。');
logBoth(fidLog, '注意：日志中的 loss 是归一化目标下的 MSE，R2 是反归一化后的原始量纲结果。');

for epoch = 1:cfg.maxEpochs
    lr = cfg.initialLearnRate * cfg.learnRateDropFactor ^ floor((epoch-1)/cfg.learnRateDropPeriod);
    order = randperm(numTrain);
    epochLoss = 0;

    for it = 1:numIterationsPerEpoch
        b1 = (it-1)*cfg.miniBatchSize + 1;
        b2 = min(it*cfg.miniBatchSize, numTrain);
        batchIds = order(b1:b2);

        dlX = dlarray(Xtrain(:, batchIds), 'CB');
        dlY = dlarray(Ytrain(:, batchIds), 'CB');

        [loss, gradients] = dlfeval(@modelGradients, net, dlX, dlY, cfg.l2Lambda);
        epochLoss = epochLoss + double(gather(extractdata(loss))) * numel(batchIds);

        globalIter = (epoch-1)*numIterationsPerEpoch + it;
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
            trailingAvg, trailingAvgSq, globalIter, lr, ...
            cfg.gradientDecayFactor, cfg.squaredGradientDecayFactor);
    end

    epochLoss = epochLoss / numTrain;

    % 评估
    predTrainNorm = predictNorm(net, Xtrain);
    predValNorm   = predictNorm(net, Xval);

    trainLossNorm = mean((predTrainNorm - double(Ytrain)).^2, 'omitnan');
    valLossNorm   = mean((predValNorm - double(Yval)).^2, 'omitnan');

    predTrainRaw = predTrainNorm .* yStd + yMu;
    trueTrainRaw = double(Ytrain) .* yStd + yMu;
    predValRaw   = predValNorm .* yStd + yMu;
    trueValRaw   = double(Yval) .* yStd + yMu;

    trainR2 = calcR2(trueTrainRaw(:), predTrainRaw(:));
    valR2   = calcR2(trueValRaw(:), predValRaw(:));

    history.epoch(epoch) = epoch;
    history.learn_rate(epoch) = lr;
    history.train_loss_norm(epoch) = trainLossNorm;
    history.val_loss_norm(epoch) = valLossNorm;
    history.train_R2_raw(epoch) = trainR2;
    history.val_R2_raw(epoch) = valR2;
    history.elapsed_sec(epoch) = toc(tStart);

    if valLossNorm < bestValLoss
        bestValLoss = valLossNorm;
        bestNet = net;
        bestEpoch = epoch;
        badEpochs = 0;
    else
        badEpochs = badEpochs + 1;
    end

    if mod(epoch, cfg.printEveryEpoch) == 0 || epoch == 1
        logBoth(fidLog, ['Epoch %4d | 学习率 %.3g | 训练loss %.6g | 验证loss %.6g | ', ...
                         '训练R2 %.4f | 验证R2 %.4f | 最优Epoch %d'], ...
                         epoch, lr, trainLossNorm, valLossNorm, trainR2, valR2, bestEpoch);
    end

    if badEpochs >= cfg.earlyStopPatience
        logBoth(fidLog, '触发早停：连续 %d 个 epoch 验证集未改善。', cfg.earlyStopPatience);
        break;
    end
end

history = history(1:epoch, :);
net = bestNet;

logBoth(fidLog, '训练结束。最优 Epoch = %d，最优验证 loss = %.6g。', bestEpoch, bestValLoss);
writetable(history, fullfile(dirs.tables, 'Training_History.csv'));

%% ========================= 7. 总体评估 ===================================
predAllNorm   = nan(1, height(T));
predTrainNorm = predictNorm(net, Xtrain);
predValNorm   = predictNorm(net, Xval);
predTestNorm  = predictNorm(net, Xtest);

predAllNorm(trainIdx) = predTrainNorm;
predAllNorm(valIdx)   = predValNorm;
predAllNorm(testIdx)  = predTestNorm;

predAllRaw = predAllNorm(:) .* yStd + yMu;

ResultPoint = T(:, ["case_id","sensor_id","x_cm","y_cm","layer","direction", ...
    "load_type","load_kg","p1_x","p1_y","p2_x","p2_y", ...
    "delta_lambda_resp_nm","strain_ue"]);
ResultPoint.target_raw = yRaw;
ResultPoint.pred_target_raw = predAllRaw;
ResultPoint.residual_raw = ResultPoint.pred_target_raw - ResultPoint.target_raw;

% 同时给出预测的 nm 与 microstrain
switch lower(cfg.targetMode)
    case 'strain'
        ResultPoint.pred_strain_ue = ResultPoint.pred_target_raw;
        ResultPoint.pred_dlambda_nm = ResultPoint.pred_strain_ue .* cfg.k_eps_nm_per_ue;
    case 'dlambda'
        ResultPoint.pred_dlambda_nm = ResultPoint.pred_target_raw;
        ResultPoint.pred_strain_ue = ResultPoint.pred_dlambda_nm ./ cfg.k_eps_nm_per_ue;
end

ResultPoint.split = strings(height(T),1);
ResultPoint.split(trainIdx) = "train";
ResultPoint.split(valIdx)   = "validation";
ResultPoint.split(testIdx)  = "test";

writetable(ResultPoint, fullfile(dirs.tables, 'Pointwise_Predictions.csv'));

% 汇总指标
metricsTrain = calcMetrics(yRaw(trainIdx), predAllRaw(trainIdx));
metricsVal   = calcMetrics(yRaw(valIdx),   predAllRaw(valIdx));
metricsTest  = calcMetrics(yRaw(testIdx),  predAllRaw(testIdx));
metricsAll   = calcMetrics(yRaw,           predAllRaw);

MetricsOverall = table( ...
    ["train";"validation";"test";"all"], ...
    [metricsTrain.R2; metricsVal.R2; metricsTest.R2; metricsAll.R2], ...
    [metricsTrain.RMSE; metricsVal.RMSE; metricsTest.RMSE; metricsAll.RMSE], ...
    [metricsTrain.MAE; metricsVal.MAE; metricsTest.MAE; metricsAll.MAE], ...
    [metricsTrain.Bias; metricsVal.Bias; metricsTest.Bias; metricsAll.Bias], ...
    'VariableNames', {'split','R2','RMSE','MAE','Bias'});
writetable(MetricsOverall, fullfile(dirs.tables, 'Overall_Metrics.csv'));

logBoth(fidLog, '总体指标：');
logBoth(fidLog, '  Train      R2=%.4f, RMSE=%.6g %s', metricsTrain.R2, metricsTrain.RMSE, targetUnit);
logBoth(fidLog, '  Validation R2=%.4f, RMSE=%.6g %s', metricsVal.R2, metricsVal.RMSE, targetUnit);
logBoth(fidLog, '  Test       R2=%.4f, RMSE=%.6g %s', metricsTest.R2, metricsTest.RMSE, targetUnit);
logBoth(fidLog, '  All        R2=%.4f, RMSE=%.6g %s', metricsAll.R2, metricsAll.RMSE, targetUnit);

% 分工况指标
caseList = unique(T.case_id, 'stable');
CaseMetrics = table();
for i = 1:numel(caseList)
    cid = caseList(i);
    mask = T.case_id == cid;
    mt = calcMetrics(yRaw(mask), predAllRaw(mask));
    one = table(cid, unique(T.load_type(mask),'stable'), unique(T.load_kg(mask),'stable'), ...
        sum(mask), mt.R2, mt.RMSE, mt.MAE, mt.Bias, ...
        'VariableNames', {'case_id','load_type','load_kg','n_points','R2','RMSE','MAE','Bias'});
    CaseMetrics = [CaseMetrics; one]; %#ok<AGROW>
end
writetable(CaseMetrics, fullfile(dirs.tables, 'Case_Metrics.csv'));

%% ========================= 8. 绘制训练与评估图片 =========================
plotTrainingHistory(history, dirs.figs);
plotParity(yRaw(trainIdx), predAllRaw(trainIdx), ...
           yRaw(valIdx), predAllRaw(valIdx), ...
           yRaw(testIdx), predAllRaw(testIdx), ...
           targetName, targetUnit, dirs.figs);
plotResiduals(yRaw, predAllRaw, targetUnit, dirs.figs);
plotCaseMetrics(CaseMetrics, targetUnit, dirs.figs);

%% ========================= 9. 连续场预测与 displacement proxy =============
logBoth(fidLog, '开始输出连续场。');

FieldSummary = table();
fieldCounter = 0;

for ic = 1:numel(cfg.fieldCaseList)
    cid = cfg.fieldCaseList(ic);
    if ~any(T.case_id == cid)
        logBoth(fidLog, '跳过连续场输出：工况不存在：%s', cid);
        continue;
    end

    refRows = T(T.case_id == cid, :);
    refRow = refRows(1,:);

    for il = 1:numel(cfg.fieldLayers)
        lay = cfg.fieldLayers(il);
        if ~any(T.case_id == cid & T.layer == lay)
            logBoth(fidLog, '跳过连续场输出：工况 %s 无层 %s 数据。', cid, lay);
            continue;
        end

        fieldCounter = fieldCounter + 1;

        G = makeGridTable(refRow, lay, cfg);
        [XgridRaw, ~] = buildFeatureMatrix(G, cfg, featureInfo);
        Xgrid = (XgridRaw - xMu) ./ xStd;

        predGridNorm = predictNorm(net, single(Xgrid));
        predGridTarget = predGridNorm(:) .* yStd + yMu;

        switch lower(cfg.targetMode)
            case 'strain'
                predStrain = predGridTarget;
                predDLambda = predStrain .* cfg.k_eps_nm_per_ue;
            case 'dlambda'
                predDLambda = predGridTarget;
                predStrain = predDLambda ./ cfg.k_eps_nm_per_ue;
        end

        nx = cfg.gridNX;
        ny = cfg.gridNY;
        Xg = reshape(G.x_cm, [ny, nx]);
        Yg = reshape(G.y_cm, [ny, nx]);
        strainGrid = reshape(predStrain, [ny, nx]);
        dlambdaGrid = reshape(predDLambda, [ny, nx]);

        dispProxyGrid_um = displacementProxyFromStrain(strainGrid, Xg, cfg.z_depth_m);

        FieldTable = table(G.x_cm, G.y_cm, repmat(cid,height(G),1), repmat(lay,height(G),1), ...
            predDLambda(:), predStrain(:), dispProxyGrid_um(:), ...
            'VariableNames', {'x_cm','y_cm','case_id','layer','pred_dlambda_nm','pred_strain_ue','disp_proxy_um'});

        safeName = sprintf('Field_%02d_%s_%s', fieldCounter, englishCaseFileName(refRow), char(lay));
        csvName = fullfile(dirs.tables, safeName + ".csv");
        writetable(FieldTable, csvName);

        englishTitle = englishCaseTitle(refRow, lay);

        plotFieldContour(Xg, Yg, dlambdaGrid, ...
            'NN Predicted Wavelength Shift Field', 'X (cm)', 'Y (cm)', ...
            'Wavelength Shift (nm)', englishTitle, ...
            fullfile(dirs.figs, safeName + "_Dlambda.png"));

        plotFieldContour(Xg, Yg, strainGrid, ...
            'NN Predicted Strain Field', 'X (cm)', 'Y (cm)', ...
            'Strain (microstrain)', englishTitle, ...
            fullfile(dirs.figs, safeName + "_Strain.png"));

        plotFieldContour(Xg, Yg, dispProxyGrid_um, ...
            'Displacement Proxy Field', 'X (cm)', 'Y (cm)', ...
            'Displacement Proxy (um)', englishTitle, ...
            fullfile(dirs.figs, safeName + "_DisplacementProxy.png"));

        one = table(cid, lay, max(abs(predDLambda(:))), max(abs(predStrain(:))), ...
            max(abs(dispProxyGrid_um(:))), csvName, ...
            'VariableNames', {'case_id','layer','max_abs_dlambda_nm','max_abs_strain_ue', ...
                              'max_abs_disp_proxy_um','csv_path'});
        FieldSummary = [FieldSummary; one]; %#ok<AGROW>

        logBoth(fidLog, '已输出连续场：%s | %s', cid, lay);
    end
end

writetable(FieldSummary, fullfile(dirs.tables, 'Field_Summary.csv'));

%% ========================= 10. 保存模型与报告 =============================
modelFile = fullfile(dirs.models, 'NN_FBG_Field_Model.mat');
save(modelFile, 'net', 'bestNet', 'cfg', 'featureInfo', 'featureNames', ...
    'xMu', 'xStd', 'yMu', 'yStd', 'targetName', 'targetUnit', ...
    'MetricsOverall', 'CaseMetrics', 'history');

reportFile = fullfile(dirs.logs, 'Analysis_Report_CN.txt');
writeChineseReport(reportFile, cfg, MetricsOverall, CaseMetrics, FieldSummary, ...
    targetName, targetUnit, modelFile);

logBoth(fidLog, '模型已保存：%s', modelFile);
logBoth(fidLog, '中文分析报告已保存：%s', reportFile);
logBoth(fidLog, '全部完成。');
fclose(fidLog);

disp(' ');
disp('==================== Step13 完成 ====================');
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

function featureInfo = buildFeatureInfo(T, cfg)
    featureInfo = struct();
    featureInfo.loadTypes = unique(T.load_type, 'stable');
    featureInfo.layers = ["Asphalt","Soil"];
    featureInfo.directions = ["纵","横"];
    featureInfo.fourierFreqs = cfg.fourierFreqs;
end

function [X, names] = buildFeatureMatrix(T, cfg, featureInfo)
    n = height(T);
    L = cfg.plateLength_cm;
    W = cfg.plateWidth_cm;

    x = double(T.x_cm);
    y = double(T.y_cm);
    loadKg = double(T.load_kg);
    p1x = double(T.p1_x);
    p1y = double(T.p1_y);
    p2x = double(T.p2_x);
    p2y = double(T.p2_y);

    p2Mask = isfinite(p2x) & isfinite(p2y);
    p2xFill = p2x;
    p2yFill = p2y;
    p2xFill(~p2Mask) = p1x(~p2Mask);
    p2yFill(~p2Mask) = p1y(~p2Mask);

    xN = x ./ L;
    yN = y ./ W;
    p1xN = p1x ./ L;
    p1yN = p1y ./ W;
    p2xN = p2xFill ./ L;
    p2yN = p2yFill ./ W;

    maxLoad = max(double(T.load_kg), [], 'omitnan');
    if maxLoad <= 0 || ~isfinite(maxLoad)
        maxLoad = 1;
    end
    loadN = loadKg ./ maxLoad;

    layerSoil = double(T.layer == "Soil");
    dirLong = double(T.direction == "纵");

    dLeft = x ./ L;
    dRight = (L - x) ./ L;
    dBottom = y ./ W;
    dTop = (W - y) ./ W;

    dP1 = sqrt((x - p1x).^2 + (y - p1y).^2) ./ sqrt(L^2 + W^2);
    dP2 = sqrt((x - p2xFill).^2 + (y - p2yFill).^2) ./ sqrt(L^2 + W^2);

    % 基础特征
    Xlist = {};
    names = strings(0,1);

    add = @(v,nm) addFeature(v,nm);

    add(xN, "x_norm");
    add(yN, "y_norm");
    add(loadN, "load_norm");
    add(p1xN, "p1_x_norm");
    add(p1yN, "p1_y_norm");
    add(p2xN, "p2_x_norm");
    add(p2yN, "p2_y_norm");
    add(double(p2Mask), "has_second_load");
    add(layerSoil, "layer_soil");
    add(dirLong, "direction_long");
    add(dLeft, "dist_left_norm");
    add(dRight, "dist_right_norm");
    add(dBottom, "dist_bottom_norm");
    add(dTop, "dist_top_norm");
    add(dP1, "dist_to_load1_norm");
    add(dP2, "dist_to_load2_norm");

    % 交互项
    add(loadN .* dP1, "load_x_dist_load1");
    add(loadN .* dP2, "load_x_dist_load2");
    add(loadN .* layerSoil, "load_x_layer_soil");

    % Fourier 坐标特征
    freqs = featureInfo.fourierFreqs;
    for k = 1:numel(freqs)
        f = freqs(k);
        add(sin(2*pi*f*xN), sprintf("sin_x_f%d", f));
        add(cos(2*pi*f*xN), sprintf("cos_x_f%d", f));
        add(sin(2*pi*f*yN), sprintf("sin_y_f%d", f));
        add(cos(2*pi*f*yN), sprintf("cos_y_f%d", f));
    end

    % 荷载类型 one-hot
    for k = 1:numel(featureInfo.loadTypes)
        lt = featureInfo.loadTypes(k);
        add(double(T.load_type == lt), "load_type_" + matlab.lang.makeValidName(char(lt)));
    end

    X = zeros(numel(Xlist), n);
    for i = 1:numel(Xlist)
        X(i,:) = Xlist{i}(:)';
    end

    function addFeature(v, nm)
        Xlist{end+1} = double(v); %#ok<AGROW>
        names(end+1,1) = string(nm); %#ok<AGROW>
    end
end

function [trainIdx, valIdx, testIdx] = splitData(T, cfg)
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

        otherwise
            error('cfg.splitMode 只能是 case 或 point。');
    end
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
        fullyConnectedLayer(1, 'Name','output')
    ];

    net = dlnetwork(layerGraph(layers));
end

function [loss, gradients] = modelGradients(net, dlX, dlY, l2Lambda)
    dlYPred = forward(net, dlX);
    dataLoss = mean((dlYPred - dlY).^2, 'all');
    l2Loss = l2Regularization(net);
    loss = dataLoss + l2Lambda * l2Loss;
    gradients = dlgradient(loss, net.Learnables);
end

function l2 = l2Regularization(net)
    l2 = dlarray(0);
    L = net.Learnables;
    for i = 1:size(L,1)
        name = string(L.Parameter(i));
        if contains(name, "Weights")
            W = L.Value{i};
            l2 = l2 + sum(W.^2, 'all');
        end
    end
end

function y = predictNorm(net, X)
    if isempty(X)
        y = [];
        return;
    end
    dlX = dlarray(single(X), 'CB');
    dlY = forward(net, dlX);
    y = double(gather(extractdata(dlY)));
end

function r2 = calcR2(yTrue, yPred)
    yTrue = yTrue(:);
    yPred = yPred(:);
    mask = isfinite(yTrue) & isfinite(yPred);
    yTrue = yTrue(mask);
    yPred = yPred(mask);
    if numel(yTrue) < 2 || sum((yTrue - mean(yTrue)).^2) < 1e-12
        r2 = NaN;
    else
        r2 = 1 - sum((yTrue - yPred).^2) / sum((yTrue - mean(yTrue)).^2);
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
    semilogy(history.epoch, history.train_loss_norm, 'LineWidth', 1.8); hold on;
    semilogy(history.epoch, history.val_loss_norm, 'LineWidth', 1.8);
    xlabel('Epoch');
    ylabel('Normalized MSE Loss');
    title('Training and Validation Loss');
    legend('Train','Validation','Location','northeast');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Training_Loss.png'));
    close(fig);

    fig = figure('Visible','off','Position',[100 100 900 520]);
    plot(history.epoch, history.train_R2_raw, 'LineWidth', 1.8); hold on;
    plot(history.epoch, history.val_R2_raw, 'LineWidth', 1.8);
    xlabel('Epoch');
    ylabel('R^2');
    title('Training and Validation R^2');
    legend('Train','Validation','Location','southeast');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Training_R2.png'));
    close(fig);
end

function plotParity(yTrain, pTrain, yVal, pVal, yTest, pTest, targetName, targetUnit, figDir)
    fig = figure('Visible','off','Position',[100 100 720 640]);
    scatter(yTrain, pTrain, 28, 'filled', 'MarkerFaceAlpha', 0.55); hold on;
    scatter(yVal, pVal, 32, 'filled', 'MarkerFaceAlpha', 0.75);
    scatter(yTest, pTest, 36, 'filled', 'MarkerFaceAlpha', 0.75);

    allY = [yTrain(:); yVal(:); yTest(:); pTrain(:); pVal(:); pTest(:)];
    mn = min(allY, [], 'omitnan');
    mx = max(allY, [], 'omitnan');
    plot([mn mx], [mn mx], 'k--', 'LineWidth', 1.4);

    xlabel(['Measured ', targetName, ' (', targetUnit, ')']);
    ylabel(['Predicted ', targetName, ' (', targetUnit, ')']);
    title('Measured vs Predicted Response');
    legend('Train','Validation','Test','1:1 Line','Location','best');
    grid on; axis equal;
    xlim([mn mx]); ylim([mn mx]);
    saveas(fig, fullfile(figDir, 'Fig_Parity_Measured_vs_Predicted.png'));
    close(fig);
end

function plotResiduals(yTrue, yPred, targetUnit, figDir)
    res = yPred(:) - yTrue(:);

    fig = figure('Visible','off','Position',[100 100 850 520]);
    histogram(res, 30);
    xlabel(['Residual (', targetUnit, ')']);
    ylabel('Count');
    title('Residual Distribution');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Residual_Histogram.png'));
    close(fig);

    fig = figure('Visible','off','Position',[100 100 850 520]);
    scatter(yPred, res, 28, 'filled', 'MarkerFaceAlpha', 0.60);
    yline(0, 'k--', 'LineWidth', 1.2);
    xlabel(['Predicted Value (', targetUnit, ')']);
    ylabel(['Residual (', targetUnit, ')']);
    title('Residuals vs Predicted Values');
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Residual_vs_Predicted.png'));
    close(fig);
end

function plotCaseMetrics(CaseMetrics, targetUnit, figDir)
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
    bar(C.RMSE);
    xlabel('Case Index');
    ylabel(['RMSE (', targetUnit, ')']);
    title('Case-wise RMSE');
    xticks(1:height(C));
    xticklabels(labels);
    grid on;
    saveas(fig, fullfile(figDir, 'Fig_Casewise_RMSE.png'));
    close(fig);

    mapTable = table((1:height(C))', C.case_id, C.load_type, C.load_kg, C.R2, C.RMSE, ...
        'VariableNames', {'case_index_in_fig','case_id','load_type','load_kg','R2','RMSE'});
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
end

function dispProxy_um = displacementProxyFromStrain(strainGrid_ue, Xg_cm, zDepth_m)
    % 输入 strainGrid_ue 为 microstrain。
    % 采用每一条 y 线上沿 x 的一维曲率积分：
    %     curvature = -epsilon / z
    %     d2w/dx2 = curvature
    % 边界条件：
    %     w(x_min)=0, w(x_max)=0
    % 这是位移代理，不是真实全场位移测量。
    [ny, nx] = size(strainGrid_ue);
    x_m = Xg_cm(1,:) ./ 100;
    dx = mean(diff(x_m));

    A = zeros(nx, nx);
    rhsTemplate = zeros(nx,1);

    A(1,1) = 1;
    A(nx,nx) = 1;
    for i = 2:nx-1
        A(i,i-1) = 1 / dx^2;
        A(i,i)   = -2 / dx^2;
        A(i,i+1) = 1 / dx^2;
    end

    dispProxy = zeros(ny, nx);
    for j = 1:ny
        eps = strainGrid_ue(j,:)' * 1e-6;
        curvature = -eps ./ zDepth_m;
        rhs = rhsTemplate;
        rhs(2:nx-1) = curvature(2:nx-1);
        w = A \ rhs;
        dispProxy(j,:) = w';
    end

    dispProxy_um = dispProxy .* 1e6;
end

function plotFieldContour(Xg, Yg, Zg, figTitle, xLabel, yLabel, cbarLabel, subTitle, outPath)
    fig = figure('Visible','off','Position',[100 100 900 650]);
    contourf(Xg, Yg, Zg, 30, 'LineColor','none');
    colorbar;
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

function writeChineseReport(reportFile, cfg, MetricsOverall, CaseMetrics, FieldSummary, targetName, targetUnit, modelFile)
    fid = fopen(reportFile, 'w');
    if fid < 0
        warning('无法写入中文分析报告。');
        return;
    end

    fprintf(fid, 'Step13 神经网络 FBG 连续响应场 / 应变场拟合报告\n');
    fprintf(fid, '============================================================\n\n');

    fprintf(fid, '一、方法概述\n');
    fprintf(fid, '本脚本采用坐标型多层感知机 MLP 建立如下映射：\n');
    fprintf(fid, '  空间坐标 + 层信息 + 方向信息 + 荷载大小 + 加载位置 + 荷载类型 -> FBG 响应/应变\n');
    fprintf(fid, '模型不预设单峰、高斯峰、双峰或边界非对称函数形式，因此可用于探索复杂边界工况下的非参数连续场重构。\n\n');

    fprintf(fid, '二、训练设置\n');
    fprintf(fid, '训练目标：%s (%s)\n', targetName, targetUnit);
    fprintf(fid, '数据划分方式：%s\n', cfg.splitMode);
    fprintf(fid, '隐藏层数：%d\n', cfg.numHiddenLayers);
    fprintf(fid, '每层神经元数：%d\n', cfg.hiddenUnits);
    fprintf(fid, 'Fourier 坐标频率：%s\n', mat2str(cfg.fourierFreqs));
    fprintf(fid, 'FBG 灵敏系数 k_eps：%.6g nm/ue\n', cfg.k_eps_nm_per_ue);
    fprintf(fid, '中性层距离 z_depth：%.6g m\n\n', cfg.z_depth_m);

    fprintf(fid, '三、总体性能\n');
    for i = 1:height(MetricsOverall)
        fprintf(fid, '%s: R2 = %.4f, RMSE = %.6g %s, MAE = %.6g %s, Bias = %.6g %s\n', ...
            MetricsOverall.split(i), MetricsOverall.R2(i), MetricsOverall.RMSE(i), targetUnit, ...
            MetricsOverall.MAE(i), targetUnit, MetricsOverall.Bias(i), targetUnit);
    end
    fprintf(fid, '\n');

    fprintf(fid, '四、分工况表现提示\n');
    [~, worstOrder] = sort(CaseMetrics.R2, 'ascend', 'MissingPlacement','last');
    nShow = min(5, height(CaseMetrics));
    fprintf(fid, 'R2 最低的 %d 个工况如下，建议优先检查这些工况的边界效应、加载位置或传感器异常：\n', nShow);
    for k = 1:nShow
        row = CaseMetrics(worstOrder(k), :);
        fprintf(fid, '  %s | %s | %.0f kg | R2=%.4f | RMSE=%.6g\n', ...
            row.case_id, row.load_type, row.load_kg, row.R2, row.RMSE);
    end
    fprintf(fid, '\n');

    fprintf(fid, '五、displacement proxy 的含义\n');
    fprintf(fid, '本脚本输出的 displacement proxy 不是直接实测位移，而是通过以下链条得到：\n');
    fprintf(fid, '  delta_lambda -> strain -> curvature -> 1D integration along x\n');
    fprintf(fid, '其中采用 w(x_min)=0, w(x_max)=0 的边界条件，并对每条 y 方向剖面独立积分。\n');
    fprintf(fid, '因此该结果更适合作为位移形态、相对幅值和边界影响的代理指标，不宜直接等同于真实绝对沉降。\n\n');

    fprintf(fid, '六、连续场输出\n');
    fprintf(fid, '共输出 %d 个连续场 CSV。\n', height(FieldSummary));
    if height(FieldSummary) > 0
        fprintf(fid, '最大绝对值摘要见 Tables/Field_Summary.csv。\n');
    end
    fprintf(fid, '\n');

    fprintf(fid, '七、后续建议\n');
    fprintf(fid, '1. 如果目的是重构已知工况内的连续场，可将 cfg.splitMode 改为 point。\n');
    fprintf(fid, '2. 如果目的是检验对未知工况的泛化能力，应保持 cfg.splitMode = case。\n');
    fprintf(fid, '3. 若后续获得真实位移计、DIC 或 LVDT 位移数据，可将本脚本的训练目标改为真实位移，或用其标定 displacement proxy 的比例系数。\n');
    fprintf(fid, '4. 对边界工况，可重点比较 NN 连续场、原高斯模型和 PDE 闭合结果之间的差异。\n\n');

    fprintf(fid, '模型文件：%s\n', modelFile);

    fclose(fid);
end
