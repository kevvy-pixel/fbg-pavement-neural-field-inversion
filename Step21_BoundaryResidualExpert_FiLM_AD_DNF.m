%% =========================================================================
% Step21_BoundaryResidualExpert_FiLM_AD_DNF.m
%
% Boundary-aware FiLM-conditioned displacement neural field with autodiff strain supervision.
%
% 关键修复：
%   1. 不再把 spaceX 预先算成普通数值特征后输入网络；
%   2. 在 dlfeval 内部由 tracked dlarray 坐标 xN/yN 动态构造空间特征；
%   3. 所有 dlarray 输入采用无标签格式，避免 W*x 的标签冲突；
%   4. 因此 w(x,y,c) -> wx -> wxx 的自动微分链条不会断开。
%
% 输入：
%   Response_Matrix_Table.csv
%
% 输出：
%   Step21_BoundaryResidualExpert_Results/
%     Figs / Tables / Models / Logs
%
% 图片标题、坐标轴、图例均为英文；日志和报告为中文。
% =========================================================================

clear; clc; close all;

%% ------------------------- 0. 配置 ---------------------------------------
cfg = struct();
cfg.inputCSV = 'Response_Matrix_Table.csv';
cfg.outBase  = 'Step21_BoundaryResidualExpert_Results';

cfg.plateLength_cm = 120;
cfg.plateWidth_cm  = 50;
cfg.k_eps_nm_per_ue = 1.2e-3;
cfg.z_asphalt_m = 0.020;
cfg.w_scale_m = 1.0e-3;

cfg.trainLayers = ["Asphalt"];
cfg.fieldLayers = ["Asphalt"];

% 默认 case 用于严格跨工况泛化；可改为 point 或 mixed_edge 做诊断
cfg.splitMode = 'case';
cfg.valRatio = 0.15;
cfg.testRatio = 0.15;
cfg.randomSeed = 2026;

% Step21 mixed-edge split:
% 目标：训练集中保留 90cm 边界样本，测试集固定留出 95cm 极边界工况。
% 这不是随机泛化评估，而是边界外推专项评估。
cfg.mixedEdgeTestKeyword = "95cm";
cfg.mixedEdgeMustTrainKeyword = "90cm";
cfg.mixedEdgeValCaseList = [ ...
    "20kg单点中心加载", ...
    "10kg双点对称加载", ...
    "40kg纵中心线30cm加载" ...
];


cfg.useMirrorAugmentation = true;
cfg.augmentMirrorX = true;
cfg.augmentMirrorY = false;

% FiLM 结构
cfg.spaceHiddenUnits = 96;
cfg.numFiLMLayers = 5;
cfg.conditionHiddenUnits = 64;
cfg.fourierFreqsXY = [1 2 4 8];
cfg.fourierFreqsRel = [1 2 4];

% 训练
cfg.maxEpochs = 1800;
cfg.miniBatchSize = 64;
cfg.initialLearnRate = 4e-4;
cfg.learnRateDropPeriod = 450;
cfg.learnRateDropFactor = 0.5;
cfg.gradientDecayFactor = 0.9;
cfg.squaredGradientDecayFactor = 0.999;
cfg.earlyStopPatience = 240;
cfg.printEveryEpoch = 20;

% Loss
cfg.lambdaL2 = 1e-6;
cfg.lambdaSlope = 1e-7;
cfg.lambdaCurvReg = 5e-8;

cfg.useLoadWeightedLoss = true;
cfg.loadWeightOffset = 0.15;
cfg.loadWeightPower = 0.5;
cfg.maxLoadWeight = 3.0;

% 只做补充统计，不删除数据
cfg.extremeBoundaryKeyword = "95cm";

% Step21 边界感知设置
% boundaryFeatureLengthNorm 控制 exp(-d/scale) 类特征的衰减长度，d 为归一化边界距离。
cfg.useBoundaryDistanceFeatures = true;
cfg.boundaryFeatureLengthNorm = 0.18;
cfg.boundarySoftMinTau = 0.03;

% 边界样本加权：提升靠近边界加载工况在 data loss 中的影响，但不会改变验证/测试。
cfg.useBoundaryWeightedLoss = true;
cfg.boundaryWeightStrength = 0.75;
cfg.boundaryWeightScale = 0.18;
cfg.maxBoundaryWeight = 2.0;

% 用加载点最近边界距离识别极边界工况；0.21 大致覆盖 x=95cm 及更靠边位置。
cfg.extremeBoundaryThresholdN = 0.21;

% Step21 边界残差专家分支
% 最终位移代理场：w = w_global + gate(load_edge_distance) * w_edge_residual
% 其中 gate 只由加载点到边界的距离控制，不直接依赖测点坐标。
cfg.useBoundaryResidualExpert = true;
cfg.edgeGateCenter = 0.30;       % load_edge_min 小于该值时边界专家明显激活
cfg.edgeGateWidth  = 0.055;      % gate 过渡宽度
cfg.edgeGateMax    = 1.0;
cfg.edgeBranchInitScale = 0.10;  % 边界分支初始输出较小，避免一开始破坏全局分支
cfg.lambdaEdgeAway = 1.0e-5;     % 约束边界残差分支远离边界时保持弱影响
cfg.lambdaEdgeMag  = 1.0e-8;     % 很弱的边界残差位移幅值正则

% 连续场输出
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

%% ------------------------- 1. 输出目录与日志 -----------------------------
dirs.base=cfg.outBase; dirs.figs=fullfile(cfg.outBase,'Figs');
dirs.tables=fullfile(cfg.outBase,'Tables'); dirs.models=fullfile(cfg.outBase,'Models');
dirs.logs=fullfile(cfg.outBase,'Logs');
makeDir(dirs.base); makeDir(dirs.figs); makeDir(dirs.tables); makeDir(dirs.models); makeDir(dirs.logs);
logFile=fullfile(dirs.logs,'Training_Log_CN.txt');
fidLog=fopen(logFile,'w');
assert(fidLog>0,'无法创建日志文件。');

logBoth(fidLog,'============================================================');
logBoth(fidLog,'Step21 边界残差专家 FiLM 条件调制 AD-DNF 位移神经场');
logBoth(fidLog,'============================================================');
logBoth(fidLog,'输出文件夹：%s',cfg.outBase);
logBoth(fidLog,'数据划分方式：%s',cfg.splitMode);
logBoth(fidLog,'训练层：%s',strjoin(cfg.trainLayers,', '));
logBoth(fidLog,'FiLM层数=%d, 空间隐藏维度=%d, 条件隐藏维度=%d',cfg.numFiLMLayers,cfg.spaceHiddenUnits,cfg.conditionHiddenUnits);
logBoth(fidLog,'边界距离特征：%d，边界样本加权：%d，极边界阈值=%.3f。',cfg.useBoundaryDistanceFeatures,cfg.useBoundaryWeightedLoss,cfg.extremeBoundaryThresholdN);
logBoth(fidLog,'边界残差专家：%d，gateCenter=%.3f，gateWidth=%.3f，lambdaEdgeAway=%.3g。',cfg.useBoundaryResidualExpert,cfg.edgeGateCenter,cfg.edgeGateWidth,cfg.lambdaEdgeAway);
logBoth(fidLog,'关键修复：空间特征在 dlfeval 内由 xN/yN 动态构造，保证自动微分链条不断开。');

if ~license('test','Neural_Network_Toolbox')
    fclose(fidLog);
    error('需要 MATLAB Deep Learning Toolbox。');
end
rng(cfg.randomSeed);

%% ------------------------- 2. 读取数据 -----------------------------------
assert(isfile(cfg.inputCSV),'找不到输入文件：%s',cfg.inputCSV);
T0=readtable(cfg.inputCSV,'TextType','string');

need=["case_id","sensor_id","x_cm","y_cm","layer","direction","delta_lambda_resp_nm","load_type","load_kg","p1_x","p1_y","p2_x","p2_y"];
miss=setdiff(need,string(T0.Properties.VariableNames));
assert(isempty(miss),'缺少列：%s',strjoin(miss,', '));

T0.case_id=string(T0.case_id);
T0.sensor_id=string(T0.sensor_id);
T0.layer=string(T0.layer);
T0.direction=string(T0.direction);
T0.load_type=string(T0.load_type);

numCols=["x_cm","y_cm","load_kg","p1_x","p1_y","p2_x","p2_y","delta_lambda_resp_nm"];
for k=1:numel(numCols)
    T0.(numCols(k))=double(T0.(numCols(k)));
end

m425=T0.case_id=="40kg纵中心线42.5cm加载";
T0.p1_x(m425)=42.5;

valid=isfinite(T0.x_cm)&isfinite(T0.y_cm)&isfinite(T0.load_kg)&isfinite(T0.p1_x)&isfinite(T0.p1_y)& ...
      isfinite(T0.delta_lambda_resp_nm)&ismember(T0.layer,cfg.trainLayers);
T0=T0(valid,:);
T0.strain_ue=T0.delta_lambda_resp_nm./cfg.k_eps_nm_per_ue;

logBoth(fidLog,'数据读取完成：%d 行，%d 个工况。',height(T0),numel(unique(T0.case_id)));
logBoth(fidLog,'荷载类型：%s。',strjoin(unique(T0.load_type),', '));

%% ------------------------- 3. 划分与增强 ---------------------------------
[trainIdx0,valIdx0,testIdx0,splitInfo]=splitDataByMode(T0,cfg);
Ttrain=T0(trainIdx0,:);
Tval=T0(valIdx0,:);
Ttest=T0(testIdx0,:);

logBoth(fidLog,'原始训练/验证/测试样本数：%d / %d / %d。',height(Ttrain),height(Tval),height(Ttest));
logBoth(fidLog,'训练/验证/测试工况数：%d / %d / %d。',numel(splitInfo.trainCases),numel(splitInfo.valCases),numel(splitInfo.testCases));

if cfg.useMirrorAugmentation
    Ttrain=augmentTrainingTable(Ttrain,cfg);
    logBoth(fidLog,'启用镜像增强后训练样本数：%d。',height(Ttrain));
end

%% ------------------------- 4. 特征信息与数据结构 -------------------------
featureInfo=buildFeatureInfo([Ttrain;Tval;Ttest],cfg);

trainData=tableToDataStruct(Ttrain,cfg,featureInfo,[]);
valData=tableToDataStruct(Tval,cfg,featureInfo,[]);
testData=tableToDataStruct(Ttest,cfg,featureInfo,[]);
allData=tableToDataStruct(T0,cfg,featureInfo,[]);

epsMu=mean(trainData.epsMeasured_ue,'omitnan');
epsStd=std(trainData.epsMeasured_ue,'omitnan');
if epsStd<1e-12, epsStd=1; end

trainData.epsNorm=(trainData.epsMeasured_ue-epsMu)./epsStd;
valData.epsNorm=(valData.epsMeasured_ue-epsMu)./epsStd;
testData.epsNorm=(testData.epsMeasured_ue-epsMu)./epsStd;
allData.epsNorm=(allData.epsMeasured_ue-epsMu)./epsStd;

logBoth(fidLog,'空间特征维度：%d；条件特征维度：%d。',featureInfo.numSpaceFeatures,featureInfo.numConditionFeatures);
logBoth(fidLog,'应变归一化参数：均值 %.6g ue，标准差 %.6g ue。',epsMu,epsStd);

writetable(table((1:numel(featureInfo.spaceFeatureNames))',featureInfo.spaceFeatureNames(:),'VariableNames',{'idx','space_feature'}),fullfile(dirs.tables,'Space_Feature_Names.csv'));
writetable(table((1:numel(featureInfo.conditionFeatureNames))',featureInfo.conditionFeatureNames(:),'VariableNames',{'idx','condition_feature'}),fullfile(dirs.tables,'Condition_Feature_Names.csv'));

%% ------------------------- 5. 初始化 FiLM 参数 ----------------------------
params=initializeFiLMParams(featureInfo.numSpaceFeatures,featureInfo.numConditionFeatures,cfg);
logBoth(fidLog,'FiLM 参数初始化完成。');

%% ------------------------- 6. 训练 ---------------------------------------
numTrain=numel(trainData.epsMeasured_ue);
iters=max(1,ceil(numTrain/cfg.miniBatchSize));

avg=[];
avgSq=[];
bestVal=inf;
bestParams=params;
bestEpoch=0;
bad=0;

history=table('Size',[cfg.maxEpochs,14],'VariableTypes',repmat("double",1,14), ...
    'VariableNames',{'epoch','learn_rate','train_loss_total','train_loss_data','train_loss_slope','train_loss_curv','train_loss_edge_away','train_loss_edge_mag','train_loss_l2','val_loss_data','train_R2','val_R2','val_RMSE_ue','elapsed_sec'});

tStart=tic;
logBoth(fidLog,'开始训练。');

for epoch=1:cfg.maxEpochs
    lr=cfg.initialLearnRate*cfg.learnRateDropFactor^floor((epoch-1)/cfg.learnRateDropPeriod);
    order=randperm(numTrain);

    epTot=0; epData=0; epSlope=0; epCurv=0; epEdgeAway=0; epEdgeMag=0; epL2=0;

    for it=1:iters
        b1=(it-1)*cfg.miniBatchSize+1;
        b2=min(it*cfg.miniBatchSize,numTrain);
        ids=order(b1:b2);

        batch=makeDLBatch(sliceDataStruct(trainData,ids));

        [loss,grads,parts]=dlfeval(@modelLossFiLM,params,batch,cfg,epsMu,epsStd);

        giter=(epoch-1)*iters+it;
        [params,avg,avgSq]=adamupdate(params,grads,avg,avgSq,giter,lr,cfg.gradientDecayFactor,cfg.squaredGradientDecayFactor);

        nb=numel(ids);
        epTot=epTot+double(gather(extractdata(loss)))*nb;
        epData=epData+double(gather(extractdata(parts.dataLoss)))*nb;
        epSlope=epSlope+double(gather(extractdata(parts.slopeLoss)))*nb;
        epCurv=epCurv+double(gather(extractdata(parts.curvLoss)))*nb;
        epEdgeAway=epEdgeAway+double(gather(extractdata(parts.edgeAwayLoss)))*nb;
        epEdgeMag=epEdgeMag+double(gather(extractdata(parts.edgeMagLoss)))*nb;
        epL2=epL2+double(gather(extractdata(parts.l2Loss)))*nb;
    end

    epTot=epTot/numTrain;
    epData=epData/numTrain;
    epSlope=epSlope/numTrain;
    epCurv=epCurv/numTrain;
    epL2=epL2/numTrain;

    pTrain=predictDatasetStrainFiLM(params,trainData,cfg,512);
    pVal=predictDatasetStrainFiLM(params,valData,cfg,512);

    trainR2=calcR2(trainData.epsMeasured_ue(:),pTrain.epsPred_ue(:));
    valR2=calcR2(valData.epsMeasured_ue(:),pVal.epsPred_ue(:));
    valRMSE=sqrt(mean((pVal.epsPred_ue(:)-valData.epsMeasured_ue(:)).^2,'omitnan'));
    valLoss=mean(((pVal.epsPred_ue(:)-epsMu)./epsStd-valData.epsNorm(:)).^2,'omitnan');

    history(epoch,:)={epoch,lr,epTot,epData,epSlope,epCurv,epEdgeAway,epEdgeMag,epL2,valLoss,trainR2,valR2,valRMSE,toc(tStart)};

    if valLoss<bestVal
        bestVal=valLoss;
        bestParams=params;
        bestEpoch=epoch;
        bad=0;
    else
        bad=bad+1;
    end

    if mod(epoch,cfg.printEveryEpoch)==0 || epoch==1
        logBoth(fidLog,'Epoch %4d | lr %.3g | total %.6g | data %.6g | slope %.3g | curv %.3g | Train R2 %.4f | Val R2 %.4f | Val RMSE %.4g ue | Best %d', ...
            epoch,lr,epTot,epData,epSlope,epCurv,trainR2,valR2,valRMSE,bestEpoch);
    end

    if bad>=cfg.earlyStopPatience
        logBoth(fidLog,'触发早停：连续 %d 个 epoch 验证集未改善。',cfg.earlyStopPatience);
        break;
    end
end

history=history(1:epoch,:);
params=bestParams;
writetable(history,fullfile(dirs.tables,'Training_History.csv'));
logBoth(fidLog,'训练结束。最优 Epoch=%d，最优验证 data loss=%.6g。',bestEpoch,bestVal);

%% ------------------------- 7. 评估 ---------------------------------------
pTrain=predictDatasetStrainFiLM(params,tableToDataStruct(T0(trainIdx0,:),cfg,featureInfo,epsMu,epsStd),cfg,512);
pVal=predictDatasetStrainFiLM(params,valData,cfg,512);
pTest=predictDatasetStrainFiLM(params,testData,cfg,512);
pAll=predictDatasetStrainFiLM(params,allData,cfg,512);

mTrain=calcMetrics(T0.strain_ue(trainIdx0),pTrain.epsPred_ue);
mVal=calcMetrics(T0.strain_ue(valIdx0),pVal.epsPred_ue);
mTest=calcMetrics(T0.strain_ue(testIdx0),pTest.epsPred_ue);
mAll=calcMetrics(T0.strain_ue,pAll.epsPred_ue);

MetricsOverall=table(["train";"validation";"test";"all"],[mTrain.R2;mVal.R2;mTest.R2;mAll.R2], ...
    [mTrain.RMSE;mVal.RMSE;mTest.RMSE;mAll.RMSE],[mTrain.MAE;mVal.MAE;mTest.MAE;mAll.MAE],[mTrain.Bias;mVal.Bias;mTest.Bias;mAll.Bias], ...
    'VariableNames',{'split','R2','RMSE_ue','MAE_ue','Bias_ue'});
writetable(MetricsOverall,fullfile(dirs.tables,'Overall_Metrics.csv'));

p1EdgeMinAll=min([T0.p1_x'./cfg.plateLength_cm; 1-T0.p1_x'./cfg.plateLength_cm; ...
                  T0.p1_y'./cfg.plateWidth_cm;  1-T0.p1_y'./cfg.plateWidth_cm],[],1)';
p2ValidAll=isfinite(T0.p2_x) & isfinite(T0.p2_y);
p2EdgeMinAll=nan(height(T0),1);
p2EdgeMinAll(p2ValidAll)=min([T0.p2_x(p2ValidAll)'./cfg.plateLength_cm; 1-T0.p2_x(p2ValidAll)'./cfg.plateLength_cm; ...
                              T0.p2_y(p2ValidAll)'./cfg.plateWidth_cm;  1-T0.p2_y(p2ValidAll)'./cfg.plateWidth_cm],[],1)';
loadEdgeMinAll=p1EdgeMinAll;
loadEdgeMinAll(p2ValidAll)=min(p1EdgeMinAll(p2ValidAll),p2EdgeMinAll(p2ValidAll));
isExtreme=loadEdgeMinAll<=cfg.extremeBoundaryThresholdN | contains(T0.case_id,cfg.extremeBoundaryKeyword);
isBoundary=loadEdgeMinAll<=0.25 | contains(T0.case_id,"90cm") | contains(T0.case_id,"95cm");
MetricsNoExtreme=table();
if any(isExtreme)
    mtNo=calcMetrics(T0.strain_ue(~isExtreme),pAll.epsPred_ue(~isExtreme));
    MetricsNoExtreme=table("all_excluding_extreme_boundary",mtNo.R2,mtNo.RMSE,mtNo.MAE,mtNo.Bias,'VariableNames',{'scope','R2','RMSE_ue','MAE_ue','Bias_ue'});
    writetable(MetricsNoExtreme,fullfile(dirs.tables,'Metrics_Excluding_ExtremeBoundary.csv'));
end

logBoth(fidLog,'总体指标：');
logBoth(fidLog,'  Train      R2=%.4f, RMSE=%.6g ue',mTrain.R2,mTrain.RMSE);
logBoth(fidLog,'  Validation R2=%.4f, RMSE=%.6g ue',mVal.R2,mVal.RMSE);
logBoth(fidLog,'  Test       R2=%.4f, RMSE=%.6g ue',mTest.R2,mTest.RMSE);
logBoth(fidLog,'  All        R2=%.4f, RMSE=%.6g ue',mAll.R2,mAll.RMSE);
if ~isempty(MetricsNoExtreme)
    logBoth(fidLog,'  All excluding extreme-boundary R2=%.4f, RMSE=%.6g ue',mtNo.R2,mtNo.RMSE);
end

PointPred=T0(:,["case_id","sensor_id","x_cm","y_cm","layer","direction","load_type","load_kg","p1_x","p1_y","p2_x","p2_y","delta_lambda_resp_nm","strain_ue"]);
PointPred.pred_strain_ue=pAll.epsPred_ue(:);
PointPred.pred_dlambda_nm=PointPred.pred_strain_ue.*cfg.k_eps_nm_per_ue;
PointPred.pred_w_um=pAll.wPred_m(:).*1e6;
PointPred.pred_w_global_um=pAll.wGlobal_m(:).*1e6;
PointPred.pred_w_edge_residual_um=pAll.wEdgeResidual_m(:).*1e6;
PointPred.edge_gate=pAll.edgeGate(:);
PointPred.pred_eps_x_ue=pAll.epsX_ue(:);
PointPred.pred_eps_y_ue=pAll.epsY_ue(:);
PointPred.residual_strain_ue=PointPred.pred_strain_ue-PointPred.strain_ue;
PointPred.split=strings(height(T0),1);
PointPred.split(trainIdx0)="train";
PointPred.split(valIdx0)="validation";
PointPred.split(testIdx0)="test";
PointPred.load_edge_min_norm=loadEdgeMinAll;
PointPred.is_boundary=isBoundary;
PointPred.is_extreme_boundary=isExtreme;
writetable(PointPred,fullfile(dirs.tables,'Pointwise_Predictions.csv'));

caseList=unique(T0.case_id,'stable');
CaseMetrics=table();
for i=1:numel(caseList)
    cid=caseList(i);
    mk=T0.case_id==cid;
    mt=calcMetrics(T0.strain_ue(mk),pAll.epsPred_ue(mk));
    one=table(cid,unique(T0.load_type(mk),'stable'),unique(T0.load_kg(mk),'stable'),sum(mk), ...
        mean(loadEdgeMinAll(mk),'omitnan'),any(isBoundary(mk)),any(isExtreme(mk)), ...
        mt.R2,mt.RMSE,mt.MAE,mt.Bias, ...
        'VariableNames',{'case_id','load_type','load_kg','n_points','load_edge_min_norm','is_boundary','is_extreme_boundary','R2','RMSE_ue','MAE_ue','Bias_ue'});
    CaseMetrics=[CaseMetrics;one]; %#ok<AGROW>
end
writetable(CaseMetrics,fullfile(dirs.tables,'Case_Metrics.csv'));

%% ------------------------- 8. 绘图 ---------------------------------------
plotTrainingHistory(history,dirs.figs);
plotParity(PointPred,dirs.figs);
plotResiduals(PointPred,dirs.figs);
plotCaseMetrics(CaseMetrics,dirs.figs);
plotBoundaryDiagnostics(CaseMetrics,dirs.figs);

%% ------------------------- 9. 连续场输出 ---------------------------------
logBoth(fidLog,'开始输出连续位移场和自动微分应变场。');
FieldSummary=table();
fieldCounter=0;

for ic=1:numel(cfg.fieldCaseList)
    cid=cfg.fieldCaseList(ic);
    if ~any(T0.case_id==cid)
        logBoth(fidLog,'跳过连续场输出：工况不存在：%s',cid);
        continue;
    end
    refRow=T0(find(T0.case_id==cid,1),:);

    for il=1:numel(cfg.fieldLayers)
        lay=cfg.fieldLayers(il);
        if ~any(T0.case_id==cid & T0.layer==lay)
            continue;
        end

        fieldCounter=fieldCounter+1;
        G=makeGridTable(refRow,lay,cfg);
        Gdata=tableToDataStruct(G,cfg,featureInfo,epsMu,epsStd);
        Pgrid=predictDatasetStrainFiLM(params,Gdata,cfg,512);

        nx=cfg.gridNX;
        ny=cfg.gridNY;
        Xg=reshape(G.x_cm,[ny,nx]);
        Yg=reshape(G.y_cm,[ny,nx]);
        Wum=reshape(Pgrid.wPred_m.*1e6,[ny,nx]);
        WGlobalum=reshape(Pgrid.wGlobal_m.*1e6,[ny,nx]);
        WEdgeum=reshape(Pgrid.wEdgeResidual_m.*1e6,[ny,nx]);
        GateGrid=reshape(Pgrid.edgeGate,[ny,nx]);
        epsX=reshape(Pgrid.epsX_ue,[ny,nx]);
        epsY=reshape(Pgrid.epsY_ue,[ny,nx]);

        safeName=sprintf('Field_%02d_%s_%s',fieldCounter,englishCaseFileName(refRow),char(lay));
        csvName=fullfile(dirs.tables,safeName+".csv");

        writetable(table(G.x_cm,G.y_cm,repmat(cid,height(G),1),repmat(lay,height(G),1),Pgrid.wPred_m(:).*1e6,Pgrid.wGlobal_m(:).*1e6,Pgrid.wEdgeResidual_m(:).*1e6,Pgrid.edgeGate(:),Pgrid.epsX_ue(:),Pgrid.epsY_ue(:), ...
            'VariableNames',{'x_cm','y_cm','case_id','layer','w_proxy_um','w_global_um','w_edge_residual_um','edge_gate','eps_x_ue','eps_y_ue'}),csvName);

        ttl=englishCaseTitle(refRow,lay);
        plotFieldContour(Xg,Yg,Wum,'Boundary-Residual FiLM Displacement Neural Field','X (cm)','Y (cm)','Displacement Proxy w (um)',ttl,fullfile(dirs.figs,safeName+"_W_DisplacementProxy.png"));
        plotFieldContour(Xg,Yg,WGlobalum,'Global Branch Displacement Field','X (cm)','Y (cm)','Global Displacement Proxy w_g (um)',ttl,fullfile(dirs.figs,safeName+"_W_Global.png"));
        plotFieldContour(Xg,Yg,WEdgeum,'Boundary Residual Branch Displacement Field','X (cm)','Y (cm)','Boundary Residual Proxy w_e (um)',ttl,fullfile(dirs.figs,safeName+"_W_EdgeResidual.png"));
        plotFieldContour(Xg,Yg,GateGrid,'Boundary Expert Gate','X (cm)','Y (cm)','Gate Value',ttl,fullfile(dirs.figs,safeName+"_EdgeGate.png"));
        plotFieldContour(Xg,Yg,epsX,'FiLM Autodiff Longitudinal Strain Field','X (cm)','Y (cm)','Longitudinal Strain epsilon_x (microstrain)',ttl,fullfile(dirs.figs,safeName+"_EpsX.png"));
        plotFieldContour(Xg,Yg,epsY,'FiLM Autodiff Transverse Strain Field','X (cm)','Y (cm)','Transverse Strain epsilon_y (microstrain)',ttl,fullfile(dirs.figs,safeName+"_EpsY.png"));

        refEdgeMin=min([refRow.p1_x/cfg.plateLength_cm,1-refRow.p1_x/cfg.plateLength_cm,refRow.p1_y/cfg.plateWidth_cm,1-refRow.p1_y/cfg.plateWidth_cm]);
        refExtreme=refEdgeMin<=cfg.extremeBoundaryThresholdN || contains(cid,cfg.extremeBoundaryKeyword);
        FieldSummary=[FieldSummary;table(cid,lay,refEdgeMin,refExtreme,max(abs(Wum(:))),max(abs(WGlobalum(:))),max(abs(WEdgeum(:))),mean(GateGrid(:),'omitnan'),max(abs(epsX(:))),max(abs(epsY(:))),csvName, ...
            'VariableNames',{'case_id','layer','load_edge_min_norm','is_extreme_boundary','max_abs_w_um','max_abs_w_global_um','max_abs_w_edge_residual_um','mean_edge_gate','max_abs_eps_x_ue','max_abs_eps_y_ue','csv_path'})]; %#ok<AGROW>

        logBoth(fidLog,'已输出连续场：%s | %s',cid,lay);
    end
end

writetable(FieldSummary,fullfile(dirs.tables,'Field_Summary.csv'));

%% ------------------------- 10. 保存 --------------------------------------
modelFile=fullfile(dirs.models,'Step21_BoundaryResidualExpert_FiLM_AD_DNF_Model.mat');
save(modelFile,'params','bestParams','cfg','featureInfo','epsMu','epsStd','MetricsOverall','MetricsNoExtreme','CaseMetrics','history','splitInfo');

reportFile=fullfile(dirs.logs,'Analysis_Report_CN.txt');
writeChineseReport(reportFile,cfg,MetricsOverall,MetricsNoExtreme,CaseMetrics,FieldSummary,modelFile,bestEpoch,bestVal);

logBoth(fidLog,'模型已保存：%s',modelFile);
logBoth(fidLog,'中文分析报告已保存：%s',reportFile);
logBoth(fidLog,'全部完成。');
fclose(fidLog);

disp(' ');
disp('==================== Step21 Boundary Residual Expert FiLM AD-DNF 完成 ====================');
disp(['输出文件夹：',cfg.outBase]);

%% =========================================================================
% 本地函数
% =========================================================================

function makeDir(p)
if ~exist(p,'dir'), mkdir(p); end
end

function logBoth(fid,fmt,varargin)
msg=sprintf(fmt,varargin{:});
fprintf('%s\n',msg);
fprintf(fid,'%s\n',msg);
end

function [trainIdx,valIdx,testIdx,splitInfo]=splitDataByMode(T,cfg)
n=height(T);
trainIdx=false(n,1); valIdx=false(n,1); testIdx=false(n,1);

switch lower(cfg.splitMode)
    case 'case'
        cases=unique(T.case_id,'stable');
        nc=numel(cases);
        order=randperm(nc);
        nTest=max(1,round(cfg.testRatio*nc));
        nVal=max(1,round(cfg.valRatio*nc));
        if nc-nVal-nTest<1
            nTest=1; nVal=1;
        end
        testCases=cases(order(1:nTest));
        valCases=cases(order(nTest+1:nTest+nVal));
        trainCases=setdiff(cases,[testCases;valCases],'stable');

        trainIdx=ismember(T.case_id,trainCases);
        valIdx=ismember(T.case_id,valCases);
        testIdx=ismember(T.case_id,testCases);

    case 'point'
        order=randperm(n);
        nTest=max(1,round(cfg.testRatio*n));
        nVal=max(1,round(cfg.valRatio*n));
        testIdx(order(1:nTest))=true;
        valIdx(order(nTest+1:nTest+nVal))=true;
        trainIdx=~(valIdx|testIdx);

        trainCases=unique(T.case_id(trainIdx),'stable');
        valCases=unique(T.case_id(valIdx),'stable');
        testCases=unique(T.case_id(testIdx),'stable');

    case 'mixed_edge'
        cases=unique(T.case_id,'stable');

        testCases=cases(contains(cases,cfg.mixedEdgeTestKeyword));
        if isempty(testCases)
            error('mixed_edge 未找到测试工况关键词：%s',cfg.mixedEdgeTestKeyword);
        end

        mustTrainCases=cases(contains(cases,cfg.mixedEdgeMustTrainKeyword));

        valCases=strings(0,1);
        for kk=1:numel(cfg.mixedEdgeValCaseList)
            hit=cases(cases==cfg.mixedEdgeValCaseList(kk));
            if ~isempty(hit)
                valCases=[valCases;hit(:)]; %#ok<AGROW>
            end
        end
        valCases=unique(valCases,'stable');

        % 避免验证集与测试集或必须训练边界工况冲突。
        valCases=setdiff(valCases,[testCases;mustTrainCases],'stable');

        % 如果指定验证集没有命中，则从非测试、非90cm工况中稳定选取少量验证工况。
        if isempty(valCases)
            candidates=setdiff(cases,[testCases;mustTrainCases],'stable');
            nVal=max(1,min(3,round(cfg.valRatio*numel(candidates))));
            valCases=candidates(1:nVal);
        end

        trainCases=setdiff(cases,[testCases;valCases],'stable');

        % 强制确认 90cm 之类相邻边界工况进入训练集。
        if ~isempty(mustTrainCases)
            trainCases=unique([trainCases;mustTrainCases],'stable');
            valCases=setdiff(valCases,mustTrainCases,'stable');
            testCases=setdiff(testCases,mustTrainCases,'stable');
        end

        trainIdx=ismember(T.case_id,trainCases);
        valIdx=ismember(T.case_id,valCases);
        testIdx=ismember(T.case_id,testCases);

    otherwise
        error('cfg.splitMode 只能是 case、point 或 mixed_edge。');
end

splitInfo=struct('trainCases',trainCases,'valCases',valCases,'testCases',testCases);
end

function Tout=augmentTrainingTable(Tin,cfg)
Tout=Tin;
L=cfg.plateLength_cm;
W=cfg.plateWidth_cm;
aug={};

if cfg.augmentMirrorX
    Tx=Tin;
    Tx.x_cm=L-Tx.x_cm;
    Tx.p1_x=L-Tx.p1_x;
    m=isfinite(Tx.p2_x);
    Tx.p2_x(m)=L-Tx.p2_x(m);
    Tx.case_id=Tx.case_id+"_aug_mirrorX";
    aug{end+1}=Tx; %#ok<AGROW>
end

if cfg.augmentMirrorY
    Ty=Tin;
    Ty.y_cm=W-Ty.y_cm;
    Ty.p1_y=W-Ty.p1_y;
    m=isfinite(Ty.p2_y);
    Ty.p2_y(m)=W-Ty.p2_y(m);
    Ty.case_id=Ty.case_id+"_aug_mirrorY";
    aug{end+1}=Ty; %#ok<AGROW>
end

for i=1:numel(aug)
    Tout=[Tout;aug{i}]; %#ok<AGROW>
end
end

function featureInfo=buildFeatureInfo(T,cfg)
featureInfo=struct();
featureInfo.loadTypes=unique(T.load_type,'stable');
featureInfo.fourierFreqsXY=cfg.fourierFreqsXY;
featureInfo.fourierFreqsRel=cfg.fourierFreqsRel;
featureInfo.boundaryFeatureLengthNorm=cfg.boundaryFeatureLengthNorm;
featureInfo.boundarySoftMinTau=cfg.boundarySoftMinTau;

dummy=tableToDataStruct(T(1:min(2,height(T)),:),cfg,featureInfo,[]);
featureInfo.spaceFeatureNames=dummy.spaceFeatureNames;
featureInfo.conditionFeatureNames=dummy.conditionFeatureNames;
featureInfo.numSpaceFeatures=numel(dummy.spaceFeatureNames);
featureInfo.numConditionFeatures=numel(dummy.conditionFeatureNames);
end

function data=tableToDataStruct(T,cfg,featureInfo,epsMu,epsStd)
L=cfg.plateLength_cm;
W=cfg.plateWidth_cm;

x=double(T.x_cm);
y=double(T.y_cm);
p1x=double(T.p1_x);
p1y=double(T.p1_y);
p2x=double(T.p2_x);
p2y=double(T.p2_y);

hasP2=isfinite(p2x)&isfinite(p2y);
p2x(~hasP2)=p1x(~hasP2);
p2y(~hasP2)=p1y(~hasP2);

maxLoad=80;
loadN=double(T.load_kg)./maxLoad;

xN=x./L;
yN=y./W;
p1xN=p1x./L;
p1yN=p1y./W;
p2xN=p2x./L;
p2yN=p2y./W;

dirLong=double(T.direction=="纵");
epsMeasured=double(T.strain_ue);

if nargin>=5 && ~isempty(epsMu)
    epsNorm=(epsMeasured-epsMu)./epsStd;
else
    epsNorm=nan(size(epsMeasured));
end

onehot=zeros(numel(featureInfo.loadTypes),height(T));
for k=1:numel(featureInfo.loadTypes)
    onehot(k,:)=double(T.load_type==featureInfo.loadTypes(k))';
end

if cfg.useLoadWeightedLoss
    lw=1./((loadN(:)'+cfg.loadWeightOffset).^cfg.loadWeightPower);
    lw=min(lw,cfg.maxLoadWeight);
    lw=lw./mean(lw,'omitnan');
else
    lw=ones(1,height(T));
end

% Step21: 按加载点到最近边界距离进行温和加权，重点照顾边界加载工况。
p1EdgeMin=min([p1xN(:)';1-p1xN(:)';p1yN(:)';1-p1yN(:)'],[],1);
p2EdgeMin=min([p2xN(:)';1-p2xN(:)';p2yN(:)';1-p2yN(:)'],[],1);
loadEdgeMin=p1EdgeMin;
loadEdgeMin(hasP2(:)')=min(p1EdgeMin(hasP2(:)'),p2EdgeMin(hasP2(:)'));
if isfield(cfg,'useBoundaryWeightedLoss') && cfg.useBoundaryWeightedLoss
    bw=1+cfg.boundaryWeightStrength.*exp(-loadEdgeMin./cfg.boundaryWeightScale);
    bw=min(bw,cfg.maxBoundaryWeight);
    bw=bw./mean(bw,'omitnan');
    lw=lw.*bw;
    lw=lw./mean(lw,'omitnan');
end

[~,spaceNames]=spaceFeaturesNumeric(xN(:)',yN(:)',p1xN(:)',p1yN(:)',p2xN(:)',p2yN(:)',hasP2(:)',cfg,featureInfo);
[~,condNames]=condFeaturesNumeric(xN(:)',yN(:)',loadN(:)',p1xN(:)',p1yN(:)',p2xN(:)',p2yN(:)',hasP2(:)',onehot,featureInfo);

data=struct();
data.xN=single(xN(:)');
data.yN=single(yN(:)');
data.p1xN=single(p1xN(:)');
data.p1yN=single(p1yN(:)');
data.p2xN=single(p2xN(:)');
data.p2yN=single(p2yN(:)');
data.hasP2=single(hasP2(:)');
data.loadN=single(loadN(:)');
data.onehotLoadType=single(onehot);
data.dirLong=single(dirLong(:)');
data.epsMeasured_ue=single(epsMeasured(:)');
data.epsNorm=single(epsNorm(:)');
data.loadWeight=single(lw);
data.p1EdgeMinN=single(p1EdgeMin);
data.p2EdgeMinN=single(p2EdgeMin);
data.loadEdgeMinN=single(loadEdgeMin);
data.spaceFeatureNames=spaceNames;
data.conditionFeatureNames=condNames;
end

function batch=sliceDataStruct(data,idx)
batch=data;
batch.xN=data.xN(idx);
batch.yN=data.yN(idx);
batch.p1xN=data.p1xN(idx);
batch.p1yN=data.p1yN(idx);
batch.p2xN=data.p2xN(idx);
batch.p2yN=data.p2yN(idx);
batch.hasP2=data.hasP2(idx);
batch.loadN=data.loadN(idx);
batch.onehotLoadType=data.onehotLoadType(:,idx);
batch.dirLong=data.dirLong(idx);
batch.epsMeasured_ue=data.epsMeasured_ue(idx);
batch.epsNorm=data.epsNorm(idx);
batch.loadWeight=data.loadWeight(idx);
if isfield(data,'p1EdgeMinN')
    batch.p1EdgeMinN=data.p1EdgeMinN(idx);
    batch.p2EdgeMinN=data.p2EdgeMinN(idx);
    batch.loadEdgeMinN=data.loadEdgeMinN(idx);
end
end

function b=makeDLBatch(batch)
b=batch;
% 无标签 dlarray，避免 W*x 标签冲突；同时保留 xN/yN 的 tracked 计算图。
b.xN=dlarray(single(batch.xN));
b.yN=dlarray(single(batch.yN));
b.p1xN=dlarray(single(batch.p1xN));
b.p1yN=dlarray(single(batch.p1yN));
b.p2xN=dlarray(single(batch.p2xN));
b.p2yN=dlarray(single(batch.p2yN));
b.hasP2=dlarray(single(batch.hasP2));
b.loadN=dlarray(single(batch.loadN));
b.onehotLoadType=dlarray(single(batch.onehotLoadType));
b.dirLong=dlarray(single(batch.dirLong));
b.epsNorm=dlarray(single(batch.epsNorm));
b.loadWeight=dlarray(single(batch.loadWeight));
end

function [X,names]=spaceFeaturesNumeric(xN,yN,p1xN,p1yN,p2xN,p2yN,hasP2,cfg,featureInfo)
dx1=xN-p1xN;
dy1=yN-p1yN;
dx2=xN-p2xN;
dy2=yN-p2yN;
d1=sqrt(dx1.^2+dy1.^2+1e-8);
d2=sqrt(dx2.^2+dy2.^2+1e-8);

tau=cfg.boundarySoftMinTau;
edgeSoft=-tau.*log(exp(-xN./tau)+exp(-(1-xN)./tau)+exp(-yN./tau)+exp(-(1-yN)./tau));
edgeExp=exp(-edgeSoft./cfg.boundaryFeatureLengthNorm);

X=[xN;yN;dx1;dy1;dx2;dy2;d1;d2;xN;1-xN;yN;1-yN;edgeSoft;edgeExp;double(hasP2)];
names=["x_norm";"y_norm";"dx1";"dy1";"dx2";"dy2";"dist1";"dist2";"dist_left";"dist_right";"dist_front";"dist_back";"sensor_edge_softmin";"sensor_edge_exp";"hasP2"];

for f=featureInfo.fourierFreqsXY
    X=[X;sin(2*pi*f*xN);cos(2*pi*f*xN);sin(2*pi*f*yN);cos(2*pi*f*yN)]; %#ok<AGROW>
    names=[names;sprintf("sin_x_%d",f);sprintf("cos_x_%d",f);sprintf("sin_y_%d",f);sprintf("cos_y_%d",f)]; %#ok<AGROW>
end

for f=featureInfo.fourierFreqsRel
    X=[X;sin(2*pi*f*dx1);cos(2*pi*f*dx1);sin(2*pi*f*dy1);cos(2*pi*f*dy1);sin(2*pi*f*dx2);cos(2*pi*f*dx2);sin(2*pi*f*dy2);cos(2*pi*f*dy2)]; %#ok<AGROW>
    names=[names;sprintf("sin_dx1_%d",f);sprintf("cos_dx1_%d",f);sprintf("sin_dy1_%d",f);sprintf("cos_dy1_%d",f);sprintf("sin_dx2_%d",f);sprintf("cos_dx2_%d",f);sprintf("sin_dy2_%d",f);sprintf("cos_dy2_%d",f)]; %#ok<AGROW>
end
end

function [X,names]=condFeaturesNumeric(xN,yN,loadN,p1xN,p1yN,p2xN,p2yN,hasP2,onehot,featureInfo)
tau=featureInfo.boundarySoftMinTau;
p1EdgeMin=-tau.*log(exp(-p1xN./tau)+exp(-(1-p1xN)./tau)+exp(-p1yN./tau)+exp(-(1-p1yN)./tau));
p2EdgeRaw=-tau.*log(exp(-p2xN./tau)+exp(-(1-p2xN)./tau)+exp(-p2yN./tau)+exp(-(1-p2yN)./tau));
p2EdgeMin=p2EdgeRaw.*double(hasP2);
loadEdgeMin=p1EdgeMin;
m=logical(hasP2);
loadEdgeMin(m)=0.5.*(p1EdgeMin(m)+p2EdgeRaw(m)-sqrt((p1EdgeMin(m)-p2EdgeRaw(m)).^2+1e-6));
p1EdgeExp=exp(-p1EdgeMin./featureInfo.boundaryFeatureLengthNorm);
loadEdgeExp=exp(-loadEdgeMin./featureInfo.boundaryFeatureLengthNorm);

X=[loadN;p1xN;p1yN;p2xN;p2yN;double(hasP2); ...
   p1xN;1-p1xN;p1yN;1-p1yN;p2xN;1-p2xN;p2yN;1-p2yN; ...
   p1EdgeMin;p2EdgeMin;loadEdgeMin;p1EdgeExp;loadEdgeExp; ...
   2*p1xN-1;2*p1yN-1;loadN.*(xN-p1xN);loadN.*(yN-p1yN); ...
   loadN.*double(hasP2);onehot];
names=["load";"p1x";"p1y";"p2x";"p2y";"hasP2"; ...
       "p1_left";"p1_right";"p1_front";"p1_back";"p2_left";"p2_right";"p2_front";"p2_back"; ...
       "p1_edge_softmin";"p2_edge_softmin_masked";"load_edge_softmin";"p1_edge_exp";"load_edge_exp"; ...
       "p1_x_asym";"p1_y_asym";"load_dx1";"load_dy1";"p2_relative_strength"];

for k=1:numel(featureInfo.loadTypes)
    names=[names;"load_type_"+string(matlab.lang.makeValidName(char(featureInfo.loadTypes(k))))]; %#ok<AGROW>
end
end

function X=spaceFeaturesDL(batch,cfg)
xN=batch.xN;
yN=batch.yN;
p1xN=batch.p1xN;
p1yN=batch.p1yN;
p2xN=batch.p2xN;
p2yN=batch.p2yN;
hasP2=batch.hasP2;

dx1=xN-p1xN;
dy1=yN-p1yN;
dx2=xN-p2xN;
dy2=yN-p2yN;
d1=sqrt(dx1.^2+dy1.^2+1e-8);
d2=sqrt(dx2.^2+dy2.^2+1e-8);

tau=cfg.boundarySoftMinTau;
edgeSoft=-tau.*log(exp(-xN./tau)+exp(-(1-xN)./tau)+exp(-yN./tau)+exp(-(1-yN)./tau));
edgeExp=exp(-edgeSoft./cfg.boundaryFeatureLengthNorm);

X=[xN;yN;dx1;dy1;dx2;dy2;d1;d2;xN;1-xN;yN;1-yN;edgeSoft;edgeExp;hasP2];

for f=cfg.fourierFreqsXY
    X=[X;sin(2*pi*f*xN);cos(2*pi*f*xN);sin(2*pi*f*yN);cos(2*pi*f*yN)]; %#ok<AGROW>
end

for f=cfg.fourierFreqsRel
    X=[X;sin(2*pi*f*dx1);cos(2*pi*f*dx1);sin(2*pi*f*dy1);cos(2*pi*f*dy1);sin(2*pi*f*dx2);cos(2*pi*f*dx2);sin(2*pi*f*dy2);cos(2*pi*f*dy2)]; %#ok<AGROW>
end
end

function X=condFeaturesDL(batch,cfg)
xN=batch.xN;
yN=batch.yN;
loadN=batch.loadN;
p1xN=batch.p1xN;
p1yN=batch.p1yN;
p2xN=batch.p2xN;
p2yN=batch.p2yN;
hasP2=batch.hasP2;

tau=cfg.boundarySoftMinTau;
p1EdgeMin=-tau.*log(exp(-p1xN./tau)+exp(-(1-p1xN)./tau)+exp(-p1yN./tau)+exp(-(1-p1yN)./tau));
p2EdgeRaw=-tau.*log(exp(-p2xN./tau)+exp(-(1-p2xN)./tau)+exp(-p2yN./tau)+exp(-(1-p2yN)./tau));
p2EdgeMin=p2EdgeRaw.*hasP2;
smoothMin12=0.5.*(p1EdgeMin+p2EdgeRaw-sqrt((p1EdgeMin-p2EdgeRaw).^2+1e-6));
loadEdgeMin=(1-hasP2).*p1EdgeMin+hasP2.*smoothMin12;
p1EdgeExp=exp(-p1EdgeMin./cfg.boundaryFeatureLengthNorm);
loadEdgeExp=exp(-loadEdgeMin./cfg.boundaryFeatureLengthNorm);

X=[loadN;p1xN;p1yN;p2xN;p2yN;hasP2; ...
   p1xN;1-p1xN;p1yN;1-p1yN;p2xN;1-p2xN;p2yN;1-p2yN; ...
   p1EdgeMin;p2EdgeMin;loadEdgeMin;p1EdgeExp;loadEdgeExp; ...
   2*p1xN-1;2*p1yN-1;loadN.*(xN-p1xN);loadN.*(yN-p1yN); ...
   loadN.*hasP2;batch.onehotLoadType];
end

function params=initializeFiLMParams(nS,nC,cfg)
params=struct();
params.global=initializeFiLMBranch(nS,nC,cfg);
params.edge=initializeFiLMBranch(nS,nC,cfg);

% 边界残差分支初始输出较小，避免训练初期破坏全局场。
params.edge.out.W=params.edge.out.W.*cfg.edgeBranchInitScale;
params.edge.out.b=params.edge.out.b.*cfg.edgeBranchInitScale;
end

function branch=initializeFiLMBranch(nS,nC,cfg)
H=cfg.spaceHiddenUnits;
L=cfg.numFiLMLayers;
CH=cfg.conditionHiddenUnits;

branch=struct();
branch.cond.fc1.W=glorot(CH,nC);
branch.cond.fc1.b=zerodl(CH,1);
branch.cond.fc2.W=glorot(CH,CH);
branch.cond.fc2.b=zerodl(CH,1);

for i=1:L
    if i==1
        branch.space(i).W=glorot(H,nS);
    else
        branch.space(i).W=glorot(H,H);
    end
    branch.space(i).b=zerodl(H,1);

    branch.film(i).gamma.W=glorot(H,CH);
    branch.film(i).gamma.b=dlarray(ones(H,1,'single'));

    branch.film(i).beta.W=glorot(H,CH);
    branch.film(i).beta.b=zerodl(H,1);
end

branch.out.W=glorot(1,H);
branch.out.b=zerodl(1,1);
end

function A=glorot(o,i)
lim=sqrt(6/(i+o));
A=dlarray(single((rand(o,i)*2-1)*lim));
end

function z=zerodl(m,n)
z=dlarray(zeros(m,n,'single'));
end

function [loss,grads,parts]=modelLossFiLM(params,batch,cfg,epsMu,epsStd)
pred=forwardFiLMAD(params,batch,cfg);

epsTrue=batch.epsNorm;
epsPred=(pred.epsPred_ue-epsMu)./epsStd;

dataLoss=mean(batch.loadWeight.*(epsPred-epsTrue).^2,'all');
slopeLoss=mean(pred.wxN,'all').^2+mean(pred.wyN,'all').^2;
curvLoss=mean(pred.wxx_m.^2+pred.wyy_m.^2,'all');

% 边界残差分支只应在边界加载时显著激活。
edgeAwayLoss=mean(((1-pred.edgeGate).^2).*(pred.wEdgeResidual_m.^2),'all');
edgeMagLoss=mean(pred.wEdgeResidual_m.^2,'all');
l2Loss=sumParams(params);

loss=dataLoss+cfg.lambdaSlope*slopeLoss+cfg.lambdaCurvReg*curvLoss+ ...
     cfg.lambdaEdgeAway*edgeAwayLoss+cfg.lambdaEdgeMag*edgeMagLoss+cfg.lambdaL2*l2Loss;
grads=dlgradient(loss,params);

parts=struct('dataLoss',dataLoss,'slopeLoss',slopeLoss,'curvLoss',curvLoss, ...
             'edgeAwayLoss',edgeAwayLoss,'edgeMagLoss',edgeMagLoss,'l2Loss',l2Loss);
end

function pred=forwardFiLMAD(params,batch,cfg)
xN=batch.xN;
yN=batch.yN;

[w_norm,w_global_norm,w_edge_norm,edgeGate]=forwardFiLMW(params,batch,cfg);
w_m=cfg.w_scale_m.*w_norm;
w_global_m=cfg.w_scale_m.*w_global_norm;
w_edge_residual_m=cfg.w_scale_m.*edgeGate.*w_edge_norm;

wSum=sum(w_m,'all');
wxN=dlgradient(wSum,xN,'EnableHigherDerivatives',true,'RetainData',true);
wyN=dlgradient(wSum,yN,'EnableHigherDerivatives',true,'RetainData',true);
wxxN=dlgradient(sum(wxN,'all'),xN,'EnableHigherDerivatives',true,'RetainData',true);
wyyN=dlgradient(sum(wyN,'all'),yN,'EnableHigherDerivatives',true);

Lm=cfg.plateLength_cm/100;
Wm=cfg.plateWidth_cm/100;

wxx_m=wxxN./(Lm^2);
wyy_m=wyyN./(Wm^2);

epsX=-cfg.z_asphalt_m.*wxx_m.*1e6;
epsY=-cfg.z_asphalt_m.*wyy_m.*1e6;

epsPred=batch.dirLong.*epsX+(1-batch.dirLong).*epsY;

pred=struct('w_norm',w_norm,'w_m',w_m, ...
    'wGlobal_norm',w_global_norm,'wGlobal_m',w_global_m, ...
    'wEdge_norm',w_edge_norm,'wEdgeResidual_m',w_edge_residual_m, ...
    'edgeGate',edgeGate,'wxN',wxN,'wyN',wyN,'wxx_m',wxx_m,'wyy_m',wyy_m, ...
    'epsX_ue',epsX,'epsY_ue',epsY,'epsPred_ue',epsPred);
end

function [w,wGlobal,wEdge,edgeGate]=forwardFiLMW(params,batch,cfg)
wGlobal=forwardFiLMWBranch(params.global,batch,cfg);

if isfield(cfg,'useBoundaryResidualExpert') && cfg.useBoundaryResidualExpert
    wEdge=forwardFiLMWBranch(params.edge,batch,cfg);
    edgeGate=boundaryGateDL(batch,cfg);
    w=wGlobal+edgeGate.*wEdge;
else
    wEdge=0.*wGlobal;
    edgeGate=0.*wGlobal;
    w=wGlobal;
end
end

function w=forwardFiLMWBranch(branch,batch,cfg)
spaceX=spaceFeaturesDL(batch,cfg);
condX=condFeaturesDL(batch,cfg);

cond=tanh(fc(branch.cond.fc1.W,branch.cond.fc1.b,condX));
cond=tanh(fc(branch.cond.fc2.W,branch.cond.fc2.b,cond));

h=spaceX;

for i=1:cfg.numFiLMLayers
    h=fc(branch.space(i).W,branch.space(i).b,h);

    gamma=fc(branch.film(i).gamma.W,branch.film(i).gamma.b,cond);
    beta=fc(branch.film(i).beta.W,branch.film(i).beta.b,cond);

    % 稳定 FiLM：初始接近恒等调制，同时允许条件改变空间场。
    gamma=1+0.25.*tanh(gamma-1);

    h=tanh(gamma.*h+beta);
end

w=fc(branch.out.W,branch.out.b,h);
end

function gate=boundaryGateDL(batch,cfg)
p1xN=batch.p1xN;
p1yN=batch.p1yN;
p2xN=batch.p2xN;
p2yN=batch.p2yN;
hasP2=batch.hasP2;

tau=cfg.boundarySoftMinTau;
p1Edge=-tau.*log(exp(-p1xN./tau)+exp(-(1-p1xN)./tau)+exp(-p1yN./tau)+exp(-(1-p1yN)./tau));
p2EdgeRaw=-tau.*log(exp(-p2xN./tau)+exp(-(1-p2xN)./tau)+exp(-p2yN./tau)+exp(-(1-p2yN)./tau));
smoothMin12=0.5.*(p1Edge+p2EdgeRaw-sqrt((p1Edge-p2EdgeRaw).^2+1e-6));
loadEdge=(1-hasP2).*p1Edge+hasP2.*smoothMin12;

gate=cfg.edgeGateMax./(1+exp((loadEdge-cfg.edgeGateCenter)./cfg.edgeGateWidth));
end

function y=fc(W,b,x)
% 无标签 dlarray 下可直接矩阵乘法，且不会断开 xN/yN 的自动微分链条。
y=W*x+b;
end

function s=sumParams(p)
s=dlarray(0);
if isa(p,'dlarray')
    s=s+sum(p.^2,'all');
elseif isstruct(p)
    f=fieldnames(p);
    for i=1:numel(f)
        v=p.(f{i});
        if isstruct(v)&&numel(v)>1
            for k=1:numel(v)
                s=s+sumParams(v(k));
            end
        else
            s=s+sumParams(v);
        end
    end
end
end

function out=predictDatasetStrainFiLM(params,data,cfg,batchSize)
n=numel(data.epsMeasured_ue);
epsPred=nan(n,1);
epsX=nan(n,1);
epsY=nan(n,1);
wPred=nan(n,1);
wGlobal=nan(n,1);
wEdge=nan(n,1);
edgeGate=nan(n,1);

for s=1:batchSize:n
    e=min(s+batchSize-1,n);
    idx=s:e;
    batch=makeDLBatch(sliceDataStruct(data,idx));
    p=dlfeval(@(pa,ba,cf) forwardFiLMAD(pa,ba,cf),params,batch,cfg);

    epsPred(idx)=double(gather(extractdata(p.epsPred_ue(:))));
    epsX(idx)=double(gather(extractdata(p.epsX_ue(:))));
    epsY(idx)=double(gather(extractdata(p.epsY_ue(:))));
    wPred(idx)=double(gather(extractdata(p.w_m(:))));
    wGlobal(idx)=double(gather(extractdata(p.wGlobal_m(:))));
    wEdge(idx)=double(gather(extractdata(p.wEdgeResidual_m(:))));
    edgeGate(idx)=double(gather(extractdata(p.edgeGate(:))));
end

out=struct('epsPred_ue',epsPred,'epsX_ue',epsX,'epsY_ue',epsY,'wPred_m',wPred, ...
           'wGlobal_m',wGlobal,'wEdgeResidual_m',wEdge,'edgeGate',edgeGate);
end

function r2=calcR2(y,p)
y=y(:);
p=p(:);
m=isfinite(y)&isfinite(p);
y=y(m);
p=p(m);
den=sum((y-mean(y)).^2);
if numel(y)<2 || den<1e-12
    r2=NaN;
else
    r2=1-sum((y-p).^2)/den;
end
end

function m=calcMetrics(y,p)
y=y(:);
p=p(:);
ok=isfinite(y)&isfinite(p);
y=y(ok);
p=p(ok);
r=p-y;
m=struct('R2',calcR2(y,p),'RMSE',sqrt(mean(r.^2,'omitnan')),'MAE',mean(abs(r),'omitnan'),'Bias',mean(r,'omitnan'));
end

function plotTrainingHistory(h,figDir)
fig=figure('Visible','off','Position',[100 100 900 520]);
semilogy(h.epoch,h.train_loss_data,'LineWidth',1.8);
hold on;
semilogy(h.epoch,h.val_loss_data,'LineWidth',1.8);
xlabel('Epoch');
ylabel('Normalized Strain MSE');
title('FiLM Training and Validation Strain Loss');
legend('Train','Validation');
grid on;
saveas(fig,fullfile(figDir,'Fig_Training_Validation_Loss.png'));
close(fig);

fig=figure('Visible','off','Position',[100 100 900 520]);
plot(h.epoch,h.train_R2,'LineWidth',1.8);
hold on;
plot(h.epoch,h.val_R2,'LineWidth',1.8);
xlabel('Epoch');
ylabel('R^2');
title('FiLM Training and Validation R^2');
legend('Train','Validation','Location','southeast');
grid on;
saveas(fig,fullfile(figDir,'Fig_Training_Validation_R2.png'));
close(fig);
end

function plotParity(P,figDir)
fig=figure('Visible','off','Position',[100 100 720 640]);
hold on;
masks={P.split=="train",P.split=="validation",P.split=="test"};
labels={'Train','Validation','Test'};

for i=1:3
    scatter(P.strain_ue(masks{i}),P.pred_strain_ue(masks{i}),30,'filled','MarkerFaceAlpha',0.6);
end

v=[P.strain_ue;P.pred_strain_ue];
mn=min(v,[],'omitnan');
mx=max(v,[],'omitnan');
plot([mn mx],[mn mx],'k--','LineWidth',1.4);

xlabel('Measured Strain (microstrain)');
ylabel('Predicted Strain from FiLM Autodiff (microstrain)');
title('Measured vs Predicted Strain');
legend([labels,{'1:1 Line'}],'Location','best');
grid on;
axis equal;
xlim([mn mx]);
ylim([mn mx]);
saveas(fig,fullfile(figDir,'Fig_Parity_Measured_vs_Predicted_Strain.png'));
close(fig);
end

function plotResiduals(P,figDir)
res=P.residual_strain_ue;

fig=figure('Visible','off','Position',[100 100 850 520]);
histogram(res,30);
xlabel('Residual Strain (microstrain)');
ylabel('Count');
title('Residual Distribution');
grid on;
saveas(fig,fullfile(figDir,'Fig_Residual_Histogram.png'));
close(fig);

fig=figure('Visible','off','Position',[100 100 850 520]);
scatter(P.pred_strain_ue,res,28,'filled','MarkerFaceAlpha',0.6);
yline(0,'k--');
xlabel('Predicted Strain (microstrain)');
ylabel('Residual Strain (microstrain)');
title('Residuals vs Predicted Strain');
grid on;
saveas(fig,fullfile(figDir,'Fig_Residual_vs_Predicted.png'));
close(fig);
end

function plotCaseMetrics(C,figDir)
[~,ord]=sort(C.R2,'ascend','MissingPlacement','last');
C=C(ord,:);
labels=arrayfun(@(i)sprintf('%02d',i),1:height(C),'UniformOutput',false);

fig=figure('Visible','off','Position',[100 100 1050 520]);
bar(C.R2);
xlabel('Case Index');
ylabel('R^2');
title('Case-wise R^2');
xticks(1:height(C));
xticklabels(labels);
grid on;
saveas(fig,fullfile(figDir,'Fig_Casewise_R2.png'));
close(fig);

fig=figure('Visible','off','Position',[100 100 1050 520]);
bar(C.RMSE_ue);
xlabel('Case Index');
ylabel('RMSE (microstrain)');
title('Case-wise RMSE');
xticks(1:height(C));
xticklabels(labels);
grid on;
saveas(fig,fullfile(figDir,'Fig_Casewise_RMSE.png'));
close(fig);

writetable(table((1:height(C))',C.case_id,C.load_type,C.load_kg,C.load_edge_min_norm,C.is_boundary,C.is_extreme_boundary,C.R2,C.RMSE_ue, ...
    'VariableNames',{'case_index','case_id','load_type','load_kg','load_edge_min_norm','is_boundary','is_extreme_boundary','R2','RMSE_ue'}), ...
    fullfile(fileparts(figDir),'Tables','Case_Index_Map_For_Figures.csv'));
end

function G=makeGridTable(refRow,layerName,cfg)
xv=linspace(0,cfg.plateLength_cm,cfg.gridNX);
yv=linspace(0,cfg.plateWidth_cm,cfg.gridNY);
[X,Y]=meshgrid(xv,yv);
n=numel(X);

G=table();
G.case_id=repmat(refRow.case_id,n,1);
G.sensor_id=repmat("grid",n,1);
G.x_cm=X(:);
G.y_cm=Y(:);
G.layer=repmat(string(layerName),n,1);
G.direction=repmat("纵",n,1);
G.channel=repmat("grid",n,1);
G.interrogator=repmat("grid",n,1);
G.delta_lambda_resp_nm=zeros(n,1);
G.delta_lambda_peak_nm=zeros(n,1);
G.raw_steady_nm=zeros(n,1);
G.raw_peak_nm=zeros(n,1);
G.baseline_nm=zeros(n,1);
G.load_type=repmat(refRow.load_type,n,1);
G.load_kg=repmat(refRow.load_kg,n,1);
G.p1_x=repmat(refRow.p1_x,n,1);
G.p1_y=repmat(refRow.p1_y,n,1);
G.p2_x=repmat(refRow.p2_x,n,1);
G.p2_y=repmat(refRow.p2_y,n,1);
G.strain_ue=zeros(n,1);
end

function plotFieldContour(X,Y,Z,figTitle,xLab,yLab,cLab,subTitle,outPath)
fig=figure('Visible','off','Position',[100 100 900 650]);

if max(Z(:))-min(Z(:)) < 1e-12
    imagesc(X(1,:),Y(:,1),Z);
    set(gca,'YDir','normal');
else
    contourf(X,Y,Z,30,'LineColor','none');
end

cb=colorbar;
ylabel(cb,cLab);
xlabel(xLab);
ylabel(yLab);
title({figTitle;subTitle},'Interpreter','none');
axis equal tight;
grid on;
saveas(fig,outPath);
close(fig);
end

function s=englishCaseTitle(refRow,layerName)
lt=englishLoadType(refRow.load_type);
if isfinite(refRow.p2_x)&&isfinite(refRow.p2_y)
    s=sprintf('P = %.0f kg | %s | Layer = %s | P1 = (%.1f, %.1f) cm | P2 = (%.1f, %.1f) cm',refRow.load_kg,lt,layerName,refRow.p1_x,refRow.p1_y,refRow.p2_x,refRow.p2_y);
else
    s=sprintf('P = %.0f kg | %s | Layer = %s | P1 = (%.1f, %.1f) cm',refRow.load_kg,lt,layerName,refRow.p1_x,refRow.p1_y);
end
end

function s=englishCaseFileName(refRow)
s=matlab.lang.makeValidName(sprintf('P%.0f_%s_X%.1f_Y%.1f',refRow.load_kg,regexprep(englishLoadType(refRow.load_type),'\s+',''),refRow.p1_x,refRow.p1_y));
end

function s=englishLoadType(t)
t=string(t);
if t=="单点中心加载"
    s='Single Center Loading';
elseif t=="双点对称加载"
    s='Dual Symmetric Loading';
elseif t=="双点不对称加载"
    s='Dual Asymmetric Loading';
elseif t=="单点偏心加载"
    s='Single Eccentric Loading';
elseif t=="纵中心线加载"
    s='Longitudinal Shifted Loading';
else
    s=char(t);
end
end


function plotBoundaryDiagnostics(C,figDir)
if ~ismember('load_edge_min_norm',C.Properties.VariableNames)
    return;
end

fig=figure('Visible','off','Position',[100 100 820 560]);
scatter(C.load_edge_min_norm,C.R2,70,double(C.is_boundary),'filled','MarkerFaceAlpha',0.75);
xlabel('Normalized Distance from Load to Nearest Boundary');
ylabel('Case-wise R^2');
title('Boundary Distance vs Case-wise R^2');
grid on;
saveas(fig,fullfile(figDir,'Fig_BoundaryDistance_vs_CaseR2.png'));
close(fig);

fig=figure('Visible','off','Position',[100 100 820 560]);
scatter(C.load_edge_min_norm,C.RMSE_ue,70,double(C.is_boundary),'filled','MarkerFaceAlpha',0.75);
xlabel('Normalized Distance from Load to Nearest Boundary');
ylabel('Case-wise RMSE (microstrain)');
title('Boundary Distance vs Case-wise RMSE');
grid on;
saveas(fig,fullfile(figDir,'Fig_BoundaryDistance_vs_CaseRMSE.png'));
close(fig);
end


function writeChineseReport(reportFile,cfg,MetricsOverall,MetricsNoExtreme,CaseMetrics,FieldSummary,modelFile,bestEpoch,bestVal)
fid=fopen(reportFile,'w');
if fid<0, return; end

fprintf(fid,'Step21 边界残差专家 FiLM 条件调制 AD-DNF 报告\n');
fprintf(fid,'============================================================\n\n');

fprintf(fid,'一、方法说明\n');
fprintf(fid,'本脚本采用边界感知 FiLM 条件调制位移神经场。空间特征在 dlfeval 内部由 tracked xN/yN 动态构造，避免预计算空间特征切断自动微分链条。\n');
fprintf(fid,'网络输出位移代理场 w，再通过自动微分得到纵向/横向应变，与 FBG 实测应变对齐。\n\n');

fprintf(fid,'二、关键设置\n');
fprintf(fid,'数据划分方式：%s\n',cfg.splitMode);
fprintf(fid,'训练层：%s\n',strjoin(cfg.trainLayers,', '));
fprintf(fid,'FiLM层数：%d，空间隐藏维度：%d，条件隐藏维度：%d\n',cfg.numFiLMLayers,cfg.spaceHiddenUnits,cfg.conditionHiddenUnits);
fprintf(fid,'镜像增强：%d, X=%d, Y=%d\n',cfg.useMirrorAugmentation,cfg.augmentMirrorX,cfg.augmentMirrorY);
fprintf(fid,'边界距离特征：%d，边界加权：%d，边界特征长度：%.3f\n',cfg.useBoundaryDistanceFeatures,cfg.useBoundaryWeightedLoss,cfg.boundaryFeatureLengthNorm);
fprintf(fid,'边界残差专家：%d，gateCenter=%.3f，gateWidth=%.3f，lambdaEdgeAway=%.3g\n',cfg.useBoundaryResidualExpert,cfg.edgeGateCenter,cfg.edgeGateWidth,cfg.lambdaEdgeAway);
fprintf(fid,'极边界阈值 normalized distance：%.3f\n',cfg.extremeBoundaryThresholdN);
fprintf(fid,'最优 Epoch：%d\n',bestEpoch);
fprintf(fid,'最优验证 data loss：%.6g\n\n',bestVal);

fprintf(fid,'三、总体性能\n');
for i=1:height(MetricsOverall)
    fprintf(fid,'%s: R2 = %.4f, RMSE = %.6g ue, MAE = %.6g ue, Bias = %.6g ue\n',MetricsOverall.split(i),MetricsOverall.R2(i),MetricsOverall.RMSE_ue(i),MetricsOverall.MAE_ue(i),MetricsOverall.Bias_ue(i));
end

if ~isempty(MetricsNoExtreme)
    fprintf(fid,'\n四、排除极边界工况后的补充指标\n');
    for i=1:height(MetricsNoExtreme)
        fprintf(fid,'%s: R2 = %.4f, RMSE = %.6g ue, MAE = %.6g ue, Bias = %.6g ue\n',MetricsNoExtreme.scope(i),MetricsNoExtreme.R2(i),MetricsNoExtreme.RMSE_ue(i),MetricsNoExtreme.MAE_ue(i),MetricsNoExtreme.Bias_ue(i));
    end
end

fprintf(fid,'\n五、分工况薄弱项\n');
[~,ord]=sort(CaseMetrics.R2,'ascend','MissingPlacement','last');
nShow=min(8,height(CaseMetrics));
for k=1:nShow
    r=CaseMetrics(ord(k),:);
    fprintf(fid,'  %s | %s | %.0f kg | edgeMin=%.3f | boundary=%d | extreme=%d | R2=%.4f | RMSE=%.6g ue\n',r.case_id,r.load_type,r.load_kg,r.load_edge_min_norm,r.is_boundary,r.is_extreme_boundary,r.R2,r.RMSE_ue);
end

fprintf(fid,'\n六、建议\n');
fprintf(fid,'本脚本为边界残差专家模型：全局分支学习主体响应，边界残差分支在边界门控下修正极边界响应。\n');
fprintf(fid,'若边界工况仍然偏差较大，优先检查 Case_Metrics.csv 中 load_edge_min_norm 较小的工况，并与 Fig_BoundaryDistance_vs_CaseR2.png 对照。\n');
fprintf(fid,'模型文件：%s\n',modelFile);

fclose(fid);
end
