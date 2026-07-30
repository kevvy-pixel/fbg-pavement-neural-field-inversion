function RUN_GLOBAL_BENCHMARK(profile,inputFile,outputBase)
%RUN_GLOBAL_BENCHMARK Case-level cross-validation and runtime benchmark.
%
% Example:
%   RUN_GLOBAL_BENCHMARK("quick", ...
%       "D:\data\Response_Matrix_Table.csv","D:\FBG_global_runs")

arguments
    profile (1,1) string = "quick"
    inputFile (1,1) string = ""
    outputBase (1,1) string = ""
end

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
cfg = fbg_config(profile);
if strlength(inputFile) == 0
    inputFile = fullfile(rootDir,cfg.data.inputFile);
end
if cfg.runtime.useGPU && ~canUseGPU
    fprintf('Compatible GPU not detected; using CPU.\n');
    cfg.runtime.useGPU = false;
end

if strlength(outputBase) == 0
    workRoot = fbg_writable_root(rootDir);
else
    workRoot = fbg_writable_root(outputBase);
end
timestamp = string(datetime('now','Format','yyyyMMdd_HHmmss'));
runDir = fullfile(workRoot,cfg.runtime.outputRoot, ...
    timestamp+"_"+profile+"_global_benchmark");
mkdir(runDir);

diary(fullfile(runDir,'run.log'));
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('FBG Global Case-Level Benchmark\n');
fprintf('Profile: %s\nInput: %s\nOutput: %s\n',profile,inputFile,runDir);
fprintf('GPU: %d\n',cfg.runtime.useGPU);

[D,meta] = fbg_load_data(inputFile,cfg);
assignments = fbg_make_case_folds(D,cfg);
writetable(assignments,fullfile(runDir,'case_fold_assignments.csv'));

foldIds = unique(assignments.fold_id,'stable');
fprintf('Cases: %d | folds: %d | rows: %d\n', ...
    height(assignments),numel(foldIds),height(D.table));

timingParts = cell(numel(foldIds)*3,1);
convergenceParts = cell(numel(foldIds)*3,1);
fullCaseParts = cell(numel(foldIds)*3,1);
fullPredictionParts = cell(numel(foldIds)*3,1);
fixedMetricParts = cell(numel(foldIds),1);
fixedPredictionParts = cell(numel(foldIds),1);
calibrationPointParts = cell(numel(foldIds),1);
timingIndex = 0;
convergenceIndex = 0;

for iFold = 1:numel(foldIds)
    foldId = foldIds(iFold);
    foldDir = fullfile(runDir,sprintf('fold_%02d',foldId));
    if ~exist(foldDir,'dir'), mkdir(foldDir); end

    testCases = assignments.case_id(assignments.fold_id == foldId);
    remainingCases = assignments.case_id(assignments.fold_id ~= foldId);
    validationCases = localValidationCases(assignments,remainingCases, ...
        cfg,foldId);
    trainCases = setdiff(remainingCases,validationCases,'stable');

    split.train = ismember(D.table.case_id,trainCases);
    split.validation = ismember(D.table.case_id,validationCases);
    split.test = ismember(D.table.case_id,testCases);
    split.trainCases = trainCases;
    split.validationCases = validationCases;
    split.testCases = testCases;

    splitTable = table([trainCases;validationCases;testCases], ...
        [repmat("train",numel(trainCases),1); ...
        repmat("validation",numel(validationCases),1); ...
        repmat("test",numel(testCases),1)], ...
        'VariableNames',{'case_id','split'});
    writetable(splitTable,fullfile(foldDir,'case_split.csv'));

    [foldD,foldMeta] = fbg_prepare_fold(D,meta,split.train);
    save(fullfile(foldDir,'fold_configuration.mat'), ...
        'cfg','foldMeta','split','-v7.3');
    normalization = table(foldId,foldMeta.maxLoad,foldMeta.targetScale, ...
        sum(split.train),sum(split.validation),sum(split.test), ...
        'VariableNames',{'fold_id','training_max_load_kg', ...
        'training_response_scale_microstrain','n_train_rows', ...
        'n_validation_rows','n_test_rows'});
    writetable(normalization,fullfile(foldDir,'fold_normalization.csv'));

    fprintf('\n=== Fold %d/%d | train/val/test cases %d/%d/%d ===\n', ...
        iFold,numel(foldIds),numel(trainCases), ...
        numel(validationCases),numel(testCases));

    % Plain-FiLM.
    rng(cfg.randomSeed+foldId*100+1);
    plainParams = fbg_model_init("plain",foldMeta,cfg);
    timerValue = tic;
    [plainParams,plainHistory] = fbg_train_model(plainParams,"plain", ...
        foldD,split.train,split.validation,foldMeta,cfg,"plain",false);
    plainSeconds = toc(timerValue);
    plainResult = fbg_save_stage(fullfile(foldDir,'01_plain_film'), ...
        plainParams,"plain",plainHistory,foldD,split,foldMeta,cfg);
    plainInference = localTimeInference(plainParams,"plain",foldD, ...
        split.test,foldMeta,cfg);

    timingIndex = timingIndex+1;
    timingParts{timingIndex} = localTimingRow(foldId,"Plain-FiLM", ...
        plainParams,plainHistory,plainSeconds,plainSeconds, ...
        plainInference,sum(split.test),0);

    % Residual-FiLM.
    rng(cfg.randomSeed+foldId*100+2);
    residualParams = fbg_model_init("residual",foldMeta,cfg);
    timerValue = tic;
    [residualParams,residualHistory] = fbg_train_model( ...
        residualParams,"residual",foldD,split.train,split.validation, ...
        foldMeta,cfg,"residual",false);
    residualSeconds = toc(timerValue);
    residualResult = fbg_save_stage(fullfile(foldDir,'02_residual_film'), ...
        residualParams,"residual",residualHistory,foldD,split,foldMeta,cfg);
    residualInference = localTimeInference(residualParams,"residual", ...
        foldD,split.test,foldMeta,cfg);

    timingIndex = timingIndex+1;
    timingParts{timingIndex} = localTimingRow(foldId,"Residual-FiLM", ...
        residualParams,residualHistory,residualSeconds,residualSeconds, ...
        residualInference,sum(split.test),0);

    % Edge expert, warm-started from the Residual-FiLM backbone.
    rng(cfg.randomSeed+foldId*100+3);
    edgeParams = fbg_model_init("edge",foldMeta,cfg,residualParams);
    timerValue = tic;
    [edgeParams,edgeHistory] = fbg_train_model(edgeParams,"edge", ...
        foldD,split.train,split.validation,foldMeta,cfg,"edge",true);
    edgeAdapterSeconds = toc(timerValue);
    edgeTotalSeconds = residualSeconds+edgeAdapterSeconds;
    edgeResult = fbg_save_stage(fullfile(foldDir, ...
        '03_spatial_edge_expert'),edgeParams,"edge",edgeHistory, ...
        foldD,split,foldMeta,cfg);
    edgeInference = localTimeInference(edgeParams,"edge",foldD, ...
        split.test,foldMeta,cfg);

    timingIndex = timingIndex+1;
    timingParts{timingIndex} = localTimingRow(foldId, ...
        "Spatial-edge-expert",edgeParams,edgeHistory, ...
        edgeAdapterSeconds,edgeTotalSeconds,edgeInference,sum(split.test), ...
        residualSeconds);

    % Full held-out case metrics before sparse calibration.
    fullCaseParts{(iFold-1)*3+1} = localFullCaseMetrics( ...
        plainResult.caseMetrics,foldId,"Plain-FiLM");
    fullCaseParts{(iFold-1)*3+2} = localFullCaseMetrics( ...
        residualResult.caseMetrics,foldId,"Residual-FiLM");
    fullCaseParts{(iFold-1)*3+3} = localFullCaseMetrics( ...
        edgeResult.caseMetrics,foldId,"Spatial-edge-expert");
    fullPredictionParts{(iFold-1)*3+1} = localFullPredictions( ...
        plainResult.predictions,foldId,"Plain-FiLM");
    fullPredictionParts{(iFold-1)*3+2} = localFullPredictions( ...
        residualResult.predictions,foldId,"Residual-FiLM");
    fullPredictionParts{(iFold-1)*3+3} = localFullPredictions( ...
        edgeResult.predictions,foldId,"Spatial-edge-expert");

    % Common fixed-coordinate comparison and 4/6-point ridge calibration.
    [fixedPredictionParts{iFold},fixedMetricParts{iFold}, ...
        calibrationPointParts{iFold}] = localFixedEvaluation( ...
        foldId,foldD,foldMeta,split,plainResult,residualResult, ...
        edgeResult,cfg);

    % Convergence comparison at a common validation RMSE.
    [convPlain,convResidual] = localCommonConvergence( ...
        foldId,plainHistory,residualHistory);
    convergenceIndex = convergenceIndex+1;
    convergenceParts{convergenceIndex} = convPlain;
    convergenceIndex = convergenceIndex+1;
    convergenceParts{convergenceIndex} = convResidual;
    convergenceIndex = convergenceIndex+1;
    convergenceParts{convergenceIndex} = localIndependentConvergence( ...
        foldId,"Spatial-edge-expert",edgeHistory);
end

trainingTiming = vertcat(timingParts{1:timingIndex});
convergence = vertcat(convergenceParts{1:convergenceIndex});
fullCaseMetrics = vertcat(fullCaseParts{:});
fullPredictions = vertcat(fullPredictionParts{:});
fixedPredictions = vertcat(fixedPredictionParts{:});
fixedCaseMetrics = vertcat(fixedMetricParts{:});
calibrationPoints = vertcat(calibrationPointParts{:});

writetable(trainingTiming,fullfile(runDir,'training_timing_by_fold.csv'));
writetable(convergence,fullfile(runDir,'convergence_by_fold.csv'));
writetable(fullCaseMetrics,fullfile(runDir, ...
    'full_zero_shot_case_metrics.csv'));
writetable(fullPredictions,fullfile(runDir, ...
    'full_zero_shot_predictions.csv'));
writetable(fixedPredictions,fullfile(runDir, ...
    'common_fixed_test_predictions.csv'));
writetable(fixedCaseMetrics,fullfile(runDir, ...
    'common_fixed_test_case_metrics.csv'));
writetable(calibrationPoints,fullfile(runDir, ...
    'calibration_point_assignments.csv'));

globalSummary = localGlobalSummary(fixedPredictions,fixedCaseMetrics);
writetable(globalSummary,fullfile(runDir,'global_model_summary.csv'));

fullZeroSummary = localCaseSummary(fullCaseMetrics,fullPredictions);
writetable(fullZeroSummary,fullfile(runDir, ...
    'full_zero_shot_summary.csv'));

timingSummary = localTimingSummary(trainingTiming,fixedCaseMetrics);
writetable(timingSummary,fullfile(runDir,'runtime_summary.csv'));

complexity = localComplexitySummary(trainingTiming);
writetable(complexity,fullfile(runDir,'model_complexity.csv'));

localPlotGlobal(globalSummary,fullfile(runDir,'global_model_comparison.png'));
localPlotTiming(timingSummary,fullfile(runDir,'runtime_comparison.png'));

fprintf('\nGlobal benchmark completed.\nResults: %s\n',runDir);
end

function validationCases = localValidationCases(assignments,remaining,cfg,foldId)
nVal = max(1,round(cfg.global.validationFraction*height(assignments)));
nVal = min(nVal,numel(remaining)-1);
rng(cfg.randomSeed+foldId*7919);
remaining = remaining(randperm(numel(remaining)));
validationCases = remaining(1:nVal);
end

function seconds = localTimeInference(params,modelKind,D,mask,meta,cfg)
% Warm-up keeps one-time function compilation out of the timing result.
fbg_predict(params,modelKind,D,mask,meta,cfg);
samples = zeros(cfg.global.inferenceTimingRepeats,1);
for i = 1:cfg.global.inferenceTimingRepeats
    timerValue = tic;
    fbg_predict(params,modelKind,D,mask,meta,cfg);
    samples(i) = toc(timerValue);
end
seconds = median(samples);
end

function row = localTimingRow(foldId,model,params,history,stageSeconds, ...
    endToEndSeconds,inferenceSeconds,nInferenceRows,backboneTimeOffset)

[bestRmse,timeToBest,bestIteration] = localBestHistory(history);
row = table(foldId,model,stageSeconds,endToEndSeconds, ...
    history.iteration(end),bestIteration,bestRmse,timeToBest, ...
    timeToBest+backboneTimeOffset, ...
    inferenceSeconds,1000*inferenceSeconds/max(nInferenceRows,1), ...
    localParameterCount(params), ...
    'VariableNames',{'fold_id','model','stage_training_seconds', ...
    'end_to_end_training_seconds','iterations_run','best_iteration', ...
    'best_validation_RMSE_microstrain','stage_time_to_best_seconds', ...
    'end_to_end_time_to_best_seconds', ...
    'test_inference_seconds','inference_milliseconds_per_row', ...
    'parameter_count'});
end

function [bestRmse,timeToBest,bestIteration] = localBestHistory(history)
valid = isfinite(history.validation_rmse_microstrain);
if ~any(valid)
    bestRmse = NaN; timeToBest = NaN; bestIteration = NaN;
    return;
end
sub = history(valid,:);
[bestRmse,index] = min(sub.validation_rmse_microstrain);
timeToBest = sub.elapsed_seconds(index);
bestIteration = sub.iteration(index);
end

function count = localParameterCount(value)
if isa(value,'dlarray')
    count = numel(extractdata(value));
elseif isa(value,'gpuArray') || isnumeric(value)
    count = numel(value);
elseif isstruct(value)
    count = 0;
    for k = 1:numel(value)
        names = fieldnames(value(k));
        for i = 1:numel(names)
            count = count+localParameterCount(value(k).(names{i}));
        end
    end
else
    count = 0;
end
end

function out = localFullCaseMetrics(caseMetrics,foldId,model)
variableNames = string(caseMetrics.Properties.VariableNames);
if any(variableNames == "split")
    splitValues = caseMetrics.split;
elseif any(variableNames == "case_split")
    splitValues = caseMetrics.case_split;
else
    error('Case-metrics table has no split or case_split variable.');
end
out = caseMetrics(splitValues == "test",:);
out = addvars(out,repmat(foldId,height(out),1), ...
    repmat(model,height(out),1),'Before',1, ...
    'NewVariableNames',{'fold_id','model'});
end

function out = localFullPredictions(predictions,foldId,model)
out = predictions(predictions.split == "test",:);
out = addvars(out,repmat(foldId,height(out),1), ...
    repmat(model,height(out),1),'Before',1, ...
    'NewVariableNames',{'fold_id','model'});
end

function [predictions,caseMetrics,calibrationPoints] = ...
    localFixedEvaluation(foldId,D,meta,split,plainResult, ...
    residualResult,edgeResult,cfg)

testCases = split.testCases;
predictionParts = {};
metricParts = {};
pointParts = {};
partIndex = 0;
metricIndex = 0;
pointIndex = 0;
maxKRequested = max(cfg.global.calibrationKValues);

for iCase = 1:numel(testCases)
    caseId = testCases(iCase);
    caseMask = split.test & D.table.case_id == caseId;
    caseRows = find(caseMask);
    coords = unique(D.table(caseRows,{'x_cm','y_cm'}),'rows','stable');
    poolSize = min(maxKRequested, ...
        height(coords)-cfg.global.minimumFixedTestCoordinates);
    poolSize = max(0,poolSize);
    if poolSize < maxKRequested
        warning(['Case %s has too few coordinates for the common 6-point ', ...
            'calibration benchmark and is excluded from the common-set ', ...
            'table. It remains in the full zero-shot table.'],caseId);
        continue;
    end

    firstRow = caseRows(1);
    order = localCoverageOrder(coords,D.p1x(firstRow),D.p1y(firstRow));
    poolCoords = coords(order(1:poolSize),:);
    fixedCoords = coords(order(poolSize+1:end),:);
    fixedMask = localCoordinateMask(D.table,caseMask,fixedCoords);

    poolTable = poolCoords;
    poolTable.selection_order = (1:height(poolTable))';
    poolTable.fold_id = repmat(foldId,height(poolTable),1);
    poolTable.case_id = repmat(caseId,height(poolTable),1);
    for k = cfg.global.calibrationKValues
        poolTable.("used_for_k"+k) = poolTable.selection_order <= k;
    end
    poolTable = movevars(poolTable,{'fold_id','case_id'},'Before',1);
    pointIndex = pointIndex+1;
    pointParts{pointIndex,1} = poolTable; %#ok<AGROW>

    modelNames = ["Plain-FiLM","Residual-FiLM","Spatial-edge-expert"];
    modelResults = {plainResult,residualResult,edgeResult};
    for iModel = 1:numel(modelNames)
        sub = localPredictionRows(modelResults{iModel}.predictions,fixedMask);
        partIndex = partIndex+1;
        predictionParts{partIndex,1} = localLabelPrediction( ...
            sub,foldId,modelNames(iModel),0,1,0); %#ok<AGROW>
        metricIndex = metricIndex+1;
        metricParts{metricIndex,1} = localMetricRow(sub,foldId, ...
            caseId,modelNames(iModel),0,0,1,0); %#ok<AGROW>
    end

    for k = cfg.global.calibrationKValues
        if k > poolSize, continue; end
        selectedCoords = poolCoords(1:k,:);
        calibrationMask = localCoordinateMask(D.table,caseMask,selectedCoords);
        calibrationPrediction = localPredictionRows( ...
            residualResult.predictions,calibrationMask);
        fixedResidual = localPredictionRows( ...
            residualResult.predictions,fixedMask);

        [scaleA,offsetB,calibrationSeconds] = localTimedRidge( ...
            calibrationPrediction,meta,cfg);
        calibrated = fixedResidual;
        calibrated.predicted_microstrain = scaleA* ...
            calibrated.predicted_microstrain+offsetB;
        calibrated.residual_microstrain = calibrated.predicted_microstrain- ...
            calibrated.measured_microstrain;
        method = "Residual+Ridge-"+k;

        partIndex = partIndex+1;
        predictionParts{partIndex,1} = localLabelPrediction( ...
            calibrated,foldId,method,k,scaleA,offsetB); %#ok<AGROW>
        metricIndex = metricIndex+1;
        metricParts{metricIndex,1} = localMetricRow(calibrated,foldId, ...
            caseId,method,k,calibrationSeconds,scaleA,offsetB); %#ok<AGROW>
    end
end

predictions = vertcat(predictionParts{:});
caseMetrics = vertcat(metricParts{:});
calibrationPoints = vertcat(pointParts{:});
end

function sub = localPredictionRows(predictions,mask)
rows = find(mask);
sub = predictions(ismember(predictions.row_index,rows),:);
end

function out = localLabelPrediction(prediction,foldId,method,k,scaleA,offsetB)
n = height(prediction);
out = addvars(prediction,repmat(foldId,n,1),repmat(method,n,1), ...
    repmat(k,n,1),repmat(scaleA,n,1),repmat(offsetB,n,1), ...
    'Before',1,'NewVariableNames',{'fold_id','method','k_points', ...
    'calibration_scale','calibration_offset_microstrain'});
end

function row = localMetricRow(prediction,foldId,caseId,method,k, ...
    calibrationSeconds,scaleA,offsetB)
[r2,rmse,mae,bias] = localMetrics(prediction.measured_microstrain, ...
    prediction.predicted_microstrain);
row = table(foldId,caseId,method,k,height(prediction), ...
    calibrationSeconds,scaleA,offsetB,r2,rmse,mae,bias, ...
    'VariableNames',{'fold_id','case_id','method','k_points', ...
    'n_fixed_test_rows','calibration_seconds','calibration_scale', ...
    'calibration_offset_microstrain','R2','RMSE_microstrain', ...
    'MAE_microstrain','Bias_microstrain'});
end

function [scaleA,offsetB,seconds] = localTimedRidge(prediction,meta,cfg)
% Warm-up and repeated tiny solves provide a measurable average duration.
[scaleA,offsetB] = localFitRidge(prediction,meta,cfg);
timerValue = tic;
for i = 1:cfg.global.calibrationTimingRepeats
    [scaleA,offsetB] = localFitRidge(prediction,meta,cfg);
end
seconds = toc(timerValue)/cfg.global.calibrationTimingRepeats;
end

function [scaleA,offsetB] = localFitRidge(prediction,meta,cfg)
z = prediction.predicted_microstrain/meta.targetScale;
t = prediction.measured_microstrain/meta.targetScale;
X = [z ones(size(z))];
targetCorrection = t-z;
penalty = diag([cfg.fewshot.affineSlopePenalty, ...
    cfg.fewshot.affineOffsetPenalty]);
theta = (X'*X+penalty)\(X'*targetCorrection);
scaleA = 1+theta(1);
bounds = cfg.fewshot.affineScaleBounds;
scaleA = min(max(scaleA,bounds(1)),bounds(2));
offsetB = theta(2)*meta.targetScale;
end

function order = localCoverageOrder(coords,p1x,p1y)
scaleX = max(max(coords.x_cm)-min(coords.x_cm),1);
scaleY = max(max(coords.y_cm)-min(coords.y_cm),1);
distanceToLoad = ((coords.x_cm-p1x)/scaleX).^2 + ...
    ((coords.y_cm-p1y)/scaleY).^2;
[~,first] = min(distanceToLoad);

n = height(coords);
order = zeros(n,1);
order(1) = first;
remaining = setdiff((1:n)',first,'stable');
for i = 2:n
    selected = order(1:i-1);
    minDistance = inf(numel(remaining),1);
    for j = 1:numel(remaining)
        dx = (coords.x_cm(remaining(j))-coords.x_cm(selected))/scaleX;
        dy = (coords.y_cm(remaining(j))-coords.y_cm(selected))/scaleY;
        minDistance(j) = min(dx.^2+dy.^2);
    end
    [~,pick] = max(minDistance);
    order(i) = remaining(pick);
    remaining(pick) = [];
end
end

function mask = localCoordinateMask(T,caseMask,coords)
mask = false(height(T),1);
for i = 1:height(coords)
    mask = mask | (caseMask & T.x_cm == coords.x_cm(i) & ...
        T.y_cm == coords.y_cm(i));
end
end

function [plainRow,residualRow] = localCommonConvergence( ...
    foldId,plainHistory,residualHistory)
[plainBest,~,~] = localBestHistory(plainHistory);
[residualBest,~,~] = localBestHistory(residualHistory);
target = max(plainBest,residualBest);
plainRow = localConvergenceRow(foldId,"Plain-FiLM",plainHistory,target, ...
    "plain_residual_common");
residualRow = localConvergenceRow(foldId,"Residual-FiLM", ...
    residualHistory,target,"plain_residual_common");
end

function row = localIndependentConvergence(foldId,model,history)
[best,~,~] = localBestHistory(history);
row = localConvergenceRow(foldId,model,history,best,"model_own_best");
end

function row = localConvergenceRow(foldId,model,history,target,targetType)
[bestRmse,timeToBest,bestIteration] = localBestHistory(history);
valid = isfinite(history.validation_rmse_microstrain);
reachedRows = find(valid & ...
    history.validation_rmse_microstrain <= target,1,'first');
reached = ~isempty(reachedRows);
if reached
    timeToTarget = history.elapsed_seconds(reachedRows);
    iterationToTarget = history.iteration(reachedRows);
else
    timeToTarget = NaN;
    iterationToTarget = NaN;
end
row = table(foldId,model,targetType,target,reached, ...
    iterationToTarget,timeToTarget, ...
    bestIteration,bestRmse,timeToBest, ...
    'VariableNames',{'fold_id','model','target_type', ...
    'target_RMSE_microstrain','reached_target','iteration_to_target', ...
    'time_to_target_seconds','best_iteration', ...
    'best_validation_RMSE_microstrain','time_to_best_seconds'});
end

function summary = localGlobalSummary(predictions,caseMetrics)
methods = unique(caseMetrics.method,'stable');
parts = cell(numel(methods),1);
for i = 1:numel(methods)
    pred = predictions(predictions.method == methods(i),:);
    cases = caseMetrics(caseMetrics.method == methods(i),:);
    [microR2,microRmse,microMae,microBias] = localMetrics( ...
        pred.measured_microstrain,pred.predicted_microstrain);
    parts{i} = table(methods(i),numel(unique(cases.fold_id)), ...
        height(cases),height(pred),microR2,microRmse,microMae,microBias, ...
        mean(cases.R2,'omitnan'),std(cases.R2,0,'omitnan'), ...
        mean(cases.RMSE_microstrain,'omitnan'), ...
        std(cases.RMSE_microstrain,0,'omitnan'), ...
        max(cases.RMSE_microstrain,[],'omitnan'), ...
        mean(cases.MAE_microstrain,'omitnan'), ...
        mean(cases.Bias_microstrain,'omitnan'), ...
        'VariableNames',{'model','n_folds','n_cases','n_rows', ...
        'micro_R2','micro_RMSE_microstrain','micro_MAE_microstrain', ...
        'micro_Bias_microstrain','macro_R2_mean','macro_R2_std', ...
        'macro_RMSE_mean_microstrain','macro_RMSE_std_microstrain', ...
        'worst_case_RMSE_microstrain','macro_MAE_mean_microstrain', ...
        'macro_Bias_mean_microstrain'});
end
summary = vertcat(parts{:});
end

function summary = localCaseSummary(caseMetrics,predictions)
models = unique(caseMetrics.model,'stable');
parts = cell(numel(models),1);
for i = 1:numel(models)
    sub = caseMetrics(caseMetrics.model == models(i),:);
    pred = predictions(predictions.model == models(i),:);
    [microR2,microRmse,microMae,microBias] = localMetrics( ...
        pred.measured_microstrain,pred.predicted_microstrain);
    parts{i} = table(models(i),height(sub),height(pred), ...
        microR2,microRmse,microMae,microBias, ...
        mean(sub.R2,'omitnan'),std(sub.R2,0,'omitnan'), ...
        mean(sub.RMSE_microstrain,'omitnan'), ...
        std(sub.RMSE_microstrain,0,'omitnan'), ...
        max(sub.RMSE_microstrain,[],'omitnan'), ...
        mean(sub.MAE_microstrain,'omitnan'), ...
        mean(sub.Bias_microstrain,'omitnan'), ...
        'VariableNames',{'model','n_cases','n_rows','micro_R2', ...
        'micro_RMSE_microstrain','micro_MAE_microstrain', ...
        'micro_Bias_microstrain','macro_R2_mean','macro_R2_std', ...
        'macro_RMSE_mean_microstrain', ...
        'macro_RMSE_std_microstrain','worst_case_RMSE_microstrain', ...
        'macro_MAE_mean_microstrain','macro_Bias_mean_microstrain'});
end
summary = vertcat(parts{:});
end

function summary = localTimingSummary(trainingTiming,fixedCaseMetrics)
models = unique(trainingTiming.model,'stable');
parts = cell(numel(models)+2,1);
for i = 1:numel(models)
    sub = trainingTiming(trainingTiming.model == models(i),:);
    parts{i} = table(models(i),"neural_training",height(sub), ...
        mean(sub.end_to_end_training_seconds), ...
        std(sub.end_to_end_training_seconds), ...
        mean(sub.end_to_end_time_to_best_seconds,'omitnan'), ...
        mean(sub.test_inference_seconds), ...
        mean(sub.inference_milliseconds_per_row),NaN, ...
        'VariableNames',{'model','stage','n_measurements', ...
        'mean_end_to_end_seconds','std_end_to_end_seconds', ...
        'mean_time_to_best_seconds','mean_inference_seconds', ...
        'mean_inference_milliseconds_per_row', ...
        'mean_calibration_seconds'});
end

for kIndex = 1:2
    k = [4 6];
    method = "Residual+Ridge-"+k(kIndex);
    sub = fixedCaseMetrics(fixedCaseMetrics.method == method,:);
    base = trainingTiming(trainingTiming.model == "Residual-FiLM",:);
    parts{numel(models)+kIndex} = table(method,"closed_form_adaptation", ...
        height(sub),mean(base.end_to_end_training_seconds), ...
        std(base.end_to_end_training_seconds), ...
        mean(base.end_to_end_time_to_best_seconds,'omitnan'), ...
        mean(base.test_inference_seconds), ...
        mean(base.inference_milliseconds_per_row), ...
        mean(sub.calibration_seconds,'omitnan'), ...
        'VariableNames',parts{1}.Properties.VariableNames);
end
summary = vertcat(parts{:});
end

function complexity = localComplexitySummary(trainingTiming)
baseModels = unique(trainingTiming.model,'stable');
parts = cell(numel(baseModels)+2,1);
for i = 1:numel(baseModels)
    sub = trainingTiming(trainingTiming.model == baseModels(i),:);
    parts{i} = table(baseModels(i),round(mean(sub.parameter_count)),0, ...
        round(mean(sub.parameter_count)), ...
        'VariableNames',{'model','backbone_parameters', ...
        'adaptation_parameters','total_parameters'});
end
residual = round(mean(trainingTiming.parameter_count( ...
    trainingTiming.model == "Residual-FiLM")));
parts{numel(baseModels)+1} = table("Residual+Ridge-4",residual,2, ...
    residual+2,'VariableNames',parts{1}.Properties.VariableNames);
parts{numel(baseModels)+2} = table("Residual+Ridge-6",residual,2, ...
    residual+2,'VariableNames',parts{1}.Properties.VariableNames);
complexity = vertcat(parts{:});
end

function [r2,rmse,mae,bias] = localMetrics(measured,predicted)
valid = isfinite(measured) & isfinite(predicted);
measured = measured(valid);
predicted = predicted(valid);
if isempty(measured)
    r2 = NaN; rmse = NaN; mae = NaN; bias = NaN;
    return;
end
residual = predicted-measured;
rmse = sqrt(mean(residual.^2));
mae = mean(abs(residual));
bias = mean(residual);
denom = sum((measured-mean(measured)).^2);
if denom <= eps
    r2 = NaN;
else
    r2 = 1-sum(residual.^2)/denom;
end
end

function localPlotGlobal(summary,outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 1100 520]);
tiledlayout(1,2);
x = 1:height(summary);
nexttile;
bar(x,summary.macro_R2_mean);
hold on;
errorbar(x,summary.macro_R2_mean, ...
    summary.macro_R2_std,'k.','LineWidth',1.2);
xticks(x); xticklabels(summary.model); xtickangle(25);
ylabel('Case-macro R^2'); title('Global held-out-case accuracy');
grid on;
nexttile;
bar(x,summary.macro_RMSE_mean_microstrain);
hold on;
errorbar(x,summary.macro_RMSE_mean_microstrain, ...
    summary.macro_RMSE_std_microstrain,'k.','LineWidth',1.2);
xticks(x); xticklabels(summary.model); xtickangle(25);
ylabel('Case-macro RMSE (\muepsilon)');
title('Global held-out-case error'); grid on;
saveas(fig,outputFile); close(fig);
end

function localPlotTiming(summary,outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 1000 520]);
tiledlayout(1,2);
x = 1:height(summary);
nexttile;
bar(x,summary.mean_end_to_end_seconds);
xticks(x); xticklabels(summary.model); xtickangle(25);
ylabel('Training seconds'); title('End-to-end backbone training');
grid on;
nexttile;
values = summary.mean_calibration_seconds;
values(~isfinite(values)) = 0;
bar(x,1000*values);
xticks(x); xticklabels(summary.model); xtickangle(25);
ylabel('Calibration milliseconds'); title('Per-case adaptation time');
grid on;
saveas(fig,outputFile); close(fig);
end
