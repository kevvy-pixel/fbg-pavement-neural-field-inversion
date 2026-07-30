function RUN_SOTA_FULL_COMPARISON(profile,inputFile,outputBase,resumeRunDir)
%RUN_SOTA_FULL_COMPARISON Fair Step19-SOTA versus Residual comparison.
%
% Full paper-grade run:
%   RUN_SOTA_FULL_COMPARISON("full","D:\data\Response_Matrix_Table.csv", ...
%       "C:\Users\name\Documents\MATLAB\FBG_NeuralField_Workspace")
%
% Resume a previously interrupted run:
%   RUN_SOTA_FULL_COMPARISON("full","D:\data\Response_Matrix_Table.csv", ...
%       "", "C:\...\outputs\20260728_120000_full_sota_comparison")

arguments
    profile (1,1) string = "full"
    inputFile (1,1) string = ""
    outputBase (1,1) string = ""
    resumeRunDir (1,1) string = ""
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

if strlength(resumeRunDir) > 0
    runDir = resumeRunDir;
    if ~exist(runDir,'dir')
        error('Resume directory does not exist: %s',runDir);
    end
else
    if strlength(outputBase) == 0
        workRoot = fbg_writable_root(rootDir);
    else
        workRoot = fbg_writable_root(outputBase);
    end
    timestamp = string(datetime('now','Format','yyyyMMdd_HHmmss'));
    runDir = fullfile(workRoot,cfg.runtime.outputRoot, ...
        timestamp+"_"+profile+"_sota_comparison");
    mkdir(runDir);
end

diary(fullfile(runDir,'run.log'));
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('FBG Step19-Compatible Full SOTA Comparison\n');
fprintf('Profile: %s\nInput: %s\nOutput: %s\n',profile,inputFile,runDir);
fprintf('GPU: %d | mirror X/Y: %d/%d | boundary weighting: %d\n', ...
    cfg.runtime.useGPU,cfg.augmentation.mirrorX, ...
    cfg.augmentation.mirrorY,cfg.loss.useBoundaryWeight);

[D,meta] = fbg_load_data(inputFile,cfg);
assignments = fbg_make_case_folds(D,cfg);
writetable(assignments,fullfile(runDir,'case_fold_assignments.csv'));
localWriteProtocol(cfg,meta,D,runDir);

foldIds = unique(assignments.fold_id,'stable');
nFolds = numel(foldIds);
fprintf('Cases: %d | folds: %d | rows: %d | iterations/model/fold: %d\n', ...
    height(assignments),nFolds,height(D.table),cfg.training.maxIterations);

fullPredictionParts = cell(2*nFolds,1);
fullMetricParts = cell(2*nFolds,1);
fixedPredictionParts = cell(nFolds,1);
fixedMetricParts = cell(nFolds,1);
calibrationParts = cell(nFolds,1);
timingParts = cell(2*nFolds,1);

for iFold = 1:nFolds
    foldId = foldIds(iFold);
    foldDir = fullfile(runDir,sprintf('fold_%02d',foldId));
    if ~exist(foldDir,'dir'), mkdir(foldDir); end

    testCases = assignments.case_id(assignments.fold_id == foldId);
    remainingCases = assignments.case_id(assignments.fold_id ~= foldId);
    validationCases = localValidationCases(assignments,remainingCases,cfg,foldId);
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

    fprintf('\n=== Fold %d/%d | train/val/test cases %d/%d/%d ===\n', ...
        iFold,nFolds,numel(trainCases),numel(validationCases),numel(testCases));

    % Identical initialization and identical minibatch sequence.
    initSeed = cfg.randomSeed+foldId*100+1;
    batchSeed = cfg.randomSeed+foldId*10000+17;
    rng(initSeed);
    sharedInitialParams = fbg_model_init("sota",foldMeta,cfg);

    [sotaResult,sotaSeconds] = localRunStage( ...
        fullfile(foldDir,'01_step19_sota_film'),sharedInitialParams, ...
        "sota","Step19-SOTA-FiLM",batchSeed,foldD,split,foldMeta,cfg);
    [residualResult,residualSeconds] = localRunStage( ...
        fullfile(foldDir,'02_step19_residual_film'),sharedInitialParams, ...
        "residual","Step19-Residual-FiLM",batchSeed,foldD,split,foldMeta,cfg);

    fullPredictionParts{2*iFold-1} = localTestPredictions( ...
        sotaResult.predictions,foldId,"Step19-SOTA-FiLM");
    fullPredictionParts{2*iFold} = localTestPredictions( ...
        residualResult.predictions,foldId,"Step19-Residual-FiLM");
    fullMetricParts{2*iFold-1} = localTestCaseMetrics( ...
        sotaResult.caseMetrics,foldId,"Step19-SOTA-FiLM");
    fullMetricParts{2*iFold} = localTestCaseMetrics( ...
        residualResult.caseMetrics,foldId,"Step19-Residual-FiLM");

    sotaInference = localTimeInference(sotaResult.params,"sota", ...
        foldD,split.test,foldMeta,cfg);
    residualInference = localTimeInference(residualResult.params,"residual", ...
        foldD,split.test,foldMeta,cfg);
    timingParts{2*iFold-1} = localTimingRow(foldId, ...
        "Step19-SOTA-FiLM",sotaResult.params,sotaResult.history, ...
        sotaSeconds,sotaInference,sum(split.test));
    timingParts{2*iFold} = localTimingRow(foldId, ...
        "Step19-Residual-FiLM",residualResult.params,residualResult.history, ...
        residualSeconds,residualInference,sum(split.test));

    [fixedPredictionParts{iFold},fixedMetricParts{iFold}, ...
        calibrationParts{iFold}] = localFixedEvaluation( ...
        foldId,foldD,foldMeta,split,sotaResult,residualResult,cfg);

    foldStatus = table(foldId,true,string(datetime('now')), ...
        'VariableNames',{'fold_id','completed','completion_time'});
    writetable(foldStatus,fullfile(foldDir,'FOLD_COMPLETE.csv'));
end

fullPredictions = vertcat(fullPredictionParts{:});
fullCaseMetrics = vertcat(fullMetricParts{:});
fixedPredictions = vertcat(fixedPredictionParts{:});
fixedCaseMetrics = vertcat(fixedMetricParts{:});
calibrationPoints = vertcat(calibrationParts{:});
timingByFold = vertcat(timingParts{:});

writetable(fullPredictions,fullfile(runDir,'full_zero_shot_predictions.csv'));
writetable(fullCaseMetrics,fullfile(runDir,'full_zero_shot_case_metrics.csv'));
writetable(fixedPredictions,fullfile(runDir,'common_fixed_test_predictions.csv'));
writetable(fixedCaseMetrics,fullfile(runDir,'common_fixed_test_case_metrics.csv'));
writetable(calibrationPoints,fullfile(runDir,'calibration_point_assignments.csv'));
writetable(timingByFold,fullfile(runDir,'training_timing_by_fold.csv'));

zeroSummary = localSummary(fullPredictions,fullCaseMetrics);
fixedSummary = localSummary(fixedPredictions,fixedCaseMetrics);
paired = localPairedSummary(fixedCaseMetrics);
target95 = localTargetSummary(fixedCaseMetrics,cfg.data.targetCasePattern);
runtimeSummary = localRuntimeSummary(timingByFold,fixedCaseMetrics);
complexity = localComplexitySummary(timingByFold);

writetable(zeroSummary,fullfile(runDir,'full_zero_shot_summary.csv'));
writetable(fixedSummary,fullfile(runDir,'common_fixed_test_summary.csv'));
writetable(paired,fullfile(runDir,'paired_improvement_summary.csv'));
writetable(target95,fullfile(runDir,'target_95cm_summary.csv'));
writetable(runtimeSummary,fullfile(runDir,'runtime_summary.csv'));
writetable(complexity,fullfile(runDir,'model_complexity.csv'));
localPlotComparison(zeroSummary,fixedSummary, ...
    fullfile(runDir,'sota_full_comparison.png'));

fprintf('\nFull SOTA comparison completed.\nResults: %s\n',runDir);
end

function [result,seconds] = localRunStage(stageDir,initialParams,modelKind, ...
    stageLabel,batchSeed,D,split,meta,cfg)
modelFile = fullfile(stageDir,'model.mat');
historyFile = fullfile(stageDir,'training_history.csv');
if cfg.runtime.resumeCompletedStages && isfile(modelFile) && isfile(historyFile)
    saved = load(modelFile,'paramsCPU');
    params = localMoveParams(saved.paramsCPU,cfg.runtime.useGPU);
    history = readtable(historyFile);
    seconds = max(history.elapsed_seconds,[],'omitnan');
    fprintf('[resume] Loaded completed stage %s.\n',stageLabel);
else
    params = initialParams;
    rng(batchSeed);
    timerValue = tic;
    [params,history] = fbg_train_model(params,modelKind,D, ...
        split.train,split.validation,meta,cfg,stageLabel);
    seconds = toc(timerValue);
end
result = fbg_save_stage(stageDir,params,modelKind,history, ...
    D,split,meta,cfg);
end

function value = localMoveParams(value,useGPU)
if isa(value,'dlarray')
    numeric = gather(extractdata(value));
    if useGPU, numeric = gpuArray(numeric); end
    value = dlarray(numeric);
elseif isstruct(value)
    for k = 1:numel(value)
        names = fieldnames(value(k));
        for i = 1:numel(names)
            value(k).(names{i}) = localMoveParams(value(k).(names{i}),useGPU);
        end
    end
end
end

function validationCases = localValidationCases(assignments,remaining,cfg,foldId)
nVal = max(1,round(cfg.global.validationFraction*height(assignments)));
nVal = min(nVal,numel(remaining)-1);
rng(cfg.randomSeed+foldId*7919);
remaining = remaining(randperm(numel(remaining)));
validationCases = remaining(1:nVal);
end

function out = localTestPredictions(predictions,foldId,method)
out = predictions(predictions.split == "test",:);
out = addvars(out,repmat(foldId,height(out),1), ...
    repmat(method,height(out),1),'Before',1, ...
    'NewVariableNames',{'fold_id','method'});
end

function out = localTestCaseMetrics(caseMetrics,foldId,method)
out = caseMetrics(caseMetrics.split == "test",:);
out = addvars(out,repmat(foldId,height(out),1), ...
    repmat(method,height(out),1),'Before',1, ...
    'NewVariableNames',{'fold_id','method'});
end

function [predictions,metrics,pointAssignments] = localFixedEvaluation( ...
    foldId,D,meta,split,sotaResult,residualResult,cfg)
testCases = split.testCases;
predictionParts = {};
metricParts = {};
pointParts = {};
pi = 0; mi = 0; ai = 0;
k = cfg.global.calibrationK;

for iCase = 1:numel(testCases)
    caseId = testCases(iCase);
    caseMask = split.test & D.table.case_id == caseId;
    rows = find(caseMask);
    coords = unique(D.table(rows,{'x_cm','y_cm'}),'rows','stable');
    if height(coords) < k+cfg.global.minimumFixedTestCoordinates
        warning('Case %s has too few coordinates for Ridge-%d.',caseId,k);
        continue;
    end
    firstRow = rows(1);
    order = localCoverageOrder(coords,D.p1x(firstRow),D.p1y(firstRow));
    calibrationCoords = coords(order(1:k),:);
    fixedCoords = coords(order(k+1:end),:);
    calibrationMask = localCoordinateMask(D.table,caseMask,calibrationCoords);
    fixedMask = localCoordinateMask(D.table,caseMask,fixedCoords);

    assigned = coords(order,:);
    assigned.selection_order = (1:height(assigned))';
    assigned.role = repmat("fixed_test",height(assigned),1);
    assigned.role(1:k) = "calibration";
    assigned.fold_id = repmat(foldId,height(assigned),1);
    assigned.case_id = repmat(caseId,height(assigned),1);
    assigned = movevars(assigned,{'fold_id','case_id'},'Before',1);
    ai = ai+1; pointParts{ai,1} = assigned; %#ok<AGROW>

    sotaFixed = localPredictionRows(sotaResult.predictions,fixedMask);
    residualFixed = localPredictionRows(residualResult.predictions,fixedMask);
    pi = pi+1; predictionParts{pi,1} = localLabelPrediction( ...
        sotaFixed,foldId,"Step19-SOTA-FiLM",0,1,0); %#ok<AGROW>
    mi = mi+1; metricParts{mi,1} = localMetricRow( ...
        sotaFixed,foldId,caseId,"Step19-SOTA-FiLM",0,0,1,0); %#ok<AGROW>
    pi = pi+1; predictionParts{pi,1} = localLabelPrediction( ...
        residualFixed,foldId,"Step19-Residual-FiLM",0,1,0); %#ok<AGROW>
    mi = mi+1; metricParts{mi,1} = localMetricRow( ...
        residualFixed,foldId,caseId,"Step19-Residual-FiLM",0,0,1,0); %#ok<AGROW>

    calibrationPrediction = localPredictionRows( ...
        residualResult.predictions,calibrationMask);
    [scaleA,offsetB,seconds] = localTimedRidge( ...
        calibrationPrediction,meta,cfg);
    calibrated = residualFixed;
    calibrated.predicted_microstrain = ...
        scaleA*calibrated.predicted_microstrain+offsetB;
    calibrated.residual_microstrain = calibrated.predicted_microstrain- ...
        calibrated.measured_microstrain;
    method = "Step19-Residual+Ridge-4";
    pi = pi+1; predictionParts{pi,1} = localLabelPrediction( ...
        calibrated,foldId,method,k,scaleA,offsetB); %#ok<AGROW>
    mi = mi+1; metricParts{mi,1} = localMetricRow( ...
        calibrated,foldId,caseId,method,k,seconds,scaleA,offsetB); %#ok<AGROW>
end

predictions = vertcat(predictionParts{:});
metrics = vertcat(metricParts{:});
pointAssignments = vertcat(pointParts{:});
end

function order = localCoverageOrder(coords,loadX,loadY)
n = height(coords);
sx = max(max(coords.x_cm)-min(coords.x_cm),1);
sy = max(max(coords.y_cm)-min(coords.y_cm),1);
[~,first] = min(((coords.x_cm-loadX)/sx).^2+ ...
    ((coords.y_cm-loadY)/sy).^2);
order = zeros(n,1); order(1) = first;
remaining = setdiff((1:n)',first,'stable');
for i = 2:n
    selected = order(1:i-1);
    distance = zeros(numel(remaining),1);
    for j = 1:numel(remaining)
        dx = (coords.x_cm(remaining(j))-coords.x_cm(selected))/sx;
        dy = (coords.y_cm(remaining(j))-coords.y_cm(selected))/sy;
        distance(j) = min(dx.^2+dy.^2);
    end
    [~,pick] = max(distance);
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
nRepeats = cfg.global.calibrationTimingRepeats;
samples = zeros(nRepeats,1);
for i = 1:nRepeats
    timerValue = tic;
    [scaleA,offsetB] = localFitRidge(prediction,meta,cfg);
    samples(i) = toc(timerValue);
end
seconds = median(samples);
end

function [scaleA,offsetB] = localFitRidge(prediction,meta,cfg)
z = prediction.predicted_microstrain/meta.targetScale;
t = prediction.measured_microstrain/meta.targetScale;
X = [z ones(size(z))];
penalty = diag([cfg.fewshot.affineSlopePenalty, ...
    cfg.fewshot.affineOffsetPenalty]);
theta = (X'*X+penalty)\(X'*(t-z));
scaleA = 1+theta(1);
bounds = cfg.fewshot.affineScaleBounds;
scaleA = min(max(scaleA,bounds(1)),bounds(2));
offsetB = theta(2)*meta.targetScale;
end

function seconds = localTimeInference(params,modelKind,D,mask,meta,cfg)
fbg_predict(params,modelKind,D,mask,meta,cfg);
samples = zeros(cfg.global.inferenceTimingRepeats,1);
for i = 1:numel(samples)
    timerValue = tic;
    fbg_predict(params,modelKind,D,mask,meta,cfg);
    samples(i) = toc(timerValue);
end
seconds = median(samples);
end

function row = localTimingRow(foldId,method,params,history, ...
    trainingSeconds,inferenceSeconds,nRows)
valid = isfinite(history.validation_rmse_microstrain);
sub = history(valid,:);
if isempty(sub)
    bestRmse = NaN; bestIteration = NaN; timeToBest = NaN;
else
    [bestRmse,index] = min(sub.validation_rmse_microstrain);
    bestIteration = sub.iteration(index);
    timeToBest = sub.elapsed_seconds(index);
end
row = table(foldId,method,trainingSeconds,history.iteration(end), ...
    bestIteration,bestRmse,timeToBest,inferenceSeconds, ...
    1000*inferenceSeconds/max(nRows,1),localParameterCount(params), ...
    'VariableNames',{'fold_id','method','training_seconds', ...
    'iterations_run','best_iteration','best_validation_RMSE_microstrain', ...
    'time_to_best_seconds','test_inference_seconds', ...
    'inference_milliseconds_per_row','parameter_count'});
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

function summary = localSummary(predictions,caseMetrics)
methods = unique(predictions.method,'stable');
parts = cell(numel(methods),1);
for i = 1:numel(methods)
    p = predictions(predictions.method == methods(i),:);
    c = caseMetrics(caseMetrics.method == methods(i),:);
    [r2,rmse,mae,bias] = localMetrics( ...
        p.measured_microstrain,p.predicted_microstrain);
    parts{i} = table(methods(i),numel(unique(c.case_id)),height(p), ...
        r2,rmse,mae,bias,median(c.R2,'omitnan'), ...
        sum(c.R2 > 0),mean(c.RMSE_microstrain,'omitnan'), ...
        std(c.RMSE_microstrain,0,'omitnan'), ...
        max(c.RMSE_microstrain,[],'omitnan'), ...
        mean(c.MAE_microstrain,'omitnan'), ...
        'VariableNames',{'method','n_cases','n_rows','micro_R2', ...
        'micro_RMSE_microstrain','micro_MAE_microstrain', ...
        'micro_Bias_microstrain','median_case_R2','n_cases_R2_positive', ...
        'macro_RMSE_mean_microstrain','macro_RMSE_std_microstrain', ...
        'worst_case_RMSE_microstrain','macro_MAE_mean_microstrain'});
end
summary = vertcat(parts{:});
end

function paired = localPairedSummary(metrics)
cases = unique(metrics.case_id,'stable');
parts = cell(2,1);
comparisons = {
    "Step19-SOTA-FiLM","Step19-Residual-FiLM";
    "Step19-Residual-FiLM","Step19-Residual+Ridge-4"};
for i = 1:2
    baseline = comparisons{i,1};
    candidate = comparisons{i,2};
    deltaRmse = nan(numel(cases),1);
    deltaR2 = nan(numel(cases),1);
    for j = 1:numel(cases)
        b = metrics(metrics.case_id == cases(j) & metrics.method == baseline,:);
        c = metrics(metrics.case_id == cases(j) & metrics.method == candidate,:);
        if ~isempty(b) && ~isempty(c)
            deltaRmse(j) = b.RMSE_microstrain(1)-c.RMSE_microstrain(1);
            deltaR2(j) = c.R2(1)-b.R2(1);
        end
    end
    valid = isfinite(deltaRmse);
    parts{i} = table(string(baseline),string(candidate),sum(valid), ...
        mean(deltaRmse(valid),'omitnan'),median(deltaRmse(valid),'omitnan'), ...
        sum(deltaRmse(valid)>0),mean(deltaR2(valid),'omitnan'), ...
        'VariableNames',{'baseline','candidate','n_paired_cases', ...
        'mean_RMSE_reduction_microstrain','median_RMSE_reduction_microstrain', ...
        'n_cases_RMSE_improved','mean_R2_improvement'});
end
paired = vertcat(parts{:});
end

function target = localTargetSummary(metrics,pattern)
target = metrics(contains(lower(metrics.case_id),lower(pattern)),:);
end

function summary = localRuntimeSummary(timing,fixedMetrics)
methods = unique(timing.method,'stable');
parts = cell(numel(methods)+1,1);
for i = 1:numel(methods)
    sub = timing(timing.method == methods(i),:);
    parts{i} = table(methods(i),mean(sub.training_seconds), ...
        std(sub.training_seconds),mean(sub.time_to_best_seconds,'omitnan'), ...
        mean(sub.test_inference_seconds), ...
        mean(sub.inference_milliseconds_per_row), ...
        mean(sub.parameter_count),0, ...
        'VariableNames',{'method','mean_training_seconds', ...
        'std_training_seconds','mean_time_to_best_seconds', ...
        'mean_test_inference_seconds','mean_inference_ms_per_row', ...
        'parameter_count','mean_calibration_seconds'});
end
ridge = fixedMetrics(fixedMetrics.method == "Step19-Residual+Ridge-4",:);
residual = timing(timing.method == "Step19-Residual-FiLM",:);
parts{end} = table("Step19-Residual+Ridge-4", ...
    mean(residual.training_seconds),std(residual.training_seconds), ...
    mean(residual.time_to_best_seconds,'omitnan'), ...
    mean(residual.test_inference_seconds), ...
    mean(residual.inference_milliseconds_per_row), ...
    mean(residual.parameter_count)+2,mean(ridge.calibration_seconds,'omitnan'), ...
    'VariableNames',parts{1}.Properties.VariableNames);
summary = vertcat(parts{:});
end

function complexity = localComplexitySummary(timing)
methods = unique(timing.method,'stable');
parts = cell(numel(methods)+1,1);
for i = 1:numel(methods)
    sub = timing(timing.method == methods(i),:);
    parts{i} = table(methods(i),round(mean(sub.parameter_count)),0, ...
        'VariableNames',{'method','backbone_parameters','added_parameters'});
end
residual = timing(timing.method == "Step19-Residual-FiLM",:);
parts{end} = table("Step19-Residual+Ridge-4", ...
    round(mean(residual.parameter_count)),2, ...
    'VariableNames',parts{1}.Properties.VariableNames);
complexity = vertcat(parts{:});
end

function [r2,rmse,mae,bias] = localMetrics(measured,predicted)
valid = isfinite(measured) & isfinite(predicted);
measured = measured(valid); predicted = predicted(valid);
if isempty(measured)
    r2 = NaN; rmse = NaN; mae = NaN; bias = NaN; return;
end
residual = predicted-measured;
rmse = sqrt(mean(residual.^2));
mae = mean(abs(residual));
bias = mean(residual);
denom = sum((measured-mean(measured)).^2);
if denom <= eps, r2 = NaN;
else, r2 = 1-sum(residual.^2)/denom;
end
end

function localWriteProtocol(cfg,meta,D,runDir)
name = ["comparison_design";"case_split";"model_width"; ...
    "condition_width";"film_blocks";"mirror_x";"mirror_y"; ...
    "boundary_feature_length";"boundary_weighting"; ...
    "iterations_per_model_per_fold";"folds";"ridge_points"];
value = ["shared initialization and minibatches";"held-out whole case"; ...
    string(cfg.model.width);string(cfg.model.conditionWidth); ...
    string(cfg.model.numBlocks);string(cfg.augmentation.mirrorX); ...
    string(cfg.augmentation.mirrorY); ...
    string(cfg.model.boundaryFeatureLength); ...
    string(cfg.loss.useBoundaryWeight); ...
    string(cfg.training.maxIterations);string(cfg.global.numFolds); ...
    string(cfg.global.calibrationK)];
writetable(table(name,value),fullfile(runDir,'comparison_protocol.csv'));
dataset = table(height(D.table),numel(unique(D.table.case_id)), ...
    meta.xMin,meta.xMax,meta.yMin,meta.yMax,meta.targetScale, ...
    'VariableNames',{'n_rows','n_cases','x_min_cm','x_max_cm', ...
    'y_min_cm','y_max_cm','response_scale_microstrain'});
writetable(dataset,fullfile(runDir,'dataset_summary.csv'));
end

function localPlotComparison(zeroSummary,fixedSummary,outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 1050 480]);
tiledlayout(1,2);
nexttile;
bar(categorical(zeroSummary.method),zeroSummary.micro_R2);
ylabel('Micro R^2'); title('Full zero-shot held-out cases'); grid on;
nexttile;
bar(categorical(fixedSummary.method),fixedSummary.micro_RMSE_microstrain);
ylabel('RMSE (\muepsilon)'); title('Common fixed test coordinates'); grid on;
saveas(fig,outputFile); close(fig);
end
