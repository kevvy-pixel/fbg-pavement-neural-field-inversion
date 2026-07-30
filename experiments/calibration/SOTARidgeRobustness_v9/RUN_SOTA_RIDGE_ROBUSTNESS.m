function outputDir = RUN_SOTA_RIDGE_ROBUSTNESS( ...
    runDir,numRepeats,outputDir)
%RUN_SOTA_RIDGE_ROBUSTNESS Repeated nested Ridge-4/Ridge-6 evaluation.
%
% No neural network is trained. Each repeat selects six spatially covering
% calibration coordinates. Ridge-4 uses the first four, Ridge-6 uses all
% six, and every method is evaluated on the same remaining coordinates.

arguments
    runDir (1,1) string
    numRepeats (1,1) double {mustBeInteger,mustBePositive} = 20
    outputDir (1,1) string = ""
end

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
cfg = fbg_config("full");
cfg.robust.numRepeats = numRepeats;
if ~exist(runDir,'dir')
    error('Existing full-comparison directory not found: %s',runDir);
end
foldFile = fullfile(runDir,'case_fold_assignments.csv');
if ~isfile(foldFile)
    error('Missing case_fold_assignments.csv in: %s',runDir);
end
if strlength(outputDir) == 0
    outputDir = fullfile(runDir,cfg.robust.outputFolder);
end
if ~exist(outputDir,'dir'), mkdir(outputDir); end

diary(fullfile(outputDir,'run.log'));
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('FBG SOTA Ridge-4/6 Repeated Spatial Robustness Study\n');
fprintf('Input: %s\nOutput: %s\nRepeats: %d\n', ...
    runDir,outputDir,numRepeats);
fprintf(['Nested design: Ridge-4 uses points 1:4; Ridge-6 uses 1:6; ', ...
    'all methods share the fixed set after point 6.\n']);

foldAssignments = readtable(foldFile,'TextType','string', ...
    'VariableNamingRule','preserve');
foldIds = unique(foldAssignments.fold_id,'stable');

predictionParts = {};
metricParts = {};
splitParts = {};
gateParts = {};
pi = 0; mi = 0; si = 0; gi = 0;

for iFold = 1:numel(foldIds)
    foldId = foldIds(iFold);
    foldDir = fullfile(runDir,sprintf('fold_%02d',foldId));
    predictionFile = fullfile(foldDir,'01_step19_sota_film','predictions.csv');
    configurationFile = fullfile(foldDir,'fold_configuration.mat');
    if ~isfile(predictionFile) || ~isfile(configurationFile)
        error('Missing saved SOTA predictions/configuration in fold %d.',foldId);
    end
    prediction = readtable(predictionFile,'TextType','string', ...
        'VariableNamingRule','preserve');
    saved = load(configurationFile,'foldMeta');
    foldMeta = saved.foldMeta;
    foldCases = foldAssignments.case_id( ...
        foldAssignments.fold_id == foldId);

    fprintf('\nFold %d: %d cases.\n',foldId,numel(foldCases));
    for iCase = 1:numel(foldCases)
        caseId = foldCases(iCase);
        casePrediction = prediction(prediction.case_id == caseId & ...
            prediction.split == "test",:);
        coords = unique(casePrediction(:,{'x_cm','y_cm'}),'rows','stable');
        if height(coords) < cfg.robust.maxK+ ...
                cfg.robust.minimumFixedTestCoordinates
            error('Case %s has too few coordinates for the design.',caseId);
        end

        splitSeed = cfg.randomSeed+foldId*100000+iCase*1000;
        [orders,splitInfo] = localUniqueCoverageOrders( ...
            coords,cfg.robust.maxK,numRepeats,splitSeed,cfg);
        splitInfo.fold_id = repmat(foldId,height(splitInfo),1);
        splitInfo.case_id = repmat(caseId,height(splitInfo),1);
        splitInfo = movevars(splitInfo,{'fold_id','case_id'},'Before',1);

        for repeatIndex = 1:numRepeats
            order = orders(:,repeatIndex);
            calibration6 = coords(order(1:6),:);
            calibration4 = coords(order(1:4),:);
            fixedCoords = coords(order(7:end),:);
            fixedPrediction = casePrediction( ...
                localCoordinateMask(casePrediction,fixedCoords),:);
            calibrationPrediction4 = casePrediction( ...
                localCoordinateMask(casePrediction,calibration4),:);
            calibrationPrediction6 = casePrediction( ...
                localCoordinateMask(casePrediction,calibration6),:);

            timerValue = tic;
            [scale4,offset4] = localFitRidge( ...
                calibrationPrediction4,foldMeta,cfg);
            seconds4 = toc(timerValue);
            timerValue = tic;
            [scale6,offset6] = localFitRidge( ...
                calibrationPrediction6,foldMeta,cfg);
            seconds6 = toc(timerValue);

            [cvBaseline6,cvRidge6] = localCoordinateLOOCV( ...
                calibrationPrediction6,calibration6,foldMeta,cfg);
            cvAbsoluteGain6 = cvBaseline6-cvRidge6;
            cvRelativeGain6 = 100*cvAbsoluteGain6/max(cvBaseline6,eps);
            bounds = cfg.fewshot.affineScaleBounds;
            scaleInterior = scale6 > bounds(1)+cfg.gate.scaleBoundMargin && ...
                scale6 < bounds(2)-cfg.gate.scaleBoundMargin;
            finiteFit = all(isfinite([cvBaseline6,cvRidge6,scale6,offset6]));
            gate6 = finiteFit && scaleInterior && ...
                cvAbsoluteGain6 >= ...
                cfg.gate.minAbsoluteCVImprovementMicrostrain && ...
                cvRelativeGain6 >= ...
                cfg.gate.minRelativeCVImprovementPercent;
            gateReason = localGateReason(finiteFit,scaleInterior, ...
                cvRelativeGain6,cvAbsoluteGain6,cfg);

            baseline = fixedPrediction;
            ridge4 = localApplyAffine(fixedPrediction,scale4,offset4);
            ridge6 = localApplyAffine(fixedPrediction,scale6,offset6);
            if gate6, gated6 = ridge6;
            else, gated6 = baseline;
            end

            methods = ["Step19-SOTA-FiLM";"Step19-SOTA+Ridge-4"; ...
                "Step19-SOTA+Ridge-6";"Step19-SOTA+Gated-Ridge-6"];
            methodPredictions = {baseline;ridge4;ridge6;gated6};
            kPoints = [0;4;6;6];
            scales = [1;scale4;scale6;scale6];
            offsets = [0;offset4;offset6;offset6];
            seconds = [0;seconds4;seconds6;seconds6];
            for iMethod = 1:numel(methods)
                p = methodPredictions{iMethod};
                pi = pi+1;
                predictionParts{pi,1} = localLabelPrediction( ...
                    p,foldId,caseId,repeatIndex, ...
                    splitInfo.seed(repeatIndex),methods(iMethod), ...
                    kPoints(iMethod),gate6,scales(iMethod), ...
                    offsets(iMethod)); %#ok<AGROW>
                mi = mi+1;
                metricParts{mi,1} = localMetricRow( ...
                    p,foldId,caseId,repeatIndex, ...
                    splitInfo.seed(repeatIndex),methods(iMethod), ...
                    kPoints(iMethod),gate6,seconds(iMethod), ...
                    scales(iMethod),offsets(iMethod), ...
                    cvBaseline6,cvRidge6,cvAbsoluteGain6, ...
                    cvRelativeGain6,gateReason); %#ok<AGROW>
            end

            assigned = calibration6;
            assigned.selection_order = (1:6)';
            assigned.used_for_ridge4 = assigned.selection_order <= 4;
            assigned.used_for_ridge6 = true(6,1);
            assigned.fold_id = repmat(foldId,6,1);
            assigned.case_id = repmat(caseId,6,1);
            assigned.repeat_index = repmat(repeatIndex,6,1);
            assigned.seed = repmat(splitInfo.seed(repeatIndex),6,1);
            assigned = movevars(assigned, ...
                {'fold_id','case_id','repeat_index','seed'},'Before',1);
            si = si+1; splitParts{si,1} = assigned; %#ok<AGROW>

            [~,baselineRmse,~,~] = localMetrics( ...
                baseline.measured_microstrain,baseline.predicted_microstrain);
            [~,ridge6Rmse,~,~] = localMetrics( ...
                ridge6.measured_microstrain,ridge6.predicted_microstrain);
            gi = gi+1;
            gateParts{gi,1} = table(foldId,caseId,repeatIndex, ...
                splitInfo.seed(repeatIndex),gate6,gateReason, ...
                cvBaseline6,cvRidge6,cvAbsoluteGain6,cvRelativeGain6, ...
                scale6,offset6,baselineRmse,ridge6Rmse, ...
                baselineRmse-ridge6Rmse, ...
                'VariableNames',{'fold_id','case_id','repeat_index', ...
                'seed','gate_applied','gate_reason', ...
                'cv_baseline_RMSE_microstrain', ...
                'cv_ridge_RMSE_microstrain', ...
                'cv_absolute_gain_microstrain', ...
                'cv_relative_gain_percent','calibration_scale', ...
                'calibration_offset_microstrain', ...
                'baseline_RMSE_microstrain','ridge6_RMSE_microstrain', ...
                'fixed_RMSE_gain_if_ridge6_microstrain'}); %#ok<AGROW>
        end
        fprintf('  %s: %d unique nested splits completed.\n', ...
            caseId,numRepeats);
    end
end

allPredictions = vertcat(predictionParts{:});
repeatCaseMetrics = vertcat(metricParts{:});
calibrationAssignments = vertcat(splitParts{:});
gateDecisions = vertcat(gateParts{:});

globalByRepeat = localGlobalByRepeat(allPredictions);
methodSummary = localMethodSummary(globalByRepeat);
paired = localPairedSummary(repeatCaseMetrics);
caseRobustness = localCaseRobustness(repeatCaseMetrics);
target95Repeat = repeatCaseMetrics( ...
    contains(lower(repeatCaseMetrics.case_id),"95cm"),:);
target95Summary = caseRobustness( ...
    contains(lower(caseRobustness.case_id),"95cm"),:);
gateAudit = localGateAudit(gateDecisions);

writetable(methodSummary,fullfile(outputDir,'method_summary.csv'));
writetable(globalByRepeat,fullfile(outputDir,'global_metrics_by_repeat.csv'));
writetable(repeatCaseMetrics,fullfile(outputDir, ...
    'repeat_case_metrics.csv'));
writetable(caseRobustness,fullfile(outputDir, ...
    'case_robustness_summary.csv'));
writetable(paired,fullfile(outputDir,'paired_improvement_summary.csv'));
writetable(target95Repeat,fullfile(outputDir, ...
    'target_95cm_repeat_metrics.csv'));
writetable(target95Summary,fullfile(outputDir, ...
    'target_95cm_summary.csv'));
writetable(gateDecisions,fullfile(outputDir, ...
    'ridge6_gate_decisions.csv'));
writetable(gateAudit,fullfile(outputDir,'ridge6_gate_audit.csv'));
writetable(calibrationAssignments,fullfile(outputDir, ...
    'calibration_point_assignments.csv'));
writetable(allPredictions,fullfile(outputDir,'predictions.csv'));
localWriteConfiguration(cfg,outputDir);
localPlotMethodSummary(methodSummary, ...
    fullfile(outputDir,'method_robustness.png'));
localPlotCaseWinRates(caseRobustness, ...
    fullfile(outputDir,'case_win_rates.png'));

fprintf('\nSOTA Ridge robustness study completed.\nResults: %s\n',outputDir);
end

function [orders,splitInfo] = localUniqueCoverageOrders( ...
    coords,poolSize,nRepeats,baseSeed,cfg)
n = height(coords);
if nRepeats > n
    error('numRepeats=%d exceeds the %d available start coordinates.', ...
        nRepeats,n);
end
rng(baseSeed);
firstCandidates = randperm(n);
orders = zeros(n,nRepeats);
signatures = strings(nRepeats,1);
firstIndices = zeros(nRepeats,1);
seeds = zeros(nRepeats,1);
accepted = 0;
attempt = 0;
while accepted < nRepeats && ...
        attempt < cfg.robust.maxUniqueSplitAttempts
    attempt = attempt+1;
    position = mod(attempt-1,n)+1;
    firstIndex = firstCandidates(position);
    cycle = floor((attempt-1)/n);
    seed = baseSeed+attempt*cfg.robust.seedStride+cycle;
    order = localCoverageOrder(coords,firstIndex,seed);
    signature = strjoin(string(sort(order(1:poolSize)))',"-");
    if ~any(signatures(1:accepted) == signature)
        accepted = accepted+1;
        orders(:,accepted) = order;
        signatures(accepted) = signature;
        firstIndices(accepted) = firstIndex;
        seeds(accepted) = seed;
    end
end
if accepted < nRepeats
    error('Only %d unique calibration pools could be generated.',accepted);
end
repeat_index = (1:nRepeats)';
seed = seeds;
first_coordinate_index = firstIndices;
calibration_pool_signature = signatures;
is_unique = true(nRepeats,1);
splitInfo = table(repeat_index,seed,first_coordinate_index, ...
    calibration_pool_signature,is_unique);
end

function order = localCoverageOrder(coords,firstIndex,seed)
rng(seed);
n = height(coords);
sx = max(max(coords.x_cm)-min(coords.x_cm),1);
sy = max(max(coords.y_cm)-min(coords.y_cm),1);
order = zeros(n,1); order(1) = firstIndex;
remaining = setdiff((1:n)',firstIndex,'stable');
for i = 2:n
    selected = order(1:i-1);
    distance = zeros(numel(remaining),1);
    for j = 1:numel(remaining)
        dx = (coords.x_cm(remaining(j))-coords.x_cm(selected))/sx;
        dy = (coords.y_cm(remaining(j))-coords.y_cm(selected))/sy;
        distance(j) = min(dx.^2+dy.^2);
    end
    jitter = 0.01*max(distance)*rand(size(distance));
    [~,pick] = max(distance+jitter);
    order(i) = remaining(pick);
    remaining(pick) = [];
end
end

function mask = localCoordinateMask(prediction,coords)
mask = false(height(prediction),1);
for i = 1:height(coords)
    mask = mask | (prediction.x_cm == coords.x_cm(i) & ...
        prediction.y_cm == coords.y_cm(i));
end
end

function [baselineRmse,ridgeRmse] = localCoordinateLOOCV( ...
    calibrationPrediction,coords,meta,cfg)
measured = zeros(0,1);
baseline = zeros(0,1);
ridge = zeros(0,1);
for i = 1:height(coords)
    heldMask = calibrationPrediction.x_cm == coords.x_cm(i) & ...
        calibrationPrediction.y_cm == coords.y_cm(i);
    trainPrediction = calibrationPrediction(~heldMask,:);
    heldPrediction = calibrationPrediction(heldMask,:);
    [scaleA,offsetB] = localFitRidge(trainPrediction,meta,cfg);
    measured = [measured;heldPrediction.measured_microstrain]; %#ok<AGROW>
    baseline = [baseline;heldPrediction.predicted_microstrain]; %#ok<AGROW>
    ridge = [ridge;scaleA*heldPrediction.predicted_microstrain+ ...
        offsetB]; %#ok<AGROW>
end
[~,baselineRmse,~,~] = localMetrics(measured,baseline);
[~,ridgeRmse,~,~] = localMetrics(measured,ridge);
end

function prediction = localApplyAffine(prediction,scaleA,offsetB)
prediction.predicted_microstrain = ...
    scaleA*prediction.predicted_microstrain+offsetB;
prediction.residual_microstrain = prediction.predicted_microstrain- ...
    prediction.measured_microstrain;
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

function reason = localGateReason(finiteFit,scaleInterior, ...
    relativeGain,absoluteGain,cfg)
if ~finiteFit
    reason = "nonfinite_fit";
elseif ~scaleInterior
    reason = "scale_near_bound";
elseif absoluteGain < cfg.gate.minAbsoluteCVImprovementMicrostrain
    reason = "absolute_cv_gain_below_threshold";
elseif relativeGain < cfg.gate.minRelativeCVImprovementPercent
    reason = "relative_cv_gain_below_threshold";
else
    reason = "cv_gate_passed";
end
end

function out = localLabelPrediction(prediction,foldId,caseId, ...
    repeatIndex,seed,method,k,gate6,scaleA,offsetB)
n = height(prediction);
out = addvars(prediction,repmat(foldId,n,1),repmat(caseId,n,1), ...
    repmat(repeatIndex,n,1),repmat(seed,n,1),repmat(method,n,1), ...
    repmat(k,n,1),repmat(gate6,n,1),repmat(scaleA,n,1), ...
    repmat(offsetB,n,1),'Before',1, ...
    'NewVariableNames',{'fold_id','evaluation_case_id', ...
    'repeat_index','seed','method','k_points','ridge6_gate_applied', ...
    'calibration_scale','calibration_offset_microstrain'});
end

function row = localMetricRow(prediction,foldId,caseId,repeatIndex, ...
    seed,method,k,gate6,seconds,scaleA,offsetB,cvBaseline6, ...
    cvRidge6,cvAbsoluteGain6,cvRelativeGain6,gateReason)
[r2,rmse,mae,bias] = localMetrics(prediction.measured_microstrain, ...
    prediction.predicted_microstrain);
row = table(foldId,caseId,repeatIndex,seed,method,k,gate6, ...
    height(prediction),seconds,scaleA,offsetB,cvBaseline6,cvRidge6, ...
    cvAbsoluteGain6,cvRelativeGain6,gateReason,r2,rmse,mae,bias, ...
    'VariableNames',{'fold_id','case_id','repeat_index','seed', ...
    'method','k_points','ridge6_gate_applied','n_fixed_test_rows', ...
    'calibration_seconds','calibration_scale', ...
    'calibration_offset_microstrain','cv6_baseline_RMSE_microstrain', ...
    'cv6_ridge_RMSE_microstrain','cv6_absolute_gain_microstrain', ...
    'cv6_relative_gain_percent','ridge6_gate_reason','R2', ...
    'RMSE_microstrain','MAE_microstrain','Bias_microstrain'});
end

function summary = localGlobalByRepeat(predictions)
methods = unique(predictions.method,'stable');
repeats = unique(predictions.repeat_index,'stable');
parts = cell(numel(methods)*numel(repeats),1);
index = 0;
for i = 1:numel(methods)
    for j = 1:numel(repeats)
        p = predictions(predictions.method == methods(i) & ...
            predictions.repeat_index == repeats(j),:);
        [r2,rmse,mae,bias] = localMetrics( ...
            p.measured_microstrain,p.predicted_microstrain);
        index = index+1;
        parts{index} = table(methods(i),repeats(j), ...
            numel(unique(p.evaluation_case_id)),height(p), ...
            r2,rmse,mae,bias, ...
            'VariableNames',{'method','repeat_index','n_cases','n_rows', ...
            'micro_R2','micro_RMSE_microstrain', ...
            'micro_MAE_microstrain','micro_Bias_microstrain'});
    end
end
summary = vertcat(parts{:});
end

function summary = localMethodSummary(globalByRepeat)
methods = unique(globalByRepeat.method,'stable');
parts = cell(numel(methods),1);
for i = 1:numel(methods)
    s = globalByRepeat(globalByRepeat.method == methods(i),:);
    parts{i} = table(methods(i),height(s), ...
        mean(s.micro_R2,'omitnan'),std(s.micro_R2,0,'omitnan'), ...
        mean(s.micro_RMSE_microstrain,'omitnan'), ...
        std(s.micro_RMSE_microstrain,0,'omitnan'), ...
        mean(s.micro_MAE_microstrain,'omitnan'), ...
        std(s.micro_MAE_microstrain,0,'omitnan'), ...
        mean(s.micro_Bias_microstrain,'omitnan'), ...
        std(s.micro_Bias_microstrain,0,'omitnan'), ...
        'VariableNames',{'method','n_repeats','micro_R2_mean', ...
        'micro_R2_std','micro_RMSE_mean_microstrain', ...
        'micro_RMSE_std_microstrain','micro_MAE_mean_microstrain', ...
        'micro_MAE_std_microstrain','micro_Bias_mean_microstrain', ...
        'micro_Bias_std_microstrain'});
end
summary = vertcat(parts{:});
end

function paired = localPairedSummary(metrics)
baselineName = "Step19-SOTA-FiLM";
candidates = ["Step19-SOTA+Ridge-4";"Step19-SOTA+Ridge-6"; ...
    "Step19-SOTA+Gated-Ridge-6"];
parts = cell(numel(candidates),1);
for i = 1:numel(candidates)
    baseline = metrics(metrics.method == baselineName,:);
    candidate = metrics(metrics.method == candidates(i),:);
    keys = intersect(baseline(:,{'case_id','repeat_index'}), ...
        candidate(:,{'case_id','repeat_index'}),'rows','stable');
    gain = nan(height(keys),1);
    deltaR2 = nan(height(keys),1);
    for j = 1:height(keys)
        b = baseline(baseline.case_id == keys.case_id(j) & ...
            baseline.repeat_index == keys.repeat_index(j),:);
        c = candidate(candidate.case_id == keys.case_id(j) & ...
            candidate.repeat_index == keys.repeat_index(j),:);
        gain(j) = b.RMSE_microstrain(1)-c.RMSE_microstrain(1);
        deltaR2(j) = c.R2(1)-b.R2(1);
    end
    parts{i} = table(baselineName,candidates(i),height(keys), ...
        mean(gain,'omitnan'),median(gain,'omitnan'),std(gain,0,'omitnan'), ...
        sum(gain>0),100*mean(gain>0),mean(deltaR2,'omitnan'), ...
        'VariableNames',{'baseline','candidate','n_case_repeat_pairs', ...
        'mean_RMSE_reduction_microstrain', ...
        'median_RMSE_reduction_microstrain', ...
        'std_RMSE_reduction_microstrain','n_pairs_RMSE_improved', ...
        'win_rate_percent','mean_R2_improvement'});
end
paired = vertcat(parts{:});
end

function summary = localCaseRobustness(metrics)
baselineName = "Step19-SOTA-FiLM";
candidates = ["Step19-SOTA+Ridge-4";"Step19-SOTA+Ridge-6"; ...
    "Step19-SOTA+Gated-Ridge-6"];
cases = unique(metrics.case_id,'stable');
parts = cell(numel(cases)*numel(candidates),1);
index = 0;
for i = 1:numel(cases)
    baseline = metrics(metrics.case_id == cases(i) & ...
        metrics.method == baselineName,:);
    for j = 1:numel(candidates)
        candidate = metrics(metrics.case_id == cases(i) & ...
            metrics.method == candidates(j),:);
        gain = baseline.RMSE_microstrain-candidate.RMSE_microstrain;
        index = index+1;
        parts{index} = table(cases(i),candidates(j),numel(gain), ...
            mean(candidate.R2,'omitnan'),std(candidate.R2,0,'omitnan'), ...
            mean(candidate.RMSE_microstrain,'omitnan'), ...
            std(candidate.RMSE_microstrain,0,'omitnan'), ...
            mean(gain,'omitnan'),std(gain,0,'omitnan'), ...
            median(gain,'omitnan'),100*mean(gain>0), ...
            localPercentile(gain,5),localPercentile(gain,95), ...
            'VariableNames',{'case_id','method','n_repeats', ...
            'R2_mean','R2_std','RMSE_mean_microstrain', ...
            'RMSE_std_microstrain','RMSE_gain_mean_microstrain', ...
            'RMSE_gain_std_microstrain','RMSE_gain_median_microstrain', ...
            'win_rate_percent','RMSE_gain_p05_microstrain', ...
            'RMSE_gain_p95_microstrain'});
    end
end
summary = vertcat(parts{1:index});
end

function audit = localGateAudit(decisions)
applied = decisions.gate_applied;
gain = decisions.fixed_RMSE_gain_if_ridge6_microstrain;
audit = table(height(decisions),sum(applied),sum(~applied), ...
    sum(applied & gain > 0),sum(applied & gain <= 0), ...
    100*mean(applied),100*mean(gain(applied)>0), ...
    mean(gain(applied),'omitnan'), ...
    'VariableNames',{'n_case_repeat_pairs','n_gate_applied', ...
    'n_gate_rejected','n_applied_pairs_improved', ...
    'n_applied_pairs_degraded','gate_application_rate_percent', ...
    'applied_pair_win_rate_percent', ...
    'mean_gain_when_applied_microstrain'});
end

function value = localPercentile(x,percent)
x = sort(x(isfinite(x)));
if isempty(x), value = NaN; return;
end
position = 1+(numel(x)-1)*percent/100;
lowerIndex = floor(position);
upperIndex = ceil(position);
if lowerIndex == upperIndex
    value = x(lowerIndex);
else
    weight = position-lowerIndex;
    value = (1-weight)*x(lowerIndex)+weight*x(upperIndex);
end
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

function localWriteConfiguration(cfg,outputDir)
parameter = ["num_repeats";"ridge4_points";"ridge6_points"; ...
    "minimum_fixed_test_coordinates"; ...
    "gate_minimum_relative_cv_gain_percent"; ...
    "gate_minimum_absolute_cv_gain_microstrain"; ...
    "fixed_test_used_by_gate"];
value = [cfg.robust.numRepeats;4;6; ...
    cfg.robust.minimumFixedTestCoordinates; ...
    cfg.gate.minRelativeCVImprovementPercent; ...
    cfg.gate.minAbsoluteCVImprovementMicrostrain;0];
writetable(table(parameter,value), ...
    fullfile(outputDir,'experiment_configuration.csv'));
end

function localPlotMethodSummary(summary,outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 1050 480]);
tiledlayout(1,2);
nexttile;
bar(categorical(summary.method),summary.micro_R2_mean);
ylabel('Mean micro R^2'); title('Repeated spatial splits'); grid on;
nexttile;
bar(categorical(summary.method),summary.micro_RMSE_mean_microstrain);
hold on;
errorbar(1:height(summary),summary.micro_RMSE_mean_microstrain, ...
    summary.micro_RMSE_std_microstrain,'.k','LineWidth',1.2);
ylabel('RMSE (\muepsilon)'); title('Mean +/- SD over repeats'); grid on;
saveas(fig,outputFile); close(fig);
end

function localPlotCaseWinRates(summary,outputFile)
sub = summary(summary.method == "Step19-SOTA+Ridge-6",:);
sub = sortrows(sub,'win_rate_percent','descend');
fig = figure('Visible','off','Color','w','Position',[100 100 1150 650]);
bar(categorical(sub.case_id,sub.case_id),sub.win_rate_percent);
yline(50,'k--','LineWidth',1.2);
ylabel('Ridge-6 win rate (%)');
title('Case-level robustness across spatial splits');
xtickangle(45); grid on;
saveas(fig,outputFile); close(fig);
end
