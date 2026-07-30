function outputDir = RUN_SOTA_CALIBRATION_GATE(runDir,outputDir)
%RUN_SOTA_CALIBRATION_GATE Post-process a completed v7 full comparison.
%
% This function does not train a neural network. It reads the saved
% Step19-SOTA predictions, fits Ridge-4, and makes a calibration decision
% using leave-one-coordinate-out cross-validation on the four calibration
% coordinates only. Fixed-test observations never affect the gate.

arguments
    runDir (1,1) string
    outputDir (1,1) string = ""
end

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
postCfg = fbg_config("full");
if ~exist(runDir,'dir')
    error('Existing run directory not found: %s',runDir);
end

assignmentFile = fullfile(runDir,'calibration_point_assignments.csv');
foldFile = fullfile(runDir,'case_fold_assignments.csv');
if ~isfile(assignmentFile) || ~isfile(foldFile)
    error(['The selected directory is not a completed v7 run. Required: ', ...
        'calibration_point_assignments.csv and case_fold_assignments.csv.']);
end
if strlength(outputDir) == 0
    outputDir = fullfile(runDir,postCfg.gate.outputFolder);
end
if ~exist(outputDir,'dir'), mkdir(outputDir); end

diary(fullfile(outputDir,'run.log'));
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('FBG SOTA Ridge-4 Calibration Gate\n');
fprintf('Input run: %s\nOutput: %s\n',runDir,outputDir);
fprintf(['Gate: relative CV gain >= %.1f%%, absolute CV gain >= %.1f ', ...
    'microstrain, scale margin %.3f.\n'], ...
    postCfg.gate.minRelativeCVImprovementPercent, ...
    postCfg.gate.minAbsoluteCVImprovementMicrostrain, ...
    postCfg.gate.scaleBoundMargin);

assignments = readtable(assignmentFile,'TextType','string', ...
    'VariableNamingRule','preserve');
foldAssignments = readtable(foldFile,'TextType','string', ...
    'VariableNamingRule','preserve');
foldIds = unique(foldAssignments.fold_id,'stable');

predictionParts = {};
metricParts = {};
decisionParts = {};
cvPredictionParts = {};
pi = 0; mi = 0; di = 0; ci = 0;

for iFold = 1:numel(foldIds)
    foldId = foldIds(iFold);
    foldDir = fullfile(runDir,sprintf('fold_%02d',foldId));
    predictionFile = fullfile(foldDir,'01_step19_sota_film','predictions.csv');
    configurationFile = fullfile(foldDir,'fold_configuration.mat');
    if ~isfile(predictionFile) || ~isfile(configurationFile)
        error('Missing saved SOTA stage in fold %d.',foldId);
    end
    prediction = readtable(predictionFile,'TextType','string', ...
        'VariableNamingRule','preserve');
    saved = load(configurationFile,'foldMeta');
    foldMeta = saved.foldMeta;

    foldCases = unique(assignments.case_id(assignments.fold_id == foldId), ...
        'stable');
    fprintf('\nFold %d: %d held-out cases.\n',foldId,numel(foldCases));
    for iCase = 1:numel(foldCases)
        caseId = foldCases(iCase);
        assigned = assignments(assignments.fold_id == foldId & ...
            assignments.case_id == caseId,:);
        calibrationCoords = assigned(assigned.role == "calibration", ...
            {'x_cm','y_cm'});
        fixedCoords = assigned(assigned.role == "fixed_test", ...
            {'x_cm','y_cm'});
        if height(calibrationCoords) ~= postCfg.global.calibrationK
            error('Case %s does not contain exactly %d calibration points.', ...
                caseId,postCfg.global.calibrationK);
        end

        casePrediction = prediction(prediction.case_id == caseId & ...
            prediction.split == "test",:);
        calibrationMask = localCoordinateMask(casePrediction,calibrationCoords);
        fixedMask = localCoordinateMask(casePrediction,fixedCoords);
        calibrationPrediction = casePrediction(calibrationMask,:);
        fixedPrediction = casePrediction(fixedMask,:);

        [cvBaselineRmse,cvRidgeRmse,cvRows] = localCoordinateLOOCV( ...
            calibrationPrediction,calibrationCoords,foldMeta,postCfg, ...
            foldId,caseId);
        cvAbsoluteGain = cvBaselineRmse-cvRidgeRmse;
        cvRelativeGain = 100*cvAbsoluteGain/max(cvBaselineRmse,eps);

        [scaleA,offsetB,calibrationSeconds] = localTimedRidge( ...
            calibrationPrediction,foldMeta,postCfg);
        bounds = postCfg.fewshot.affineScaleBounds;
        margin = postCfg.gate.scaleBoundMargin;
        scaleInterior = scaleA > bounds(1)+margin && ...
            scaleA < bounds(2)-margin;
        finiteFit = all(isfinite([cvBaselineRmse,cvRidgeRmse, ...
            scaleA,offsetB]));
        gateApplied = finiteFit && scaleInterior && ...
            cvRelativeGain >= postCfg.gate.minRelativeCVImprovementPercent && ...
            cvAbsoluteGain >= ...
            postCfg.gate.minAbsoluteCVImprovementMicrostrain;
        gateReason = localGateReason(finiteFit,scaleInterior, ...
            cvRelativeGain,cvAbsoluteGain,postCfg);

        baseline = fixedPrediction;
        always = localApplyAffine(fixedPrediction,scaleA,offsetB);
        if gateApplied
            gated = always;
        else
            gated = baseline;
        end

        methods = ["Step19-SOTA-FiLM"; ...
            "Step19-SOTA+Ridge-4"; ...
            "Step19-SOTA+Gated-Ridge-4"];
        predictionsForMethod = {baseline;always;gated};
        for iMethod = 1:numel(methods)
            methodPrediction = predictionsForMethod{iMethod};
            pi = pi+1;
            predictionParts{pi,1} = localLabelPrediction( ...
                methodPrediction,foldId,methods(iMethod),gateApplied, ...
                scaleA,offsetB,cvBaselineRmse,cvRidgeRmse); %#ok<AGROW>
            mi = mi+1;
            metricParts{mi,1} = localMetricRow( ...
                methodPrediction,foldId,caseId,methods(iMethod), ...
                gateApplied,calibrationSeconds,scaleA,offsetB, ...
                cvBaselineRmse,cvRidgeRmse,cvAbsoluteGain, ...
                cvRelativeGain,gateReason); %#ok<AGROW>
        end

        [baseR2,baseRmse,~,~] = localMetrics( ...
            baseline.measured_microstrain,baseline.predicted_microstrain);
        [alwaysR2,alwaysRmse,~,~] = localMetrics( ...
            always.measured_microstrain,always.predicted_microstrain);
        [gatedR2,gatedRmse,~,~] = localMetrics( ...
            gated.measured_microstrain,gated.predicted_microstrain);
        di = di+1;
        decisionParts{di,1} = table(foldId,caseId,gateApplied,gateReason, ...
            height(calibrationCoords),height(calibrationPrediction), ...
            cvBaselineRmse,cvRidgeRmse,cvAbsoluteGain,cvRelativeGain, ...
            scaleA,offsetB,scaleInterior,baseR2,baseRmse, ...
            alwaysR2,alwaysRmse,gatedR2,gatedRmse, ...
            baseRmse-alwaysRmse,baseRmse-gatedRmse, ...
            'VariableNames',{'fold_id','case_id','gate_applied', ...
            'gate_reason','n_calibration_coordinates', ...
            'n_calibration_rows','cv_baseline_RMSE_microstrain', ...
            'cv_ridge_RMSE_microstrain','cv_absolute_gain_microstrain', ...
            'cv_relative_gain_percent','calibration_scale', ...
            'calibration_offset_microstrain','scale_inside_bounds', ...
            'baseline_R2','baseline_RMSE_microstrain','always_ridge_R2', ...
            'always_ridge_RMSE_microstrain','gated_R2', ...
            'gated_RMSE_microstrain','fixed_RMSE_gain_always_microstrain', ...
            'fixed_RMSE_gain_gated_microstrain'}); %#ok<AGROW>
        ci = ci+1;
        cvPredictionParts{ci,1} = cvRows; %#ok<AGROW>

        fprintf(['  %s | CV %.2f -> %.2f (%.1f%%) | a=%.3f, ', ...
            'b=%.2f | gate=%d (%s)\n'],caseId,cvBaselineRmse, ...
            cvRidgeRmse,cvRelativeGain,scaleA,offsetB, ...
            gateApplied,gateReason);
    end
end

allPredictions = vertcat(predictionParts{:});
caseMetrics = vertcat(metricParts{:});
decisions = vertcat(decisionParts{:});
cvPredictions = vertcat(cvPredictionParts{:});
summary = localSummary(allPredictions,caseMetrics);
paired = localPairedSummary(caseMetrics);
target95 = caseMetrics(contains(lower(caseMetrics.case_id),"95cm"),:);
gateAudit = localGateAudit(decisions);

writetable(allPredictions,fullfile(outputDir,'predictions.csv'));
writetable(caseMetrics,fullfile(outputDir,'case_metrics.csv'));
writetable(decisions,fullfile(outputDir,'gate_decision_by_case.csv'));
writetable(cvPredictions,fullfile(outputDir,'calibration_loocv_predictions.csv'));
writetable(summary,fullfile(outputDir,'method_summary.csv'));
writetable(paired,fullfile(outputDir,'paired_improvement_summary.csv'));
writetable(target95,fullfile(outputDir,'target_95cm_summary.csv'));
writetable(gateAudit,fullfile(outputDir,'gate_audit_summary.csv'));
localWriteGateConfiguration(postCfg,outputDir);
localPlotSummary(summary,fullfile(outputDir,'method_comparison.png'));
localPlotGate(decisions,fullfile(outputDir,'gate_diagnostic.png'));

fprintf('\nSOTA calibration gate completed.\nResults: %s\n',outputDir);
end

function mask = localCoordinateMask(prediction,coords)
mask = false(height(prediction),1);
for i = 1:height(coords)
    mask = mask | (prediction.x_cm == coords.x_cm(i) & ...
        prediction.y_cm == coords.y_cm(i));
end
end

function [baselineRmse,ridgeRmse,rows] = localCoordinateLOOCV( ...
    calibrationPrediction,coords,meta,cfg,foldId,caseId)
parts = cell(height(coords),1);
for i = 1:height(coords)
    heldMask = calibrationPrediction.x_cm == coords.x_cm(i) & ...
        calibrationPrediction.y_cm == coords.y_cm(i);
    trainPrediction = calibrationPrediction(~heldMask,:);
    heldPrediction = calibrationPrediction(heldMask,:);
    [scaleA,offsetB] = localFitRidge(trainPrediction,meta,cfg);
    calibrated = localApplyAffine(heldPrediction,scaleA,offsetB);
    n = height(calibrated);
    parts{i} = table(repmat(foldId,n,1),repmat(caseId,n,1), ...
        repmat(i,n,1),calibrated.x_cm,calibrated.y_cm, ...
        calibrated.measured_microstrain, ...
        heldPrediction.predicted_microstrain, ...
        calibrated.predicted_microstrain,repmat(scaleA,n,1), ...
        repmat(offsetB,n,1), ...
        'VariableNames',{'fold_id','case_id','held_coordinate_index', ...
        'x_cm','y_cm','measured_microstrain', ...
        'baseline_predicted_microstrain','ridge_predicted_microstrain', ...
        'loocv_scale','loocv_offset_microstrain'});
end
rows = vertcat(parts{:});
[~,baselineRmse,~,~] = localMetrics(rows.measured_microstrain, ...
    rows.baseline_predicted_microstrain);
[~,ridgeRmse,~,~] = localMetrics(rows.measured_microstrain, ...
    rows.ridge_predicted_microstrain);
end

function prediction = localApplyAffine(prediction,scaleA,offsetB)
prediction.predicted_microstrain = ...
    scaleA*prediction.predicted_microstrain+offsetB;
prediction.residual_microstrain = prediction.predicted_microstrain- ...
    prediction.measured_microstrain;
end

function [scaleA,offsetB,seconds] = localTimedRidge(prediction,meta,cfg)
samples = zeros(cfg.global.calibrationTimingRepeats,1);
for i = 1:numel(samples)
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

function out = localLabelPrediction(prediction,foldId,method,gateApplied, ...
    scaleA,offsetB,cvBaselineRmse,cvRidgeRmse)
n = height(prediction);
out = addvars(prediction,repmat(foldId,n,1),repmat(method,n,1), ...
    repmat(gateApplied,n,1),repmat(scaleA,n,1),repmat(offsetB,n,1), ...
    repmat(cvBaselineRmse,n,1),repmat(cvRidgeRmse,n,1), ...
    'Before',1,'NewVariableNames',{'fold_id','method','gate_applied', ...
    'calibration_scale','calibration_offset_microstrain', ...
    'cv_baseline_RMSE_microstrain','cv_ridge_RMSE_microstrain'});
end

function row = localMetricRow(prediction,foldId,caseId,method, ...
    gateApplied,seconds,scaleA,offsetB,cvBaselineRmse,cvRidgeRmse, ...
    cvAbsoluteGain,cvRelativeGain,gateReason)
[r2,rmse,mae,bias] = localMetrics(prediction.measured_microstrain, ...
    prediction.predicted_microstrain);
row = table(foldId,caseId,method,gateApplied,height(prediction), ...
    seconds,scaleA,offsetB,cvBaselineRmse,cvRidgeRmse, ...
    cvAbsoluteGain,cvRelativeGain,gateReason,r2,rmse,mae,bias, ...
    'VariableNames',{'fold_id','case_id','method','gate_applied', ...
    'n_fixed_test_rows','calibration_seconds','calibration_scale', ...
    'calibration_offset_microstrain','cv_baseline_RMSE_microstrain', ...
    'cv_ridge_RMSE_microstrain','cv_absolute_gain_microstrain', ...
    'cv_relative_gain_percent','gate_reason','R2','RMSE_microstrain', ...
    'MAE_microstrain','Bias_microstrain'});
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
        r2,rmse,mae,bias,median(c.R2,'omitnan'),sum(c.R2 > 0), ...
        mean(c.RMSE_microstrain,'omitnan'), ...
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
baselineName = "Step19-SOTA-FiLM";
candidates = ["Step19-SOTA+Ridge-4";"Step19-SOTA+Gated-Ridge-4"];
cases = unique(metrics.case_id,'stable');
parts = cell(numel(candidates),1);
for i = 1:numel(candidates)
    deltaRmse = nan(numel(cases),1);
    deltaR2 = nan(numel(cases),1);
    for j = 1:numel(cases)
        b = metrics(metrics.case_id == cases(j) & ...
            metrics.method == baselineName,:);
        c = metrics(metrics.case_id == cases(j) & ...
            metrics.method == candidates(i),:);
        if ~isempty(b) && ~isempty(c)
            deltaRmse(j) = b.RMSE_microstrain(1)-c.RMSE_microstrain(1);
            deltaR2(j) = c.R2(1)-b.R2(1);
        end
    end
    valid = isfinite(deltaRmse);
    parts{i} = table(baselineName,candidates(i),sum(valid), ...
        mean(deltaRmse(valid),'omitnan'),median(deltaRmse(valid),'omitnan'), ...
        sum(deltaRmse(valid)>0),mean(deltaR2(valid),'omitnan'), ...
        'VariableNames',{'baseline','candidate','n_paired_cases', ...
        'mean_RMSE_reduction_microstrain', ...
        'median_RMSE_reduction_microstrain','n_cases_RMSE_improved', ...
        'mean_R2_improvement'});
end
paired = vertcat(parts{:});
end

function audit = localGateAudit(decisions)
applied = decisions.gate_applied;
fixedGain = decisions.fixed_RMSE_gain_gated_microstrain;
audit = table(height(decisions),sum(applied),sum(~applied), ...
    sum(applied & fixedGain > 0),sum(applied & fixedGain <= 0), ...
    mean(fixedGain,'omitnan'),median(fixedGain,'omitnan'), ...
    mean(fixedGain(applied),'omitnan'), ...
    'VariableNames',{'n_cases','n_gate_applied','n_gate_rejected', ...
    'n_applied_cases_improved','n_applied_cases_degraded', ...
    'mean_gated_RMSE_gain_microstrain', ...
    'median_gated_RMSE_gain_microstrain', ...
    'mean_gain_when_applied_microstrain'});
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

function localWriteGateConfiguration(cfg,outputDir)
parameter = ["ridge_points";"slope_penalty";"offset_penalty"; ...
    "scale_lower_bound";"scale_upper_bound";"scale_bound_margin"; ...
    "minimum_relative_cv_gain_percent"; ...
    "minimum_absolute_cv_gain_microstrain"; ...
    "gate_uses_fixed_test_observations"];
value = [cfg.global.calibrationK;cfg.fewshot.affineSlopePenalty; ...
    cfg.fewshot.affineOffsetPenalty;cfg.fewshot.affineScaleBounds(1); ...
    cfg.fewshot.affineScaleBounds(2);cfg.gate.scaleBoundMargin; ...
    cfg.gate.minRelativeCVImprovementPercent; ...
    cfg.gate.minAbsoluteCVImprovementMicrostrain;0];
writetable(table(parameter,value), ...
    fullfile(outputDir,'gate_configuration.csv'));
end

function localPlotSummary(summary,outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 1050 480]);
tiledlayout(1,2);
nexttile;
bar(categorical(summary.method),summary.micro_R2);
ylabel('Micro R^2'); title('SOTA calibration comparison'); grid on;
nexttile;
bar(categorical(summary.method),summary.micro_RMSE_microstrain);
ylabel('RMSE (\muepsilon)'); title('Common fixed test coordinates'); grid on;
saveas(fig,outputFile); close(fig);
end

function localPlotGate(decisions,outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 700 560]);
hold on;
applied = decisions.gate_applied;
scatter(decisions.cv_absolute_gain_microstrain(~applied), ...
    decisions.fixed_RMSE_gain_always_microstrain(~applied),60,'o', ...
    'DisplayName','Gate rejected');
scatter(decisions.cv_absolute_gain_microstrain(applied), ...
    decisions.fixed_RMSE_gain_always_microstrain(applied),70,'filled', ...
    'DisplayName','Gate applied');
xline(0,'k:'); yline(0,'k:');
xlabel('Calibration LOOCV RMSE gain (\muepsilon)');
ylabel('Fixed-test RMSE gain if calibrated (\muepsilon)');
title('No-leakage calibration gate diagnostic');
legend('Location','best'); grid on;
saveas(fig,outputFile); close(fig);
end
