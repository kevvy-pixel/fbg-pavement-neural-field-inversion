function [summary, repeatMetrics] = fbg_run_fixed_fewshot( ...
    residualParams, edgeParams, D, split, meta, cfg, outputDir)
%FBG_RUN_FIXED_FEWSHOT Fair 95 cm calibration with a fixed test set.
%
% For every repeat, a calibration pool is selected once. All k values and
% all methods are then evaluated on the same untouched coordinates.

if ~exist(outputDir,'dir'), mkdir(outputDir); end

targetMask = split.test;
targetRows = find(targetMask);
if isempty(targetRows)
    warning('No target rows are available for fixed-holdout calibration.');
    summary = table();
    repeatMetrics = table();
    return;
end

coords = unique(D.table(targetRows,{'x_cm','y_cm'}),'rows','stable');
nCoords = height(coords);
poolSize = min(cfg.fewshot.calibrationPoolSize, ...
    nCoords-cfg.fewshot.minimumFixedTestCoordinates);
poolSize = max(0,poolSize);
if poolSize < 1
    warning(['Too few target coordinates for a calibration pool while ', ...
        'retaining the requested fixed test set.']);
    summary = table();
    repeatMetrics = table();
    return;
end

kValues = unique(min(cfg.fewshot.kValues,poolSize),'stable');
if ~any(kValues == 0)
    kValues = [0 kValues];
end

fprintf(['\n[fixed-fewshot] %d repeats, %d calibration candidates, ', ...
    '%d fixed-test coordinates per repeat.\n'], ...
    cfg.fewshot.numRepeats,poolSize,nCoords-poolSize);

metricParts = cell(cfg.fewshot.numRepeats*numel(kValues)*4,1);
predictionParts = cell(size(metricParts));
partIndex = 0;

for repeatIndex = 1:cfg.fewshot.numRepeats
    repeatSeed = cfg.randomSeed + ...
        (repeatIndex-1)*cfg.fewshot.repeatSeedStride;
    repeatDir = fullfile(outputDir,sprintf('repeat_%02d',repeatIndex));
    if ~exist(repeatDir,'dir'), mkdir(repeatDir); end

    order = localRandomCoverageOrder(coords,repeatSeed);
    poolCoords = coords(order(1:poolSize),:);
    fixedTestCoords = coords(order(poolSize+1:end),:);
    poolCoords.selection_order = (1:height(poolCoords))';
    writetable(poolCoords,fullfile(repeatDir,'calibration_pool.csv'));
    writetable(fixedTestCoords,fullfile(repeatDir,'fixed_test_points.csv'));

    fixedTestMask = localCoordinateMask(D.table,targetMask,fixedTestCoords);
    residualFixed = fbg_predict(residualParams,"residual",D, ...
        fixedTestMask,meta,cfg);
    edgeFixed = fbg_predict(edgeParams,"edge",D, ...
        fixedTestMask,meta,cfg);

    for ik = 1:numel(kValues)
        k = kValues(ik);
        kDir = fullfile(repeatDir,sprintf('k_%02d',k));
        if ~exist(kDir,'dir'), mkdir(kDir); end

        selectedCoords = poolCoords(1:k,{'x_cm','y_cm'});
        writetable(selectedCoords,fullfile(kDir,'calibration_points.csv'));
        calibrationMask = localCoordinateMask(D.table,targetMask,selectedCoords);

        % Method 1: zero-shot Residual-FiLM.
        partIndex = partIndex+1;
        [metricParts{partIndex},predictionParts{partIndex}] = ...
            localRecord(residualFixed,0,repeatIndex,repeatSeed, ...
            "residual_zero",k,sum(calibrationMask),height(selectedCoords), ...
            height(fixedTestCoords));

        % Method 2: one closed-form scalar bias fitted on calibration rows.
        biasCorrection = 0;
        if k > 0
            calibrationPrediction = fbg_predict(residualParams,"residual", ...
                D,calibrationMask,meta,cfg);
            biasCorrection = mean(calibrationPrediction.measured_microstrain- ...
                calibrationPrediction.predicted_microstrain,'omitnan');
        end
        partIndex = partIndex+1;
        [metricParts{partIndex},predictionParts{partIndex}] = ...
            localRecord(residualFixed,biasCorrection,repeatIndex,repeatSeed, ...
            "bias_only",k,sum(calibrationMask),height(selectedCoords), ...
            height(fixedTestCoords));

        % Method 3: update only the Residual-FiLM output head.
        headParams = residualParams;
        if k > 0
            rng(repeatSeed + 100*k + 31);
            falseValidation = false(height(D.table),1);
            [headParams,headHistory] = fbg_train_model(headParams, ...
                "residual",D,calibrationMask,falseValidation,meta,cfg, ...
                "fewshot",false,"output");
            headDir = fullfile(kDir,'output_head');
            if ~exist(headDir,'dir'), mkdir(headDir); end
            writetable(headHistory,fullfile(headDir,'training_history.csv'));
        end
        headFixed = fbg_predict(headParams,"residual",D,fixedTestMask,meta,cfg);
        partIndex = partIndex+1;
        [metricParts{partIndex},predictionParts{partIndex}] = ...
            localRecord(headFixed,0,repeatIndex,repeatSeed, ...
            "output_head",k,sum(calibrationMask),height(selectedCoords), ...
            height(fixedTestCoords));

        % Method 4: update only the gate and local edge expert.
        adaptedEdgeParams = edgeParams;
        if k > 0
            rng(repeatSeed + 100*k + 67);
            falseValidation = false(height(D.table),1);
            [adaptedEdgeParams,edgeHistory] = fbg_train_model( ...
                adaptedEdgeParams,"edge",D,calibrationMask,falseValidation, ...
                meta,cfg,"fewshot",true,"edge");
            edgeDir = fullfile(kDir,'edge_expert');
            if ~exist(edgeDir,'dir'), mkdir(edgeDir); end
            writetable(edgeHistory,fullfile(edgeDir,'training_history.csv'));
        end
        adaptedEdgeFixed = fbg_predict(adaptedEdgeParams,"edge",D, ...
            fixedTestMask,meta,cfg);
        partIndex = partIndex+1;
        [metricParts{partIndex},predictionParts{partIndex}] = ...
            localRecord(adaptedEdgeFixed,0,repeatIndex,repeatSeed, ...
            "edge_expert",k,sum(calibrationMask),height(selectedCoords), ...
            height(fixedTestCoords));
    end
end

metricParts = metricParts(1:partIndex);
predictionParts = predictionParts(1:partIndex);
repeatMetrics = vertcat(metricParts{:});
writetable(repeatMetrics,fullfile(outputDir,'fixed_holdout_repeat_metrics.csv'));

if cfg.fewshot.saveAllPredictions
    allPredictions = vertcat(predictionParts{:});
    writetable(allPredictions, ...
        fullfile(outputDir,'fixed_holdout_predictions.csv'));
end

summary = localSummarize(repeatMetrics);
writetable(summary,fullfile(outputDir,'fixed_holdout_summary.csv'));

decision = localDecisionTable(summary);
writetable(decision,fullfile(outputDir,'method_decision_table.csv'));
localPlotSummary(summary,fullfile(outputDir,'fixed_holdout_performance.png'));
end

function mask = localCoordinateMask(T,targetMask,coords)
mask = false(height(T),1);
for i = 1:height(coords)
    mask = mask | (targetMask & T.x_cm == coords.x_cm(i) & ...
        T.y_cm == coords.y_cm(i));
end
end

function order = localRandomCoverageOrder(coords,seed)
% Random first point, then greedy farthest-point sampling.
rng(seed);
n = height(coords);
scaleX = max(max(coords.x_cm)-min(coords.x_cm),1);
scaleY = max(max(coords.y_cm)-min(coords.y_cm),1);
order = zeros(n,1);
order(1) = randi(n);
remaining = setdiff((1:n)',order(1),'stable');

for i = 2:n
    selected = order(1:i-1);
    minDistance = inf(numel(remaining),1);
    for j = 1:numel(remaining)
        dx = (coords.x_cm(remaining(j))-coords.x_cm(selected))/scaleX;
        dy = (coords.y_cm(remaining(j))-coords.y_cm(selected))/scaleY;
        minDistance(j) = min(dx.^2+dy.^2);
    end
    jitter = 1e-12*rand(size(minDistance));
    [~,pick] = max(minDistance+jitter);
    order(i) = remaining(pick);
    remaining(pick) = [];
end
end

function [metricRow,prediction] = localRecord(prediction,biasCorrection, ...
    repeatIndex,repeatSeed,method,k,nCalibrationRows,nCalibrationCoords, ...
    nFixedTestCoords)

prediction.predicted_microstrain = prediction.predicted_microstrain + ...
    biasCorrection;
prediction.residual_microstrain = prediction.predicted_microstrain - ...
    prediction.measured_microstrain;
[r2,rmse,mae,bias] = localMetrics(prediction.measured_microstrain, ...
    prediction.predicted_microstrain);

metricRow = table(repeatIndex,repeatSeed,method,k,nCalibrationCoords, ...
    nCalibrationRows,nFixedTestCoords,height(prediction),biasCorrection, ...
    r2,rmse,mae,bias, ...
    'VariableNames',{'repeat_index','seed','method','k_points', ...
    'n_calibration_coordinates','n_calibration_rows', ...
    'n_fixed_test_coordinates','n_fixed_test_rows', ...
    'bias_correction_microstrain','R2','RMSE_microstrain', ...
    'MAE_microstrain','Bias_microstrain'});

n = height(prediction);
prediction = addvars(prediction, ...
    repmat(repeatIndex,n,1),repmat(repeatSeed,n,1), ...
    repmat(method,n,1),repmat(k,n,1), ...
    repmat(nCalibrationCoords,n,1), ...
    repmat(biasCorrection,n,1), ...
    'Before',1,'NewVariableNames',{'repeat_index','seed','method', ...
    'k_points','n_calibration_coordinates','bias_correction_microstrain'});
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

function summary = localSummarize(repeatMetrics)
methods = unique(repeatMetrics.method,'stable');
kValues = unique(repeatMetrics.k_points,'stable');
parts = cell(numel(methods)*numel(kValues),1);
index = 0;
for i = 1:numel(methods)
    for j = 1:numel(kValues)
        sub = repeatMetrics(repeatMetrics.method == methods(i) & ...
            repeatMetrics.k_points == kValues(j),:);
        if isempty(sub), continue; end
        index = index+1;
        parts{index} = table(methods(i),kValues(j),height(sub), ...
            mean(sub.n_calibration_coordinates), ...
            mean(sub.n_calibration_rows), ...
            mean(sub.n_fixed_test_coordinates), ...
            mean(sub.n_fixed_test_rows), ...
            mean(sub.R2,'omitnan'),std(sub.R2,0,'omitnan'), ...
            mean(sub.RMSE_microstrain,'omitnan'), ...
            std(sub.RMSE_microstrain,0,'omitnan'), ...
            mean(sub.MAE_microstrain,'omitnan'), ...
            std(sub.MAE_microstrain,0,'omitnan'), ...
            mean(sub.Bias_microstrain,'omitnan'), ...
            std(sub.Bias_microstrain,0,'omitnan'), ...
            'VariableNames',{'method','k_points','n_repeats', ...
            'n_calibration_coordinates','mean_n_calibration_rows', ...
            'n_fixed_test_coordinates','mean_n_fixed_test_rows', ...
            'R2_mean','R2_std','RMSE_mean_microstrain', ...
            'RMSE_std_microstrain','MAE_mean_microstrain', ...
            'MAE_std_microstrain','Bias_mean_microstrain', ...
            'Bias_std_microstrain'});
    end
end
summary = vertcat(parts{1:index});
end

function decision = localDecisionTable(summary)
kValues = unique(summary.k_points(summary.k_points > 0),'stable');
methods = ["output_head";"edge_expert"];
parts = cell(numel(kValues)*numel(methods),1);
index = 0;
for i = 1:numel(kValues)
    biasRow = summary(summary.method == "bias_only" & ...
        summary.k_points == kValues(i),:);
    if isempty(biasRow), continue; end
    for j = 1:numel(methods)
        row = summary(summary.method == methods(j) & ...
            summary.k_points == kValues(i),:);
        if isempty(row), continue; end
        deltaR2 = row.R2_mean-biasRow.R2_mean;
        rmseReductionPercent = 100*(biasRow.RMSE_mean_microstrain- ...
            row.RMSE_mean_microstrain)/max(biasRow.RMSE_mean_microstrain,eps);
        passesThreshold = deltaR2 >= 0.02 && rmseReductionPercent >= 5;
        index = index+1;
        parts{index} = table(methods(j),kValues(i),deltaR2, ...
            rmseReductionPercent,passesThreshold, ...
            'VariableNames',{'method','k_points','delta_R2_vs_bias', ...
            'RMSE_reduction_percent_vs_bias','passes_complexity_threshold'});
    end
end
if index == 0
    decision = table();
else
    decision = vertcat(parts{1:index});
end
end

function localPlotSummary(summary,outputFile)
methods = unique(summary.method,'stable');
colors = lines(numel(methods));
fig = figure('Visible','off','Color','w','Position',[100 100 1100 520]);
tiledlayout(1,2);

nexttile; hold on;
for i = 1:numel(methods)
    sub = sortrows(summary(summary.method == methods(i),:),'k_points');
    errorbar(sub.k_points,sub.R2_mean,sub.R2_std,'-o', ...
        'Color',colors(i,:),'LineWidth',1.4,'DisplayName',methods(i));
end
xlabel('Calibration coordinates'); ylabel('Fixed-test R^2');
title('Repeated fixed-holdout R^2'); grid on;
legend('Interpreter','none','Location','best');

nexttile; hold on;
for i = 1:numel(methods)
    sub = sortrows(summary(summary.method == methods(i),:),'k_points');
    errorbar(sub.k_points,sub.RMSE_mean_microstrain, ...
        sub.RMSE_std_microstrain,'-s','Color',colors(i,:), ...
        'LineWidth',1.4,'DisplayName',methods(i));
end
xlabel('Calibration coordinates'); ylabel('Fixed-test RMSE (\muepsilon)');
title('Repeated fixed-holdout RMSE'); grid on;
legend('Interpreter','none','Location','best');

saveas(fig,outputFile);
close(fig);
end
