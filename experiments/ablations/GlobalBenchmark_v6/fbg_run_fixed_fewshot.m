function [summary, repeatMetrics] = fbg_run_fixed_fewshot( ...
    residualParams, D, split, meta, cfg, outputDir)
%FBG_RUN_FIXED_FEWSHOT Final repeated calibration with unique spatial splits.
%
% Each repeat has a unique calibration-pool coordinate set. All k values
% inside one repeat share the same fixed test set and nested calibration
% points. Four Residual-FiLM calibration strategies are compared.

if ~exist(outputDir,'dir'), mkdir(outputDir); end

targetMask = split.test;
targetRows = find(targetMask);
if isempty(targetRows)
    warning('No target rows are available for final calibration.');
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
if ~any(kValues == 0), kValues = [0 kValues]; end

[coverageOrders,splitCheck] = localUniqueCoverageOrders(coords,poolSize,cfg);
writetable(splitCheck,fullfile(outputDir,'unique_split_check.csv'));

fprintf(['\n[final-calibration] %d unique repeats, %d calibration ', ...
    'candidates, %d fixed-test coordinates per repeat.\n'], ...
    cfg.fewshot.numRepeats,poolSize,nCoords-poolSize);

nMethods = 4;
maxParts = cfg.fewshot.numRepeats*numel(kValues)*nMethods;
metricParts = cell(maxParts,1);
predictionParts = cell(maxParts,1);
partIndex = 0;

for repeatIndex = 1:cfg.fewshot.numRepeats
    repeatSeed = splitCheck.seed(repeatIndex);
    repeatDir = fullfile(outputDir,sprintf('repeat_%02d',repeatIndex));
    if ~exist(repeatDir,'dir'), mkdir(repeatDir); end

    order = coverageOrders(:,repeatIndex);
    poolCoords = coords(order(1:poolSize),:);
    fixedTestCoords = coords(order(poolSize+1:end),:);
    poolCoords.selection_order = (1:height(poolCoords))';
    writetable(poolCoords,fullfile(repeatDir,'calibration_pool.csv'));
    writetable(fixedTestCoords,fullfile(repeatDir,'fixed_test_points.csv'));

    fixedTestMask = localCoordinateMask(D.table,targetMask,fixedTestCoords);
    residualFixed = fbg_predict(residualParams,"residual",D, ...
        fixedTestMask,meta,cfg);

    for ik = 1:numel(kValues)
        k = kValues(ik);
        kDir = fullfile(repeatDir,sprintf('k_%02d',k));
        if ~exist(kDir,'dir'), mkdir(kDir); end

        selectedCoords = poolCoords(1:k,{'x_cm','y_cm'});
        writetable(selectedCoords,fullfile(kDir,'calibration_points.csv'));
        calibrationMask = localCoordinateMask(D.table,targetMask,selectedCoords);

        calibrationPrediction = table();
        if k > 0
            calibrationPrediction = fbg_predict(residualParams,"residual", ...
                D,calibrationMask,meta,cfg);
        end

        % Method 1: no target-case calibration.
        partIndex = partIndex+1;
        [metricParts{partIndex},predictionParts{partIndex}] = ...
            localRecord(residualFixed,1,0,repeatIndex,repeatSeed, ...
            "residual_zero",k,sum(calibrationMask),height(selectedCoords), ...
            height(fixedTestCoords));

        % Method 2: one closed-form scalar bias.
        biasOffset = 0;
        if k > 0
            biasOffset = mean(calibrationPrediction.measured_microstrain- ...
                calibrationPrediction.predicted_microstrain,'omitnan');
        end
        partIndex = partIndex+1;
        [metricParts{partIndex},predictionParts{partIndex}] = ...
            localRecord(residualFixed,1,biasOffset,repeatIndex,repeatSeed, ...
            "bias_only",k,sum(calibrationMask),height(selectedCoords), ...
            height(fixedTestCoords));

        % Method 3: regularized affine calibration y* = a*yhat+b.
        affineScale = 1;
        affineOffset = 0;
        if k > 0
            [affineScale,affineOffset] = localFitRidgeAffine( ...
                calibrationPrediction,meta,cfg);
        end
        affineParameters = table(affineScale,affineOffset, ...
            cfg.fewshot.affineSlopePenalty,cfg.fewshot.affineOffsetPenalty, ...
            'VariableNames',{'scale_a','offset_b_microstrain', ...
            'slope_penalty','offset_penalty'});
        writetable(affineParameters, ...
            fullfile(kDir,'ridge_affine_parameters.csv'));
        partIndex = partIndex+1;
        [metricParts{partIndex},predictionParts{partIndex}] = ...
            localRecord(residualFixed,affineScale,affineOffset, ...
            repeatIndex,repeatSeed,"ridge_affine",k,sum(calibrationMask), ...
            height(selectedCoords),height(fixedTestCoords));

        % Method 4: freeze the field and update only the output head.
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
            localRecord(headFixed,1,0,repeatIndex,repeatSeed, ...
            "output_head",k,sum(calibrationMask),height(selectedCoords), ...
            height(fixedTestCoords));
    end
end

metricParts = metricParts(1:partIndex);
predictionParts = predictionParts(1:partIndex);
repeatMetrics = vertcat(metricParts{:});
writetable(repeatMetrics,fullfile(outputDir, ...
    'fixed_holdout_repeat_metrics.csv'));

if cfg.fewshot.saveAllPredictions
    allPredictions = vertcat(predictionParts{:});
    writetable(allPredictions,fullfile(outputDir, ...
        'fixed_holdout_predictions.csv'));
end

summary = localSummarize(repeatMetrics);
writetable(summary,fullfile(outputDir,'fixed_holdout_summary.csv'));

paired = localPairedImprovements(repeatMetrics);
writetable(paired,fullfile(outputDir,'paired_improvement_summary.csv'));

decision = localDecisionTable(summary,cfg);
writetable(decision,fullfile(outputDir,'method_decision_table.csv'));

selection = localSelectMethod(summary,paired);
writetable(selection,fullfile(outputDir,'best_method_by_k.csv'));

localPlotSummary(summary,fullfile(outputDir, ...
    'fixed_holdout_performance.png'));
end

function [orders,splitCheck] = localUniqueCoverageOrders(coords,poolSize,cfg)
n = height(coords);
nRequested = cfg.fewshot.numRepeats;
if nRequested > n
    error(['Requested %d unique spatial starts, but only %d target ', ...
        'coordinates exist. Reduce cfg.fewshot.numRepeats.'],nRequested,n);
end

rng(cfg.randomSeed);
firstCandidates = randperm(n);
orders = zeros(n,nRequested);
signatures = strings(nRequested,1);
firstIndices = zeros(nRequested,1);
seeds = zeros(nRequested,1);
nAccepted = 0;
attempt = 0;

while nAccepted < nRequested && attempt < cfg.fewshot.maxUniqueSplitAttempts
    attempt = attempt+1;
    candidatePosition = mod(attempt-1,n)+1;
    firstIndex = firstCandidates(candidatePosition);
    cycle = floor((attempt-1)/n);
    tieSeed = cfg.randomSeed + attempt*cfg.fewshot.repeatSeedStride + cycle;
    order = localCoverageOrder(coords,firstIndex,tieSeed);
    signature = strjoin(string(sort(order(1:poolSize)))',"-");

    if ~any(signatures(1:nAccepted) == signature)
        nAccepted = nAccepted+1;
        orders(:,nAccepted) = order;
        signatures(nAccepted) = signature;
        firstIndices(nAccepted) = firstIndex;
        seeds(nAccepted) = tieSeed;
    end
end

if nAccepted < nRequested
    error(['Could create only %d unique calibration pools after %d ', ...
        'attempts. Reduce numRepeats or calibrationPoolSize.'], ...
        nAccepted,attempt);
end

repeat_index = (1:nRequested)';
first_coordinate_index = firstIndices;
first_x_cm = coords.x_cm(firstIndices);
first_y_cm = coords.y_cm(firstIndices);
calibration_pool_signature = signatures;
is_unique = true(nRequested,1);
seed = seeds;
splitCheck = table(repeat_index,seed,first_coordinate_index, ...
    first_x_cm,first_y_cm,calibration_pool_signature,is_unique);
end

function order = localCoverageOrder(coords,firstIndex,tieSeed)
rng(tieSeed);
n = height(coords);
scaleX = max(max(coords.x_cm)-min(coords.x_cm),1);
scaleY = max(max(coords.y_cm)-min(coords.y_cm),1);
order = zeros(n,1);
order(1) = firstIndex;
remaining = setdiff((1:n)',firstIndex,'stable');

for i = 2:n
    selected = order(1:i-1);
    minDistance = inf(numel(remaining),1);
    for j = 1:numel(remaining)
        dx = (coords.x_cm(remaining(j))-coords.x_cm(selected))/scaleX;
        dy = (coords.y_cm(remaining(j))-coords.y_cm(selected))/scaleY;
        minDistance(j) = min(dx.^2+dy.^2);
    end
    % Small randomized perturbation preserves farthest-point coverage while
    % allowing additional unique pools when several candidates are similar.
    jitter = 0.01*max(minDistance)*rand(size(minDistance));
    [~,pick] = max(minDistance+jitter);
    order(i) = remaining(pick);
    remaining(pick) = [];
end
end

function mask = localCoordinateMask(T,targetMask,coords)
mask = false(height(T),1);
for i = 1:height(coords)
    mask = mask | (targetMask & T.x_cm == coords.x_cm(i) & ...
        T.y_cm == coords.y_cm(i));
end
end

function [scaleA,offsetB] = localFitRidgeAffine(prediction,meta,cfg)
% Fit measured = a*predicted+b with shrinkage toward a=1 and b=0.
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

function [metricRow,prediction] = localRecord(prediction,scaleA,offsetB, ...
    repeatIndex,repeatSeed,method,k,nCalibrationRows,nCalibrationCoords, ...
    nFixedTestCoords)

prediction.predicted_microstrain = scaleA* ...
    prediction.predicted_microstrain+offsetB;
prediction.residual_microstrain = prediction.predicted_microstrain- ...
    prediction.measured_microstrain;
[r2,rmse,mae,bias] = localMetrics(prediction.measured_microstrain, ...
    prediction.predicted_microstrain);

metricRow = table(repeatIndex,repeatSeed,method,k,nCalibrationCoords, ...
    nCalibrationRows,nFixedTestCoords,height(prediction),scaleA,offsetB, ...
    r2,rmse,mae,bias, ...
    'VariableNames',{'repeat_index','seed','method','k_points', ...
    'n_calibration_coordinates','n_calibration_rows', ...
    'n_fixed_test_coordinates','n_fixed_test_rows','calibration_scale', ...
    'calibration_offset_microstrain','R2','RMSE_microstrain', ...
    'MAE_microstrain','Bias_microstrain'});

n = height(prediction);
prediction = addvars(prediction,repmat(repeatIndex,n,1), ...
    repmat(repeatSeed,n,1),repmat(method,n,1),repmat(k,n,1), ...
    repmat(nCalibrationCoords,n,1),repmat(scaleA,n,1), ...
    repmat(offsetB,n,1),'Before',1, ...
    'NewVariableNames',{'repeat_index','seed','method','k_points', ...
    'n_calibration_coordinates','calibration_scale', ...
    'calibration_offset_microstrain'});
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
            mean(sub.calibration_scale,'omitnan'), ...
            std(sub.calibration_scale,0,'omitnan'), ...
            mean(sub.calibration_offset_microstrain,'omitnan'), ...
            std(sub.calibration_offset_microstrain,0,'omitnan'), ...
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
            'calibration_scale_mean','calibration_scale_std', ...
            'calibration_offset_mean_microstrain', ...
            'calibration_offset_std_microstrain','R2_mean','R2_std', ...
            'RMSE_mean_microstrain','RMSE_std_microstrain', ...
            'MAE_mean_microstrain','MAE_std_microstrain', ...
            'Bias_mean_microstrain','Bias_std_microstrain'});
    end
end
summary = vertcat(parts{1:index});
end

function paired = localPairedImprovements(repeatMetrics)
methods = unique(repeatMetrics.method,'stable');
kValues = unique(repeatMetrics.k_points,'stable');
parts = cell(numel(methods)*numel(kValues),1);
index = 0;
for i = 1:numel(methods)
    for j = 1:numel(kValues)
        sub = repeatMetrics(repeatMetrics.method == methods(i) & ...
            repeatMetrics.k_points == kValues(j),:);
        base = repeatMetrics(repeatMetrics.method == "residual_zero" & ...
            repeatMetrics.k_points == kValues(j),:);
        sub = sortrows(sub,'repeat_index');
        base = sortrows(base,'repeat_index');
        if isempty(sub) || height(sub) ~= height(base), continue; end

        deltaR2 = sub.R2-base.R2;
        rmseReduction = base.RMSE_microstrain-sub.RMSE_microstrain;
        rmseReductionPercent = 100*rmseReduction./base.RMSE_microstrain;
        maeReduction = base.MAE_microstrain-sub.MAE_microstrain;
        absoluteBiasReduction = abs(base.Bias_microstrain)- ...
            abs(sub.Bias_microstrain);

        index = index+1;
        parts{index} = table(methods(i),kValues(j),height(sub), ...
            mean(deltaR2,'omitnan'),std(deltaR2,0,'omitnan'), ...
            mean(rmseReduction,'omitnan'), ...
            mean(rmseReductionPercent,'omitnan'), ...
            std(rmseReductionPercent,0,'omitnan'), ...
            mean(rmseReduction > 0,'omitnan'), ...
            mean(maeReduction,'omitnan'), ...
            mean(absoluteBiasReduction,'omitnan'), ...
            'VariableNames',{'method','k_points','n_pairs', ...
            'delta_R2_mean','delta_R2_std', ...
            'RMSE_reduction_mean_microstrain', ...
            'RMSE_reduction_percent_mean', ...
            'RMSE_reduction_percent_std','RMSE_win_rate', ...
            'MAE_reduction_mean_microstrain', ...
            'absolute_Bias_reduction_mean_microstrain'});
    end
end
paired = vertcat(parts{1:index});
end

function decision = localDecisionTable(summary,cfg)
kValues = unique(summary.k_points(summary.k_points > 0),'stable');
methods = ["ridge_affine";"output_head"];
parts = cell(numel(kValues)*numel(methods),1);
index = 0;
for i = 1:numel(kValues)
    biasRow = summary(summary.method == "bias_only" & ...
        summary.k_points == kValues(i),:);
    zeroRow = summary(summary.method == "residual_zero" & ...
        summary.k_points == kValues(i),:);
    if isempty(biasRow) || isempty(zeroRow), continue; end
    for j = 1:numel(methods)
        row = summary(summary.method == methods(j) & ...
            summary.k_points == kValues(i),:);
        if isempty(row), continue; end

        deltaR2VsBias = row.R2_mean-biasRow.R2_mean;
        rmseVsBias = 100*(biasRow.RMSE_mean_microstrain- ...
            row.RMSE_mean_microstrain)/max(biasRow.RMSE_mean_microstrain,eps);
        deltaR2VsZero = row.R2_mean-zeroRow.R2_mean;
        rmseVsZero = 100*(zeroRow.RMSE_mean_microstrain- ...
            row.RMSE_mean_microstrain)/max(zeroRow.RMSE_mean_microstrain,eps);

        beatsBias = deltaR2VsBias >= cfg.fewshot.complexityMinDeltaR2 && ...
            rmseVsBias >= cfg.fewshot.complexityMinRMSEReductionPercent;
        beatsZero = deltaR2VsZero >= cfg.fewshot.complexityMinDeltaR2 && ...
            rmseVsZero >= cfg.fewshot.complexityMinRMSEReductionPercent;
        passesThreshold = beatsBias && beatsZero;

        index = index+1;
        parts{index} = table(methods(j),kValues(i),deltaR2VsBias, ...
            rmseVsBias,deltaR2VsZero,rmseVsZero,beatsBias,beatsZero, ...
            passesThreshold, ...
            'VariableNames',{'method','k_points','delta_R2_vs_bias', ...
            'RMSE_reduction_percent_vs_bias','delta_R2_vs_zero', ...
            'RMSE_reduction_percent_vs_zero','beats_bias_threshold', ...
            'beats_zero_threshold','passes_complexity_threshold'});
    end
end
decision = vertcat(parts{1:index});
end

function selection = localSelectMethod(summary,paired)
kValues = unique(summary.k_points,'stable');
parts = cell(numel(kValues),1);
for i = 1:numel(kValues)
    sub = summary(summary.k_points == kValues(i),:);
    [~,bestIndex] = min(sub.RMSE_mean_microstrain);
    best = sub(bestIndex,:);
    pairedRow = paired(paired.method == best.method & ...
        paired.k_points == kValues(i),:);
    winRate = NaN;
    if ~isempty(pairedRow), winRate = pairedRow.RMSE_win_rate(1); end
    parts{i} = table(kValues(i),best.method,best.R2_mean, ...
        best.RMSE_mean_microstrain,best.RMSE_std_microstrain,winRate, ...
        'VariableNames',{'k_points','best_method','R2_mean', ...
        'RMSE_mean_microstrain','RMSE_std_microstrain', ...
        'RMSE_win_rate_vs_zero'});
end
selection = vertcat(parts{:});
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
title('Unique repeated fixed-holdout R^2'); grid on;
legend('Interpreter','none','Location','best');

nexttile; hold on;
for i = 1:numel(methods)
    sub = sortrows(summary(summary.method == methods(i),:),'k_points');
    errorbar(sub.k_points,sub.RMSE_mean_microstrain, ...
        sub.RMSE_std_microstrain,'-s','Color',colors(i,:), ...
        'LineWidth',1.4,'DisplayName',methods(i));
end
xlabel('Calibration coordinates'); ylabel('Fixed-test RMSE (\muepsilon)');
title('Unique repeated fixed-holdout RMSE'); grid on;
legend('Interpreter','none','Location','best');

saveas(fig,outputFile);
close(fig);
end
