function [summary, results] = fbg_run_fewshot(baseParams, D, split, meta, cfg, outputDir)
%FBG_RUN_FEWSHOT Calibrate only the edge adapter with k target coordinates.

if ~exist(outputDir,'dir'), mkdir(outputDir); end
targetMask = split.test;
targetRows = find(targetMask);
if isempty(targetRows)
    warning('No target rows are available for few-shot calibration.');
    summary = table();
    results = {};
    return;
end

coords = unique(D.table(targetRows,{'x_cm','y_cm'}),'rows','stable');
maxK = height(coords);
kValues = unique(min(cfg.fewshot.kValues,maxK),'stable');

k_points = zeros(numel(kValues),1);
n_calibration_rows = zeros(numel(kValues),1);
n_test_rows = zeros(numel(kValues),1);
R2 = nan(numel(kValues),1);
RMSE_microstrain = nan(numel(kValues),1);
MAE_microstrain = nan(numel(kValues),1);
Bias_microstrain = nan(numel(kValues),1);
results = cell(numel(kValues),1);

order = localCoverageOrder(coords, D, targetRows);

for ik = 1:numel(kValues)
    k = kValues(ik);
    stageDir = fullfile(outputDir,sprintf('k_%02d',k));
    if ~exist(stageDir,'dir'), mkdir(stageDir); end

    calibrationMask = false(height(D.table),1);
    if k > 0
        selected = coords(order(1:k),:);
        for j = 1:height(selected)
            calibrationMask = calibrationMask | ...
                (targetMask & D.table.x_cm == selected.x_cm(j) & ...
                D.table.y_cm == selected.y_cm(j));
        end
        writetable(selected,fullfile(stageDir,'calibration_points.csv'));
    else
        writetable(coords([],:),fullfile(stageDir,'calibration_points.csv'));
    end

    localSplit = split;
    localSplit.train = split.train | calibrationMask;
    localSplit.test = targetMask & ~calibrationMask;
    localSplit.validation = split.validation;

    params = baseParams;
    if k > 0
        [params,history] = fbg_train_model(params,"edge",D,calibrationMask, ...
            split.validation,meta,cfg,"fewshot",true);
    else
        history = table(0,NaN,NaN,NaN,NaN,NaN,0,false, ...
            'VariableNames',{'iteration','loss','data_loss','gauge_loss', ...
            'edge_regularization','validation_rmse_microstrain','elapsed_seconds', ...
            'is_selected_checkpoint'});
    end

    result = fbg_save_stage(stageDir,params,"edge",history,D,localSplit,meta,cfg);
    results{ik} = result;

    testRow = result.metrics(result.metrics.split == "test",:);
    k_points(ik) = k;
    n_calibration_rows(ik) = sum(calibrationMask);
    n_test_rows(ik) = sum(localSplit.test);
    if ~isempty(testRow)
        R2(ik) = testRow.R2(1);
        RMSE_microstrain(ik) = testRow.RMSE_microstrain(1);
        MAE_microstrain(ik) = testRow.MAE_microstrain(1);
        Bias_microstrain(ik) = testRow.Bias_microstrain(1);
    end
end

summary = table(k_points,n_calibration_rows,n_test_rows,R2,RMSE_microstrain, ...
    MAE_microstrain,Bias_microstrain);
writetable(summary,fullfile(outputDir,'fewshot_summary.csv'));
localPlotSummary(summary,fullfile(outputDir,'fewshot_performance.png'));
end

function order = localCoverageOrder(coords,D,targetRows)
% Begin near the load and then greedily maximize spatial coverage.
px = median(D.p1x(targetRows));
py = median(D.p1y(targetRows));
scaleX = max(max(coords.x_cm)-min(coords.x_cm),1);
scaleY = max(max(coords.y_cm)-min(coords.y_cm),1);
distLoad = ((coords.x_cm-px)/scaleX).^2 + ((coords.y_cm-py)/scaleY).^2;
[~,first] = min(distLoad);

order = zeros(height(coords),1);
order(1) = first;
remaining = setdiff((1:height(coords))',first,'stable');

for i = 2:height(coords)
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

function localPlotSummary(summary,outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 900 500]);
yyaxis left;
plot(summary.k_points,summary.R2,'-o','LineWidth',1.8);
ylabel('Target-case R^2');
yyaxis right;
plot(summary.k_points,summary.RMSE_microstrain,'-s','LineWidth',1.8);
ylabel('Target-case RMSE (\muepsilon)');
xlabel('Number of calibration coordinates');
title('95 cm few-shot calibration');
grid on;
saveas(fig,outputFile); close(fig);
end
