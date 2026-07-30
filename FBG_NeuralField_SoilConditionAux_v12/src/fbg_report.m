function fbg_report(P, outputDir, auxiliary)
%FBG_REPORT Build asphalt and soil-summary evaluation tables.
if nargin < 3
    auxiliary = table();
end
models = unique(P.model, "stable");
globalRows = table();
targetRows = table();
directionRows = table();

for i = 1:numel(models)
    modelName = models(i);
    asphalt = P.model == modelName & P.layer == "asphalt";
    globalRows = [globalRows; metricRow(P, asphalt, modelName)]; %#ok<AGROW>

    target = asphalt & abs(P.load_x_cm - 95) <= 0.6;
    targetRows = [targetRows; metricRow(P, target, modelName)]; %#ok<AGROW>

    for directionName = ["x", "y"]
        mask = asphalt & P.direction == directionName;
        directionRows = [directionRows; ...
            directionMetricRow(P, mask, modelName, directionName)]; %#ok<AGROW>
    end
end

writetable(globalRows, fullfile(outputDir, "global_asphalt_summary.csv"));
writetable(targetRows, fullfile(outputDir, "target_95cm_summary.csv"));
writetable(directionRows, fullfile(outputDir, ...
    "direction_asphalt_summary.csv"));
soilRows = soilSummaryRows(auxiliary);
writetable(soilRows, fullfile(outputDir, ...
    "soil_summary_metrics.csv"));

baselineName = "Step19-SOTA-FiLM";
baseMask = P.model == baselineName & P.layer == "asphalt";
base = fbg_metrics(P.actual_ue(baseMask), P.predicted_ue(baseMask));
model = strings(0, 1);
delta_r2 = zeros(0, 1);
rmse_improvement_ue = zeros(0, 1);
rmse_improvement_percent = zeros(0, 1);
mae_improvement_ue = zeros(0, 1);
for i = 1:numel(models)
    if models(i) == baselineName
        continue;
    end
    mask = P.model == models(i) & P.layer == "asphalt";
    M = fbg_metrics(P.actual_ue(mask), P.predicted_ue(mask));
    model(end + 1, 1) = models(i); %#ok<AGROW>
    delta_r2(end + 1, 1) = M.r2 - base.r2; %#ok<AGROW>
    rmse_improvement_ue(end + 1, 1) = base.rmse - M.rmse; %#ok<AGROW>
    rmse_improvement_percent(end + 1, 1) = ...
        100 * (base.rmse - M.rmse) / max(base.rmse, eps); %#ok<AGROW>
    mae_improvement_ue(end + 1, 1) = base.mae - M.mae; %#ok<AGROW>
end
paired = table(model, delta_r2, rmse_improvement_ue, ...
    rmse_improvement_percent, mae_improvement_ue);
writetable(paired, fullfile(outputDir, "paired_improvement_summary.csv"));

end

function R = metricRow(P, mask, modelName)
M = fbg_metrics(P.actual_ue(mask), P.predicted_ue(mask));
model = modelName;
split = "test";
n = M.n;
r2 = M.r2;
rmse_ue = M.rmse;
mae_ue = M.mae;
bias_ue = M.bias;
R = table(model, split, n, r2, rmse_ue, mae_ue, bias_ue);
end

function R = directionMetricRow(P, mask, modelName, directionName)
R = metricRow(P, mask, modelName);
R.direction = directionName;
R = movevars(R, "direction", "After", "split");
end

function T = soilSummaryRows(A)
if isempty(A)
    T = emptySoilMetricTable();
    return;
end
models = unique(A.model, "stable");
T = table();
for i = 1:numel(models)
    modelName = models(i);
    modelMask = A.model == modelName;
    fields = {
        "soil_mean_x", "actual_soil_mean_x_ue", ...
            "predicted_soil_mean_x_ue";
        "soil_mean_y", "actual_soil_mean_y_ue", ...
            "predicted_soil_mean_y_ue"};
    for j = 1:size(fields, 1)
        actual = A.(fields{j, 2})(modelMask);
        predicted = A.(fields{j, 3})(modelMask);
        M = fbg_metrics(actual, predicted);
        if M.n == 0
            continue;
        end
        model = modelName;
        target = string(fields{j, 1});
        split = "test";
        n = M.n;
        r2 = M.r2;
        rmse_ue = M.rmse;
        mae_ue = M.mae;
        bias_ue = M.bias;
        row = table(model, target, split, n, r2, rmse_ue, ...
            mae_ue, bias_ue);
        T = [T; row]; %#ok<AGROW>
    end
end
if isempty(T)
    T = emptySoilMetricTable();
end
end

function T = emptySoilMetricTable()
T = table(strings(0, 1), strings(0, 1), strings(0, 1), ...
    zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    'VariableNames', {'model', 'target', 'split', 'n', 'r2', ...
    'rmse_ue', 'mae_ue', 'bias_ue'});
end
