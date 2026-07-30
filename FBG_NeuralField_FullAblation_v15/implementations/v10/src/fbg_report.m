function fbg_report(P, etaTable, outputDir)
%FBG_REPORT Build the cross-fold summary tables expected by the workflow.
models = unique(P.model, "stable");
globalRows = table();
targetRows = table();
soilRows = table();

for i = 1:numel(models)
    modelName = models(i);
    asphalt = P.model == modelName & P.layer == "asphalt";
    globalRows = [globalRows; metricRow(P, asphalt, modelName)]; %#ok<AGROW>

    target = asphalt & abs(P.load_x_cm - 95) <= 0.6;
    targetRows = [targetRows; metricRow(P, target, modelName)]; %#ok<AGROW>

    soil = P.model == modelName & P.layer == "soil";
    if any(soil)
        soilRows = [soilRows; metricRow(P, soil, modelName)]; %#ok<AGROW>
    end
end

writetable(globalRows, fullfile(outputDir, "global_asphalt_summary.csv"));
writetable(targetRows, fullfile(outputDir, "target_95cm_summary.csv"));
if isempty(soilRows)
    soilRows = emptyMetricTable();
end
writetable(soilRows, fullfile(outputDir, "soil_auxiliary_summary.csv"));

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

if ~isempty(etaTable)
    cases = unique(etaTable.case_id, "stable");
    case_id = strings(numel(cases), 1);
    load_x_cm = zeros(numel(cases), 1);
    load_kN = zeros(numel(cases), 1);
    mean_eta_predicted = zeros(numel(cases), 1);
    std_eta_predicted = zeros(numel(cases), 1);
    for i = 1:numel(cases)
        mask = etaTable.case_id == cases(i);
        case_id(i) = cases(i);
        load_x_cm(i) = median(etaTable.load_x_cm(mask));
        load_kN(i) = median(etaTable.load_kN(mask));
        mean_eta_predicted(i) = mean(etaTable.eta_predicted(mask));
        std_eta_predicted(i) = std(etaTable.eta_predicted(mask));
    end
    etaSummary = table(case_id, load_x_cm, load_kN, ...
        mean_eta_predicted, std_eta_predicted);
    writetable(etaSummary, fullfile(outputDir, ...
        "transfer_eta_by_case_summary.csv"));
end
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

function T = emptyMetricTable()
T = table(strings(0, 1), strings(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    'VariableNames', {'model', 'split', 'n', 'r2', 'rmse_ue', ...
    'mae_ue', 'bias_ue'});
end
