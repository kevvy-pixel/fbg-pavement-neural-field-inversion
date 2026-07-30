function [P, taskWeights] = fbg_run_one(T, fold, modelName, cfg, modelDir)
%FBG_RUN_ONE Train one model and export fold-level predictions and metrics.
data = fbg_prepare_data(T, fold, modelName, cfg);
fprintf("train/val/test rows: %d/%d/%d\n", numel(data.train.indices), ...
    numel(data.val.indices), numel(data.test.indices));

[params, history] = fbg_train(data, modelName, cfg);
writetable(history, fullfile(modelDir, "training_history.csv"));

serialParams = dlupdate(@(x) gather(extractdata(x)), params);
normalizer = data.normalizer; %#ok<NASGU>
save(fullfile(modelDir, "best_model.mat"), "serialParams", ...
    "normalizer", "modelName", "cfg");

result = fbg_predict(params, data.test.X, modelName, cfg);
Q = data.test.table;
n = height(Q);
actual_ue = Q.strain_ue(:);
predicted_ue = result.prediction(:);
residual_ue = predicted_ue - actual_ue;
eps_x_ue = result.epsX(:);
eps_y_ue = result.epsY(:);
w_proxy_m = result.wM(:);
soil_residual_proxy = result.soilResidual(:);
split = repmat("test", n, 1);
model = repmat(modelName, n, 1);

P = table(Q.row_id(:), Q.case_id(:), split, model, Q.layer(:), ...
    Q.direction(:), Q.sensor_x_cm(:), Q.sensor_y_cm(:), ...
    Q.load_x_cm(:), Q.load_y_cm(:), Q.load2_x_cm(:), ...
    Q.load2_y_cm(:), Q.has_load2(:), Q.load_kN(:), ...
    actual_ue, predicted_ue, residual_ue, eps_x_ue, eps_y_ue, ...
    w_proxy_m, soil_residual_proxy, ...
    'VariableNames', {'row_id', 'case_id', 'split', 'model', ...
    'layer', 'direction', 'sensor_x_cm', 'sensor_y_cm', ...
    'load_x_cm', 'load_y_cm', 'load2_x_cm', 'load2_y_cm', ...
    'has_load2', 'load_kN', 'actual_ue', 'predicted_ue', ...
    'residual_ue', 'eps_x_ue', 'eps_y_ue', 'w_proxy_m', ...
    'soil_residual_proxy'});
writetable(P, fullfile(modelDir, "test_predictions.csv"));

caseMetrics = makeCaseMetrics(P);
writetable(caseMetrics, fullfile(modelDir, "case_metrics.csv"));

model = modelName;
if modelName == "SharedMultilayer-FiLM"
    asphalt_weight = double(gather(extractdata( ...
        exp(-params.taskLogVarianceA))));
    soil_weight = double(gather(extractdata( ...
        exp(-params.taskLogVarianceS))));
else
    asphalt_weight = 1;
    soil_weight = 0;
end
taskWeights = table(model, asphalt_weight, soil_weight);
end

function S = makeCaseMetrics(P)
cases = unique(P.case_id, "stable");
case_id = strings(0, 1);
model = strings(0, 1);
layer = strings(0, 1);
split = strings(0, 1);
n = zeros(0, 1);
r2 = zeros(0, 1);
rmse_ue = zeros(0, 1);
mae_ue = zeros(0, 1);
bias_ue = zeros(0, 1);
for i = 1:numel(cases)
    for layerName = ["asphalt", "soil"]
        mask = P.case_id == cases(i) & P.layer == layerName;
        if ~any(mask)
            continue;
        end
        M = fbg_metrics(P.actual_ue(mask), P.predicted_ue(mask));
        case_id(end + 1, 1) = cases(i); %#ok<AGROW>
        model(end + 1, 1) = P.model(find(mask, 1)); %#ok<AGROW>
        layer(end + 1, 1) = layerName; %#ok<AGROW>
        split(end + 1, 1) = "test"; %#ok<AGROW>
        n(end + 1, 1) = M.n; %#ok<AGROW>
        r2(end + 1, 1) = M.r2; %#ok<AGROW>
        rmse_ue(end + 1, 1) = M.rmse; %#ok<AGROW>
        mae_ue(end + 1, 1) = M.mae; %#ok<AGROW>
        bias_ue(end + 1, 1) = M.bias; %#ok<AGROW>
    end
end
S = table(case_id, model, layer, split, n, r2, rmse_ue, mae_ue, bias_ue);
end
