function P = fbg_run_one(T, fold, modelName, cfg, modelDir)
%FBG_RUN_ONE Train/evaluate one model and export anchor diagnostics.
data = fbg_prepare_data(T, fold, modelName, cfg);
fprintf("train/val/test rows: %d/%d/%d\n", numel(data.train.indices), ...
    numel(data.val.indices), numel(data.test.indices));
fprintf("test anchor coverage: %.1f%% | mean condition distance %.4f\n", ...
    100 * mean(data.test.AnchorAvailable), ...
    mean(data.test.AnchorDistance(data.test.AnchorAvailable > 0)));

if modelName == "NearestAnchor-Persistence"
    history = table();
    result = persistencePrediction(data.test);
    normalizer = data.normalizer; %#ok<NASGU>
    save(fullfile(modelDir, "anchor_baseline.mat"), ...
        "normalizer", "modelName", "cfg");
else
    [params, history] = fbg_train(data, modelName, cfg);
    serialParams = dlupdate(@(x) gather(extractdata(x)), params);
    normalizer = data.normalizer; %#ok<NASGU>
    save(fullfile(modelDir, "best_model.mat"), "serialParams", ...
        "normalizer", "modelName", "cfg");
    result = fbg_predict(params, data.test.X, modelName, cfg);
end
writetable(history, fullfile(modelDir, "training_history.csv"));

Q = data.test.table;
n = height(Q);
actual_ue = Q.strain_ue(:);
predicted_ue = result.prediction(:);
residual_ue = predicted_ue - actual_ue;
eps_x_ue = result.epsX(:);
eps_y_ue = result.epsY(:);
w_proxy_m = result.wM(:);
anchor_case_id = data.test.AnchorCaseId(:);
anchor_strain_ue = data.test.AnchorStrainUe(:);
anchor_condition_distance = data.test.AnchorDistance(:);
anchor_sensor_distance = data.test.AnchorSensorDistanceNorm(:);
anchor_available = data.test.AnchorAvailable(:);
anchor_gate = result.anchorGate(:);
correction_ue = result.correction(:);
split = repmat("test", n, 1);
model = repmat(modelName, n, 1);

P = table(Q.row_id(:), Q.case_id(:), split, model, Q.layer(:), ...
    Q.direction(:), Q.sensor_x_cm(:), Q.sensor_y_cm(:), ...
    Q.load_x_cm(:), Q.load_y_cm(:), Q.load2_x_cm(:), ...
    Q.load2_y_cm(:), Q.has_load2(:), Q.load_kN(:), ...
    actual_ue, predicted_ue, residual_ue, eps_x_ue, eps_y_ue, ...
    w_proxy_m, anchor_case_id, anchor_strain_ue, ...
    anchor_condition_distance, anchor_sensor_distance, ...
    anchor_available, anchor_gate, correction_ue, ...
    'VariableNames', {'row_id', 'case_id', 'split', 'model', ...
    'layer', 'direction', 'sensor_x_cm', 'sensor_y_cm', ...
    'load_x_cm', 'load_y_cm', 'load2_x_cm', 'load2_y_cm', ...
    'has_load2', 'load_kN', 'actual_ue', 'predicted_ue', ...
    'residual_ue', 'eps_x_ue', 'eps_y_ue', 'w_proxy_m', ...
    'anchor_case_id', 'anchor_strain_ue', ...
    'anchor_condition_distance', 'anchor_sensor_distance_norm', ...
    'anchor_available', 'anchor_gate', 'correction_ue'});
writetable(P, fullfile(modelDir, "test_predictions.csv"));
writetable(makeCaseMetrics(P), fullfile(modelDir, "case_metrics.csv"));
writetable(makeAnchorAudit(P), fullfile(modelDir, "anchor_audit.csv"));
end

function result = persistencePrediction(testData)
n = numel(testData.Y);
available = testData.AnchorAvailable(:) > 0;
prediction = nan(n, 1);
prediction(available) = testData.AnchorStrainUe(available);
result.prediction = prediction;
result.epsX = nan(n, 1);
result.epsY = nan(n, 1);
result.wM = nan(n, 1);
result.fieldPrediction = nan(n, 1);
result.correction = zeros(n, 1);
result.anchorGate = zeros(n, 1);
end

function S = makeAnchorAudit(P)
cases = unique(P.case_id, "stable");
case_id = strings(numel(cases), 1);
anchor_case_id = strings(numel(cases), 1);
n = zeros(numel(cases), 1);
coverage_percent = zeros(numel(cases), 1);
condition_distance = zeros(numel(cases), 1);
max_sensor_distance_norm = zeros(numel(cases), 1);
for i = 1:numel(cases)
    mask = P.case_id == cases(i);
    case_id(i) = cases(i);
    n(i) = sum(mask);
    coverage_percent(i) = 100 * mean(P.anchor_available(mask));
    firstAvailable = find(mask & P.anchor_available > 0, 1);
    if isempty(firstAvailable)
        anchor_case_id(i) = "";
        condition_distance(i) = NaN;
        max_sensor_distance_norm(i) = NaN;
    else
        anchor_case_id(i) = P.anchor_case_id(firstAvailable);
        condition_distance(i) = mean( ...
            P.anchor_condition_distance(mask & P.anchor_available > 0));
        max_sensor_distance_norm(i) = max( ...
            P.anchor_sensor_distance_norm(mask & P.anchor_available > 0));
    end
end
S = table(case_id, anchor_case_id, n, coverage_percent, ...
    condition_distance, max_sensor_distance_norm);
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
    mask = P.case_id == cases(i) & P.layer == "asphalt";
    if ~any(mask)
        continue;
    end
    M = fbg_metrics(P.actual_ue(mask), P.predicted_ue(mask));
    case_id(end + 1, 1) = cases(i); %#ok<AGROW>
    model(end + 1, 1) = P.model(find(mask, 1)); %#ok<AGROW>
    layer(end + 1, 1) = "asphalt"; %#ok<AGROW>
    split(end + 1, 1) = "test"; %#ok<AGROW>
    n(end + 1, 1) = M.n; %#ok<AGROW>
    r2(end + 1, 1) = M.r2; %#ok<AGROW>
    rmse_ue(end + 1, 1) = M.rmse; %#ok<AGROW>
    mae_ue(end + 1, 1) = M.mae; %#ok<AGROW>
    bias_ue(end + 1, 1) = M.bias; %#ok<AGROW>
end
S = table(case_id, model, layer, split, n, r2, rmse_ue, ...
    mae_ue, bias_ue);
end
