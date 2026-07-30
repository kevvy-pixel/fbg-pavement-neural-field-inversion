function aggregate_full_ablation(runRoot, outputDir)
% Recompute the cross-version benchmark directly from every point prediction.

arguments
    runRoot (1,1) string
    outputDir (1,1) string
end

versions = ["v10","v11","v12","v13","v14"];
if ~isfolder(outputDir)
    mkdir(outputDir);
end

allPred = table();
runManifest = table();
coreColumns = ["row_id","case_id","split","model","layer","direction", ...
    "sensor_x_cm","sensor_y_cm","load_x_cm","load_y_cm","load_kN", ...
    "actual_ue","predicted_ue","residual_ue","fold"];
for v = versions
    candidates = dir(fullfile(runRoot, v, "outputs", "*_full_" + v));
    candidates = candidates([candidates.isdir]);
    complete = false(size(candidates));
    for j = 1:numel(candidates)
        complete(j) = isfile(fullfile(candidates(j).folder, candidates(j).name, ...
            "predictions_all.csv"));
    end
    candidates = candidates(complete);
    assert(~isempty(candidates), "No completed full run found for %s.", v);
    [~, newest] = max([candidates.datenum]);
    runDir = string(fullfile(candidates(newest).folder, candidates(newest).name));

    P = readtable(fullfile(runDir, "predictions_all.csv"), ...
        TextType="string", VariableNamingRule="preserve");
    assert(all(ismember(coreColumns, string(P.Properties.VariableNames))), ...
        "%s predictions are missing a required core column.", v);
    P = P(:, cellstr(coreColumns));
    P.version = repmat(v, height(P), 1);
    P.model_key = v + "/" + string(P.model);
    allPred = [allPred; P]; %#ok<AGROW>

    runManifest = [runManifest; table(v, runDir, height(P), ...
        numel(unique(string(P.model))), ...
        VariableNames=["version","run_directory","prediction_rows","model_count"])]; %#ok<AGROW>
end

writetable(runManifest, fullfile(outputDir, "run_manifest.csv"));
writetable(allPred, fullfile(outputDir, "all_predictions_versioned.csv"));

A = allPred(string(allPred.layer) == "asphalt", :);
keys = unique(A.model_key, "stable");
coverage = table();
globalMetrics = table();
directionMetrics = table();
foldMetrics = table();
targetMetrics = table();

for key = keys'
    M = A(A.model_key == key, :);
    ids = double(M.row_id);
    coverage = [coverage; table(key, height(M), numel(unique(ids)), ...
        sum(histcounts(ids, [unique(ids); inf]) > 1), ...
        numel(unique(double(M.fold))), ...
        VariableNames=["model_key","asphalt_rows","unique_row_ids", ...
        "duplicate_row_id_groups","fold_count"])]; %#ok<AGROW>

    globalMetrics = [globalMetrics; metricRow(key, "all", M)]; %#ok<AGROW>
    for direction = ["x","y"]
        D = M(string(M.direction) == direction, :);
        directionMetrics = [directionMetrics; metricRow(key, direction, D)]; %#ok<AGROW>
    end
    for fold = 1:5
        F = M(double(M.fold) == fold, :);
        r = metricRow(key, string(fold), F);
        r.fold = fold;
        foldMetrics = [foldMetrics; r]; %#ok<AGROW>
    end

    T95 = M(abs(double(M.load_x_cm) - 95) < 1e-9, :);
    targetMetrics = [targetMetrics; metricRow(key, "95cm", T95)]; %#ok<AGROW>
end

assert(all(coverage.asphalt_rows == 616), ...
    "At least one model does not have all 616 asphalt predictions.");
assert(all(coverage.unique_row_ids == 616), ...
    "At least one model does not cover 616 unique asphalt rows.");
assert(all(coverage.duplicate_row_id_groups == 0), ...
    "At least one model predicts an asphalt row more than once.");
assert(all(coverage.fold_count == 5), ...
    "At least one model is missing a fold.");
assert(all(targetMetrics.n == 28), ...
    "At least one model does not contain all 28 target-95 observations.");

globalMetrics = sortrows(globalMetrics, "rmse_ue");
directionMetrics = sortrows(directionMetrics, ["group","rmse_ue"]);
targetMetrics = sortrows(targetMetrics, "rmse_ue");
writetable(coverage, fullfile(outputDir, "coverage_audit.csv"));
writetable(globalMetrics, fullfile(outputDir, "global_asphalt_metrics.csv"));
writetable(directionMetrics, fullfile(outputDir, "direction_metrics.csv"));
writetable(foldMetrics, fullfile(outputDir, "fold_metrics.csv"));
writetable(targetMetrics, fullfile(outputDir, "target_95cm_metrics.csv"));

foldStability = groupsummary(foldMetrics, "model_key", ["mean","std"], ...
    ["rmse_ue","mae_ue","r2"]);
writetable(foldStability, fullfile(outputDir, "fold_stability.csv"));

baselineKey = "v14/AnchoredResidual-XOnly-LowFreq";
assert(any(globalMetrics.model_key == baselineKey), "v14 low-frequency baseline is absent.");
baseGlobal = globalMetrics(globalMetrics.model_key == baselineKey, :);
baseTarget = targetMetrics(targetMetrics.model_key == baselineKey, :);
comparison = globalMetrics(:, ["model_key","rmse_ue","mae_ue","r2"]);
comparison.delta_rmse_vs_v14_lowfreq = comparison.rmse_ue - baseGlobal.rmse_ue;
comparison.delta_mae_vs_v14_lowfreq = comparison.mae_ue - baseGlobal.mae_ue;
comparison.delta_r2_vs_v14_lowfreq = comparison.r2 - baseGlobal.r2;
comparison.target95_rmse_ue = nan(height(comparison), 1);
comparison.target95_r2 = nan(height(comparison), 1);
for i = 1:height(comparison)
    q = targetMetrics.model_key == comparison.model_key(i);
    comparison.target95_rmse_ue(i) = targetMetrics.rmse_ue(q);
    comparison.target95_r2(i) = targetMetrics.r2(q);
end
comparison.delta_target95_rmse_vs_v14_lowfreq = ...
    comparison.target95_rmse_ue - baseTarget.rmse_ue;
writetable(comparison, fullfile(outputDir, "comparison_vs_v14_lowfreq.csv"));

training = collectTraining(runManifest);
writetable(training, fullfile(outputDir, "training_budget_audit.csv"));

save(fullfile(outputDir, "full_ablation_aggregate.mat"), ...
    "allPred", "runManifest", "coverage", "globalMetrics", ...
    "directionMetrics", "foldMetrics", "foldStability", ...
    "targetMetrics", "comparison", "training", "-v7.3");
fprintf("Full ablation aggregation completed: %s\n", outputDir);
end

function R = metricRow(key, group, T)
y = double(T.actual_ue);
yhat = double(T.predicted_ue);
e = yhat - y;
rmse = sqrt(mean(e.^2));
mae = mean(abs(e));
r2 = 1 - sum(e.^2) / sum((y - mean(y)).^2);
bias = mean(e);
R = table(key, group, numel(y), rmse, mae, r2, bias, ...
    VariableNames=["model_key","group","n","rmse_ue","mae_ue","r2","bias_ue"]);
end

function training = collectTraining(runManifest)
training = table();
for i = 1:height(runManifest)
    v = runManifest.version(i);
    runDir = runManifest.run_directory(i);
    for fold = 1:5
        foldDir = fullfile(runDir, sprintf("fold_%02d", fold));
        modelDirs = dir(foldDir);
        modelDirs = modelDirs([modelDirs.isdir] & ~ismember({modelDirs.name}, {'.','..'}));
        for j = 1:numel(modelDirs)
            model = string(modelDirs(j).name);
            historyFile = fullfile(modelDirs(j).folder, modelDirs(j).name, ...
                "training_history.csv");
            if isfile(historyFile)
                H = readtable(historyFile, VariableNamingRule="preserve");
                if isempty(H)
                    iterations = 0;
                    validations = 0;
                    bestVal = NaN;
                else
                    iterations = max(double(H.iteration));
                    validations = height(H);
                    if ismember("val_rmse_ue", string(H.Properties.VariableNames))
                        bestVal = min(double(H.val_rmse_ue));
                    else
                        bestVal = NaN;
                    end
                end
                training = [training; table(v, v + "/" + model, fold, ...
                    iterations, validations, bestVal, ...
                    VariableNames=["version","model_key","fold","iterations", ...
                    "validation_records","best_validation_rmse_ue"])]; %#ok<AGROW>
            end
        end
    end
end
end
