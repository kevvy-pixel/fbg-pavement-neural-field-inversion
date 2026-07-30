function outputDir = RUN_EXPERIMENT(mode, csvPath, workspaceDir)
%RUN_EXPERIMENT Main entry point for all v11 experiments.
if nargin < 3
    workspaceDir = "";
end
rootDir = fileparts(mfilename("fullpath"));
addpath(fullfile(rootDir, "src"));

cfg = fbg_config(mode);
workspaceDir = fbg_workspace(workspaceDir);
stamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
outputDir = fullfile(workspaceDir, "outputs", stamp + "_" + mode + "_v11");
mkdir(outputDir);

diary(fullfile(outputDir, "run.log"));
cleanup = onCleanup(@() diary("off")); %#ok<NASGU>
fprintf("FBG Neural Field Shared Multilayer v11\n");
fprintf("Mode: %s\nInput: %s\nOutput: %s\n", mode, csvPath, outputDir);
rng(cfg.seed, "twister");

T = fbg_load_data(csvPath);
fprintf("Rows: %d | cases: %d | asphalt: %d | soil: %d\n", ...
    height(T), numel(unique(T.case_id)), sum(T.layer == "asphalt"), ...
    sum(T.layer == "soil"));

fbg_diagnostics(T, fullfile(outputDir, "00_data_diagnostics"));
folds = fbg_make_folds(T, cfg);
models = ["Step19-SOTA-FiLM", "SharedMultilayer-FiLM"];

allPredictions = table();
allTaskWeights = table();
for f = 1:numel(folds)
    foldDir = fullfile(outputDir, sprintf("fold_%02d", f));
    mkdir(foldDir);
    fprintf("\n=== Fold %d/%d ===\n", f, numel(folds));
    for m = 1:numel(models)
        modelName = models(m);
        modelDir = fullfile(foldDir, fbg_safe_name(modelName));
        mkdir(modelDir);
        fprintf("\n--- %s ---\n", modelName);
        [predTable, taskWeights] = fbg_run_one( ...
            T, folds(f), modelName, cfg, modelDir);
        predTable.fold = repmat(f, height(predTable), 1);
        allPredictions = [allPredictions; predTable]; %#ok<AGROW>
        if ~isempty(taskWeights)
            taskWeights.fold = repmat(f, height(taskWeights), 1);
            allTaskWeights = [allTaskWeights; taskWeights]; %#ok<AGROW>
        end
    end
end

writetable(allPredictions, fullfile(outputDir, "predictions_all.csv"));
if ~isempty(allTaskWeights)
    writetable(allTaskWeights, fullfile(outputDir, ...
        "learned_task_weights.csv"));
end
fbg_report(allPredictions, outputDir);
fprintf("\nExperiment completed: %s\n", outputDir);
end
