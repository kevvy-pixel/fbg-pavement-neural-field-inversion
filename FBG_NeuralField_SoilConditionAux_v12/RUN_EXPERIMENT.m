function outputDir = RUN_EXPERIMENT(mode, csvPath, workspaceDir)
%RUN_EXPERIMENT Main entry point for all v12 experiments.
if nargin < 3
    workspaceDir = "";
end
rootDir = fileparts(mfilename("fullpath"));
addpath(fullfile(rootDir, "src"));

cfg = fbg_config(mode);
workspaceDir = fbg_workspace(workspaceDir);
stamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
outputDir = fullfile(workspaceDir, "outputs", stamp + "_" + mode + "_v12");
mkdir(outputDir);

diary(fullfile(outputDir, "run.log"));
cleanup = onCleanup(@() diary("off")); %#ok<NASGU>
fprintf("FBG Neural Field Soil Condition Auxiliary v12\n");
fprintf("Mode: %s\nInput: %s\nOutput: %s\n", mode, csvPath, outputDir);
rng(cfg.seed, "twister");

T = fbg_load_data(csvPath);
fprintf("Rows: %d | cases: %d | asphalt: %d | soil: %d\n", ...
    height(T), numel(unique(T.case_id)), sum(T.layer == "asphalt"), ...
    sum(T.layer == "soil"));

fbg_diagnostics(T, fullfile(outputDir, "00_data_diagnostics"));
folds = fbg_make_folds(T, cfg);
models = ["Step19-SOTA-FiLM", "SoilMeanAux-X-FiLM", ...
    "SoilMeanAux-XY-FiLM"];

allPredictions = table();
allAuxiliary = table();
for f = 1:numel(folds)
    foldDir = fullfile(outputDir, sprintf("fold_%02d", f));
    mkdir(foldDir);
    fprintf("\n=== Fold %d/%d ===\n", f, numel(folds));
    for m = 1:numel(models)
        modelName = models(m);
        modelDir = fullfile(foldDir, fbg_safe_name(modelName));
        mkdir(modelDir);
        modelSeed = cfg.seed + f - 1;
        rng(modelSeed, "twister");
        fprintf("\n--- %s | initialization seed: %d ---\n", ...
            modelName, modelSeed);
        [predTable, auxiliaryTable] = fbg_run_one( ...
            T, folds(f), modelName, cfg, modelDir);
        predTable.fold = repmat(f, height(predTable), 1);
        allPredictions = [allPredictions; predTable]; %#ok<AGROW>
        if ~isempty(auxiliaryTable)
            auxiliaryTable.fold = repmat(f, height(auxiliaryTable), 1);
            allAuxiliary = [allAuxiliary; auxiliaryTable]; %#ok<AGROW>
        end
    end
end

writetable(allPredictions, fullfile(outputDir, "predictions_all.csv"));
if ~isempty(allAuxiliary)
    writetable(allAuxiliary, fullfile(outputDir, ...
        "soil_summary_predictions.csv"));
end
fbg_report(allPredictions, outputDir, allAuxiliary);
fprintf("\nExperiment completed: %s\n", outputDir);
end
