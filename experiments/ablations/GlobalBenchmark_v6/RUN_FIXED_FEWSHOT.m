function RUN_FIXED_FEWSHOT(existingRunDir,inputFile,profile,outputBase)
%RUN_FIXED_FEWSHOT Reuse a trained Residual-FiLM for final calibration.
%
% Example:
%   RUN_FIXED_FEWSHOT("D:\FBG_runs\outputs\20260727_120000_quick", ...
%       "D:\data\Response_Matrix_Table.csv","quick","D:\FBG_runs")

arguments
    existingRunDir (1,1) string
    inputFile (1,1) string
    profile (1,1) string = "quick"
    outputBase (1,1) string = ""
end

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);

residualFile = fullfile(existingRunDir,'02_residual_film','model.mat');
if ~isfile(residualFile)
    error('Residual model not found: %s',residualFile);
end
if ~isfile(inputFile)
    error('Input CSV not found: %s',inputFile);
end

residualSaved = load(residualFile,'paramsCPU','meta','cfg');
residualParams = residualSaved.paramsCPU;
meta = residualSaved.meta;
cfg = residualSaved.cfg;

profileCfg = fbg_config(profile);
cfg.profile = profile+"-fixed-fewshot";
cfg.training.fewshotIterations = profileCfg.training.fewshotIterations;
cfg.training.fewshotLearningRate = profileCfg.training.fewshotLearningRate;
cfg.training.logEvery = profileCfg.training.logEvery;
cfg.training.validationEvery = profileCfg.training.validationEvery;
cfg.fewshot = profileCfg.fewshot;
cfg.runtime.useGPU = profileCfg.runtime.useGPU && canUseGPU;

[D,reloadedMeta] = fbg_load_data(inputFile,cfg);
localAssertCompatible(meta,reloadedMeta);
split = fbg_make_splits(D,cfg);

if strlength(outputBase) == 0
    workRoot = fbg_writable_root(rootDir);
else
    workRoot = fbg_writable_root(outputBase);
end
timestamp = string(datetime('now','Format','yyyyMMdd_HHmmss'));
outputDir = fullfile(workRoot,cfg.runtime.outputRoot, ...
    timestamp+"_"+profile+"_final_calibration_only");
mkdir(outputDir);

diary(fullfile(outputDir,'run.log'));
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('Final unique fixed-holdout calibration\n');
fprintf('Existing run: %s\nInput: %s\nOutput: %s\n', ...
    existingRunDir,inputFile,outputDir);
fprintf('Profile: %s | repeats: %d | GPU: %d\n', ...
    profile,cfg.fewshot.numRepeats,cfg.runtime.useGPU);

fbg_run_fixed_fewshot(residualParams,D,split,meta,cfg,outputDir);
fprintf('\nFinal calibration completed.\nResults: %s\n',outputDir);
end

function localAssertCompatible(savedMeta,currentMeta)
checks = ["spatialDim","conditionDim"];
for i = 1:numel(checks)
    field = checks(i);
    if savedMeta.(field) ~= currentMeta.(field)
        error(['The input data is incompatible with the saved model: ', ...
            '%s changed from %g to %g.'],field,savedMeta.(field), ...
            currentMeta.(field));
    end
end

numericChecks = ["xMin","xScale","yMin","yScale","targetScale"];
for i = 1:numel(numericChecks)
    field = numericChecks(i);
    tolerance = 1e-6*max(1,abs(savedMeta.(field)));
    if abs(savedMeta.(field)-currentMeta.(field)) > tolerance
        error(['The input data does not match the saved model normalization: ', ...
            '%s changed. Use the same Response_Matrix_Table.csv.'],field);
    end
end
end
