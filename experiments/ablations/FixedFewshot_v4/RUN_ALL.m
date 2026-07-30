function RUN_ALL(profile, inputFile, outputBase)
%RUN_ALL Execute all FBG neural-field experiments and save every artifact.
%
% Examples:
%   RUN_ALL("quick")
%   RUN_ALL("research","D:\data\Response_Matrix_Table.csv")
%   RUN_ALL("research","D:\data\Response_Matrix_Table.csv","D:\FBG_runs")

arguments
    profile (1,1) string = "quick"
    inputFile (1,1) string = ""
    outputBase (1,1) string = ""
end

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
cfg = fbg_config(profile);
if strlength(inputFile) == 0
    inputFile = fullfile(rootDir,cfg.data.inputFile);
end

if cfg.runtime.useGPU && ~canUseGPU
    fprintf('Compatible GPU not detected; using CPU.\n');
    cfg.runtime.useGPU = false;
end

rng(cfg.randomSeed);
timestamp = string(datetime('now','Format','yyyyMMdd_HHmmss'));
if strlength(outputBase) == 0
    workRoot = fbg_writable_root(rootDir);
else
    workRoot = fbg_writable_root(outputBase);
end
runDir = fullfile(workRoot,cfg.runtime.outputRoot,timestamp+"_"+profile);
mkdir(runDir);

logFile = fullfile(runDir,'run.log');
diary(logFile);
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('FBG Neural Field Experiment Package\n');
fprintf('Profile: %s\nInput: %s\nOutput: %s\n',profile,inputFile,runDir);
fprintf('Writable workspace: %s\n',workRoot);
fprintf('GPU enabled: %d\n',cfg.runtime.useGPU);

validationDir = fullfile(runDir,'00_data_validation');
mkdir(validationDir);

[D,meta] = fbg_load_data(inputFile,cfg);
split = fbg_make_splits(D,cfg);
localSaveConfiguration(cfg,meta,validationDir);
localSaveDataDiagnostics(D,split,cfg,validationDir);

fprintf('\nData: %d rows, %d cases, %d condition features.\n', ...
    height(D.table),numel(unique(D.table.case_id)),meta.conditionDim);
fprintf('Train/validation/test rows: %d/%d/%d\n', ...
    sum(split.train),sum(split.validation),sum(split.test));

% Step 1: plain FiLM baseline.
rng(cfg.randomSeed);
plainParams = fbg_model_init("plain",meta,cfg);
[plainParams,plainHistory] = fbg_train_model(plainParams,"plain",D, ...
    split.train,split.validation,meta,cfg,"plain",false);
plainResult = fbg_save_stage(fullfile(runDir,'01_plain_film'), ...
    plainParams,"plain",plainHistory,D,split,meta,cfg);

% Step 2: Residual-FiLM.
rng(cfg.randomSeed);
residualParams = fbg_model_init("residual",meta,cfg);
[residualParams,residualHistory] = fbg_train_model(residualParams,"residual",D, ...
    split.train,split.validation,meta,cfg,"residual",false);
residualResult = fbg_save_stage(fullfile(runDir,'02_residual_film'), ...
    residualParams,"residual",residualHistory,D,split,meta,cfg);

% Step 3: spatially conditioned edge residual expert, warm-started from Step 2.
edgeParams = fbg_model_init("edge",meta,cfg,residualParams);
[edgeParams,edgeHistory] = fbg_train_model(edgeParams,"edge",D, ...
    split.train,split.validation,meta,cfg,"edge",true);
edgeResult = fbg_save_stage(fullfile(runDir,'03_spatial_edge_expert'), ...
    edgeParams,"edge",edgeHistory,D,split,meta,cfg);

% Step 4: repeated fixed-holdout target-case calibration.
[fixedFewshotSummary,fixedFewshotRepeats] = fbg_run_fixed_fewshot( ...
    residualParams,edgeParams,D,split,meta,cfg, ...
    fullfile(runDir,'04_fixed_holdout_fewshot_95cm')); %#ok<NASGU>

% Step 5: compact cross-model comparison.
comparisonDir = fullfile(runDir,'05_comparison');
mkdir(comparisonDir);
comparison = localBuildComparison(plainResult,residualResult,edgeResult,cfg);
writetable(comparison,fullfile(comparisonDir,'model_comparison.csv'));
localPlotComparison(comparison,fullfile(comparisonDir,'model_comparison.png'));

manifest.profile = profile;
manifest.input_file = inputFile;
manifest.output_directory = string(runDir);
manifest.created_at = string(datetime('now'));
manifest.target_case_pattern = cfg.data.targetCasePattern;
manifest.files_are_grouped_by_stage = true;
manifest.fixed_holdout_fewshot = true;
manifest.fewshot_repeats = cfg.fewshot.numRepeats;
localWriteJson(fullfile(runDir,'run_manifest.json'),manifest);

fprintf('\nAll experiments completed.\nResults: %s\n',runDir);
end

function localSaveConfiguration(cfg,meta,outputDir)
save(fullfile(outputDir,'config.mat'),'cfg','meta');
localWriteJson(fullfile(outputDir,'config.json'),cfg);
end

function localWriteJson(fileName,value)
text = jsonencode(value,'PrettyPrint',true);
fid = fopen(fileName,'w','n','UTF-8');
if fid < 0, error('Cannot write JSON file: %s',fileName); end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid,'%s',text);
end

function localSaveDataDiagnostics(D,split,cfg,outputDir)
T = D.table;
splitLabel = repmat("unused",height(T),1);
splitLabel(split.train) = "train";
splitLabel(split.validation) = "validation";
splitLabel(split.test) = "test";

splitAssignments = table(T.case_id,T.sensor_id,T.x_cm,T.y_cm,splitLabel, ...
    'VariableNames',{'case_id','sensor_id','x_cm','y_cm','split'});
writetable(splitAssignments,fullfile(outputDir,'split_assignments.csv'));

cases = unique(T.case_id,'stable');
case_id = strings(numel(cases),1);
load_type = strings(numel(cases),1);
load_kg = zeros(numel(cases),1);
n_rows = zeros(numel(cases),1);
split_name = strings(numel(cases),1);
max_abs_microstrain = zeros(numel(cases),1);
for i = 1:numel(cases)
    subMask = T.case_id == cases(i);
    sub = T(subMask,:);
    case_id(i) = cases(i);
    load_type(i) = sub.load_type(1);
    load_kg(i) = sub.load_kg(1);
    n_rows(i) = height(sub);
    split_name(i) = splitLabel(find(subMask,1));
    max_abs_microstrain(i) = max(abs(sub.response_microstrain));
end
caseSummary = table(case_id,load_type,load_kg,n_rows,split_name,max_abs_microstrain);
writetable(caseSummary,fullfile(outputDir,'case_summary.csv'));

datasetSummary = table(height(T),numel(cases),sum(split.train), ...
    sum(split.validation),sum(split.test),min(T.x_cm),max(T.x_cm), ...
    min(T.y_cm),max(T.y_cm),mean(T.response_microstrain), ...
    std(T.response_microstrain),max(abs(T.response_microstrain)), ...
    'VariableNames',{'n_rows','n_cases','n_train','n_validation','n_test', ...
    'x_min_cm','x_max_cm','y_min_cm','y_max_cm','response_mean_microstrain', ...
    'response_std_microstrain','response_max_abs_microstrain'});
writetable(datasetSummary,fullfile(outputDir,'dataset_summary.csv'));

target = T(contains(lower(T.case_id),lower(cfg.data.targetCasePattern)),:);
writetable(target,fullfile(outputDir,'target_95cm_raw.csv'));
localMeasuredShapeDiagnostic(T,cfg,outputDir);
end

function localMeasuredShapeDiagnostic(T,cfg,outputDir)
patterns = [cfg.data.forceTrainPattern,cfg.data.targetCasePattern];
rows = T([] ,{'case_id','x_cm','y_cm'});
rows.response_microstrain = zeros(0,1);
rows.normalized_response = zeros(0,1);
for i = 1:numel(patterns)
    sub = T(contains(lower(T.case_id),lower(patterns(i))) & ...
        T.direction_longitudinal == 1,:);
    if isempty(sub), continue; end
    cases = unique(sub.case_id,'stable');
    for j = 1:numel(cases)
        one = sub(sub.case_id == cases(j),:);
        centerDistance = abs(one.y_cm-one.p1_y);
        one = one(centerDistance <= min(centerDistance)+1e-9,:);
        one = sortrows(one,'x_cm');
        scale = max(abs(one.response_microstrain));
        one.normalized_response = one.response_microstrain/max(scale,eps);
        rows = [rows; one(:,{'case_id','x_cm','y_cm', ...
            'response_microstrain','normalized_response'})]; %#ok<AGROW>
    end
end
writetable(rows,fullfile(outputDir,'measured_90cm_vs_95cm_profiles.csv'));

fig = figure('Visible','off','Color','w','Position',[100 100 900 500]);
if isempty(rows)
    axis off; text(0.1,0.5,'90/95 cm profiles were not found.','FontSize',14);
else
    hold on;
    cases = unique(rows.case_id,'stable');
    colors = lines(numel(cases));
    for i = 1:numel(cases)
        sub = rows(rows.case_id == cases(i),:);
        plot(sub.x_cm,sub.normalized_response,'-o','Color',colors(i,:), ...
            'LineWidth',1.4,'DisplayName',cases(i));
    end
    xlabel('x (cm)'); ylabel('Normalized measured strain');
    title('Measured shape diagnostic: 90 cm vs 95 cm');
    legend('Interpreter','none','Location','best'); grid on;
end
saveas(fig,fullfile(outputDir,'measured_90cm_vs_95cm_profiles.png'));
close(fig);
end

function comparison = localBuildComparison(plainResult,residualResult, ...
    edgeResult,cfg)
names = ["Plain-FiLM";"Residual-FiLM";"Spatial-edge-expert"];
results = {plainResult,residualResult,edgeResult};
n = numel(names);
test_R2 = nan(n,1);
test_RMSE_microstrain = nan(n,1);
target_R2 = nan(n,1);
target_RMSE_microstrain = nan(n,1);
for i = 1:n
    row = results{i}.metrics(results{i}.metrics.split == "test",:);
    if ~isempty(row)
        test_R2(i) = row.R2(1);
        test_RMSE_microstrain(i) = row.RMSE_microstrain(1);
    end
    target = results{i}.caseMetrics(contains(lower(results{i}.caseMetrics.case_id), ...
        lower(cfg.data.targetCasePattern)),:);
    if ~isempty(target)
        target_R2(i) = mean(target.R2,'omitnan');
        target_RMSE_microstrain(i) = mean(target.RMSE_microstrain,'omitnan');
    end
end

comparison = table(names,test_R2,test_RMSE_microstrain,target_R2, ...
    target_RMSE_microstrain,'VariableNames',{'model','test_R2', ...
    'test_RMSE_microstrain','target_R2','target_RMSE_microstrain'});
end

function localPlotComparison(comparison,outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 1000 520]);
tiledlayout(1,2);
nexttile;
bar(categorical(comparison.model),comparison.target_R2);
ylabel('Target R^2'); grid on; title('95 cm accuracy');
nexttile;
bar(categorical(comparison.model),comparison.target_RMSE_microstrain);
ylabel('Target RMSE (\muepsilon)'); grid on; title('95 cm error');
saveas(fig,outputFile); close(fig);
end
