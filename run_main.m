%% 运行当前最优Direct RBF零目标观测模型
% 要求三个仓库位于同一父目录，或设置FBG_CORRECTED_DATA环境变量。

clear; clc;
repoRoot = fileparts(mfilename('fullpath'));
dataFile = locate_corrected_dataset(repoRoot);
run(fullfile(repoRoot, 'check_data_schema.m'));

pythonCommand = string(getenv('FBG_PYTHON'));
if strlength(pythonCommand) == 0
    pythonCommand = "python";
end
scriptFile = fullfile(repoRoot, 'best_model', 'direct_rbf_zero_observation.py');
foldFile = fullfile(repoRoot, 'best_model', 'fixed_case_folds.csv');
resultDir = fullfile(repoRoot, 'results', 'direct_rbf_cv');
modelFile = fullfile(repoRoot, 'models', 'direct_rbf_corrected_v2.npz');

cvCommand = sprintf('"%s" "%s" cross-validate "%s" --folds "%s" --output "%s"', ...
    pythonCommand, scriptFile, dataFile, foldFile, resultDir);
fprintf('执行固定五折验证：\n%s\n', cvCommand);
[status, output] = system(cvCommand);
fprintf('%s\n', output);
assert(status == 0, 'Direct RBF五折验证失败。');

fitCommand = sprintf('"%s" "%s" fit "%s" --model "%s"', ...
    pythonCommand, scriptFile, dataFile, modelFile);
fprintf('训练并保存全数据模型：\n%s\n', fitCommand);
[status, output] = system(fitCommand);
fprintf('%s\n', output);
assert(status == 0, 'Direct RBF全数据训练失败。');

fprintf('完成。五折结果：%s\n', resultDir);
fprintf('部署模型：%s\n', modelFile);
