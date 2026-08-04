%% 检查CorrectedData_v2结构，不启动训练

if ~exist('repoRoot', 'var')
    repoRoot = fileparts(mfilename('fullpath'));
end
if ~exist('dataFile', 'var')
    dataFile = locate_corrected_dataset(repoRoot);
end

requiredColumns = [ ...
    "case_id","sensor_id","x_cm","y_cm","z_cm","layer","direction", ...
    "channel","interrogator","zero_load_baseline_nm", ...
    "delta_lambda_initial_nm","delta_lambda_final_nm", ...
    "delta_lambda_peak_signed_nm","loaded_creep_change_nm", ...
    "preload_noise_std_nm","load_kg","load_type", ...
    "p1_x","p1_y","p2_x","p2_y","load_onset_s","analysis_quality"];

T = readtable(dataFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
actualColumns = string(T.Properties.VariableNames);
missingColumns = setdiff(requiredColumns, actualColumns);
assert(isempty(missingColumns), '输入表缺少字段：%s', strjoin(missingColumns, ', '));
assert(height(T) == 1122, '期望1122行，实际%d行。', height(T));
assert(numel(unique(T.case_id)) == 22, '期望22个独立工况。');
assert(sum(T.layer == "Asphalt") == 616, '期望616条沥青记录。');
assert(sum(T.layer == "Soil") == 484, '期望484条土层记录。');
assert(sum(T.layer == "Temperature") == 22, '期望22条温度参考。');
assert(all(T.direction(T.layer == "Asphalt") == "Longitudinal" | ...
    T.direction(T.layer == "Asphalt") == "Transverse"), ...
    '沥青层存在不支持的方向标签。');

fprintf('CorrectedData_v2结构检查通过。\n');
fprintf('文件：%s\n', dataFile);
fprintf('行数：%d；工况：%d；沥青：%d；土层：%d；温度：%d。\n', ...
    height(T), numel(unique(T.case_id)), sum(T.layer == "Asphalt"), ...
    sum(T.layer == "Soil"), sum(T.layer == "Temperature"));
