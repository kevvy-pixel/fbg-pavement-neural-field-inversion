function correctedFile = locate_corrected_dataset(repoRoot)
%LOCATE_CORRECTED_DATASET 定位CorrectedData_v2权威响应表。
if nargin < 1 || strlength(string(repoRoot)) == 0
    repoRoot = fileparts(mfilename('fullpath'));
end
environmentPath = string(getenv('FBG_CORRECTED_DATA'));
candidates = strings(0,1);
if strlength(environmentPath) > 0
    candidates(end+1) = environmentPath; %#ok<AGROW>
end
candidates(end+1) = fullfile(repoRoot, '..', ...
    'fbg-multilayer-pavement-data', 'CorrectedData_v2', ...
    'Corrected_Response_Matrix_Table_v2.csv');
for index = 1:numel(candidates)
    if isfile(candidates(index))
        correctedFile = char(candidates(index));
        return;
    end
end
error('FBG:DataNotFound', [ ...
    '找不到CorrectedData_v2。请把数据仓库放在本仓库同级目录，', ...
    '或设置环境变量FBG_CORRECTED_DATA。']);
end
