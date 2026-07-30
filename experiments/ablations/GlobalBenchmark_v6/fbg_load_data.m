function [D, meta] = fbg_load_data(inputFile, cfg)
%FBG_LOAD_DATA Read, validate, filter, aggregate, and encode the response table.

if ~isfile(inputFile)
    error(['Input file not found: %s\n' ...
        'Generate it with the original run_02_build_response_matrix.m, ' ...
        'or pass the complete CSV path to RUN_ALL.'], inputFile);
end

T = readtable(inputFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
required = ["case_id","sensor_id","x_cm","y_cm","layer","direction", ...
    "delta_lambda_resp_nm","load_type","load_kg","p1_x","p1_y","p2_x","p2_y"];
missing = required(~ismember(required, string(T.Properties.VariableNames)));
if ~isempty(missing)
    error('Missing required columns: %s', strjoin(missing, ', '));
end

numericColumns = ["x_cm","y_cm","delta_lambda_resp_nm","load_kg", ...
    "p1_x","p1_y","p2_x","p2_y"];
for name = numericColumns
    T.(name) = localToDouble(T.(name));
end

T.case_id = string(T.case_id);
T.sensor_id = string(T.sensor_id);
T.layer = string(T.layer);
T.direction = string(T.direction);
T.load_type = string(T.load_type);

% Preserve the correction already used by the original analysis scripts.
mask425 = contains(T.case_id,"42.5cm");
T.p1_x(mask425) = 42.5;

layerMask = strcmpi(strtrim(T.layer), cfg.data.layer);
if ~any(layerMask)
    error('No rows found for layer "%s". Available layers: %s', ...
        cfg.data.layer, strjoin(unique(T.layer), ', '));
end
T = T(layerMask,:);

valid = isfinite(T.x_cm) & isfinite(T.y_cm) & ...
    isfinite(T.delta_lambda_resp_nm) & isfinite(T.load_kg) & ...
    isfinite(T.p1_x) & isfinite(T.p1_y);
T = T(valid,:);

dirText = lower(strtrim(T.direction));
isLong = contains(dirText, "纵") | contains(dirText, "long");
isTran = contains(dirText, "横") | contains(dirText, "trans");
if any(~(isLong | isTran))
    bad = unique(T.direction(~(isLong | isTran)));
    error('Unknown direction labels: %s', strjoin(bad, ', '));
end
T.direction_longitudinal = double(isLong);

isDual = contains(lower(T.load_type),"dual") | contains(T.load_type,"双点");
T.p2_present = double(isDual & isfinite(T.p2_x) & isfinite(T.p2_y));
replaceP2 = T.p2_present == 0;
T.p2_x(replaceP2) = T.p1_x(replaceP2);
T.p2_y(replaceP2) = T.p1_y(replaceP2);

T.response_microstrain = T.delta_lambda_resp_nm ./ ...
    cfg.data.kEps_nm_per_microstrain;

if cfg.data.aggregateDuplicates
    T = localAggregate(T);
end

meta.xMin = min(T.x_cm);
meta.xMax = max(T.x_cm);
meta.yMin = min(T.y_cm);
meta.yMax = max(T.y_cm);
meta.xScale = max(meta.xMax - meta.xMin, eps);
meta.yScale = max(meta.yMax - meta.yMin, eps);
meta.maxLoad = max(T.load_kg);
meta.targetScale = max(std(T.response_microstrain, 'omitnan'), 1);
meta.zNeutral_cm = cfg.data.zNeutral_cm;
meta.typeNames = unique(T.load_type, 'stable');
meta.numTypes = numel(meta.typeNames);
meta.inputFile = string(inputFile);

xN = (T.x_cm - meta.xMin) ./ meta.xScale;
yN = (T.y_cm - meta.yMin) ./ meta.yScale;
p1xN = (T.p1_x - meta.xMin) ./ meta.xScale;
p1yN = (T.p1_y - meta.yMin) ./ meta.yScale;
p2xN = (T.p2_x - meta.xMin) ./ meta.xScale;
p2yN = (T.p2_y - meta.yMin) ./ meta.yScale;

d1 = min([p1xN, 1-p1xN, p1yN, 1-p1yN], [], 2);
d2 = min([p2xN, 1-p2xN, p2yN, 1-p2yN], [], 2);
dLoadEdge = min(d1, d2);

oneHot = zeros(height(T), meta.numTypes);
for i = 1:meta.numTypes
    oneHot(:,i) = double(T.load_type == meta.typeNames(i));
end

cond = [ ...
    T.load_kg ./ max(meta.maxLoad, eps), ...
    p1xN, p1yN, p2xN, p2yN, ...
    T.p2_present, dLoadEdge, ...
    exp(-max(dLoadEdge,0) ./ cfg.model.edgeLengthNormalized), ...
    oneHot]';

D.table = T;
D.x = T.x_cm';
D.y = T.y_cm';
D.target = T.response_microstrain';
D.dirLong = T.direction_longitudinal';
D.loadKg = T.load_kg';
D.p1x = T.p1_x';
D.p1y = T.p1_y';
D.dLoadEdge = dLoadEdge';
D.cond = cond;
D.rowId = 1:height(T);
D.xN = xN';
D.yN = yN';

meta.conditionDim = size(cond,1);
meta.spatialDim = 8 + 4*numel(cfg.model.fourierFrequencies);
end

function T = localAggregate(T)
% Average repeated measurements at the same physical point and condition.
[G, case_id, x_cm, y_cm, layer, direction, load_type, load_kg, ...
    p1_x, p1_y, p2_x, p2_y, p2_present, direction_longitudinal] = ...
    findgroups(T.case_id, T.x_cm, T.y_cm, T.layer, T.direction, ...
    T.load_type, T.load_kg, T.p1_x, T.p1_y, T.p2_x, T.p2_y, ...
    T.p2_present, T.direction_longitudinal);

response = splitapply(@(v) mean(v, 'omitnan'), T.delta_lambda_resp_nm, G);
responseMicro = splitapply(@(v) mean(v, 'omitnan'), T.response_microstrain, G);
sensor_id = "AGG_" + compose("%05d", (1:numel(response))');

T = table(case_id, sensor_id, x_cm, y_cm, layer, direction, response, ...
    load_type, load_kg, p1_x, p1_y, p2_x, p2_y, p2_present, ...
    direction_longitudinal, responseMicro, ...
    'VariableNames', {'case_id','sensor_id','x_cm','y_cm','layer','direction', ...
    'delta_lambda_resp_nm','load_type','load_kg','p1_x','p1_y','p2_x','p2_y', ...
    'p2_present','direction_longitudinal','response_microstrain'});
end

function x = localToDouble(x)
if isnumeric(x)
    x = double(x);
else
    x = str2double(string(x));
end
end
