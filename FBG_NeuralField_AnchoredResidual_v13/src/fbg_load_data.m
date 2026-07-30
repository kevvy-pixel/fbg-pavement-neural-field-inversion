function T = fbg_load_data(csvPath)
%FBG_LOAD_DATA Read a response matrix and normalize common schema variants.
if ~isfile(csvPath)
    error("FBG:Input", "Input table does not exist: %s", csvPath);
end
raw = readtable(csvPath, 'VariableNamingRule', 'preserve', ...
    'TextType', 'string');
if isempty(raw)
    error("FBG:Input", "Input table is empty: %s", csvPath);
end

[caseRaw, ~] = pick(raw, ["case_id", "caseid", "case", "loadcase", ...
    "condition_id", "condition"], false);
[sensorXRaw, ~] = pick(raw, ["sensor_x_cm", "sensorxcm", ...
    "sensor_x", "sensorx", "x_cm", "x"]);
[sensorYRaw, ~] = pick(raw, ["sensor_y_cm", "sensorycm", ...
    "sensor_y", "sensory", "y_cm", "y"]);
[loadXRaw, ~] = pick(raw, ["load_x_cm", "loadxcm", "load_x", ...
    "loadx", "px_cm", "px", "loading_x_cm", "p1_x", "p1x"]);
[loadYRaw, ~] = pick(raw, ["load_y_cm", "loadycm", "load_y", ...
    "loady", "py_cm", "py", "loading_y_cm", "p1_y", "p1y"]);
[load2XRaw, ~] = pick(raw, ["load2_x_cm", "load2xcm", "load2_x", ...
    "load2x", "p2_x", "p2x"], false);
[load2YRaw, ~] = pick(raw, ["load2_y_cm", "load2ycm", "load2_y", ...
    "load2y", "p2_y", "p2y"], false);
[loadRaw, loadSource] = pick(raw, ["load_kn", "loadkn", "p_kn", ...
    "load_level", "loadlevel", "load_magnitude", "loadmagnitude", ...
    "load", "load_kg", "loadkg", "mass_kg", "masskg", ...
    "p_total_n", "ptotaln", "force_n", "forcen", "load_n", "p"]);
[directionRaw, ~] = pick(raw, ["direction", "strain_direction", ...
    "axis", "orientation", "sensor_direction"]);
[strainRaw, strainSource] = pick(raw, ["strain_ue", ...
    "measured_strain_ue", "measuredstrainue", "measured_strain", ...
    "strain", "response", "epsilon", "strain_microstrain", ...
    "delta_lambda_resp_nm", "deltalambdarespnm", "shift_nm"]);

sensor_x_cm = numericColumn(sensorXRaw);
sensor_y_cm = numericColumn(sensorYRaw);
load_x_cm = numericColumn(loadXRaw);
load_y_cm = numericColumn(loadYRaw);
direction = normalizeDirection(directionRaw);

% Preserve both load points. Missing p2 coordinates are replaced by p1 only
% for feature construction, while has_load2 keeps the topology explicit.
if ~isempty(load2XRaw) && ~isempty(load2YRaw)
    load2_x_cm = numericColumn(load2XRaw);
    load2_y_cm = numericColumn(load2YRaw);
    has_load2 = isfinite(load2_x_cm) & isfinite(load2_y_cm);
    load2_x_cm(~has_load2) = load_x_cm(~has_load2);
    load2_y_cm(~has_load2) = load_y_cm(~has_load2);
else
    load2_x_cm = load_x_cm;
    load2_y_cm = load_y_cm;
    has_load2 = false(size(load_x_cm));
end

load_kN = numericColumn(loadRaw);
if contains(loadSource, "kg")
    kgForceToKN = 9.80665e-3;
    load_kN = load_kN * kgForceToKN;
elseif (endsWith(loadSource, "n") || contains(loadSource, "forcen") || ...
        contains(loadSource, "totaln")) && ~contains(loadSource, "kn")
    load_kN = load_kN / 1000;
end

strain_ue = numericColumn(strainRaw);
if contains(strainSource, "nm") || contains(strainSource, "deltalambda")
    % Same wavelength-to-strain sensitivity as the retained Step19 model.
    wavelengthSensitivityNmPerUe = 1.2e-3;
    strain_ue = strain_ue / wavelengthSensitivityNmPerUe;
end

[layerRaw, ~] = pick(raw, ["layer", "sensor_layer", "sensorlayer", ...
    "material", "depth_layer"], false);
if isempty(layerRaw)
    layer = repmat("asphalt", height(raw), 1);
    warning("FBG:Layer", "No layer column found; all rows are treated as asphalt.");
else
    layer = normalizeLayer(layerRaw);
end

[loadTypeRaw, ~] = pick(raw, ["load_type", "loadtype", "type", ...
    "loading_type"], false);
if isempty(loadTypeRaw)
    load_type = repmat("single", height(raw), 1);
else
    load_type = string(loadTypeRaw);
end

if isempty(caseRaw)
    case_id = compose("case_x%.3g_y%.3g_p%.3g_%s", ...
        load_x_cm, load_y_cm, load_kN, load_type);
else
    case_id = string(caseRaw);
    missingCase = ismissing(case_id) | strlength(case_id) == 0;
    if any(missingCase)
        generated = compose("case_x%.3g_y%.3g_p%.3g_%s", ...
            load_x_cm, load_y_cm, load_kN, load_type);
        case_id(missingCase) = generated(missingCase);
    end
end

T = table(case_id(:), sensor_x_cm(:), sensor_y_cm(:), load_x_cm(:), ...
    load_y_cm(:), load2_x_cm(:), load2_y_cm(:), has_load2(:), ...
    load_kN(:), direction(:), layer(:), strain_ue(:), load_type(:), ...
    'VariableNames', {'case_id', 'sensor_x_cm', 'sensor_y_cm', ...
    'load_x_cm', 'load_y_cm', 'load2_x_cm', 'load2_y_cm', ...
    'has_load2', 'load_kN', 'direction', 'layer', 'strain_ue', ...
    'load_type'});

valid = isfinite(T.sensor_x_cm) & isfinite(T.sensor_y_cm) & ...
    isfinite(T.load_x_cm) & isfinite(T.load_y_cm) & ...
    isfinite(T.load2_x_cm) & isfinite(T.load2_y_cm) & ...
    isfinite(T.load_kN) & isfinite(T.strain_ue) & ...
    ~ismissing(T.case_id) & (T.direction == "x" | T.direction == "y") & ...
    (T.layer == "asphalt" | T.layer == "soil");
if any(~valid)
    warning("FBG:Rows", "Removed %d rows with missing or invalid required values.", ...
        sum(~valid));
    T = T(valid, :);
end
if isempty(T)
    error("FBG:Input", "No valid rows remained after schema normalization.");
end
T.row_id = (1:height(T))';
end

function [value, source] = pick(T, aliases, required)
if nargin < 3
    required = true;
end
names = string(T.Properties.VariableNames);
keys = lower(regexprep(names, "[^A-Za-z0-9]", ""));
aliasKeys = lower(regexprep(string(aliases), "[^A-Za-z0-9]", ""));
idx = [];
for i = 1:numel(aliasKeys)
    hit = find(keys == aliasKeys(i), 1);
    if ~isempty(hit)
        idx = hit;
        break;
    end
end
if isempty(idx)
    if required
        error("FBG:Schema", "Missing required column. Accepted aliases: %s", ...
            strjoin(aliases, ", "));
    end
    value = [];
    source = "";
else
    value = T.(names(idx));
    source = keys(idx);
end
end

function value = numericColumn(x)
if isnumeric(x) || islogical(x)
    value = double(x);
else
    value = str2double(string(x));
end
end

function d = normalizeDirection(x)
s = lower(strtrim(string(x)));
d = strings(size(s));
isX = s == "x" | s == "1" | contains(s, "long") | ...
    contains(s, "纵") | contains(s, "轴");
isY = s == "y" | s == "2" | contains(s, "trans") | ...
    contains(s, "横");
d(isX) = "x";
d(isY) = "y";
end

function layer = normalizeLayer(x)
s = lower(strtrim(string(x)));
layer = strings(size(s));
isAsphalt = contains(s, "asphalt") | contains(s, "surface") | ...
    contains(s, "upper") | contains(s, "沥青") | s == "a" | s == "1";
isSoil = contains(s, "soil") | contains(s, "base") | ...
    contains(s, "subgrade") | contains(s, "基层") | ...
    contains(s, "土") | s == "s" | s == "2";
layer(isAsphalt) = "asphalt";
layer(isSoil) = "soil";
unknown = strlength(layer) == 0;
if any(unknown)
    warning("FBG:Layer", "%d rows have unsupported layer labels; excluded.", ...
        sum(unknown));
end
end
