function data = fbg_prepare_data(T, fold, ~, cfg)
%FBG_PREPARE_DATA Build asphalt field data and case-level soil summaries.
usable = T.layer == "asphalt";
trainMask = usable & ismember(T.case_id, fold.trainCases);
valMask = usable & ismember(T.case_id, fold.valCases);
testMask = usable & ismember(T.case_id, fold.testCases);
if ~any(trainMask) || ~any(testMask)
    error("FBG:Split", "Empty asphalt training or test split.");
end

maxLoadKN = max(abs(T.load_kN(trainMask)));
if ~isfinite(maxLoadKN) || maxLoadKN < eps
    maxLoadKN = 1;
end
trainTypes = unique(T.load_type(trainMask), "stable");
if isempty(trainTypes)
    trainTypes = "unknown";
end

soilSummary = buildSoilSummary(T);
trainSummary = soilSummary(ismember(soilSummary.case_id, ...
    fold.trainCases), :);
[soilMeanX, soilScaleX] = safeLocationScale(trainSummary.mean_x_ue);
[soilMeanY, soilScaleY] = safeLocationScale(trainSummary.mean_y_ue);

normalizer.maxLoadKN = maxLoadKN;
normalizer.trainTypes = trainTypes;
normalizer.numLoadTypes = numel(trainTypes);
normalizer.strainScaleA = safeStrainScale(T.strain_ue(trainMask));
normalizer.lengthCm = cfg.specimenLengthCm;
normalizer.widthCm = cfg.specimenWidthCm;
normalizer.soilMeanX = soilMeanX;
normalizer.soilScaleX = soilScaleX;
normalizer.soilMeanY = soilMeanY;
normalizer.soilScaleY = soilScaleY;

data.train = makeSubset(T, find(trainMask), normalizer, soilSummary);
data.val = makeSubset(T, find(valMask), normalizer, soilSummary);
data.test = makeSubset(T, find(testMask), normalizer, soilSummary);
data.normalizer = normalizer;
end

function scale = safeStrainScale(values)
values = double(values(:));
scale = std(values);
if ~isfinite(scale) || scale < 1
    scale = max(sqrt(mean(values .^ 2)), 1);
end
end

function summary = buildSoilSummary(T)
cases = unique(T.case_id, "stable");
case_id = cases(:);
mean_x_ue = nan(numel(cases), 1);
mean_y_ue = nan(numel(cases), 1);
for i = 1:numel(cases)
    soil = T(T.case_id == cases(i) & T.layer == "soil", :);
    valuesX = soil.strain_ue(soil.direction == "x");
    valuesY = soil.strain_ue(soil.direction == "y");
    if ~isempty(valuesX)
        mean_x_ue(i) = mean(valuesX);
    end
    if ~isempty(valuesY)
        mean_y_ue(i) = mean(valuesY);
    end
end
summary = table(case_id, mean_x_ue, mean_y_ue);
end

function [location, scale] = safeLocationScale(values)
values = double(values(:));
values = values(isfinite(values));
if isempty(values)
    location = 0;
    scale = 1;
    return;
end
location = mean(values);
scale = std(values);
if ~isfinite(scale) || scale < 1
    scale = max(sqrt(mean((values - location) .^ 2)), 1);
end
end

function S = makeSubset(T, idx, N, soilSummary)
idx = idx(:);
Q = T(idx, :);
xN = Q.sensor_x_cm' / N.lengthCm;
yN = Q.sensor_y_cm' / N.widthCm;
p1xN = Q.load_x_cm' / N.lengthCm;
p1yN = Q.load_y_cm' / N.widthCm;
p2xN = Q.load2_x_cm' / N.lengthCm;
p2yN = Q.load2_y_cm' / N.widthCm;
loadN = Q.load_kN' / N.maxLoadKN;
hasP2 = double(Q.has_load2');
directionLong = double(Q.direction' == "x");
layerSoil = zeros(1, height(Q));

onehot = zeros(N.numLoadTypes, height(Q));
for i = 1:N.numLoadTypes
    onehot(i, :) = double(Q.load_type == N.trainTypes(i))';
end

[found, location] = ismember(Q.case_id, soilSummary.case_id);
if any(~found)
    error("FBG:SoilSummary", ...
        "Missing soil summary for %d asphalt rows.", sum(~found));
end
soilX = soilSummary.mean_x_ue(location)';
soilY = soilSummary.mean_y_ue(location)';
auxX = (soilX - N.soilMeanX) / N.soilScaleX;
auxY = (soilY - N.soilMeanY) / N.soilScaleY;

S.X = [xN; yN; p1xN; p1yN; p2xN; p2yN; loadN; hasP2; ...
    directionLong; layerSoil; onehot];
S.Y = Q.strain_ue';
S.AuxY = [auxX; auxY];
S.AuxPhysical = [soilX; soilY];
S.indices = idx;
S.table = Q;
end
