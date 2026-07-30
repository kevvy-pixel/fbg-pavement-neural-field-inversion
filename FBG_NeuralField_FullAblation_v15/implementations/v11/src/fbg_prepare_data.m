function data = fbg_prepare_data(T, fold, modelName, cfg)
%FBG_PREPARE_DATA Build Step19-style features using training statistics only.
isMulti = modelName == "SharedMultilayer-FiLM";
if isMulti
    usable = true(height(T), 1);
else
    usable = T.layer == "asphalt";
end

trainMask = usable & ismember(T.case_id, fold.trainCases);
valMask = usable & ismember(T.case_id, fold.valCases);
testMask = usable & ismember(T.case_id, fold.testCases);
if ~any(trainMask) || ~any(testMask)
    error("FBG:Split", "Empty training or test split for %s.", modelName);
end

maxLoadKN = max(abs(T.load_kN(trainMask)));
if ~isfinite(maxLoadKN) || maxLoadKN < eps
    maxLoadKN = 1;
end
trainTypes = unique(T.load_type(trainMask), "stable");
if isempty(trainTypes)
    trainTypes = "unknown";
end

strainScaleA = safeScale(T.strain_ue(trainMask & T.layer == "asphalt"));
strainScaleS = safeScale(T.strain_ue(trainMask & T.layer == "soil"));
if ~isfinite(strainScaleS)
    strainScaleS = strainScaleA;
end

normalizer.maxLoadKN = maxLoadKN;
normalizer.trainTypes = trainTypes;
normalizer.numLoadTypes = numel(trainTypes);
normalizer.strainScaleA = strainScaleA;
normalizer.strainScaleS = strainScaleS;
normalizer.lengthCm = cfg.specimenLengthCm;
normalizer.widthCm = cfg.specimenWidthCm;

data.train = makeSubset(T, find(trainMask), normalizer);
data.val = makeSubset(T, find(valMask), normalizer);
data.test = makeSubset(T, find(testMask), normalizer);
data.normalizer = normalizer;
end

function scale = safeScale(values)
values = double(values(:));
if isempty(values)
    scale = NaN;
    return;
end
scale = std(values);
if ~isfinite(scale) || scale < 1
    scale = max(sqrt(mean(values.^2)), 1);
end
end

function S = makeSubset(T, idx, N)
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
layerSoil = double(Q.layer' == "soil");

onehot = zeros(N.numLoadTypes, height(Q));
for i = 1:N.numLoadTypes
    onehot(i, :) = double(Q.load_type == N.trainTypes(i))';
end

% Rows: x, y, p1 x/y, p2 x/y, load, has p2, longitudinal,
% soil indicator, followed by load-type one-hot codes.
S.X = [xN; yN; p1xN; p1yN; p2xN; p2yN; loadN; hasP2; ...
    directionLong; layerSoil; onehot];
S.Y = Q.strain_ue';
S.indices = idx;
S.table = Q;
end
