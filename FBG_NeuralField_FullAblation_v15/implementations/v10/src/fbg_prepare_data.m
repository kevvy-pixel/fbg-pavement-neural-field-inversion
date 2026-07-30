function data = fbg_prepare_data(T, fold, modelName, cfg)
%FBG_PREPARE_DATA Normalize coordinates and targets using training data only.
isMulti = modelName == "Multilayer-Transfer-FiLM";
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

trainLoad = T.load_kN(trainMask);
muP = mean(trainLoad);
sdP = std(trainLoad);
if ~isfinite(sdP) || sdP < 1e-9
    sdP = 1;
end
trainTypes = unique(T.load_type(trainMask), "stable");
strainScale = std(T.strain_ue(trainMask & T.layer == "asphalt"));
if ~isfinite(strainScale) || strainScale < 1
    strainScale = max(std(T.strain_ue(trainMask)), 1);
end

normalizer.muP = muP;
normalizer.sdP = sdP;
normalizer.trainTypes = trainTypes;
normalizer.strainScale = strainScale;
normalizer.lengthCm = cfg.specimenLengthCm;
normalizer.widthCm = cfg.specimenWidthCm;

data.train = makeSubset(T, find(trainMask), normalizer);
data.val = makeSubset(T, find(valMask), normalizer);
data.test = makeSubset(T, find(testMask), normalizer);
data.normalizer = normalizer;
end

function S = makeSubset(T, idx, N)
idx = idx(:);
Q = T(idx, :);
xn = 2 * Q.sensor_x_cm' / N.lengthCm - 1;
yn = 2 * Q.sensor_y_cm' / N.widthCm - 1;
lxn = 2 * Q.load_x_cm' / N.lengthCm - 1;
lyn = 2 * Q.load_y_cm' / N.widthCm - 1;
pn = (Q.load_kN' - N.muP) / N.sdP;

typeCode = zeros(1, height(Q));
for i = 1:numel(N.trainTypes)
    typeCode(Q.load_type == N.trainTypes(i)) = i - 1;
end
if numel(N.trainTypes) > 1
    typeCode = 2 * typeCode / (numel(N.trainTypes) - 1) - 1;
end
dirCode = double(Q.direction' == "y");
layerCode = double(Q.layer' == "soil");

S.X = [xn; yn; lxn; lyn; pn; typeCode; dirCode; layerCode];
S.Y = Q.strain_ue' / N.strainScale;
S.indices = idx;
S.table = Q;
end
