function data = fbg_prepare_data(T, fold, modelName, cfg)
%FBG_PREPARE_DATA Build asphalt data with leakage-safe training anchors.
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

normalizer.maxLoadKN = maxLoadKN;
normalizer.trainTypes = trainTypes;
normalizer.numLoadTypes = numel(trainTypes);
normalizer.strainScaleA = safeStrainScale(T.strain_ue(trainMask));
normalizer.lengthCm = cfg.specimenLengthCm;
normalizer.widthCm = cfg.specimenWidthCm;

referenceCases = fold.trainCases;
data.train = makeSubset(T, find(trainMask), referenceCases, ...
    modelName, normalizer, cfg);
data.val = makeSubset(T, find(valMask), referenceCases, ...
    modelName, normalizer, cfg);
data.test = makeSubset(T, find(testMask), referenceCases, ...
    modelName, normalizer, cfg);
data.normalizer = normalizer;
end

function scale = safeStrainScale(values)
values = double(values(:));
scale = std(values);
if ~isfinite(scale) || scale < 1
    scale = max(sqrt(mean(values .^ 2)), 1);
end
end

function S = makeSubset(T, idx, referenceCases, modelName, N, cfg)
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

[anchorStrain, anchorDeltaX, anchorDeltaY, anchorDeltaLoad, ...
    anchorDistance, anchorAvailable, anchorCaseId, ...
    anchorSensorDistance] = buildAnchors(T, Q, referenceCases, N, cfg);

useAnchor = modelName == "NearestAnchor-Persistence" || ...
    startsWith(modelName, "AnchoredResidual");
if useAnchor
    networkAnchorStrain = anchorStrain;
    networkAnchorDeltaX = anchorDeltaX;
    networkAnchorDeltaY = anchorDeltaY;
    networkAnchorDeltaLoad = anchorDeltaLoad;
    networkAnchorDistance = anchorDistance;
    networkAnchorAvailable = anchorAvailable;
else
    networkAnchorStrain = zeros(size(anchorStrain));
    networkAnchorDeltaX = zeros(size(anchorDeltaX));
    networkAnchorDeltaY = zeros(size(anchorDeltaY));
    networkAnchorDeltaLoad = zeros(size(anchorDeltaLoad));
    networkAnchorDistance = zeros(size(anchorDistance));
    networkAnchorAvailable = zeros(size(anchorAvailable));
end

S.X = [xN; yN; p1xN; p1yN; p2xN; p2yN; loadN; hasP2; ...
    directionLong; layerSoil; networkAnchorStrain; ...
    networkAnchorDeltaX; networkAnchorDeltaY; ...
    networkAnchorDeltaLoad; networkAnchorDistance; ...
    networkAnchorAvailable; onehot];
S.Y = Q.strain_ue';
S.indices = idx;
S.table = Q;
S.AnchorStrainUe = anchorStrain;
S.AnchorDeltaXNorm = anchorDeltaX;
S.AnchorDeltaYNorm = anchorDeltaY;
S.AnchorDeltaLoadNorm = anchorDeltaLoad;
S.AnchorDistance = anchorDistance;
S.AnchorAvailable = anchorAvailable;
S.AnchorCaseId = anchorCaseId;
S.AnchorSensorDistanceNorm = anchorSensorDistance;
end

function [anchorStrain, deltaX, deltaY, deltaLoad, distance, ...
    available, anchorCaseId, sensorDistance] = buildAnchors( ...
    T, Q, referenceCases, N, cfg)
n = height(Q);
anchorStrain = zeros(1, n);
deltaX = zeros(1, n);
deltaY = zeros(1, n);
deltaLoad = zeros(1, n);
distance = zeros(1, n);
available = zeros(1, n);
anchorCaseId = repmat("", n, 1);
sensorDistance = nan(1, n);

queryCases = unique(Q.case_id, "stable");
for i = 1:numel(queryCases)
    queryCase = queryCases(i);
    queryRows = find(Q.case_id == queryCase);
    queryMeta = Q(queryRows(1), :);
    candidateCases = referenceCases(referenceCases ~= queryCase);
    if isempty(candidateCases)
        continue;
    end

    candidateFirstRows = zeros(numel(candidateCases), 1);
    for j = 1:numel(candidateCases)
        candidateFirstRows(j) = find(T.case_id == candidateCases(j), 1);
    end
    candidateMeta = T(candidateFirstRows, :);
    compatible = candidateMeta.load_type == queryMeta.load_type & ...
        candidateMeta.has_load2 == queryMeta.has_load2;
    if any(compatible)
        candidateCases = candidateCases(compatible);
        candidateMeta = candidateMeta(compatible, :);
    end

    dx = (queryMeta.load_x_cm - candidateMeta.load_x_cm) / N.lengthCm;
    dy = (queryMeta.load_y_cm - candidateMeta.load_y_cm) / N.widthCm;
    dp2x = (queryMeta.load2_x_cm - candidateMeta.load2_x_cm) / ...
        N.lengthCm;
    dp2y = (queryMeta.load2_y_cm - candidateMeta.load2_y_cm) / ...
        N.widthCm;
    dload = (queryMeta.load_kN - candidateMeta.load_kN) / N.maxLoadKN;
    secondPoint = double(queryMeta.has_load2);
    score = sqrt(dx .^ 2 + dy .^ 2 + ...
        secondPoint .* (dp2x .^ 2 + dp2y .^ 2) + ...
        (cfg.anchorLoadDistanceWeight .* dload) .^ 2);
    [caseDistance, best] = min(score);
    selectedCase = candidateCases(best);
    selectedMeta = candidateMeta(best, :);

    caseDeltaX = (queryMeta.load_x_cm - selectedMeta.load_x_cm) / ...
        N.lengthCm;
    caseDeltaY = (queryMeta.load_y_cm - selectedMeta.load_y_cm) / ...
        N.widthCm;
    caseDeltaLoad = (queryMeta.load_kN - selectedMeta.load_kN) / ...
        N.maxLoadKN;
    anchorRows = T(T.case_id == selectedCase & T.layer == "asphalt", :);

    for rowIndex = queryRows'
        candidates = find(anchorRows.direction == Q.direction(rowIndex));
        if isempty(candidates)
            continue;
        end
        pointDistance = hypot( ...
            (anchorRows.sensor_x_cm(candidates) - ...
                Q.sensor_x_cm(rowIndex)) / N.lengthCm, ...
            (anchorRows.sensor_y_cm(candidates) - ...
                Q.sensor_y_cm(rowIndex)) / N.widthCm);
        [minimumDistance, nearest] = min(pointDistance);
        if minimumDistance > cfg.anchorMaxSensorDistanceNorm
            continue;
        end
        selectedRow = candidates(nearest);
        anchorStrain(rowIndex) = anchorRows.strain_ue(selectedRow);
        deltaX(rowIndex) = caseDeltaX;
        deltaY(rowIndex) = caseDeltaY;
        deltaLoad(rowIndex) = caseDeltaLoad;
        distance(rowIndex) = caseDistance;
        available(rowIndex) = 1;
        anchorCaseId(rowIndex) = selectedCase;
        sensorDistance(rowIndex) = minimumDistance;
    end
end
end
