function result = fbg_model_forward(params, X, modelName, cfg)
%FBG_MODEL_FORWARD Absolute Step19 or conservative anchor residual field.
% X rows: x, y, p1 x/y, p2 x/y, load, hasP2, longitudinal,
% soil indicator, anchor values/deltas, and load-type one-hot codes.
x = X(1, :);
y = X(2, :);
p1x = X(3, :);
p1y = X(4, :);
p2x = X(5, :);
p2y = X(6, :);
loadN = X(7, :);
hasP2 = X(8, :);
directionLong = X(9, :);
anchorStrainUe = X(11, :);
anchorDeltaX = X(12, :);
anchorDeltaY = X(13, :);
anchorDeltaLoad = X(14, :);
anchorDistance = X(15, :);
anchorAvailable = X(16, :);
onehotLoadType = X(17:end, :);

space = spaceFeatures(x, y, p1x, p1y, p2x, p2y, hasP2, cfg);
conditionFeatures = conditionInputs(x, y, p1x, p1y, p2x, p2y, ...
    loadN, hasP2, onehotLoadType, anchorStrainUe, anchorDeltaX, ...
    anchorDeltaY, anchorDeltaLoad, anchorDistance, anchorAvailable, ...
    modelName, cfg);
condition = encodeCondition(params, conditionFeatures);

h = space;
for i = 1:cfg.depth
    h = params.space(i).W * h + params.space(i).B;
    gamma = params.film(i).gammaW * condition + ...
        params.film(i).gammaB;
    beta = params.film(i).betaW * condition + params.film(i).betaB;
    gamma = 1 + 0.25 * tanh(gamma - 1);
    h = tanh(gamma .* h + beta);
end

wBase = params.outBaseW * h + params.outBaseB;
wProxy = wBase;

wM = cfg.wScaleM * wProxy;
firstDerivative = dlgradient(sum(wM, "all"), X, ...
    "EnableHigherDerivatives", true, "RetainData", true);
wxN = firstDerivative(1, :);
wyN = firstDerivative(2, :);
secondX = dlgradient(sum(wxN, "all"), X, ...
    "EnableHigherDerivatives", true, "RetainData", true);
secondY = dlgradient(sum(wyN, "all"), X, ...
    "EnableHigherDerivatives", true);

lengthM = cfg.specimenLengthCm / 100;
widthM = cfg.specimenWidthCm / 100;
wxxM = secondX(1, :) / (lengthM ^ 2);
wyyM = secondY(2, :) / (widthM ^ 2);
epsX = -cfg.effectiveDepthM * wxxM * 1e6;
epsY = -cfg.effectiveDepthM * wyyM * 1e6;
fieldPrediction = directionLong .* epsX + (1 - directionLong) .* epsY;

if modelName == "AnchoredResidual-FiLM" || ...
        modelName == "AnchoredResidual-XOnly-FiLM"
    anchorGate = anchorAvailable .* ...
        (1 - exp(-anchorDistance ./ cfg.anchorCorrectionLengthNorm));
    if modelName == "AnchoredResidual-XOnly-FiLM"
        anchorGate = anchorGate .* directionLong;
    end
    correction = anchorGate .* fieldPrediction;
    prediction = anchorAvailable .* anchorStrainUe + correction + ...
        (1 - anchorAvailable) .* fieldPrediction;
else
    anchorGate = 0 * fieldPrediction;
    correction = 0 * fieldPrediction;
    prediction = fieldPrediction;
end

result.prediction = prediction;
result.fieldPrediction = fieldPrediction;
result.correction = correction;
result.anchorGate = anchorGate;
result.epsX = epsX;
result.epsY = epsY;
result.wM = wM;
result.wBase = wBase;
end

function condition = encodeCondition(params, features)
condition = tanh(params.condition1W * features + params.condition1B);
condition = tanh(params.condition2W * condition + params.condition2B);
end

function features = spaceFeatures(x, y, p1x, p1y, p2x, p2y, hasP2, cfg)
dx1 = x - p1x;
dy1 = y - p1y;
dx2 = x - p2x;
dy2 = y - p2y;
d1 = sqrt(dx1 .^ 2 + dy1 .^ 2 + 1e-8);
d2 = sqrt(dx2 .^ 2 + dy2 .^ 2 + 1e-8);

tau = cfg.boundarySoftMinTau;
edgeSoft = -tau .* log(exp(-x ./ tau) + exp(-(1 - x) ./ tau) + ...
    exp(-y ./ tau) + exp(-(1 - y) ./ tau));
edgeExp = exp(-edgeSoft ./ cfg.boundaryFeatureLengthNorm);

features = [x; y; dx1; dy1; dx2; dy2; d1; d2; ...
    x; 1 - x; y; 1 - y; edgeSoft; edgeExp; hasP2];
for frequency = cfg.fourierFreqsXY
    features = [features; sin(2 * pi * frequency * x); ...
        cos(2 * pi * frequency * x); ...
        sin(2 * pi * frequency * y); ...
        cos(2 * pi * frequency * y)]; %#ok<AGROW>
end
for frequency = cfg.fourierFreqsRelative
    features = [features; ...
        sin(2 * pi * frequency * dx1); cos(2 * pi * frequency * dx1); ...
        sin(2 * pi * frequency * dy1); cos(2 * pi * frequency * dy1); ...
        sin(2 * pi * frequency * dx2); cos(2 * pi * frequency * dx2); ...
        sin(2 * pi * frequency * dy2); cos(2 * pi * frequency * dy2)]; %#ok<AGROW>
end
end

function features = conditionInputs(x, y, p1x, p1y, p2x, p2y, ...
    loadN, hasP2, onehotLoadType, anchorStrainUe, anchorDeltaX, ...
    anchorDeltaY, anchorDeltaLoad, anchorDistance, anchorAvailable, ...
    modelName, cfg)
tau = cfg.boundarySoftMinTau;
p1Edge = -tau .* log(exp(-p1x ./ tau) + exp(-(1 - p1x) ./ tau) + ...
    exp(-p1y ./ tau) + exp(-(1 - p1y) ./ tau));
p2EdgeRaw = -tau .* log(exp(-p2x ./ tau) + ...
    exp(-(1 - p2x) ./ tau) + exp(-p2y ./ tau) + ...
    exp(-(1 - p2y) ./ tau));
p2Edge = p2EdgeRaw .* hasP2;
smoothMin = 0.5 .* (p1Edge + p2EdgeRaw - ...
    sqrt((p1Edge - p2EdgeRaw) .^ 2 + 1e-6));
loadEdge = (1 - hasP2) .* p1Edge + hasP2 .* smoothMin;
p1EdgeExp = exp(-p1Edge ./ cfg.boundaryFeatureLengthNorm);
loadEdgeExp = exp(-loadEdge ./ cfg.boundaryFeatureLengthNorm);

features = [loadN; p1x; p1y; p2x; p2y; hasP2; ...
    p1x; 1 - p1x; p1y; 1 - p1y; ...
    p2x; 1 - p2x; p2y; 1 - p2y; ...
    p1Edge; p2Edge; loadEdge; p1EdgeExp; loadEdgeExp; ...
    2 * p1x - 1; 2 * p1y - 1; ...
    loadN .* (x - p1x); loadN .* (y - p1y); ...
    loadN .* hasP2];
if modelName == "AnchoredResidual-FiLM" || ...
        modelName == "AnchoredResidual-XOnly-FiLM"
    features = [features; anchorStrainUe ./ cfg.anchorStrainScaleUe; ...
        anchorDeltaX; anchorDeltaY; anchorDeltaLoad; ...
        anchorDistance; anchorAvailable];
end
features = [features; onehotLoadType];
end
