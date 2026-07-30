function [epsA, epsS, wA, wS, eta] = fbg_model_forward(params, X, modelName, cfg)
%FBG_MODEL_FORWARD Evaluate fields and derive directional strain by AD.
% X rows: x, y, load x, load y, load magnitude, load type, direction, layer.
x = X(1, :);
y = X(2, :);
loadX = X(3, :);
loadY = X(4, :);
condition = X(3:6, :);

baseFeatures = [x; y; sin(pi * x); cos(pi * x); ...
    sin(pi * y); cos(pi * y)];
if modelName == "Step19-SOTA-FiLM"
    spatial = baseFeatures;
else
    dLeft = 0.5 * (x + 1);
    dRight = 0.5 * (1 - x);
    dBottom = 0.5 * (y + 1);
    dTop = 0.5 * (1 - y);
    dx = x - loadX;
    dy = y - loadY;
    spatial = [baseFeatures; dLeft; dRight; dBottom; dTop; ...
        dx; dy; abs(dx); abs(dy)];
end

h = tanh(params.spatialW * spatial + params.spatialB);
film = params.conditionW * condition + params.conditionB;
width = size(h, 1);
gamma = 1 + 0.1 * tanh(film(1:width, :));
beta = 0.1 * tanh(film(width + 1:end, :));
h = tanh(gamma .* h + beta);
h = tanh(params.hiddenW1 * h + params.hiddenB1);
h = tanh(params.hiddenW2 * h + params.hiddenB2);
h = tanh(params.hiddenW3 * h + params.hiddenB3);

wA = params.outAW * h + params.outAB;
directionY = X(7, :);
directionX = 1 - directionY;

gradA = dlgradient(sum(wA, "all"), X, ...
    "EnableHigherDerivatives", true, "RetainData", true);
directionalGradA = gradA(1, :) .* directionX + gradA(2, :) .* directionY;
curvatureA = dlgradient(sum(directionalGradA, "all"), X, ...
    "EnableHigherDerivatives", true, "RetainData", true);
epsA = -(curvatureA(1, :) .* directionX + ...
    curvatureA(2, :) .* directionY);

if modelName == "Multilayer-Transfer-FiLM"
    wS = params.outSW * h + params.outSB;
    gradS = dlgradient(sum(wS, "all"), X, ...
        "EnableHigherDerivatives", true, "RetainData", true);
    directionalGradS = gradS(1, :) .* directionX + ...
        gradS(2, :) .* directionY;
    curvatureS = dlgradient(sum(directionalGradS, "all"), X, ...
        "EnableHigherDerivatives", true);
    epsS = -(curvatureS(1, :) .* directionX + ...
        curvatureS(2, :) .* directionY);
    etaRaw = params.etaW * condition + params.etaB;
    eta = cfg.etaMin + (cfg.etaMax - cfg.etaMin) ./ (1 + exp(-etaRaw));
else
    wS = 0 * wA;
    epsS = 0 * epsA;
    eta = 0.15 + 0 * wA;
end
end
