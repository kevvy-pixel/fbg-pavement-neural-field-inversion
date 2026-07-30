function [loss, gradients, components] = fbg_loss_gradients( ...
    params, X, Y, modelName, cfg, normalizer)
%FBG_LOSS_GRADIENTS Per-layer data losses with learned task uncertainty.
result = fbg_model_forward(params, X, modelName, cfg);
layerSoil = X(10, :);
layerAsphalt = 1 - layerSoil;

residualA = (result.prediction - Y) / normalizer.strainScaleA;
countA = max(sum(layerAsphalt, "all"), 1);
lossA = sum((residualA .^ 2) .* layerAsphalt, "all") / countA;

regularization = mean(params.outBaseW .^ 2, "all");
if modelName == "SharedMultilayer-FiLM"
    residualS = (result.prediction - Y) / normalizer.strainScaleS;
    countS = max(sum(layerSoil, "all"), 1);
    lossS = sum((residualS .^ 2) .* layerSoil, "all") / countS;
    weightA = exp(-params.taskLogVarianceA);
    weightS = exp(-params.taskLogVarianceS);
    regularization = regularization + ...
        mean(params.outSoilResidualW .^ 2, "all");
    loss = weightA * lossA + params.taskLogVarianceA + ...
        weightS * lossS + params.taskLogVarianceS + ...
        cfg.parameterL2 * regularization;
else
    lossS = 0 * lossA;
    weightA = 1 + 0 * lossA;
    weightS = 0 * lossA;
    loss = lossA + cfg.parameterL2 * regularization;
end

gradients = dlgradient(loss, params);
components = [lossA; lossS; weightA; weightS];
end
