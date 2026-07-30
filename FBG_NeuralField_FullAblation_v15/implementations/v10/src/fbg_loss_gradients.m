function [loss, gradients, components] = fbg_loss_gradients( ...
    params, X, Y, modelName, cfg)
%FBG_LOSS_GRADIENTS Data, soil, transfer, and regularization losses.
[epsA, epsS, ~, ~, eta] = fbg_model_forward(params, X, modelName, cfg);
layerSoil = X(8, :);
layerAsphalt = 1 - layerSoil;

countA = max(sum(layerAsphalt, "all"), 1);
lossA = sum(((epsA - Y).^2) .* layerAsphalt, "all") / countA;

if modelName == "Multilayer-Transfer-FiLM"
    countS = max(sum(layerSoil, "all"), 1);
    lossS = sum(((epsS - Y).^2) .* layerSoil, "all") / countS;
    transferResidual = abs(epsS) - eta .* abs(epsA);
    lossTransfer = mean(transferResidual.^2, "all");
    regularization = mean(params.outAW.^2, "all") + ...
        mean(params.outSW.^2, "all");
    loss = lossA + cfg.soilLossWeight * lossS + ...
        cfg.transferLossWeight * lossTransfer + ...
        cfg.parameterL2 * regularization;
else
    lossS = 0 * lossA;
    lossTransfer = 0 * lossA;
    regularization = mean(params.outAW.^2, "all");
    loss = lossA + cfg.parameterL2 * regularization;
end

gradients = dlgradient(loss, params);
components = [lossA; lossS; lossTransfer];
end
