function [loss, gradients, components] = fbg_loss_gradients( ...
    params, X, Y, modelName, cfg, normalizer)
%FBG_LOSS_GRADIENTS Asphalt field loss plus soil condition-summary loss.
result = fbg_model_forward(params, X, modelName, cfg);
strainTarget = Y(1, :);
mainResidual = (result.prediction - strainTarget) / ...
    normalizer.strainScaleA;
lossMain = mean(mainResidual .^ 2, "all");

lossAuxX = 0 * lossMain;
lossAuxY = 0 * lossMain;
regularization = mean(params.outBaseW .^ 2, "all");
if modelName == "SoilMeanAux-X-FiLM" || ...
        modelName == "SoilMeanAux-XY-FiLM"
    lossAuxX = mean((result.auxiliaryPrediction(1, :) - ...
        Y(2, :)) .^ 2, "all");
    regularization = regularization + ...
        mean(params.outSoilSummaryW .^ 2, "all");
    if modelName == "SoilMeanAux-XY-FiLM"
        lossAuxY = mean((result.auxiliaryPrediction(2, :) - ...
            Y(3, :)) .^ 2, "all");
        lossAuxiliary = 0.5 * (lossAuxX + lossAuxY);
    else
        lossAuxiliary = lossAuxX;
    end
    loss = lossMain + cfg.soilSummaryAuxWeight * lossAuxiliary + ...
        cfg.parameterL2 * regularization;
else
    loss = lossMain + cfg.parameterL2 * regularization;
end

gradients = dlgradient(loss, params);
components = [lossMain; lossAuxX; lossAuxY];
end
