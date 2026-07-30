function [loss, gradients, components] = fbg_loss_gradients( ...
    params, X, Y, modelName, cfg, normalizer)
%FBG_LOSS_GRADIENTS Main asphalt loss and conservative correction penalty.
result = fbg_model_forward(params, X, modelName, cfg);
mainResidual = (result.prediction - Y) / normalizer.strainScaleA;
lossMain = mean(mainResidual .^ 2, "all");

lossCorrection = 0 * lossMain;
if modelName == "AnchoredResidual-FiLM" || ...
        modelName == "AnchoredResidual-XOnly-FiLM"
    normalizedCorrection = result.correction / normalizer.strainScaleA;
    lossCorrection = mean(normalizedCorrection .^ 2, "all");
end

regularization = mean(params.outBaseW .^ 2, "all");
loss = lossMain + cfg.anchorCorrectionPenalty * lossCorrection + ...
    cfg.parameterL2 * regularization;
gradients = dlgradient(loss, params);
components = [lossMain; lossCorrection];
end
