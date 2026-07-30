function [loss, gradients, components] = fbg_loss_gradients( ...
    params, X, XPlus, XMinus, Y, modelName, cfg, normalizer)
%FBG_LOSS_GRADIENTS Main asphalt loss and conservative correction penalty.
result = fbg_model_forward(params, X, modelName, cfg);
mainResidual = (result.prediction - Y) / normalizer.strainScaleA;
lossMain = mean(mainResidual .^ 2, "all");

lossCorrection = 0 * lossMain;
if startsWith(modelName, "AnchoredResidual")
    normalizedCorrection = result.correction / normalizer.strainScaleA;
    lossCorrection = mean(normalizedCorrection .^ 2, "all");
end

lossPositionSmooth = 0 * lossMain;
if contains(modelName, "LowFreqSmooth")
    resultPlus = fbg_model_forward(params, XPlus, modelName, cfg);
    resultMinus = fbg_model_forward(params, XMinus, modelName, cfg);
    normalizedPlusChange = ...
        (resultPlus.prediction - result.prediction) / ...
        normalizer.strainScaleA;
    normalizedMinusChange = ...
        (resultMinus.prediction - result.prediction) / ...
        normalizer.strainScaleA;
    lossPositionSmooth = 0.5 * ( ...
        mean(normalizedPlusChange .^ 2, "all") + ...
        mean(normalizedMinusChange .^ 2, "all"));
end

regularization = mean(params.outBaseW .^ 2, "all");
loss = lossMain + cfg.anchorCorrectionPenalty * lossCorrection + ...
    cfg.positionSmoothWeight * lossPositionSmooth + ...
    cfg.parameterL2 * regularization;
gradients = dlgradient(loss, params);
components = [lossMain; lossCorrection; lossPositionSmooth];
end
