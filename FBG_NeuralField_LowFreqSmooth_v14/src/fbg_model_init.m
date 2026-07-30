function params = fbg_model_init(modelName, cfg, numLoadTypes)
%FBG_MODEL_INIT Initialize the absolute or anchor-residual neural field.
spaceDim = 15 + 4 * numel(cfg.fourierFreqsXY) + ...
    8 * numel(cfg.fourierFreqsRelative);
conditionDim = 24 + numLoadTypes;
if startsWith(modelName, "AnchoredResidual")
    conditionDim = conditionDim + 6;
end
h = cfg.hiddenWidth;
ch = cfg.conditionHiddenWidth;

params.condition1W = xavier(ch, conditionDim);
params.condition1B = zerosdl(ch, 1);
params.condition2W = xavier(ch, ch);
params.condition2B = zerosdl(ch, 1);

for i = 1:cfg.depth
    if i == 1
        params.space(i).W = xavier(h, spaceDim);
    else
        params.space(i).W = xavier(h, h);
    end
    params.space(i).B = zerosdl(h, 1);
    params.film(i).gammaW = xavier(h, ch);
    params.film(i).gammaB = dlarray(ones(h, 1, "single"));
    params.film(i).betaW = xavier(h, ch);
    params.film(i).betaB = zerosdl(h, 1);
end

params.outBaseW = cfg.outputInitScale * xavier(1, h);
params.outBaseB = zerosdl(1, 1);
end

function value = xavier(rows, cols)
limit = sqrt(6 / (rows + cols));
value = dlarray(single((2 * rand(rows, cols) - 1) * limit));
end

function value = zerosdl(rows, cols)
value = dlarray(zeros(rows, cols, "single"));
end
