function params = fbg_model_init(modelName, cfg, numLoadTypes)
%FBG_MODEL_INIT Initialize Step19 trunk and optional soil-summary head.
spaceDim = 15 + 4 * numel(cfg.fourierFreqsXY) + ...
    8 * numel(cfg.fourierFreqsRelative);
conditionDim = 24 + numLoadTypes;
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

if modelName == "SoilMeanAux-X-FiLM"
    params.outSoilSummaryW = cfg.outputInitScale * xavier(1, ch);
    params.outSoilSummaryB = zerosdl(1, 1);
elseif modelName == "SoilMeanAux-XY-FiLM"
    params.outSoilSummaryW = cfg.outputInitScale * xavier(2, ch);
    params.outSoilSummaryB = zerosdl(2, 1);
end
end

function value = xavier(rows, cols)
limit = sqrt(6 / (rows + cols));
value = dlarray(single((2 * rand(rows, cols) - 1) * limit));
end

function value = zerosdl(rows, cols)
value = dlarray(zeros(rows, cols, "single"));
end
