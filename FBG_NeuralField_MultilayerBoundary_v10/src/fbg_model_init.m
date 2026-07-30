function params = fbg_model_init(modelName, cfg)
%FBG_MODEL_INIT Initialize a compact smooth FiLM coordinate network.
if modelName == "Step19-SOTA-FiLM"
    spatialDim = 6;
else
    spatialDim = 14;
end
h = cfg.hiddenWidth;
conditionDim = 4;

params.spatialW = xavier(h, spatialDim);
params.spatialB = dlarray(zeros(h, 1, "single"));
params.conditionW = xavier(2 * h, conditionDim);
params.conditionB = dlarray(zeros(2 * h, 1, "single"));
params.hiddenW1 = xavier(h, h);
params.hiddenB1 = dlarray(zeros(h, 1, "single"));
params.hiddenW2 = xavier(h, h);
params.hiddenB2 = dlarray(zeros(h, 1, "single"));
params.hiddenW3 = xavier(h, h);
params.hiddenB3 = dlarray(zeros(h, 1, "single"));
params.outAW = xavier(1, h);
params.outAB = dlarray(zeros(1, 1, "single"));

if modelName == "Multilayer-Transfer-FiLM"
    params.outSW = 0.2 * xavier(1, h);
    params.outSB = dlarray(zeros(1, 1, "single"));
    params.etaW = 0.1 * xavier(1, conditionDim);
    etaMid = (0.15 - cfg.etaMin) / (cfg.etaMax - cfg.etaMin);
    etaMid = min(max(etaMid, 1e-3), 1 - 1e-3);
    params.etaB = dlarray(single(log(etaMid / (1 - etaMid))));
end
end

function value = xavier(rows, cols)
limit = sqrt(6 / (rows + cols));
value = dlarray(single((2 * rand(rows, cols) - 1) * limit));
end
