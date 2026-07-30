function result = fbg_predict(params, X, modelName, cfg)
%FBG_PREDICT Evaluate predictions in bounded chunks.
n = size(X, 2);
prediction = zeros(n, 1);
epsAAll = zeros(n, 1);
epsSAll = zeros(n, 1);
etaAll = zeros(n, 1);
chunkSize = 512;
for first = 1:chunkSize:n
    last = min(first + chunkSize - 1, n);
    idx = first:last;
    dlX = dlarray(single(X(:, idx)));
    [epsA, epsS, ~, ~, eta] = dlfeval(@fbg_model_forward, ...
        params, dlX, modelName, cfg);
    epsA = double(gather(extractdata(epsA)))';
    epsS = double(gather(extractdata(epsS)))';
    eta = double(gather(extractdata(eta)))';
    layerSoil = X(8, idx)' > 0.5;
    combined = epsA;
    combined(layerSoil) = epsS(layerSoil);
    prediction(idx) = combined;
    epsAAll(idx) = epsA;
    epsSAll(idx) = epsS;
    etaAll(idx) = eta;
end
result.prediction = prediction;
result.epsA = epsAAll;
result.epsS = epsSAll;
result.eta = etaAll;
end
