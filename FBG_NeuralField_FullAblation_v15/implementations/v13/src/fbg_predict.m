function result = fbg_predict(params, X, modelName, cfg)
%FBG_PREDICT Evaluate physical microstrain predictions in bounded chunks.
n = size(X, 2);
prediction = zeros(n, 1);
epsX = zeros(n, 1);
epsY = zeros(n, 1);
wM = zeros(n, 1);
fieldPrediction = zeros(n, 1);
correction = zeros(n, 1);
anchorGate = zeros(n, 1);
chunkSize = 512;
for first = 1:chunkSize:n
    last = min(first + chunkSize - 1, n);
    idx = first:last;
    dlX = dlarray(single(X(:, idx)));
    values = dlfeval(@fbg_model_forward, params, dlX, modelName, cfg);
    prediction(idx) = double(gather(extractdata(values.prediction)))';
    epsX(idx) = double(gather(extractdata(values.epsX)))';
    epsY(idx) = double(gather(extractdata(values.epsY)))';
    wM(idx) = double(gather(extractdata(values.wM)))';
    fieldPrediction(idx) = double(gather( ...
        extractdata(values.fieldPrediction)))';
    correction(idx) = double(gather(extractdata(values.correction)))';
    anchorGate(idx) = double(gather(extractdata(values.anchorGate)))';
end
result.prediction = prediction;
result.epsX = epsX;
result.epsY = epsY;
result.wM = wM;
result.fieldPrediction = fieldPrediction;
result.correction = correction;
result.anchorGate = anchorGate;
end
