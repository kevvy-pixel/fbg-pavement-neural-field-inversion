function prediction = fbg_predict(params, modelKind, D, mask, meta, cfg)
%FBG_PREDICT Predict strains and diagnostic latent outputs.

idx = find(mask);
if isempty(idx)
    prediction = table();
    return;
end

batchSize = max(16, min(256, cfg.training.batchSize));
n = numel(idx);
predAll = nan(n,1);
wAll = nan(n,1);
gateAll = nan(n,1);
edgeAll = nan(n,1);

for startIndex = 1:batchSize:n
    stopIndex = min(n, startIndex+batchSize-1);
    localIdx = idx(startIndex:stopIndex);
    batch = localMakeBatch(D, localIdx, cfg);
    [pred, w, gate, edgeResidual] = dlfeval(@localPredictBatch, ...
        params, modelKind, batch, meta, cfg);

    range = startIndex:stopIndex;
    predAll(range) = localNumeric(pred)';
    wAll(range) = localNumeric(w)';
    gateAll(range) = localNumeric(gate)';
    edgeAll(range) = localNumeric(edgeResidual)';
end

prediction = table( ...
    idx, D.table.case_id(idx), D.table.sensor_id(idx), ...
    D.table.x_cm(idx), D.table.y_cm(idx), D.table.direction(idx), ...
    D.table.load_type(idx), D.table.load_kg(idx), ...
    D.table.p1_x(idx), D.table.p1_y(idx), ...
    D.target(idx)', predAll, predAll-D.target(idx)', ...
    wAll, gateAll, edgeAll, ...
    'VariableNames', {'row_index','case_id','sensor_id','x_cm','y_cm', ...
    'direction','load_type','load_kg','p1_x','p1_y', ...
    'measured_microstrain','predicted_microstrain','residual_microstrain', ...
    'displacement_proxy_cm','gate','edge_residual_cm'});
end

function [pred, w, gate, edgeResidual] = localPredictBatch( ...
    params, modelKind, batch, meta, cfg)

[w, aux] = fbg_model_forward(params, batch.x, batch.y, batch.cond, ...
    batch.p1x, batch.p1y, batch.dLoadEdge, meta, cfg, modelKind);

[wx, wy] = dlgradient(sum(w,'all'), batch.x, batch.y, ...
    'EnableHigherDerivatives', true, 'RetainData', true);
wxx = dlgradient(sum(wx,'all'), batch.x, ...
    'EnableHigherDerivatives', false, 'RetainData', true);
wyy = dlgradient(sum(wy,'all'), batch.y, ...
    'EnableHigherDerivatives', false);

pred = -meta.zNeutral_cm*1e6 .* ...
    (batch.dirLong.*wxx + (1-batch.dirLong).*wyy);
gate = aux.gate;
edgeResidual = aux.edgeResidual;
end

function batch = localMakeBatch(D, idx, cfg)
useGPU = cfg.runtime.useGPU && canUseGPU;
batch.x = localDl(D.x(idx), useGPU);
batch.y = localDl(D.y(idx), useGPU);
batch.dirLong = localDl(D.dirLong(idx), useGPU);
batch.p1x = localDl(D.p1x(idx), useGPU);
batch.p1y = localDl(D.p1y(idx), useGPU);
batch.dLoadEdge = localDl(D.dLoadEdge(idx), useGPU);
batch.cond = localDl(D.cond(:,idx), useGPU);
end

function value = localDl(value, useGPU)
value = single(value);
if useGPU, value = gpuArray(value); end
value = dlarray(value);
end

function value = localNumeric(value)
value = double(gather(extractdata(value)));
end
