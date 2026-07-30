function [params, history] = fbg_train_model(params, modelKind, D, ...
    trainMask, validationMask, meta, cfg, stageName, freezeBase)
%FBG_TRAIN_MODEL Custom AD-DNF training loop.

arguments
    params struct
    modelKind (1,1) string
    D struct
    trainMask (:,1) logical
    validationMask (:,1) logical
    meta struct
    cfg struct
    stageName (1,1) string
    freezeBase (1,1) logical = false
end

switch stageName
    case "plain"
        maxIterations = cfg.training.plainIterations;
        learnRate = cfg.training.learningRate;
    case "residual"
        maxIterations = cfg.training.residualIterations;
        learnRate = cfg.training.learningRate;
    case "edge"
        maxIterations = cfg.training.edgeIterations;
        learnRate = cfg.training.edgeLearningRate;
    case "fewshot"
        maxIterations = cfg.training.fewshotIterations;
        learnRate = cfg.training.fewshotLearningRate;
    otherwise
        error('Unknown training stage: %s', stageName);
end

trainIdx = find(trainMask);
if isempty(trainIdx)
    error('Training mask is empty.');
end
validationIdx = find(validationMask);
batchSize = min(cfg.training.batchSize, numel(trainIdx));

avgGrad = [];
avgSqGrad = [];
iterationLog = [];
lossLog = [];
dataLossLog = [];
gaugeLossLog = [];
edgeLossLog = [];
valRmseLog = [];
elapsedLog = [];
selectedCheckpointLog = false(0,1);
timerValue = tic;
bestMetric = inf;
bestIteration = 0;
bestParams = params;

fprintf('\n[%s] Training %s model for %d iterations (%d samples)...\n', ...
    stageName, modelKind, maxIterations, numel(trainIdx));

for iteration = 1:maxIterations
    selected = trainIdx(randi(numel(trainIdx), [batchSize,1]));
    batch = localMakeBatch(D, selected, cfg);

    [loss, gradients, components] = dlfeval(@localModelGradients, ...
        params, modelKind, batch, meta, cfg);

    gradients = dlupdate(@(g) max(min(g,cfg.training.gradientClip), ...
        -cfg.training.gradientClip),gradients);

    if freezeBase
        gradients = localFreezeBase(gradients);
    end

    [params, avgGrad, avgSqGrad] = adamupdate(params, gradients, ...
        avgGrad, avgSqGrad, iteration, learnRate, ...
        cfg.training.gradientDecay, cfg.training.squaredGradientDecay);

    shouldLog = iteration == 1 || mod(iteration,cfg.training.logEvery) == 0 || ...
        iteration == maxIterations;
    valRmse = NaN;
    shouldValidate = ~isempty(validationIdx) && ...
        (iteration == 1 || mod(iteration,cfg.training.validationEvery) == 0 || ...
        iteration == maxIterations);
    if shouldValidate
        valPred = fbg_predict(params, modelKind, D, validationMask, meta, cfg);
        valRmse = sqrt(mean((valPred.predicted_microstrain - ...
            valPred.measured_microstrain).^2, 'omitnan'));
    end

    selectedNow = false;
    if stageName == "fewshot"
        selectionMetric = localScalar(loss);
        if selectionMetric < bestMetric
            bestMetric = selectionMetric;
            bestIteration = iteration;
            bestParams = params;
            selectedNow = true;
        end
    elseif isfinite(valRmse) && valRmse < bestMetric
        bestMetric = valRmse;
        bestIteration = iteration;
        bestParams = params;
        selectedNow = true;
    elseif isempty(validationIdx)
        selectionMetric = localScalar(loss);
        if selectionMetric < bestMetric
            bestMetric = selectionMetric;
            bestIteration = iteration;
            bestParams = params;
            selectedNow = true;
        end
    end

    if shouldLog
        iterationLog(end+1,1) = iteration; %#ok<AGROW>
        lossLog(end+1,1) = localScalar(loss); %#ok<AGROW>
        dataLossLog(end+1,1) = localScalar(components.data); %#ok<AGROW>
        gaugeLossLog(end+1,1) = localScalar(components.gauge); %#ok<AGROW>
        edgeLossLog(end+1,1) = localScalar(components.edge); %#ok<AGROW>
        valRmseLog(end+1,1) = valRmse; %#ok<AGROW>
        elapsedLog(end+1,1) = toc(timerValue); %#ok<AGROW>
        selectedCheckpointLog(end+1,1) = selectedNow; %#ok<AGROW>

        fprintf('[%s] iter %5d/%5d | loss %.5g | data %.5g | val RMSE %.3f\n', ...
            stageName, iteration, maxIterations, lossLog(end), ...
            dataLossLog(end), valRmse);
    end
end

if bestIteration > 0
    params = bestParams;
    fprintf('[%s] selected checkpoint: iteration %d, metric %.6g\n', ...
        stageName,bestIteration,bestMetric);
end

selectedCheckpointLog = iterationLog == bestIteration;
history = table(iterationLog, lossLog, dataLossLog, gaugeLossLog, ...
    edgeLossLog, valRmseLog, elapsedLog, selectedCheckpointLog, ...
    'VariableNames', {'iteration','loss','data_loss','gauge_loss', ...
    'edge_regularization','validation_rmse_microstrain','elapsed_seconds', ...
    'is_selected_checkpoint'});
end

function [loss, gradients, components] = localModelGradients( ...
    params, modelKind, batch, meta, cfg)

[w, aux] = fbg_model_forward(params, batch.x, batch.y, batch.cond, ...
    batch.p1x, batch.p1y, batch.dLoadEdge, meta, cfg, modelKind);

[wx, wy] = dlgradient(sum(w,'all'), batch.x, batch.y, ...
    'EnableHigherDerivatives', true, 'RetainData', true);
wxx = dlgradient(sum(wx,'all'), batch.x, ...
    'EnableHigherDerivatives', true, 'RetainData', true);
wyy = dlgradient(sum(wy,'all'), batch.y, ...
    'EnableHigherDerivatives', true, 'RetainData', true);

pred = -meta.zNeutral_cm*1e6 .* ...
    (batch.dirLong.*wxx + (1-batch.dirLong).*wyy);

loadN = batch.loadKg/max(meta.maxLoad,eps);
sampleWeight = 1 ./ ((loadN + cfg.loss.loadBalanceAlpha) .^ ...
    cfg.loss.loadBalanceBeta);
sampleWeight = sampleWeight / mean(sampleWeight,'all');

normalizedError = (pred-batch.target)/meta.targetScale;
dataLoss = mean(sampleWeight.*normalizedError.^2,'all');
gaugeLoss = (mean(w,'all')/cfg.model.displacementScale_cm)^2;

edgeRegularization = 0*dataLoss;
if modelKind == "edge"
    xN = (batch.x-meta.xMin)/meta.xScale;
    yN = (batch.y-meta.yMin)/meta.yScale;
    dQueryEdge = min(min(xN,1-xN), min(yN,1-yN));
    interiorWeight = min(max(dQueryEdge,0)/0.25,1);
    gatePenalty = mean(aux.gate.*interiorWeight,'all');
    deltaPenalty = mean((aux.edgeResidual/ ...
        cfg.model.displacementScale_cm).^2,'all');
    edgeRegularization = cfg.loss.gateInteriorWeight*gatePenalty + ...
        cfg.loss.edgeResidualWeight*deltaPenalty;
end

loss = dataLoss + cfg.loss.gaugeWeight*gaugeLoss + edgeRegularization;
gradients = dlgradient(loss, params);

components.data = dataLoss;
components.gauge = gaugeLoss;
components.edge = edgeRegularization;
end

function batch = localMakeBatch(D, idx, cfg)
useGPU = cfg.runtime.useGPU && canUseGPU;
batch.x = localDl(D.x(idx), useGPU);
batch.y = localDl(D.y(idx), useGPU);
batch.target = localDl(D.target(idx), useGPU);
batch.dirLong = localDl(D.dirLong(idx), useGPU);
batch.loadKg = localDl(D.loadKg(idx), useGPU);
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

function value = localScalar(value)
value = double(gather(extractdata(value)));
end

function gradients = localFreezeBase(gradients)
% Freeze the global field during few-shot calibration.
baseFields = {'spatialIn','condition','blocks','output'};
for i = 1:numel(baseFields)
    name = baseFields{i};
    if isfield(gradients,name)
        gradients.(name) = dlupdate(@(x) 0*x, gradients.(name));
    end
end
end
