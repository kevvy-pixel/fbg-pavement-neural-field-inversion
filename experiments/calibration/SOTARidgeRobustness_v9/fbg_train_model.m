function [params,history] = fbg_train_model(params,modelKind,D, ...
    trainMask,validationMask,meta,cfg,stageName)
%FBG_TRAIN_MODEL Identical full-budget training for SOTA and Residual.
%
% The two calls are made after resetting the same minibatch seed. Therefore
% they see the same sampled rows and differ only in the residual connection.

arguments
    params struct
    modelKind (1,1) string
    D struct
    trainMask (:,1) logical
    validationMask (:,1) logical
    meta struct
    cfg struct
    stageName (1,1) string = ""
end

maxIterations = cfg.training.maxIterations;
learnRate = cfg.training.learningRate;
trainIdx = find(trainMask);
validationIdx = find(validationMask);
if isempty(trainIdx), error('Training mask is empty.'); end
batchSize = min(cfg.training.batchSize,numel(trainIdx));

avgGrad = [];
avgSqGrad = [];
iterationLog = zeros(0,1);
lossLog = zeros(0,1);
dataLossLog = zeros(0,1);
gaugeLossLog = zeros(0,1);
valRmseLog = zeros(0,1);
elapsedLog = zeros(0,1);
timerValue = tic;
bestMetric = inf;
bestIteration = 0;
bestParams = params;

fprintf('\n[%s] Training %s for %d iterations (%d original rows)...\n', ...
    stageName,modelKind,maxIterations,numel(trainIdx));

for iteration = 1:maxIterations
    selected = trainIdx(randi(numel(trainIdx),[batchSize,1]));
    batch = localMakeBatch(D,selected,cfg,meta);
    [loss,gradients,components] = dlfeval(@localModelGradients, ...
        params,modelKind,batch,meta,cfg);

    gradients = dlupdate(@(g) max(min(g,cfg.training.gradientClip), ...
        -cfg.training.gradientClip),gradients);
    [params,avgGrad,avgSqGrad] = adamupdate(params,gradients, ...
        avgGrad,avgSqGrad,iteration,learnRate, ...
        cfg.training.gradientDecay,cfg.training.squaredGradientDecay);

    shouldValidate = ~isempty(validationIdx) && ...
        (iteration == 1 || mod(iteration,cfg.training.validationEvery) == 0 || ...
        iteration == maxIterations);
    shouldLog = shouldValidate || iteration == 1 || ...
        mod(iteration,cfg.training.logEvery) == 0 || iteration == maxIterations;
    valRmse = NaN;
    if shouldValidate
        valPred = fbg_predict(params,modelKind,D,validationMask,meta,cfg);
        valRmse = sqrt(mean((valPred.predicted_microstrain- ...
            valPred.measured_microstrain).^2,'omitnan'));
        if isfinite(valRmse) && valRmse < bestMetric
            bestMetric = valRmse;
            bestIteration = iteration;
            bestParams = params;
        end
    end

    if shouldLog
        iterationLog(end+1,1) = iteration; %#ok<AGROW>
        lossLog(end+1,1) = localScalar(loss); %#ok<AGROW>
        dataLossLog(end+1,1) = localScalar(components.data); %#ok<AGROW>
        gaugeLossLog(end+1,1) = localScalar(components.gauge); %#ok<AGROW>
        valRmseLog(end+1,1) = valRmse; %#ok<AGROW>
        elapsedLog(end+1,1) = toc(timerValue); %#ok<AGROW>
        fprintf('[%s] iter %5d/%5d | loss %.5g | val RMSE %.3f\n', ...
            stageName,iteration,maxIterations,lossLog(end),valRmse);
    end
end

if bestIteration > 0
    params = bestParams;
    fprintf('[%s] selected checkpoint: iteration %d, RMSE %.6g\n', ...
        stageName,bestIteration,bestMetric);
end

isSelected = iterationLog == bestIteration;
history = table(iterationLog,lossLog,dataLossLog,gaugeLossLog, ...
    valRmseLog,elapsedLog,isSelected, ...
    'VariableNames',{'iteration','loss','data_loss','gauge_loss', ...
    'validation_rmse_microstrain','elapsed_seconds', ...
    'is_selected_checkpoint'});
end

function [loss,gradients,components] = localModelGradients( ...
    params,modelKind,batch,meta,cfg)

[w,~] = fbg_model_forward(params,batch.x,batch.y,batch.cond, ...
    batch.p1x,batch.p1y,batch.dLoadEdge,meta,cfg,modelKind);
[wx,wy] = dlgradient(sum(w,'all'),batch.x,batch.y, ...
    'EnableHigherDerivatives',true,'RetainData',true);
wxx = dlgradient(sum(wx,'all'),batch.x, ...
    'EnableHigherDerivatives',true,'RetainData',true);
wyy = dlgradient(sum(wy,'all'),batch.y, ...
    'EnableHigherDerivatives',true,'RetainData',true);

pred = -meta.zNeutral_cm*1e6.* ...
    (batch.dirLong.*wxx + (1-batch.dirLong).*wyy);
sampleWeight = 1+0*batch.target;

if cfg.loss.useLoadBalance
    loadN = batch.loadKg/max(meta.maxLoad,eps);
    sampleWeight = sampleWeight ./ ...
        ((loadN+cfg.loss.loadBalanceAlpha).^cfg.loss.loadBalanceBeta);
end
if cfg.loss.useBoundaryWeight
    boundaryEmphasis = exp(-max(batch.dLoadEdge,0)/ ...
        cfg.model.boundaryFeatureLength);
    sampleWeight = sampleWeight.* ...
        (1+cfg.loss.boundaryWeightStrength*boundaryEmphasis);
end
sampleWeight = sampleWeight/mean(sampleWeight,'all');

normalizedError = (pred-batch.target)/meta.targetScale;
dataLoss = mean(sampleWeight.*normalizedError.^2,'all');
gaugeLoss = (mean(w,'all')/cfg.model.displacementScale_cm)^2;
loss = dataLoss+cfg.loss.gaugeWeight*gaugeLoss;
gradients = dlgradient(loss,params);

components.data = dataLoss;
components.gauge = gaugeLoss;
end

function batch = localMakeBatch(D,idx,cfg,meta)
useGPU = cfg.runtime.useGPU && canUseGPU;
batch.x = localDl(D.x(idx),useGPU);
batch.y = localDl(D.y(idx),useGPU);
batch.target = localDl(D.target(idx),useGPU);
batch.dirLong = localDl(D.dirLong(idx),useGPU);
batch.loadKg = localDl(D.loadKg(idx),useGPU);
batch.p1x = localDl(D.p1x(idx),useGPU);
batch.p1y = localDl(D.p1y(idx),useGPU);
batch.dLoadEdge = localDl(D.dLoadEdge(idx),useGPU);
batch.cond = localDl(D.cond(:,idx),useGPU);

if cfg.augmentation.mirrorX
    mirror = batch;
    mirror.x = meta.xMin+meta.xMax-batch.x;
    mirror.p1x = meta.xMin+meta.xMax-batch.p1x;
    mirror.cond = [batch.cond(1,:); 1-batch.cond(2,:); ...
        batch.cond(3,:); 1-batch.cond(4,:); batch.cond(5:end,:)];
    batch = localConcatBatch(batch,mirror);
end
if cfg.augmentation.mirrorY
    mirror = batch;
    mirror.y = meta.yMin+meta.yMax-batch.y;
    mirror.p1y = meta.yMin+meta.yMax-batch.p1y;
    mirror.cond = [batch.cond(1:2,:); 1-batch.cond(3,:); ...
        batch.cond(4,:); 1-batch.cond(5,:); batch.cond(6:end,:)];
    batch = localConcatBatch(batch,mirror);
end
end

function out = localConcatBatch(a,b)
names = fieldnames(a);
out = a;
for i = 1:numel(names)
    out.(names{i}) = [a.(names{i}) b.(names{i})];
end
end

function value = localDl(value,useGPU)
value = single(value);
if useGPU, value = gpuArray(value); end
value = dlarray(value);
end

function value = localScalar(value)
value = double(gather(extractdata(value)));
end
