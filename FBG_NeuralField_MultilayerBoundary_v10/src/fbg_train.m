function [bestParams, history] = fbg_train(data, modelName, cfg)
%FBG_TRAIN Full-batch custom training with clipping and best validation restore.
params = fbg_model_init(modelName, cfg);
averageGrad = [];
averageSqGrad = [];
bestParams = params;
bestVal = inf;
checksWithoutImprovement = 0;

iteration = zeros(cfg.maxIterations, 1);
train_loss = nan(cfg.maxIterations, 1);
loss_asphalt = nan(cfg.maxIterations, 1);
loss_soil = nan(cfg.maxIterations, 1);
loss_transfer = nan(cfg.maxIterations, 1);
val_rmse_ue = nan(cfg.maxIterations, 1);

% The network uses explicit feature-by-observation matrices. Keep custom-loop
% arrays unformatted so they can be multiplied by unformatted parameter
% matrices on MATLAB R2024a.
XTrain = dlarray(single(data.train.X));
YTrain = dlarray(single(data.train.Y));
lastIteration = cfg.maxIterations;
for iter = 1:cfg.maxIterations
    [loss, gradients, components] = dlfeval(@fbg_loss_gradients, ...
        params, XTrain, YTrain, modelName, cfg);
    gradients = dlupdate(@(g) max(min(g, cfg.gradClip), -cfg.gradClip), ...
        gradients);
    [params, averageGrad, averageSqGrad] = adamupdate(params, gradients, ...
        averageGrad, averageSqGrad, iter, cfg.learningRate);

    iteration(iter) = iter;
    train_loss(iter) = double(gather(extractdata(loss)));
    c = double(gather(extractdata(components)));
    loss_asphalt(iter) = c(1);
    loss_soil(iter) = c(2);
    loss_transfer(iter) = c(3);

    doValidation = mod(iter, cfg.validationEvery) == 0 || ...
        iter == 1 || iter == cfg.maxIterations;
    if doValidation
        if isempty(data.val.indices)
            pred = fbg_predict(params, data.train.X, modelName, cfg);
            actual = data.train.Y(:) * data.normalizer.strainScale;
            predicted = pred.prediction(:) * data.normalizer.strainScale;
            currentVal = sqrt(mean((predicted - actual).^2));
        else
            pred = fbg_predict(params, data.val.X, modelName, cfg);
            actual = data.val.Y(:) * data.normalizer.strainScale;
            predicted = pred.prediction(:) * data.normalizer.strainScale;
            currentVal = sqrt(mean((predicted - actual).^2));
        end
        val_rmse_ue(iter) = currentVal;
        fprintf("iter %4d | loss %.5g | val RMSE %.3f ue\n", ...
            iter, train_loss(iter), currentVal);
        if isfinite(currentVal) && currentVal < bestVal
            bestVal = currentVal;
            bestParams = params;
            checksWithoutImprovement = 0;
        else
            checksWithoutImprovement = checksWithoutImprovement + 1;
        end
        if checksWithoutImprovement >= cfg.patienceChecks
            fprintf("Early stopping at iteration %d.\n", iter);
            lastIteration = iter;
            break;
        end
    end
    if ~isfinite(train_loss(iter))
        error("FBG:Training", "Non-finite training loss at iteration %d.", iter);
    end
end

keep = 1:lastIteration;
history = table(iteration(keep), train_loss(keep), loss_asphalt(keep), ...
    loss_soil(keep), loss_transfer(keep), val_rmse_ue(keep), ...
    'VariableNames', {'iteration', 'train_loss', 'loss_asphalt', ...
    'loss_soil', 'loss_transfer', 'val_rmse_ue'});
end
