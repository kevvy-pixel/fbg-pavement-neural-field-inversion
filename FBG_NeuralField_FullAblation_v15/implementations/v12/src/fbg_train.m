function [bestParams, history] = fbg_train(data, modelName, cfg)
%FBG_TRAIN Full-batch training with shared validation and early stopping.
params = fbg_model_init(modelName, cfg, ...
    data.normalizer.numLoadTypes);
averageGrad = [];
averageSqGrad = [];
bestParams = params;
bestVal = inf;
checksWithoutImprovement = 0;

iteration = zeros(cfg.maxIterations, 1);
train_loss = nan(cfg.maxIterations, 1);
loss_asphalt = nan(cfg.maxIterations, 1);
loss_aux_x = nan(cfg.maxIterations, 1);
loss_aux_y = nan(cfg.maxIterations, 1);
val_rmse_ue = nan(cfg.maxIterations, 1);

XTrain = dlarray(single(data.train.X));
YTrain = dlarray(single([data.train.Y; data.train.AuxY]));
lastIteration = cfg.maxIterations;
for iter = 1:cfg.maxIterations
    [loss, gradients, components] = dlfeval(@fbg_loss_gradients, ...
        params, XTrain, YTrain, modelName, cfg, data.normalizer);
    gradients = dlupdate(@(g) max(min(g, cfg.gradClip), -cfg.gradClip), ...
        gradients);
    [params, averageGrad, averageSqGrad] = adamupdate(params, gradients, ...
        averageGrad, averageSqGrad, iter, cfg.learningRate);

    iteration(iter) = iter;
    train_loss(iter) = double(gather(extractdata(loss)));
    values = double(gather(extractdata(components)));
    loss_asphalt(iter) = values(1);
    loss_aux_x(iter) = values(2);
    loss_aux_y(iter) = values(3);

    if ~isfinite(train_loss(iter))
        error("FBG:Training", ...
            "Non-finite training loss at iteration %d.", iter);
    end

    doValidation = mod(iter, cfg.validationEvery) == 0 || ...
        iter == 1 || iter == cfg.maxIterations;
    if doValidation
        if isempty(data.val.indices)
            prediction = fbg_predict(params, data.train.X, modelName, cfg);
            actual = data.train.Y(:);
            evaluationTable = data.train.table;
        else
            prediction = fbg_predict(params, data.val.X, modelName, cfg);
            actual = data.val.Y(:);
            evaluationTable = data.val.table;
        end
        primaryMask = evaluationTable.layer == "asphalt";
        currentVal = sqrt(mean((prediction.prediction(primaryMask) - ...
            actual(primaryMask)) .^ 2));
        val_rmse_ue(iter) = currentVal;
        fprintf(['iter %4d | loss %.5g | asphalt %.4g | auxX %.4g | ' ...
            'auxY %.4g | ' ...
            'val RMSE %.3f ue\n'], iter, train_loss(iter), ...
            loss_asphalt(iter), loss_aux_x(iter), loss_aux_y(iter), ...
            currentVal);
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
end

keep = 1:lastIteration;
history = table(iteration(keep), train_loss(keep), loss_asphalt(keep), ...
    loss_aux_x(keep), loss_aux_y(keep), val_rmse_ue(keep), ...
    'VariableNames', {'iteration', 'train_loss', 'loss_asphalt', ...
    'loss_aux_x', 'loss_aux_y', 'val_rmse_ue'});
end
