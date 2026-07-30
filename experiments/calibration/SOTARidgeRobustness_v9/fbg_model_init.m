function params = fbg_model_init(modelKind, meta, cfg, baseParams)
%FBG_MODEL_INIT Initialize the common SOTA/Residual parameter tensors.

arguments
    modelKind (1,1) string
    meta struct
    cfg struct
    baseParams = []
end

useGPU = cfg.runtime.useGPU && canUseGPU;
W = cfg.model.width;
C = cfg.model.conditionWidth;

if ~isempty(baseParams)
    params = baseParams;
else
    params.spatialIn.W = localWeight(W, meta.spatialDim, useGPU);
    params.spatialIn.b = localBias(W, useGPU);
    params.condition.W = localWeight(C, meta.conditionDim, useGPU);
    params.condition.b = localBias(C, useGPU);

    for i = 1:cfg.model.numBlocks
        params.blocks(i).W1 = localWeight(W, W, useGPU);
        params.blocks(i).b1 = localBias(W, useGPU);
        params.blocks(i).W2 = localWeight(W, W, useGPU);
        params.blocks(i).b2 = localBias(W, useGPU);
        params.blocks(i).Wg = localWeight(W, C, useGPU);
        params.blocks(i).bg = localBias(W, useGPU);
        params.blocks(i).Wb = localWeight(W, C, useGPU);
        params.blocks(i).bb = localBias(W, useGPU);
    end

    params.output.W = 0.05*localWeight(1, W, useGPU);
    params.output.b = localBias(1, useGPU);
end
end

function value = localWeight(nOut, nIn, useGPU)
limit = sqrt(6/(nIn+nOut));
value = (2*rand(nOut,nIn,'single')-1)*limit;
if useGPU, value = gpuArray(value); end
value = dlarray(value);
end

function value = localBias(n, useGPU)
value = zeros(n,1,'single');
if useGPU, value = gpuArray(value); end
value = dlarray(value);
end
