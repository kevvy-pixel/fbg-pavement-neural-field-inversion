function params = fbg_model_init(modelKind, meta, cfg, baseParams)
%FBG_MODEL_INIT Initialize plain, residual, or edge-expert parameters.

arguments
    modelKind (1,1) string
    meta struct
    cfg struct
    baseParams = []
end

useGPU = cfg.runtime.useGPU && canUseGPU;
W = cfg.model.width;
C = cfg.model.conditionWidth;

if modelKind == "edge" && ~isempty(baseParams)
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

if modelKind == "edge"
    E = cfg.model.edgeWidth;
    gateInputDim = W + C + 4;
    edgeInputDim = meta.spatialDim + C + 4;

    params.gate.W1 = localWeight(E, gateInputDim, useGPU);
    params.gate.b1 = localBias(E, useGPU);
    params.gate.W2 = localZeros(1, E, useGPU);
    params.gate.b2 = localBias(1, useGPU)-4;

    params.edge.W1 = localWeight(E, edgeInputDim, useGPU);
    params.edge.b1 = localBias(E, useGPU);
    params.edge.W2 = localWeight(E, E, useGPU);
    params.edge.b2 = localBias(E, useGPU);
    params.edge.W3 = localZeros(1, E, useGPU);
    params.edge.b3 = localBias(1, useGPU);
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

function value = localZeros(nOut, nIn, useGPU)
value = zeros(nOut,nIn,'single');
if useGPU, value = gpuArray(value); end
value = dlarray(value);
end
