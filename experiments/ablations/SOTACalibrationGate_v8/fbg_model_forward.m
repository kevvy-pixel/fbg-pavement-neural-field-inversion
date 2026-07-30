function [w, aux] = fbg_model_forward(params, x, y, cond, ~, ~, ...
    ~, meta, cfg, modelKind)
%FBG_MODEL_FORWARD Step19 FiLM field with an optional residual connection.

s = localSpatialFeatures(x,y,meta,cfg);
h = tanh(params.spatialIn.W*s + params.spatialIn.b);
c = tanh(params.condition.W*cond + params.condition.b);

for i = 1:numel(params.blocks)
    gamma = 1 + 0.10*tanh(params.blocks(i).Wg*c + params.blocks(i).bg);
    beta = 0.10*tanh(params.blocks(i).Wb*c + params.blocks(i).bb);
    u = tanh(params.blocks(i).W1*h + params.blocks(i).b1);
    u = tanh(params.blocks(i).W2*u + params.blocks(i).b2);
    u = gamma.*u + beta;
    if modelKind == "sota"
        h = tanh(u);
    elseif modelKind == "residual"
        h = h + cfg.model.residualScale*u;
    else
        error('Unknown model kind "%s". Use sota or residual.',modelKind);
    end
end

w = cfg.model.displacementScale_cm*(params.output.W*h + params.output.b);
zeroRow = 0*x;
aux.gate = zeroRow;
aux.edgeResidual = zeroRow;
aux.wBase = w;
end

function s = localSpatialFeatures(x,y,meta,cfg)
xN = (x-meta.xMin)/meta.xScale;
yN = (y-meta.yMin)/meta.yScale;
dLeft = xN;
dRight = 1-xN;
dBottom = yN;
dTop = 1-yN;
dEdge = min(min(dLeft,dRight),min(dBottom,dTop));
edgeExp = exp(-max(dEdge,0)/cfg.model.boundaryFeatureLength);

s = [xN; yN; dLeft; dRight; dBottom; dTop; dEdge; edgeExp];
for f = cfg.model.fourierFrequencies
    s = [s; ...
        sin(2*pi*f*xN); cos(2*pi*f*xN); ...
        sin(2*pi*f*yN); cos(2*pi*f*yN)]; %#ok<AGROW>
end
end
