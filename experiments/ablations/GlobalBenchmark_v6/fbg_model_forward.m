function [w, aux] = fbg_model_forward(params, x, y, cond, p1x, p1y, ...
    dLoadEdge, meta, cfg, modelKind)
%FBG_MODEL_FORWARD FiLM neural field with optional residual and edge expert.

s = localSpatialFeatures(x, y, meta, cfg);
h = tanh(params.spatialIn.W*s + params.spatialIn.b);
c = tanh(params.condition.W*cond + params.condition.b);

for i = 1:numel(params.blocks)
    gamma = 1 + 0.10*tanh(params.blocks(i).Wg*c + params.blocks(i).bg);
    beta = 0.10*tanh(params.blocks(i).Wb*c + params.blocks(i).bb);
    u = tanh(params.blocks(i).W1*h + params.blocks(i).b1);
    u = tanh(params.blocks(i).W2*u + params.blocks(i).b2);
    u = gamma.*u + beta;
    if modelKind == "plain"
        h = tanh(u);
    else
        h = h + cfg.model.residualScale*u;
    end
end

wBaseRaw = params.output.W*h + params.output.b;
wBase = cfg.model.displacementScale_cm*wBaseRaw;

zeroRow = 0*x;
aux.gate = zeroRow;
aux.edgeResidual = zeroRow;
aux.wBase = wBase;

if modelKind == "edge"
    xN = (x-meta.xMin)/meta.xScale;
    yN = (y-meta.yMin)/meta.yScale;
    p1xN = (p1x-meta.xMin)/meta.xScale;
    p1yN = (p1y-meta.yMin)/meta.yScale;
    relX = xN-p1xN;
    relY = yN-p1yN;
    dQueryEdge = min(min(xN,1-xN), min(yN,1-yN));
    localFeatures = [relX; relY; dQueryEdge; dLoadEdge];

    gateHidden = tanh(params.gate.W1*[h;c;localFeatures] + params.gate.b1);
    gate = 1./(1+exp(-(params.gate.W2*gateHidden + params.gate.b2)));

    edgeHidden = tanh(params.edge.W1*[s;c;localFeatures] + params.edge.b1);
    edgeHidden = tanh(params.edge.W2*edgeHidden + params.edge.b2);
    edgeResidual = cfg.model.displacementScale_cm * ...
        cfg.model.edgeResidualScale * ...
        (params.edge.W3*edgeHidden + params.edge.b3);

    w = wBase + gate.*edgeResidual;
    aux.gate = gate;
    aux.edgeResidual = edgeResidual;
else
    w = wBase;
end
end

function s = localSpatialFeatures(x, y, meta, cfg)
xN = (x-meta.xMin)/meta.xScale;
yN = (y-meta.yMin)/meta.yScale;
dLeft = xN;
dRight = 1-xN;
dBottom = yN;
dTop = 1-yN;
dEdge = min(min(dLeft,dRight), min(dBottom,dTop));
edgeExp = exp(-max(dEdge,0)/cfg.model.edgeLengthNormalized);

s = [xN; yN; dLeft; dRight; dBottom; dTop; dEdge; edgeExp];
for f = cfg.model.fourierFrequencies
    s = [s; ...
        sin(2*pi*f*xN); cos(2*pi*f*xN); ...
        sin(2*pi*f*yN); cos(2*pi*f*yN)]; %#ok<AGROW>
end
end
