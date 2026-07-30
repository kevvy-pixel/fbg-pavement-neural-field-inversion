function [XPlus, XMinus] = fbg_position_perturb(X, cfg)
%FBG_POSITION_PERTURB Build coherent +/- load-position perturbations.
jitter = cfg.positionJitterCm / cfg.specimenLengthCm;
eligible = double(X(8, :) < 0.5);
plusShift = min(jitter, 1 - X(3, :)) .* eligible;
minusShift = max(-jitter, -X(3, :)) .* eligible;
XPlus = applyShift(X, plusShift);
XMinus = applyShift(X, minusShift);
end

function shifted = applyShift(X, shift)
shifted = X;
shifted(3, :) = X(3, :) + shift;
shifted(5, :) = X(5, :) + shift;
shifted(12, :) = X(12, :) + shift;
remainingDistanceSquared = max(X(15, :) .^ 2 - X(12, :) .^ 2, 0);
shifted(15, :) = sqrt(remainingDistanceSquared + shifted(12, :) .^ 2);
end
