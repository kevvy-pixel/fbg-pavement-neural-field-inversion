function [foldD,foldMeta] = fbg_prepare_fold(D,meta,trainMask)
%FBG_PREPARE_FOLD Refit response/load scales using training cases only.

if ~any(trainMask)
    error('Cannot prepare fold metadata from an empty training mask.');
end

foldD = D;
foldMeta = meta;
foldMeta.maxLoad = max(D.loadKg(trainMask));
foldMeta.targetScale = max(std(D.target(trainMask),'omitnan'),1);

% The first condition feature is normalized load. Geometry and type coding
% are response-independent and remain shared across folds.
foldD.cond(1,:) = D.loadKg/max(foldMeta.maxLoad,eps);
end
