function assignments = fbg_make_case_folds(D,cfg)
%FBG_MAKE_CASE_FOLDS Deterministic case-level stratified folds.

T = D.table;
cases = unique(T.case_id,'stable');
nCases = numel(cases);
nFolds = min(cfg.global.numFolds,nCases);
if nFolds < 2
    error('At least two distinct cases are required for global validation.');
end

case_id = cases;
load_type = strings(nCases,1);
load_kg = zeros(nCases,1);
n_rows = zeros(nCases,1);
max_abs_microstrain = zeros(nCases,1);
for i = 1:nCases
    sub = T(T.case_id == cases(i),:);
    load_type(i) = sub.load_type(1);
    load_kg(i) = sub.load_kg(1);
    n_rows(i) = height(sub);
    max_abs_microstrain(i) = max(abs(sub.response_microstrain));
end

rng(cfg.randomSeed);
fold_id = zeros(nCases,1);
foldCounts = zeros(nFolds,1);
types = unique(load_type,'stable');

% Distribute every load type across the currently smallest folds.
for it = 1:numel(types)
    idx = find(load_type == types(it));
    idx = idx(randperm(numel(idx)));
    for j = 1:numel(idx)
        minimum = min(foldCounts);
        candidates = find(foldCounts == minimum);
        chosen = candidates(mod(j+it-2,numel(candidates))+1);
        fold_id(idx(j)) = chosen;
        foldCounts(chosen) = foldCounts(chosen)+1;
    end
end

assignments = table(case_id,fold_id,load_type,load_kg,n_rows, ...
    max_abs_microstrain);
assignments = sortrows(assignments,{'fold_id','load_type','load_kg'});
end
