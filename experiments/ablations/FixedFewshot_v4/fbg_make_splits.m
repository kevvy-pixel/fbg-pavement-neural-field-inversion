function split = fbg_make_splits(D, cfg)
%FBG_MAKE_SPLITS Deterministic case split with 95 cm held out.

rng(cfg.randomSeed);
cases = unique(D.table.case_id, 'stable');
targetCases = cases(contains(lower(cases), lower(cfg.data.targetCasePattern)));
forceTrainCases = cases(contains(lower(cases), lower(cfg.data.forceTrainPattern)));

if isempty(targetCases)
    warning('No case contains "%s"; using a generic case split.', ...
        cfg.data.targetCasePattern);
    nTest = max(1, round(0.15*numel(cases)));
    order = randperm(numel(cases));
    targetCases = cases(order(1:nTest));
else
    order = randperm(numel(cases));
end

candidate = setdiff(cases, [targetCases; forceTrainCases], 'stable');
nVal = min(numel(candidate), max(1, round(cfg.split.validationFraction*numel(cases))));
if isempty(candidate)
    validationCases = strings(0,1);
else
    shuffled = candidate(randperm(numel(candidate)));
    validationCases = shuffled(1:nVal);
end
trainCases = setdiff(cases, [targetCases; validationCases], 'stable');
trainCases = unique([trainCases; forceTrainCases], 'stable');

split.train = ismember(D.table.case_id, trainCases);
split.validation = ismember(D.table.case_id, validationCases);
split.test = ismember(D.table.case_id, targetCases);
split.trainCases = trainCases;
split.validationCases = validationCases;
split.testCases = targetCases;

if ~any(split.train)
    error('The split produced an empty training set.');
end
if ~any(split.test)
    error('The split produced an empty test set.');
end
end
