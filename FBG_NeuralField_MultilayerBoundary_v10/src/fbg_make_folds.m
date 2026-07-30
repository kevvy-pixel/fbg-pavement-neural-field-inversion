function folds = fbg_make_folds(T, cfg)
%FBG_MAKE_FOLDS Create case-grouped train/validation/test partitions.
cases = unique(T.case_id, "stable");
if numel(cases) < 3
    error("FBG:Folds", "At least three distinct cases are required.");
end

caseLoadX = zeros(numel(cases), 1);
for i = 1:numel(cases)
    caseLoadX(i) = median(T.load_x_cm(T.case_id == cases(i)));
end

if cfg.numFolds == 1
    testMask = abs(caseLoadX - cfg.targetLoadXCm) <= cfg.targetToleranceCm;
    if ~any(testMask)
        testMask = false(size(cases));
        testMask(end) = true;
    end
    remain = find(~testMask);
    valMask = false(size(cases));
    valCount = max(1, round(0.2 * numel(remain)));
    valMask(remain(end - valCount + 1:end)) = true;
    trainMask = ~(testMask | valMask);
    if ~any(trainMask)
        error("FBG:Folds", "Not enough non-test cases for training.");
    end
    folds = struct("trainCases", cases(trainMask), ...
        "valCases", cases(valMask), "testCases", cases(testMask));
    return;
end

k = min(cfg.numFolds, numel(cases));
assignment = mod((0:numel(cases)-1)', k) + 1;
folds = repmat(struct("trainCases", strings(0, 1), ...
    "valCases", strings(0, 1), "testCases", strings(0, 1)), k, 1);
for f = 1:k
    testMask = assignment == f;
    valMask = assignment == mod(f, k) + 1;
    trainMask = ~(testMask | valMask);
    folds(f).trainCases = cases(trainMask);
    folds(f).valCases = cases(valMask);
    folds(f).testCases = cases(testMask);
end
end
