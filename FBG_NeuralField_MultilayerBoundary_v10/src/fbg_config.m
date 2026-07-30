function cfg = fbg_config(mode)
%FBG_CONFIG Central configuration for demo, quick, and full experiments.
mode = lower(string(mode));
cfg.mode = mode;
cfg.seed = 20260729;
cfg.specimenLengthCm = 120;
cfg.specimenWidthCm = 50;
cfg.hiddenWidth = 48;
cfg.depth = 3;
cfg.learningRate = 3e-3;
cfg.gradClip = 5;
cfg.validationEvery = 10;
cfg.patienceChecks = 12;
cfg.soilLossWeight = 0.35;
cfg.transferLossWeight = 0.08;
cfg.parameterL2 = 1e-6;
cfg.etaMin = 0.02;
cfg.etaMax = 0.35;
cfg.targetLoadXCm = 95;
cfg.targetToleranceCm = 0.6;
cfg.numFolds = 5;

switch mode
    case "demo"
        cfg.maxIterations = 8;
        cfg.validationEvery = 2;
        cfg.patienceChecks = 4;
        cfg.numFolds = 1;
        cfg.hiddenWidth = 24;
    case "quick"
        cfg.maxIterations = 180;
        cfg.validationEvery = 10;
        cfg.patienceChecks = 10;
        cfg.numFolds = 1;
    case "full"
        cfg.maxIterations = 900;
        cfg.validationEvery = 20;
        cfg.patienceChecks = 15;
        cfg.numFolds = 5;
    otherwise
        error("FBG:Mode", "Unknown experiment mode: %s", mode);
end
end
