function cfg = fbg_config(mode)
%FBG_CONFIG Central configuration for v12 soil-condition auxiliaries.
mode = lower(string(mode));
cfg.mode = mode;
cfg.seed = 2026;
cfg.specimenLengthCm = 120;
cfg.specimenWidthCm = 50;
cfg.hiddenWidth = 96;
cfg.depth = 5;
cfg.conditionHiddenWidth = 64;
cfg.fourierFreqsXY = [1, 2, 4, 8];
cfg.fourierFreqsRelative = [1, 2, 4];
cfg.boundarySoftMinTau = 0.03;
cfg.boundaryFeatureLengthNorm = 0.18;
cfg.learningRate = 4e-4;
cfg.gradClip = 5;
cfg.validationEvery = 20;
cfg.patienceChecks = 12;
cfg.parameterL2 = 1e-7;
cfg.outputInitScale = 0.01;
cfg.soilSummaryAuxWeight = 0.05;
cfg.wScaleM = 1e-3;
cfg.effectiveDepthM = 0.02;
cfg.targetLoadXCm = 95;
cfg.targetToleranceCm = 0.6;
cfg.numFolds = 5;

switch mode
    case "demo"
        cfg.maxIterations = 6;
        cfg.validationEvery = 2;
        cfg.patienceChecks = 4;
        cfg.numFolds = 1;
        cfg.hiddenWidth = 32;
        cfg.depth = 3;
        cfg.conditionHiddenWidth = 24;
        cfg.fourierFreqsXY = [1, 2];
        cfg.fourierFreqsRelative = 1;
    case "quick"
        cfg.maxIterations = 400;
        cfg.validationEvery = 20;
        cfg.patienceChecks = 12;
        cfg.numFolds = 1;
    case "full"
        cfg.maxIterations = 1800;
        cfg.validationEvery = 25;
        cfg.patienceChecks = 20;
        cfg.numFolds = 5;
    otherwise
        error("FBG:Mode", "Unknown experiment mode: %s", mode);
end
end
