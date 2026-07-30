function cfg = fbg_config(profile)
%FBG_CONFIG Configuration for the Step19-compatible full comparison.

arguments
    profile (1,1) string = "full"
end

cfg.profile = profile;
cfg.randomSeed = 20260727;

cfg.data.inputFile = "Response_Matrix_Table.csv";
cfg.data.targetCasePattern = "95cm";
cfg.data.layer = "Asphalt";
cfg.data.kEps_nm_per_microstrain = 1.2e-3;
cfg.data.zNeutral_cm = 4.0;
cfg.data.aggregateDuplicates = true;

cfg.runtime.useGPU = true;
cfg.runtime.outputRoot = "outputs";
cfg.runtime.resumeCompletedStages = true;

% Step19 settings reconstructed from Analysis_Report_CN.txt.
cfg.model.fourierFrequencies = [0.5 1 2];
cfg.model.width = 96;
cfg.model.conditionWidth = 64;
cfg.model.numBlocks = 5;
cfg.model.residualScale = 0.15;
cfg.model.displacementScale_cm = 0.01;
cfg.model.boundaryFeatureLength = 0.180;
cfg.model.extremeBoundaryThreshold = 0.210;

cfg.augmentation.mirrorX = true;
cfg.augmentation.mirrorY = false;

cfg.training.batchSize = 96;
cfg.training.learningRate = 2e-4;
cfg.training.maxIterations = 2400;
cfg.training.logEvery = 20;
cfg.training.validationEvery = 50;
cfg.training.gradientDecay = 0.9;
cfg.training.squaredGradientDecay = 0.999;
cfg.training.gradientClip = 5.0;

% Both models use exactly the same objective and sample weights.
cfg.loss.useLoadBalance = false;
cfg.loss.loadBalanceAlpha = 0.15;
cfg.loss.loadBalanceBeta = 0.5;
cfg.loss.useBoundaryWeight = true;
cfg.loss.boundaryWeightStrength = 1.0;
cfg.loss.gaugeWeight = 1e-5;

cfg.global.numFolds = 5;
cfg.global.validationFraction = 0.15;
cfg.global.calibrationK = 4;
cfg.global.minimumFixedTestCoordinates = 4;
cfg.global.inferenceTimingRepeats = 5;
cfg.global.calibrationTimingRepeats = 1000;
cfg.global.saveFoldModels = true;

cfg.fewshot.affineSlopePenalty = 1.0;
cfg.fewshot.affineOffsetPenalty = 0.05;
cfg.fewshot.affineScaleBounds = [0.5 1.5];

% Pre-registered gate: fixed-test observations are never used here.
cfg.gate.minRelativeCVImprovementPercent = 5.0;
cfg.gate.minAbsoluteCVImprovementMicrostrain = 2.0;
cfg.gate.scaleBoundMargin = 0.01;
cfg.gate.outputFolder = "03_sota_ridge_gate";

switch lower(profile)
    case "demo"
        cfg.model.fourierFrequencies = [0.5 1];
        cfg.model.width = 20;
        cfg.model.conditionWidth = 12;
        cfg.model.numBlocks = 2;
        cfg.training.batchSize = 32;
        cfg.training.maxIterations = 16;
        cfg.training.logEvery = 2;
        cfg.training.validationEvery = 4;
        cfg.global.numFolds = 2;
        cfg.global.inferenceTimingRepeats = 1;
        cfg.global.calibrationTimingRepeats = 20;
    case "quick"
        cfg.model.width = 48;
        cfg.model.conditionWidth = 32;
        cfg.model.numBlocks = 3;
        cfg.training.batchSize = 64;
        cfg.training.learningRate = 5e-4;
        cfg.training.maxIterations = 600;
        cfg.training.logEvery = 10;
        cfg.training.validationEvery = 25;
        cfg.global.numFolds = 3;
        cfg.global.inferenceTimingRepeats = 3;
        cfg.global.calibrationTimingRepeats = 200;
    case "full"
        % Defaults above are the paper-grade comparison.
    otherwise
        error('Unknown profile "%s". Use demo, quick, or full.',profile);
end
end
