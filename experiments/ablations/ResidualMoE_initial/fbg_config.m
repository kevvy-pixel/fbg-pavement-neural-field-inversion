function cfg = fbg_config(profile)
%FBG_CONFIG Configuration for all neural-field experiments.

arguments
    profile (1,1) string = "quick"
end

cfg.profile = profile;
cfg.randomSeed = 20260727;

cfg.data.inputFile = "Response_Matrix_Table.csv";
cfg.data.targetCasePattern = "95cm";
cfg.data.forceTrainPattern = "90cm";
cfg.data.layer = "Asphalt";
cfg.data.kEps_nm_per_microstrain = 1.2e-3;
cfg.data.zNeutral_cm = 4.0;
cfg.data.aggregateDuplicates = true;

cfg.runtime.useGPU = true;
cfg.runtime.outputRoot = "outputs";

cfg.model.fourierFrequencies = [0.5 1 2];
cfg.model.width = 48;
cfg.model.conditionWidth = 32;
cfg.model.numBlocks = 3;
cfg.model.residualScale = 0.15;
cfg.model.displacementScale_cm = 0.01;
cfg.model.edgeWidth = 32;
cfg.model.edgeResidualScale = 0.25;
cfg.model.edgeLengthNormalized = 0.12;

cfg.training.batchSize = 64;
cfg.training.learningRate = 5e-4;
cfg.training.edgeLearningRate = 1e-4;
cfg.training.fewshotLearningRate = 5e-5;
cfg.training.plainIterations = 400;
cfg.training.residualIterations = 600;
cfg.training.edgeIterations = 300;
cfg.training.fewshotIterations = 120;
cfg.training.logEvery = 10;
cfg.training.validationEvery = 25;
cfg.training.gradientDecay = 0.9;
cfg.training.squaredGradientDecay = 0.999;
cfg.training.gradientClip = 5.0;

cfg.loss.loadBalanceAlpha = 0.15;
cfg.loss.loadBalanceBeta = 0.5;
cfg.loss.gaugeWeight = 1e-5;
cfg.loss.gateInteriorWeight = 1e-2;
cfg.loss.edgeResidualWeight = 2e-4;

cfg.split.validationFraction = 0.20;
cfg.fewshot.kValues = [0 2 4 6 8];

switch lower(profile)
    case "demo"
        cfg.model.fourierFrequencies = [0.5 1];
        cfg.model.width = 20;
        cfg.model.conditionWidth = 12;
        cfg.model.numBlocks = 2;
        cfg.model.edgeWidth = 12;
        cfg.training.batchSize = 32;
        cfg.training.plainIterations = 12;
        cfg.training.residualIterations = 18;
        cfg.training.edgeIterations = 12;
        cfg.training.fewshotIterations = 8;
        cfg.training.logEvery = 2;
        cfg.training.validationEvery = 4;
        cfg.fewshot.kValues = [0 2 4];
    case "quick"
        % Defaults above.
    case "research"
        cfg.model.width = 96;
        cfg.model.conditionWidth = 64;
        cfg.model.numBlocks = 5;
        cfg.model.edgeWidth = 64;
        cfg.training.batchSize = 96;
        cfg.training.learningRate = 2e-4;
        cfg.training.edgeLearningRate = 5e-5;
        cfg.training.fewshotLearningRate = 2e-5;
        cfg.training.plainIterations = 1600;
        cfg.training.residualIterations = 2400;
        cfg.training.edgeIterations = 1200;
        cfg.training.fewshotIterations = 500;
        cfg.training.logEvery = 20;
        cfg.training.validationEvery = 50;
    otherwise
        error('Unknown profile "%s". Use demo, quick, or research.', profile);
end
end
