function outputDir = RUN_DATA_DIAGNOSTICS(csvPath, workspaceDir)
%RUN_DATA_DIAGNOSTICS Run 90/95 cm and cross-layer checks without training.
if nargin < 2
    workspaceDir = "";
end
rootDir = fileparts(mfilename("fullpath"));
addpath(fullfile(rootDir, "src"));
workspaceDir = fbg_workspace(workspaceDir);
stamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
outputDir = fullfile(workspaceDir, "outputs", stamp + "_diagnostics_v12");
mkdir(outputDir);
logPath = fullfile(outputDir, "run.log");
diary(logPath);
cleanup = onCleanup(@() diary("off")); %#ok<NASGU>
fprintf("FBG v12 data diagnostics\nInput: %s\nOutput: %s\n", csvPath, outputDir);
T = fbg_load_data(csvPath);
fbg_diagnostics(T, fullfile(outputDir, "00_data_diagnostics"));
fprintf("Diagnostics completed.\n");
end
