function outputDir = RUN_FULL(csvPath, workspaceDir)
%RUN_FULL Run the complete grouped five-fold comparison.
if nargin < 2
    workspaceDir = "";
end
outputDir = RUN_EXPERIMENT("full", csvPath, workspaceDir);
end
