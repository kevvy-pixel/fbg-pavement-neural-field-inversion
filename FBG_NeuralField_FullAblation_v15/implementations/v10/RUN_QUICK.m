function outputDir = RUN_QUICK(csvPath, workspaceDir)
%RUN_QUICK Run one grouped holdout comparison with moderate iterations.
if nargin < 2
    workspaceDir = "";
end
outputDir = RUN_EXPERIMENT("quick", csvPath, workspaceDir);
end
