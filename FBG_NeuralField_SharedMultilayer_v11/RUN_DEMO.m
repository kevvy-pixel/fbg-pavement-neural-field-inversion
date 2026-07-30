function outputDir = RUN_DEMO(workspaceDir)
%RUN_DEMO Generate synthetic two-layer data and execute a short smoke test.
if nargin < 1
    workspaceDir = "";
end
rootDir = fileparts(mfilename("fullpath"));
addpath(fullfile(rootDir, "src"));
workspaceDir = fbg_workspace(workspaceDir);
demoDir = fullfile(workspaceDir, "demo_data");
if ~isfolder(demoDir)
    mkdir(demoDir);
end
csvPath = fullfile(demoDir, "Response_Matrix_Table_demo_v11.csv");
fbg_demo_data(csvPath);
fprintf("Synthetic demo data saved: %s\n", csvPath);
outputDir = RUN_EXPERIMENT("demo", csvPath, workspaceDir);
end
