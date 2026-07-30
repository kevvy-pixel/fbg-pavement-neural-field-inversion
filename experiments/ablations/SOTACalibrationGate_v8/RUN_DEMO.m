function RUN_DEMO()
%RUN_DEMO Generate synthetic data and execute the comparison smoke test.

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);

workRoot = fbg_writable_root(rootDir);
demoDir = fullfile(workRoot, 'demo_data');
if ~exist(demoDir, 'dir')
    mkdir(demoDir);
end

demoFile = fullfile(demoDir, 'Response_Matrix_Table_demo.csv');
fprintf('Writable workspace: %s\n',workRoot);
GENERATE_DEMO_DATA(demoFile);
runDir = RUN_SOTA_FULL_COMPARISON("demo",demoFile,workRoot);
RUN_SOTA_CALIBRATION_GATE(runDir);
end
