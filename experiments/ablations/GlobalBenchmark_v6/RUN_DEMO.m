function RUN_DEMO()
%RUN_DEMO Generate a synthetic response table and execute the full demo.

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
RUN_GLOBAL_BENCHMARK("demo", demoFile, workRoot);
end
