function RUN_FULL(inputFile,outputBase)
%RUN_FULL One-command paper-grade five-fold comparison.
arguments
    inputFile (1,1) string
    outputBase (1,1) string = ""
end
runDir = RUN_SOTA_FULL_COMPARISON("full",inputFile,outputBase);
RUN_SOTA_CALIBRATION_GATE(runDir);
RUN_SOTA_RIDGE_ROBUSTNESS(runDir);
end
