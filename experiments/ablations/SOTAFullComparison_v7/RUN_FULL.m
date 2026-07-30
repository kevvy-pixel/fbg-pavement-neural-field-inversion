function RUN_FULL(inputFile,outputBase)
%RUN_FULL One-command paper-grade five-fold comparison.
arguments
    inputFile (1,1) string
    outputBase (1,1) string = ""
end
RUN_SOTA_FULL_COMPARISON("full",inputFile,outputBase);
end
