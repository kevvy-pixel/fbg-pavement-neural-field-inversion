function RUN_POSTPROCESS_EXISTING(existingRunDir)
%RUN_POSTPROCESS_EXISTING Add SOTA Ridge-4 and the no-leakage gate.
arguments
    existingRunDir (1,1) string
end
RUN_SOTA_CALIBRATION_GATE(existingRunDir);
end
