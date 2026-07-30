function result = CHECK_ENVIRONMENT()
%CHECK_ENVIRONMENT Verify required MATLAB capabilities with a real AD test.

rootDir = fileparts(mfilename("fullpath"));
addpath(fullfile(rootDir, "src"));

names = ["dlarray"; "dlfeval"; "dlgradient"; "adamupdate"; ...
    "dlupdate"; "canUseGPU"; "readtable"; "writetable"; "jsonencode"];
available = false(size(names));
diagnostic = strings(size(names));

for i = 1:numel(names)
    name = names(i);
    if name == "dlgradient"
        try
            % dlfeval only traces differentiable values supplied as inputs.
            % Creating x inside localSecondDerivative makes dlgradient report
            % a false negative even when higher-order AD is available.
            d2 = dlfeval(@localSecondDerivative, dlarray(2));
            available(i) = isfinite(d2) && abs(d2 - 12) < 1e-4;
            if available(i)
                diagnostic(i) = "second derivative smoke test passed";
            else
                diagnostic(i) = "unexpected second derivative";
            end
        catch ME
            diagnostic(i) = string(ME.message);
        end
    else
        available(i) = exist(name, "file") > 0 || exist(name, "builtin") > 0;
        if available(i)
            diagnostic(i) = "resolved";
        else
            diagnostic(i) = "not resolved";
        end
    end
end

% 强制所有表变量统一为列向量
names = string(names(:));
available = logical(available(:));
diagnostic = string(diagnostic(:));

% 使用兼容 MATLAB R2024a 的名称—值写法
result = table( ...
    names, ...
    available, ...
    diagnostic, ...
    'VariableNames', ...
    {'function_name', 'available', 'diagnostic'});
disp(result);
fprintf("MATLAB release: %s\n", version("-release"));

gpuAvailable = false;
if available(names == "canUseGPU")
    try
        gpuAvailable = canUseGPU;
    catch
        gpuAvailable = false;
    end
end
fprintf("GPU available: %d\n", gpuAvailable);

required = ismember(names, ["dlarray", "dlfeval", "dlgradient", ...
    "adamupdate", "dlupdate", "readtable", "writetable"]);
if ~all(available(required))
    missing = strjoin(names(required & ~available), ", ");
    error("FBG:Environment", "Missing required MATLAB capabilities: %s", missing);
end

workspaceDir = fbg_workspace("");
fprintf("Writable workspace: %s\n", workspaceDir);
fprintf("Environment check passed. Next run RUN_DEMO.\n");
end

function d2 = localSecondDerivative(x)
y = x.^3;
d1 = dlgradient(y, x, "EnableHigherDerivatives", true);
d2dl = dlgradient(d1, x);
d2 = double(extractdata(d2dl));
end
