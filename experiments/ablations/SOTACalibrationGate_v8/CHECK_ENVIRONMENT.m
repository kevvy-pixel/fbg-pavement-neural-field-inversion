function report = CHECK_ENVIRONMENT()
%CHECK_ENVIRONMENT Verify MATLAB functions required by the package.

requiredFunctions = [ ...
    "dlarray","dlfeval","dlgradient","adamupdate","dlupdate", ...
    "canUseGPU","readtable","writetable","jsonencode"];

available = false(numel(requiredFunctions),1);
diagnostic = repmat("",numel(requiredFunctions),1);
for i = 1:numel(requiredFunctions)
    if requiredFunctions(i) == "dlgradient"
        [available(i),diagnostic(i)] = localCheckHigherDerivatives();
    else
        available(i) = exist(requiredFunctions(i),'file') > 0 || ...
            exist(requiredFunctions(i),'builtin') > 0 || ...
            exist(requiredFunctions(i),'class') > 0;
        if available(i)
            diagnostic(i) = "resolved";
        else
            diagnostic(i) = "not found";
        end
    end
end

report = table(requiredFunctions',available,diagnostic, ...
    'VariableNames',{'function_name','available','diagnostic'});
disp(report);

fprintf('MATLAB release: %s\n',version('-release'));

if any(~available)
    error('Missing required MATLAB functions: %s', ...
        strjoin(requiredFunctions(~available),', '));
end
fprintf('GPU available: %d\n',canUseGPU);
workRoot = fbg_writable_root(fileparts(mfilename('fullpath')));
fprintf('Writable workspace: %s\n',workRoot);
fprintf('Environment check passed. Next run RUN_DEMO.\n');
end

function [ok,message] = localCheckHigherDerivatives()
% dlgradient can be dispatched as a dlarray method, so exist(...,'file')
% is not a reliable availability test in some MATLAB releases.
try
    x = dlarray(single(2));
    secondDerivative = dlfeval(@localSecondDerivative,x);
    value = double(extractdata(secondDerivative));
    ok = isfinite(value) && abs(value-12) < 1e-4;
    if ok
        message = "second derivative smoke test passed";
    else
        message = "unexpected derivative value: " + value;
    end
catch ME
    ok = false;
    message = string(ME.message);
end
end

function secondDerivative = localSecondDerivative(x)
y = x.^3;
firstDerivative = dlgradient(sum(y,'all'),x, ...
    'EnableHigherDerivatives',true);
secondDerivative = dlgradient(sum(firstDerivative,'all'),x);
end
