function M = fbg_metrics(actual, predicted)
%FBG_METRICS Compute standard regression metrics with safe edge handling.
actual = double(actual(:));
predicted = double(predicted(:));
valid = isfinite(actual) & isfinite(predicted);
actual = actual(valid);
predicted = predicted(valid);
M.n = numel(actual);
if isempty(actual)
    M.r2 = NaN;
    M.rmse = NaN;
    M.mae = NaN;
    M.bias = NaN;
    return;
end
residual = predicted - actual;
M.rmse = sqrt(mean(residual.^2));
M.mae = mean(abs(residual));
M.bias = mean(residual);
denom = sum((actual - mean(actual)).^2);
if M.n < 2 || denom <= eps
    M.r2 = NaN;
else
    M.r2 = 1 - sum(residual.^2) / denom;
end
end
