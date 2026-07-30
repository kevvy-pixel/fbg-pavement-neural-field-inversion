function [metrics, caseMetrics] = fbg_compute_metrics(predictions)
%FBG_COMPUTE_METRICS Compute split-level and case-level regression metrics.

if isempty(predictions)
    metrics = table();
    caseMetrics = table();
    return;
end

splitNames = unique(predictions.split, 'stable');
splitNames(end+1) = "all";

name = strings(0,1);
n = zeros(0,1);
r2 = zeros(0,1);
rmse = zeros(0,1);
mae = zeros(0,1);
bias = zeros(0,1);

for i = 1:numel(splitNames)
    if splitNames(i) == "all"
        sub = predictions;
    else
        sub = predictions(predictions.split == splitNames(i),:);
    end
    [n1,r21,rmse1,mae1,bias1] = localMetrics( ...
        sub.measured_microstrain, sub.predicted_microstrain);
    name(end+1,1) = splitNames(i); %#ok<AGROW>
    n(end+1,1) = n1; %#ok<AGROW>
    r2(end+1,1) = r21; %#ok<AGROW>
    rmse(end+1,1) = rmse1; %#ok<AGROW>
    mae(end+1,1) = mae1; %#ok<AGROW>
    bias(end+1,1) = bias1; %#ok<AGROW>
end
metrics = table(name,n,r2,rmse,mae,bias, ...
    'VariableNames', {'split','n','R2','RMSE_microstrain', ...
    'MAE_microstrain','Bias_microstrain'});

cases = unique(predictions.case_id,'stable');
case_id = strings(numel(cases),1);
case_split = strings(numel(cases),1);
n = zeros(numel(cases),1);
r2 = zeros(numel(cases),1);
rmse = zeros(numel(cases),1);
mae = zeros(numel(cases),1);
bias = zeros(numel(cases),1);
mean_gate = zeros(numel(cases),1);

for i = 1:numel(cases)
    sub = predictions(predictions.case_id == cases(i),:);
    case_id(i) = cases(i);
    case_split(i) = sub.split(1);
    [n(i),r2(i),rmse(i),mae(i),bias(i)] = localMetrics( ...
        sub.measured_microstrain, sub.predicted_microstrain);
    mean_gate(i) = mean(sub.gate,'omitnan');
end

caseMetrics = table(case_id,case_split,n,r2,rmse,mae,bias,mean_gate, ...
    'VariableNames', {'case_id','split','n','R2','RMSE_microstrain', ...
    'MAE_microstrain','Bias_microstrain','mean_gate'});
end

function [n,r2,rmse,mae,bias] = localMetrics(measured,predicted)
valid = isfinite(measured) & isfinite(predicted);
measured = measured(valid);
predicted = predicted(valid);
n = numel(measured);
if n == 0
    r2 = NaN; rmse = NaN; mae = NaN; bias = NaN;
    return;
end
residual = predicted-measured;
rmse = sqrt(mean(residual.^2));
mae = mean(abs(residual));
bias = mean(residual);
denom = sum((measured-mean(measured)).^2);
if denom <= eps
    r2 = NaN;
else
    r2 = 1-sum(residual.^2)/denom;
end
end
