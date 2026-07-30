function fbg_diagnostics(T, outputDir)
%FBG_DIAGNOSTICS Export data-only checks for the 90/95 cm edge response.
if ~isfolder(outputDir)
    mkdir(outputDir);
end

shiftSummary = buildShiftSummary(T);
writetable(shiftSummary, fullfile(outputDir, "shift_90_95_summary.csv"));

clamp = buildClampProfiles(T);
writetable(clamp, fullfile(outputDir, "cross_layer_clamp_profiles.csv"));

transfer = buildMeasuredTransfer(T);
writetable(transfer, fullfile(outputDir, "measured_transfer_by_case.csv"));
fprintf("Data diagnostics saved: %s\n", outputDir);
end

function S = buildShiftSummary(T)
layers = ["asphalt", "soil"];
dirs = ["x", "y"];
records = cell(0, 8);
for il = 1:numel(layers)
    for id = 1:numel(dirs)
        A = T(abs(T.load_x_cm - 90) < 0.6 & T.layer == layers(il) & ...
            T.direction == dirs(id), :);
        B = T(abs(T.load_x_cm - 95) < 0.6 & T.layer == layers(il) & ...
            T.direction == dirs(id), :);
        if isempty(A) || isempty(B)
            records(end + 1, :) = {layers(il), dirs(id), 0, NaN, NaN, ...
                NaN, NaN, NaN}; %#ok<AGROW>
            continue;
        end
        used = false(height(B), 1);
        actual90 = [];
        actual95 = [];
        distances = [];
        for i = 1:height(A)
            targetX = A.sensor_x_cm(i) + 5;
            sameLoad = abs(B.load_kN - A.load_kN(i)) < 1e-6;
            candidates = find(sameLoad & ~used);
            if isempty(candidates)
                continue;
            end
            d = hypot(B.sensor_x_cm(candidates) - targetX, ...
                B.sensor_y_cm(candidates) - A.sensor_y_cm(i));
            [dmin, j] = min(d);
            if dmin <= 7.5
                k = candidates(j);
                used(k) = true;
                actual90(end + 1, 1) = A.strain_ue(i); %#ok<AGROW>
                actual95(end + 1, 1) = B.strain_ue(k); %#ok<AGROW>
                distances(end + 1, 1) = dmin; %#ok<AGROW>
            end
        end
        n = numel(actual90);
        if n >= 2
            delta = actual95 - actual90;
            rmse = sqrt(mean(delta.^2));
            mae = mean(abs(delta));
            bias = mean(delta);
            C = corrcoef(actual90, actual95);
            correlation = C(1, 2);
            meanDistance = mean(distances);
        else
            rmse = NaN; mae = NaN; bias = NaN;
            correlation = NaN; meanDistance = NaN;
        end
        records(end + 1, :) = {layers(il), dirs(id), n, rmse, mae, ...
            bias, correlation, meanDistance}; %#ok<AGROW>
    end
end
S = cell2table(records, 'VariableNames', {'layer', 'direction', ...
    'matched_pairs', 'shift_rmse_ue', 'shift_mae_ue', 'shift_bias_ue', ...
    'correlation', 'mean_match_distance_cm'});
end

function C = buildClampProfiles(T)
maskLoad = abs(T.load_x_cm - 90) < 0.6 | abs(T.load_x_cm - 95) < 0.6;
maskAsphalt = T.layer == "asphalt" & abs(T.sensor_x_cm - 101) <= 3;
maskSoil = T.layer == "soil" & abs(T.sensor_x_cm - 90) <= 3;
C = T(maskLoad & (maskAsphalt | maskSoil), ...
    ["case_id", "load_x_cm", "load_y_cm", "load_kN", "layer", ...
    "direction", "sensor_x_cm", "sensor_y_cm", "strain_ue"]);
if isempty(C)
    C = table(strings(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
        strings(0, 1), strings(0, 1), zeros(0, 1), zeros(0, 1), ...
        zeros(0, 1), 'VariableNames', {'case_id', 'load_x_cm', ...
        'load_y_cm', 'load_kN', 'layer', 'direction', 'sensor_x_cm', ...
        'sensor_y_cm', 'strain_ue'});
end
end

function R = buildMeasuredTransfer(T)
cases = unique(T.case_id, "stable");
case_id = strings(0, 1);
load_x_cm = zeros(0, 1);
load_kN = zeros(0, 1);
rms_asphalt_ue = zeros(0, 1);
rms_soil_ue = zeros(0, 1);
eta_measured = zeros(0, 1);
for i = 1:numel(cases)
    Q = T(T.case_id == cases(i), :);
    a = Q.strain_ue(Q.layer == "asphalt");
    s = Q.strain_ue(Q.layer == "soil");
    if isempty(a) || isempty(s)
        continue;
    end
    ra = sqrt(mean(a.^2));
    rs = sqrt(mean(s.^2));
    case_id(end + 1, 1) = cases(i); %#ok<AGROW>
    load_x_cm(end + 1, 1) = median(Q.load_x_cm); %#ok<AGROW>
    load_kN(end + 1, 1) = median(Q.load_kN); %#ok<AGROW>
    rms_asphalt_ue(end + 1, 1) = ra; %#ok<AGROW>
    rms_soil_ue(end + 1, 1) = rs; %#ok<AGROW>
    eta_measured(end + 1, 1) = rs / max(ra, eps); %#ok<AGROW>
end
R = table(case_id, load_x_cm, load_kN, rms_asphalt_ue, ...
    rms_soil_ue, eta_measured);
end
