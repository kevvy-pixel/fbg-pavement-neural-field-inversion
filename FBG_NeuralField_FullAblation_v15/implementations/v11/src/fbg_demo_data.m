function T = fbg_demo_data(csvPath)
%FBG_DEMO_DATA Create deterministic synthetic asphalt/soil FBG observations.
rng(21, "twister");
loadXs = [30, 42.5, 60, 75, 90, 95];
loadLevels = [5, 10];

sensorX = [50:5:85, repmat(101, 1, 9), repmat(90, 1, 9), 45:10:95];
sensorY = [repmat(25, 1, 8), 5:5:45, 5:5:45, repmat(25, 1, 6)];
direction = [repmat("x", 1, 8), repmat("y", 1, 9), ...
    repmat("y", 1, 9), repmat("x", 1, 6)];
layer = [repmat("asphalt", 1, 17), repmat("soil", 1, 15)];

rows = numel(loadXs) * numel(loadLevels) * numel(sensorX);
case_id = strings(rows, 1);
sensor_x_cm = zeros(rows, 1);
sensor_y_cm = zeros(rows, 1);
load_x_cm = zeros(rows, 1);
load_y_cm = zeros(rows, 1);
load_kN = zeros(rows, 1);
directionOut = strings(rows, 1);
layerOut = strings(rows, 1);
strain_ue = zeros(rows, 1);
load_type = repmat("single", rows, 1);

r = 0;
for ix = 1:numel(loadXs)
    for ip = 1:numel(loadLevels)
        lx = loadXs(ix);
        p = loadLevels(ip);
        cid = sprintf("single_x%g_p%g", lx, p);
        for s = 1:numel(sensorX)
            r = r + 1;
            x = sensorX(s);
            y = sensorY(s);
            dx = x - lx;
            dy = y - 25;
            edgeGain = 1 + 0.42 * max(0, (lx - 85) / 15) .* (dx > 0);
            if direction(s) == "x"
                base = -24 * p * exp(-(dx / 16).^2 - (dy / 11).^2);
                side = 9 * p * (dx / 17).^2 .* exp(-(dx / 24).^2 - (dy / 16).^2);
            else
                base = -18 * p * exp(-(dx / 18).^2 - (dy / 9).^2);
                side = 7 * p * (dy / 10).^2 .* exp(-(dx / 23).^2 - (dy / 15).^2);
            end
            asphaltResponse = edgeGain * (base + side);
            attenuation = 0.045 + 0.17 * (1 - exp(-p / 6)) + ...
                0.025 * (lx >= 90);
            if layer(s) == "soil"
                value = attenuation * asphaltResponse;
            else
                value = asphaltResponse;
            end
            value = value + randn * (2.0 + 0.004 * abs(value));
            case_id(r) = cid;
            sensor_x_cm(r) = x;
            sensor_y_cm(r) = y;
            load_x_cm(r) = lx;
            load_y_cm(r) = 25;
            load_kN(r) = p;
            directionOut(r) = direction(s);
            layerOut(r) = layer(s);
            strain_ue(r) = value;
        end
    end
end

T = table(case_id, sensor_x_cm, sensor_y_cm, load_x_cm, load_y_cm, ...
    load_kN, directionOut, layerOut, strain_ue, load_type, ...
    'VariableNames', {'case_id', 'sensor_x_cm', 'sensor_y_cm', ...
    'load_x_cm', 'load_y_cm', 'load_kN', 'direction', 'layer', ...
    'strain_ue', 'load_type'});
writetable(T, csvPath);
end
