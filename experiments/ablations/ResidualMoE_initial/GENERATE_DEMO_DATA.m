function GENERATE_DEMO_DATA(outputFile)
%GENERATE_DEMO_DATA Create a small synthetic table with a difficult 95 cm case.

arguments
    outputFile (1,1) string = "Response_Matrix_Table_demo.csv"
end

rng(42);
[outDir,~,~] = fileparts(outputFile);
if strlength(outDir) > 0 && ~exist(outDir, 'dir')
    mkdir(outDir);
end

xVals = (10:10:110)';
yVals = [10 18 25 32 40]';

caseDefs = {
    "10kg单点中心加载",       "单点中心加载", 10, 60.0, 25.0;
    "20kg单点中心加载",       "单点中心加载", 20, 60.0, 25.0;
    "40kg单点中心加载",       "单点中心加载", 40, 60.0, 25.0;
    "80kg单点中心加载",       "单点中心加载", 80, 60.0, 25.0;
    "40kg纵中心线30cm加载",   "纵中心线加载", 40, 30.0, 25.0;
    "40kg纵中心线42.5cm加载", "纵中心线加载", 40, 42.5, 25.0;
    "40kg纵中心线90cm加载",   "纵中心线加载", 40, 90.0, 25.0;
    "40kg纵中心线95cm加载",   "纵中心线加载", 40, 95.0, 25.0
};

case_id = strings(0,1);
sensor_id = strings(0,1);
x_cm = zeros(0,1);
y_cm = zeros(0,1);
layer = strings(0,1);
direction = strings(0,1);
delta_lambda_resp_nm = zeros(0,1);
load_type = strings(0,1);
load_kg = zeros(0,1);
p1_x = zeros(0,1);
p1_y = zeros(0,1);
p2_x = nan(0,1);
p2_y = nan(0,1);

zNeutral = 4.0;
kEps = 1.2e-3;

for ic = 1:size(caseDefs,1)
    cid = caseDefs{ic,1};
    ctype = caseDefs{ic,2};
    P = caseDefs{ic,3};
    px = caseDefs{ic,4};
    py = caseDefs{ic,5};

    for ix = 1:numel(xVals)
        for iy = 1:numel(yVals)
            x = xVals(ix);
            y = yVals(iy);

            A = 0.0016 * P;
            ax = 23.0;
            by = 12.0;
            dx = x - px;
            dy = y - py;
            q = (dx/ax)^2 + (dy/by)^2;
            base = A * exp(-q);

            wxx = base * (4*dx^2/ax^4 - 2/ax^2);
            wyy = base * (4*dy^2/by^4 - 2/by^2);

            if px >= 94
                local = exp(-((x-105)/13)^2 - ((y-py)/10)^2);
                edgeTerm = 0.00075 * P * local;
                edgeXX = edgeTerm * (4*(x-105)^2/13^4 - 2/13^2);
                edgeYY = edgeTerm * (4*(y-py)^2/10^4 - 2/10^2);
                wxx = wxx + edgeXX;
                wyy = wyy + 1.8*edgeYY;
            end

            epsLong = -zNeutral * wxx * 1e6 + 8*randn();
            epsTran = -zNeutral * wyy * 1e6 + 8*randn();

            for idir = 1:2
                case_id(end+1,1) = cid;
                sensor_id(end+1,1) = sprintf("DEMO_X%03d_Y%03d_D%d", x, y, idir);
                x_cm(end+1,1) = x;
                y_cm(end+1,1) = y;
                layer(end+1,1) = "Asphalt";
                if idir == 1
                    direction(end+1,1) = "纵";
                    response = epsLong;
                else
                    direction(end+1,1) = "横";
                    response = epsTran;
                end
                delta_lambda_resp_nm(end+1,1) = response * kEps;
                load_type(end+1,1) = ctype;
                load_kg(end+1,1) = P;
                p1_x(end+1,1) = px;
                p1_y(end+1,1) = py;
                p2_x(end+1,1) = NaN;
                p2_y(end+1,1) = NaN;
            end
        end
    end
end

T = table(case_id, sensor_id, x_cm, y_cm, layer, direction, ...
    delta_lambda_resp_nm, load_type, load_kg, p1_x, p1_y, p2_x, p2_y);
writetable(T, outputFile);
fprintf('Synthetic demo data saved: %s (%d rows)\n', outputFile, height(T));
end
