function make_ablation_figures(aggregateDir, figureDir)
arguments
    aggregateDir (1,1) string
    figureDir (1,1) string
end
if ~isfolder(figureDir)
    mkdir(figureDir);
end

G = readtable(fullfile(aggregateDir, "global_asphalt_metrics.csv"), TextType="string");
T = readtable(fullfile(aggregateDir, "target_95cm_metrics.csv"), TextType="string");
F = readtable(fullfile(aggregateDir, "fold_metrics.csv"), TextType="string");

labels = replace(G.model_key, ["v10/","v11/","v12/","v13/","v14/"], ...
    ["v10 · ","v11 · ","v12 · ","v13 · ","v14 · "]);
fig = figure(Visible="off", Color="w", Position=[100 100 1200 700]);
barh(categorical(labels, flip(labels)), G.rmse_ue, FaceColor=[0.16 0.45 0.72]);
xlabel("五折拼接 RMSE (\mu\epsilon)");
title("完整五折消融：沥青层全局误差");
grid on;
exportgraphics(fig, fullfile(figureDir, "01_global_rmse_ranking.png"), Resolution=180);
close(fig);

labels95 = replace(T.model_key, ["v10/","v11/","v12/","v13/","v14/"], ...
    ["v10 · ","v11 · ","v12 · ","v13 · ","v14 · "]);
fig = figure(Visible="off", Color="w", Position=[100 100 1200 700]);
barh(categorical(labels95, flip(labels95)), T.rmse_ue, FaceColor=[0.84 0.38 0.18]);
xlabel("95 cm RMSE (\mu\epsilon)");
title("完整五折消融：95 cm 工况");
grid on;
exportgraphics(fig, fullfile(figureDir, "02_target95_rmse_ranking.png"), Resolution=180);
close(fig);

selected = ["v10/Step19-SOTA-FiLM", ...
    "v13/NearestAnchor-Persistence", ...
    "v13/AnchoredResidual-XOnly-FiLM", ...
    "v14/AnchoredResidual-XOnly-LowFreq", ...
    "v14/AnchoredResidual-XOnly-LowFreqSmooth"];
short = ["v10 Step19","锚点持久性","v13 XOnly", ...
    "v14 低频","v14 低频+平滑"];
fig = figure(Visible="off", Color="w", Position=[100 100 1100 650]);
hold on;
for i = 1:numel(selected)
    Q = F(F.model_key == selected(i), :);
    Q = sortrows(Q, "fold");
    plot(Q.fold, Q.rmse_ue, "-o", LineWidth=1.8, MarkerSize=6, ...
        DisplayName=short(i));
end
hold off;
xlabel("测试折");
ylabel("RMSE (\mu\epsilon)");
title("代表模型的折间稳定性");
xticks(1:5);
grid on;
legend(Location="best");
exportgraphics(fig, fullfile(figureDir, "03_fold_stability.png"), Resolution=180);
close(fig);
end
