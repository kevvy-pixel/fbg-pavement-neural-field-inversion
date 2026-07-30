function result = fbg_save_stage(stageDir, params, modelKind, history, ...
    D, split, meta, cfg)
%FBG_SAVE_STAGE Evaluate one model and save all stage artifacts.

if ~exist(stageDir,'dir'), mkdir(stageDir); end
figDir = fullfile(stageDir,'figures');
if ~exist(figDir,'dir'), mkdir(figDir); end

allMask = true(height(D.table),1);
predictions = fbg_predict(params, modelKind, D, allMask, meta, cfg);
predictions.split = repmat("unused",height(predictions),1);
predictions.split(split.train(predictions.row_index)) = "train";
predictions.split(split.validation(predictions.row_index)) = "validation";
predictions.split(split.test(predictions.row_index)) = "test";
predictions = movevars(predictions,'split','After','row_index');

[metrics,caseMetrics] = fbg_compute_metrics(predictions);
writetable(history, fullfile(stageDir,'training_history.csv'));
writetable(metrics, fullfile(stageDir,'metrics.csv'));
writetable(caseMetrics, fullfile(stageDir,'case_metrics.csv'));
writetable(predictions, fullfile(stageDir,'predictions.csv'));

paramsCPU = localGatherStruct(params);
save(fullfile(stageDir,'model.mat'),'paramsCPU','meta','cfg','modelKind','-v7.3');

localPlotHistory(history, fullfile(figDir,'training_history.png'));
localPlotScatter(predictions, fullfile(figDir,'measured_vs_predicted.png'));
localPlotTargetCurves(predictions, cfg, fullfile(figDir,'target_95cm_profiles.png'));
localPlotTargetResidualMap(predictions, cfg, ...
    fullfile(figDir,'target_95cm_residual_map.png'));

result.params = params;
result.history = history;
result.predictions = predictions;
result.metrics = metrics;
result.caseMetrics = caseMetrics;
end

function localPlotHistory(history, outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 900 500]);
yyaxis left;
semilogy(history.iteration,max(history.loss,eps),'-o','LineWidth',1.2);
ylabel('Training loss');
yyaxis right;
plot(history.iteration,history.validation_rmse_microstrain,'-s','LineWidth',1.2);
ylabel('Validation RMSE (\muepsilon)');
xlabel('Iteration'); grid on; title('Training history');
saveas(fig,outputFile); close(fig);
end

function localPlotScatter(predictions, outputFile)
fig = figure('Visible','off','Color','w','Position',[100 100 650 600]);
hold on;
names = ["train","validation","test"];
markers = {'o','s','^'};
colors = lines(3);
for i = 1:numel(names)
    sub = predictions(predictions.split == names(i),:);
    if isempty(sub), continue; end
    scatter(sub.measured_microstrain,sub.predicted_microstrain,28, ...
        'Marker',markers{i},'MarkerEdgeColor',colors(i,:), ...
        'DisplayName',names(i));
end
lims = [min([predictions.measured_microstrain;predictions.predicted_microstrain]), ...
    max([predictions.measured_microstrain;predictions.predicted_microstrain])];
plot(lims,lims,'k--','LineWidth',1.2,'DisplayName','1:1');
xlim(lims); ylim(lims); axis square; grid on; legend('Location','best');
xlabel('Measured strain (\muepsilon)');
ylabel('Predicted strain (\muepsilon)');
title('Measured vs predicted');
saveas(fig,outputFile); close(fig);
end

function localPlotTargetCurves(predictions, cfg, outputFile)
target = predictions(contains(lower(predictions.case_id), ...
    lower(cfg.data.targetCasePattern)),:);
fig = figure('Visible','off','Color','w','Position',[100 100 1000 550]);
if isempty(target)
    axis off; text(0.1,0.5,'No target 95 cm case found.','FontSize',14);
else
    target = target(contains(lower(target.direction),"纵") | ...
        contains(lower(target.direction),"long"),:);
    cases = unique(target.case_id,'stable');
    hold on;
    colors = lines(max(1,numel(cases)));
    for i = 1:numel(cases)
        sub = target(target.case_id == cases(i),:);
        centerDistance = abs(sub.y_cm-sub.p1_y);
        minDistance = min(centerDistance);
        sub = sub(centerDistance <= minDistance+1e-9,:);
        sub = sortrows(sub,'x_cm');
        plot(sub.x_cm,sub.measured_microstrain,'o-','Color',colors(i,:), ...
            'LineWidth',1.4,'DisplayName',cases(i)+" measured");
        plot(sub.x_cm,sub.predicted_microstrain,'--','Color',colors(i,:), ...
            'LineWidth',1.8,'DisplayName',cases(i)+" predicted");
    end
    xlabel('x (cm)'); ylabel('Longitudinal strain (\muepsilon)');
    title('Extreme-boundary centerline profile');
    legend('Location','best','Interpreter','none'); grid on;
end
saveas(fig,outputFile); close(fig);
end

function localPlotTargetResidualMap(predictions, cfg, outputFile)
target = predictions(contains(lower(predictions.case_id), ...
    lower(cfg.data.targetCasePattern)),:);
fig = figure('Visible','off','Color','w','Position',[100 100 800 550]);
if isempty(target)
    axis off; text(0.1,0.5,'No target 95 cm case found.','FontSize',14);
else
    scatter(target.x_cm,target.y_cm,75,target.residual_microstrain,'filled');
    colorbar; axis equal; grid on;
    xlabel('x (cm)'); ylabel('y (cm)');
    title('Extreme-boundary residual map (\muepsilon)');
end
saveas(fig,outputFile); close(fig);
end

function out = localGatherStruct(value)
if isa(value,'dlarray')
    out = dlarray(gather(extractdata(value)));
elseif isa(value,'gpuArray')
    out = gather(value);
elseif isstruct(value)
    out = value;
    for k = 1:numel(value)
        fields = fieldnames(value(k));
        for i = 1:numel(fields)
            out(k).(fields{i}) = localGatherStruct(value(k).(fields{i}));
        end
    end
else
    out = value;
end
end
