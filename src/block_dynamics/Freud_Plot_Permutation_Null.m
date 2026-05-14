%% Freud_Plot_Permutation_Null_Figure4D.m
% Generate Figure 4D permutation-null panels.
%
% Required input:
%   perm_null_results.mat
%
% Outputs:
%   Figure_4_D_1.svg
%   Figure_4_D_2.svg
%   Figure_4_D_3.svg

clear; clc; close all;

%% ----------------------- Load results -----------------------
load('perm_null_results.mat');

R        = perm_results.R;
bacc_obs = perm_results.obs.bacc;
auc_obs  = perm_results.obs.auc;
sens_obs = perm_results.obs.sens;
spec_obs = perm_results.obs.spec;

null_bacc = perm_results.null_bacc(:);
null_auc  = perm_results.null_auc(:);
null_sens = perm_results.null_sens(:);
null_spec = perm_results.null_spec(:);

fprintf('===== Permutation results summary, R = %d =====\n', R);
fprintf('Observed: AUC = %.4f | BACC = %.4f | Sens = %.4f | Spec = %.4f\n', ...
    auc_obs, bacc_obs, sens_obs, spec_obs);

%% ----------------------- Formatting -----------------------
labelFS = 34;
tickFS  = 28;
lineW   = 4;

histColor = [0.216 0.478 0.816];   % MATLAB-like blue
obsColor  = [0.90 0.00 0.00];      % red observed line
starColor = [0.55 0.12 0.08];       % magenta observed marker

%% ----------------------- Figure 4D-1: AUC null -----------------------
f1 = figure('Color', 'w', 'Position', [100 100 1100 850]);
ax1 = axes('Parent', f1);
hold(ax1, 'on');

histogram(ax1, null_auc, 40, ...
    'FaceColor', histColor, ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.95);

xline(ax1, auc_obs, '-', ...
    'Color', obsColor, ...
    'LineWidth', lineW);

grid(ax1, 'on');
box(ax1, 'on');

xlabel(ax1, 'Area Under the Curve (AUC)', 'FontSize', labelFS);
ylabel(ax1, 'Count', 'FontSize', labelFS);

set(ax1, 'FontSize', tickFS, 'LineWidth', 1.2);
xlim(ax1, [0.25 0.82]);
ylim(ax1, [0 70]);

set(f1, 'Renderer', 'painters');
set(f1, 'PaperPositionMode', 'auto');
print(f1, 'Figure_4_D_1.svg', '-dsvg', '-painters');

%% ----------------------- Figure 4D-2: Balanced accuracy null -----------------------
f2 = figure('Color', 'w', 'Position', [100 100 1100 850]);
ax2 = axes('Parent', f2);
hold(ax2, 'on');

histogram(ax2, null_bacc, 40, ...
    'FaceColor', histColor, ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.95);

xline(ax2, bacc_obs, '-', ...
    'Color', obsColor, ...
    'LineWidth', lineW);

grid(ax2, 'on');
box(ax2, 'on');

xlabel(ax2, 'Balanced Accuracy', 'FontSize', labelFS);
ylabel(ax2, 'Count', 'FontSize', labelFS);

set(ax2, 'FontSize', tickFS, 'LineWidth', 1.2);
xlim(ax2, [0.40 0.80]);
ylim(ax2, [0 100]);

set(f2, 'Renderer', 'painters');
set(f2, 'PaperPositionMode', 'auto');
print(f2, 'Figure_4_D_2.svg', '-dsvg', '-painters');

%% ----------------------- Figure 4D-3: Sensitivity-specificity heatmap -----------------------
f3 = figure('Color', 'w', 'Position', [100 100 1100 850]);
ax3 = axes('Parent', f3);
hold(ax3, 'on');

nBins = 20;
edges = linspace(0, 1, nBins + 1);

[N2, ~, ~] = histcounts2(null_spec, null_sens, edges, edges, ...
    'Normalization', 'probability');

imagesc(ax3, edges, edges, N2');
set(ax3, 'YDir', 'normal');

colormap(ax3, parula);
cb = colorbar(ax3);
cb.FontSize = tickFS;
cb.Ticks = 0:0.005:0.02;
cb.TickLabels = {'0', '0.005', '0.01', '0.015', '0.02'};

caxis(ax3, [0 0.02]);

plot(ax3, spec_obs, sens_obs, '*', ...
    'Color', starColor, ...
    'MarkerSize', 42, ...
    'LineWidth', 4);

xlabel(ax3, 'Specificity', 'FontSize', labelFS);
ylabel(ax3, 'Sensitivity', 'FontSize', labelFS);

xlim(ax3, [0 1]);
ylim(ax3, [0 1]);

xticks(ax3, 0:0.2:1);
yticks(ax3, 0:0.2:1);
xticklabels(ax3, {'0', '0.2', '0.4', '0.6', '0.8', '1'});
yticklabels(ax3, {'0', '0.2', '0.4', '0.6', '0.8', '1'});

axis(ax3, 'square');
box(ax3, 'on');
set(ax3, 'FontSize', tickFS, 'LineWidth', 1.2);

set(f3, 'Renderer', 'painters');
set(f3, 'PaperPositionMode', 'auto');
print(f3, 'Figure_4_D_3.svg', '-dsvg', '-painters');

disp('Saved Figure_4_D_1.svg, Figure_4_D_2.svg, and Figure_4_D_3.svg');