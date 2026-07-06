%% Freud_Compare_Preprocessing.m
clear; close all; clc;

set(groot, 'DefaultAxesFontSize', 28);
set(groot, 'DefaultTextFontSize', 28);
set(groot, 'DefaultLegendFontSize', 28);

addpath('C:\Users\prajaei\Desktop\PNAS\Github_clean\Github\src\block_dynamics');

%% ----------------------- 1. Load Original Data -------------------------
table_path = 'Freud_Cohort_N80.xlsx';
tx = readtable(table_path);
reaction_time = tx{:, 2:361};
active_score  = tx.SI_label;

rowsToRemove = [6, 48, 72];
active_score(rowsToRemove)      = [];
reaction_time(rowsToRemove, :)  = [];

N        = length(active_score);
n_trials = 360;

%% ----------------------- 2. Select Target Participant ------------------
target_subject = 70;

temp_rt = reaction_time(target_subject, :)';

ind_too_fast = find(temp_rt > 0   & temp_rt < 350);
ind_too_slow = find(temp_rt > 2000);
ind_missing  = find(temp_rt == 0  | isnan(temp_rt));
ind_above    = union(ind_too_slow, ind_missing);
n_invalid    = length(ind_too_fast) + length(ind_above);

fprintf('\n======================================================\n');
fprintf('---> SELECTED SUBJECT ID: %d (Invalid Trials: %d) <---', ...
        target_subject, n_invalid);
fprintf('\n======================================================\n\n');
fprintf('Subject %d trial breakdown:\n',   target_subject);
fprintf('  RT < 350 ms   (dots below) : %d trials\n', length(ind_too_fast));
fprintf('  RT > 2000 ms  (x above)    : %d trials\n', length(ind_too_slow));
fprintf('  Missing / 0   (x above)    : %d trials\n', length(ind_missing));
fprintf('  Total above                : %d trials\n', length(ind_above));
valid_rt = temp_rt(temp_rt > 0 & ~isnan(temp_rt));
fprintf('  Valid RT range: [%.1f, %.1f] ms\n\n', min(valid_rt), max(valid_rt));

%% ----------------------- 3. Load Preprocessed Datasets -----------------
load('Freud_Processed_BDIAT_Compass.mat',     'XF', 'active_score');
XF_Compass = XF;
labels     = active_score(:);

load('Freud_Processed_BDIAT_Alternative.mat', 'XF');
XF_Alt = XF;

classes = unique(labels(:));
y_label = double(labels == classes(2));

%% ----------------------- 4. Shared Appearance --------------------------
color_comp = [0.00, 0.60, 0.50];
color_alt  = [0.80, 0.60, 0.70];
color_dot  = [0.00, 0.45, 0.70];
color_x    = [0.90, 0.60, 0.00];
color_dash = [0.50, 0.50, 0.50];
color_raw  = [0.00, 0.00, 0.00];

%% ----------------------- 5. Compute Plot Traces ------------------------
mean_comp = mean(XF_Compass(target_subject, :), 'omitnan');
mean_alt  = mean(XF_Alt(target_subject, :),     'omitnan');

plot_comp = XF_Compass(target_subject, :) - mean_comp + 7;
plot_alt  = XF_Alt(target_subject, :)     - mean_alt  + 7;

raw_rt = reaction_time(target_subject, :)';
raw_rt(raw_rt == 0 | isnan(raw_rt)) = NaN;
raw_log  = log(raw_rt);
plot_raw = raw_log - mean(raw_log, 'omitnan') + 7;

%% ----------------------- 6. Threshold Geometry -------------------------
thresh_low  = log(350)  - mean_alt + 7;
thresh_high = log(2000) - mean_alt + 7;
band_gap    = 0.12 * (thresh_high - thresh_low);

y_dots    = thresh_low  - band_gap;
y_x_row   = thresh_high + band_gap;
y_lim_bot = y_dots   - 0.5 * band_gap;
y_lim_top = y_x_row  + 0.5 * band_gap;

%% =======================================================================
%% SHARED DRAWING SUBROUTINE
%% =======================================================================
function draw_invalid_markers(n_trials, ind_too_fast, ind_above, ...
        thresh_low, thresh_high, y_dots, y_x_row, color_dot, color_x, color_dash)

    plot([1, n_trials], [thresh_low,  thresh_low],  '--', ...
         'Color', color_dash, 'LineWidth', 1.0, 'HandleVisibility', 'off');
    plot([1, n_trials], [thresh_high, thresh_high], '--', ...
         'Color', color_dash, 'LineWidth', 1.0, 'HandleVisibility', 'off');

    if ~isempty(ind_too_fast)
        scatter(ind_too_fast, repmat(y_dots,  length(ind_too_fast), 1), ...
                120, color_dot, 'o', 'filled', 'LineWidth', 0.8, ...
                'HandleVisibility', 'off');
    end
    if ~isempty(ind_above)
        scatter(ind_above, repmat(y_x_row, length(ind_above), 1), ...
                160, color_x, 'x', 'LineWidth', 1.2, ...
                'HandleVisibility', 'off');
    end
end

%% =======================================================================
%% MANUAL LEGEND SUBROUTINE
%% =======================================================================
function draw_manual_legend(ax_h, h_raw, h_trace, trace_label, color_dot, color_x)

    ax_ylim = get(ax_h, 'YLim');
    ax_xlim = get(ax_h, 'XLim');
    x_range = ax_xlim(2) - ax_xlim(1);
    y_range = ax_ylim(2) - ax_ylim(1);

    line_x1 = ax_xlim(2) + x_range * 0.050;
    line_x2 = ax_xlim(2) + x_range * 0.067;
    glyph_x = (line_x1 + line_x2) / 2;
    text_x  = ax_xlim(2) + x_range * 0.075;

    row1_y = ax_ylim(1) + y_range * 0.550;
    row2_y = ax_ylim(1) + y_range * 0.483;
    row3_y = ax_ylim(1) + y_range * 0.416;
    row4_y = ax_ylim(1) + y_range * 0.349;

    % Row 1: Raw RT line
    raw_color = get(h_raw, 'Color');
    raw_lw    = get(h_raw, 'LineWidth');
    plot(ax_h, [line_x1, line_x2], [row1_y, row1_y], '-', ...
        'Color', raw_color, 'LineWidth', raw_lw, ...
        'Clipping', 'off', 'HandleVisibility', 'off');
    text(ax_h, text_x, row1_y, 'Raw RT (log-transformed)', ...
        'VerticalAlignment', 'middle', ...
        'HorizontalAlignment', 'left', 'Clipping', 'off');

    % Row 2: Preprocessing trace line
    trace_color = get(h_trace, 'Color');
    trace_lw    = get(h_trace, 'LineWidth');
    plot(ax_h, [line_x1, line_x2], [row2_y, row2_y], '-', ...
        'Color', trace_color, 'LineWidth', trace_lw, ...
        'Clipping', 'off', 'HandleVisibility', 'off');
    text(ax_h, text_x, row2_y, trace_label, ...
        'VerticalAlignment', 'middle', ...
        'HorizontalAlignment', 'left', 'Clipping', 'off');

    % Row 3: Missing RT — orange x
    plot(ax_h, glyph_x, row3_y, 'x', ...
        'Color', color_x, 'MarkerSize', 18, 'LineWidth', 1.6, ...
        'Clipping', 'off', 'HandleVisibility', 'off');
    text(ax_h, text_x, row3_y, 'Missing RT trial', ...
        'VerticalAlignment', 'middle', ...
        'HorizontalAlignment', 'left', 'Clipping', 'off');

    % Row 4: Censored RT — blue dot
    scatter(ax_h, glyph_x, row4_y, 100, color_dot, 'o', 'filled', ...
        'Clipping', 'off', 'HandleVisibility', 'off');
    text(ax_h, text_x, row4_y, 'Censored RT trial', ...
        'VerticalAlignment', 'middle', ...
        'HorizontalAlignment', 'left', 'Clipping', 'off');
end

%% =======================================================================
%% PLOT 1: Raw RT  +  Primary (COMPASS) preprocessing
%% =======================================================================
fig1 = figure('Color', 'w', 'Position', [100, 100, 2000, 2000]);
hold on;

h_raw1  = plot(1:n_trials, plot_raw,  'Color', color_raw,  'LineWidth', 0.9);
h_comp1 = plot(1:n_trials, plot_comp, 'Color', color_comp, 'LineWidth', 1.8);
uistack(h_raw1, 'top');

draw_invalid_markers(n_trials, ind_too_fast, ind_above, ...
    thresh_low, thresh_high, y_dots, y_x_row, ...
    color_dot, color_x, color_dash);

xlim([1, 360]);  ylim([y_lim_bot, y_lim_top]);
set(gca, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.2);
xlabel('Trial');
ylabel('Latent State (Log RT)');
set(gca, 'Position', [0.08, 0.11, 0.40, 0.55]);

ax1 = gca;
draw_manual_legend(ax1, h_raw1, h_comp1, ...
    'Primary: state-space posterior mean', color_dot, color_x);
set(fig1, 'PaperPositionMode', 'auto');
print(fig1, 'Figure_S6_A.svg', '-dsvg', '-painters');

%% =======================================================================
%% PLOT 2: Raw RT  +  Secondary (Alternative) preprocessing
%% =======================================================================
fig2 = figure('Color', 'w', 'Position', [120, 120, 2000, 2000]);
hold on;

h_raw2 = plot(1:n_trials, plot_raw, 'Color', color_raw, 'LineWidth', 0.9);
h_alt2 = plot(1:n_trials, plot_alt, 'Color', color_alt, 'LineWidth', 1.8);
uistack(h_raw2, 'top');

draw_invalid_markers(n_trials, ind_too_fast, ind_above, ...
    thresh_low, thresh_high, y_dots, y_x_row, ...
    color_dot, color_x, color_dash);

xlim([1, 360]);  ylim([y_lim_bot, y_lim_top]);
set(gca, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.2);
xlabel('Trial');
ylabel('Latent State (Log RT)');
set(gca, 'Position', [0.08, 0.11, 0.40, 0.55]);

ax2 = gca;
draw_manual_legend(ax2, h_raw2, h_alt2, ...
    'Secondary: local imputation robustness check', color_dot, color_x);

set(fig2, 'PaperPositionMode', 'auto');
print(fig2, 'Figure_S6_B.svg', '-dsvg', '-painters');

%% =======================================================================
%% PLOT 3: Scatter
%% =======================================================================
comp_all = XF_Compass - mean(XF_Compass, 2, 'omitnan') + 7;
alt_all  = XF_Alt     - mean(XF_Alt,     2, 'omitnan') + 7;

x_sc = comp_all(:);
y_sc = alt_all(:);

p_fit  = polyfit(x_sc, y_sc, 1);
y_fit  = polyval(p_fit, x_sc);
R2     = 1 - sum((y_sc - y_fit).^2) / sum((y_sc - mean(y_sc)).^2);
x_line = [min(x_sc), max(x_sc)];
y_line = polyval(p_fit, x_line);

fprintf('Trial-level R² (Primary vs Secondary): %.4f\n', R2);

fig3 = figure('Color', 'w', 'Position', [140, 140, 1700, 1500]);
ax3  = axes('Parent', fig3);
set(ax3, 'Position', [0.13, 0.13, 0.72, 0.74]);  % add this line
hold(ax3, 'on');

scatter(ax3, x_sc, y_sc, 70, [0.55 0.55 0.55], 'o', 'filled', ...
        'MarkerFaceAlpha', 0.90, 'MarkerEdgeAlpha', 0);

plot(ax3, x_line, y_line, 'k-', 'LineWidth', 1.8, 'HandleVisibility', 'off');
id_lim = [min([x_sc; y_sc]), max([x_sc; y_sc])];
plot(ax3, id_lim, id_lim, '--', 'Color', [0.6 0.6 0.6], ...
     'LineWidth', 1.2, 'HandleVisibility', 'off');

text(ax3, 0.05, 0.93, sprintf('R^{2} = %.4f', R2), ...
     'Units', 'normalized');

set(ax3, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.2);
xlabel(ax3, 'Primary: state-space posterior mean');
ylabel(ax3, 'Secondary: local imputation robustness check');

set(fig3, 'PaperPositionMode', 'auto');
print(fig3, 'Figure_S6_C.svg', '-dsvg', '-painters');

%% =======================================================================
%% PREPARE DATA FOR CLASSIFIER (J=2, Learned)
%% =======================================================================
m = 9; p = 20;
starts = 1:40:321;
idx    = cell2mat(arrayfun(@(s) s:(s+19), starts, 'UniformOutput', false));
alpha_alt = 0.99;

Xcells_compass = cell(N, 1);
Xcells_alt     = cell(N, 1);
for ii = 1:N
    x_comp             = XF_Compass(ii, idx);
    Xcells_compass{ii} = exp(-alpha_alt * reshape(x_comp, [p, m])');

    x_alt              = XF_Alt(ii, idx);
    Xcells_alt{ii}     = exp(-alpha_alt * reshape(x_alt, [p, m])');
end

cfg = struct();
cfg.rng_seed       = 42;
cfg.altIters       = 6;
cfg.vGradSteps     = 12;
cfg.vStepSize      = 0.15;
cfg.orthOn         = true;
cfg.cvFolds        = 5;
cfg.useOneSE       = true;
cfg.balanceWeights = true;
cfg.standardize    = true;
cfg.J              = 2;

%% =======================================================================
%% RUN LOOCV CLASSIFIERS
%% =======================================================================
fprintf('Running LOOCV (J=2, Learned) on COMPASS data...\n');
rng(cfg.rng_seed, 'twister');
res_compass = Freud_Model_CrossVal_Joint(Xcells_compass, y_label, cfg);
[fpr_comp, tpr_comp, auc_comp] = compute_roc_and_metrics(res_compass.p_all(:), y_label(:));
fprintf('  COMPASS AUC = %.3f\n', auc_comp);

fprintf('\nRunning LOOCV (J=2, Learned) on Alternative data...\n');
rng(cfg.rng_seed, 'twister');
res_alt = Freud_Model_CrossVal_Joint(Xcells_alt, y_label, cfg);
[fpr_alt, tpr_alt, auc_alt] = compute_roc_and_metrics(res_alt.p_all(:), y_label(:));
fprintf('  Alternative AUC = %.3f\n\n', auc_alt);

%% =======================================================================
%% PLOT 4: ROC Comparison
%% =======================================================================
fig4 = figure('Color', 'w', 'Position', [150, 150, 550, 500]);
ax4  = axes('Parent', fig4);
hold(ax4, 'on');

plot(ax4, [0 1], [0 1], 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
plot(ax4, fpr_alt,  tpr_alt,  'Color', color_alt,  'LineWidth', 2.5, ...
    'DisplayName', sprintf('Secondary (AUC=%.3f)', auc_alt));
plot(ax4, fpr_comp, tpr_comp, 'Color', color_comp, 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Primary (AUC=%.3f)', auc_comp));

axis(ax4, 'square');
xlim(ax4, [0 1]);  ylim(ax4, [0 1]);
set(ax4, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.2);
xlabel(ax4, 'False Positive Rate');
ylabel(ax4, 'True Positive Rate');
lg4 = legend(ax4, 'Location', 'southeast');
set(lg4, 'Box', 'off');
set(lg4, 'FontSize', 20);
set(fig4, 'PaperPositionMode', 'auto');
print(fig4, 'Figure_S6_D.svg', '-dsvg', '-painters');

%% =======================================================================
%% HELPER FUNCTION
%% =======================================================================
function [fpr, tpr, auc, thrStar, sensStar, specStar, balAccStar] = ...
         compute_roc_and_metrics(scores, labels_bin)

    scores     = scores(:);
    labels_bin = labels_bin(:);

    if any(isnan(scores))
        keep       = ~isnan(scores);
        scores     = scores(keep);
        labels_bin = labels_bin(keep);
    end

    usedPerfcurve = false;
    try
        [fpr, tpr, thr, auc] = perfcurve(labels_bin, scores, 1);
        usedPerfcurve = true;
    catch
        thr = linspace(0, 1, 401);
        tpr = zeros(size(thr));  fpr = zeros(size(thr));
        P   = sum(labels_bin == 1);
        Nn  = sum(labels_bin == 0);
        for k = 1:numel(thr)
            yhat   = scores >= thr(k);
            tp     = sum(yhat == 1 & labels_bin == 1);
            fp     = sum(yhat == 1 & labels_bin == 0);
            tpr(k) = tp / max(P,  1);
            fpr(k) = fp / max(Nn, 1);
        end
        [fpr, ord] = sort(fpr);
        tpr = tpr(ord);  thr = thr(ord);
        auc = trapz(fpr, tpr);
    end

    [~, kStar] = max(tpr - fpr);
    thrStar    = thr(kStar);
    if usedPerfcurve && ~isfinite(thrStar), thrStar = 0.5; end

    yhatStar   = scores >= thrStar;
    tp         = sum(yhatStar == 1 & labels_bin == 1);
    fn         = sum(yhatStar == 0 & labels_bin == 1);
    tn         = sum(yhatStar == 0 & labels_bin == 0);
    fp         = sum(yhatStar == 1 & labels_bin == 0);
    sensStar   = tp / max(tp + fn, 1);
    specStar   = tn / max(tn + fp, 1);
    balAccStar = 0.5 * (sensStar + specStar);
end