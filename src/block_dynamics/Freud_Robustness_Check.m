%% Freud_Robustness_Check.m
% Robustness analysis for block-level autocorrelation and RhythmIndex results.
%
% This script repeats the block-level ACF and shuffle analyses using the
% short-block processed dataset. It generates robustness panels:
%
%   Figure_2_F.svg : short-block block-level ACF
%   Figure_2_G.svg : destroy-ABAB null distribution for group RhythmIndex
%
% Required input:
%   Freud_Processed_BDIAT_Short.mat
%
% Required helpers:
%   Freud_Autocorr_Advanced_2.m
%   Freud_Autocorr_Advanced.m

clear all
close all

% Load short-block processed data
load('Freud_Processed_BDIAT_Short.mat');
b_len = 10;
b_num = 36;

X = exp(XF);
G = active_score;

% ===================== Figure 2F source =====================
out = Freud_Autocorr_Advanced_2(X, G, ...
    'DoTrialACF', false, 'DoBlockACF', true, ...
    'MaxLagBlock', 12, 'BlockSummary', 'mean', 'DetrendBlocks', true, ...
    'UseLog', false, 'BlockLen', b_len, 'NumBlocks', b_num);

% ===================== Camera-ready formatting + export: Figure 2F =====================
fig2f_handle = gcf;     % block ACF figure from Freud_Autocorr_Advanced_2

labelFS  = 24;
tickFS   = 20;
legendFS = 20;

figure(fig2f_handle);
set(gcf, 'Color', 'w');
set(gcf, 'Renderer', 'painters');

ax = gca;
grid(ax, 'off');
box(ax, 'off');

title(ax, '');
xlabel(ax, 'Lag (blocks)', 'FontSize', labelFS);
ylabel(ax, 'Autocorrelation', 'FontSize', labelFS);

xlim(ax, [1 12]);
ylim(ax, [-0.4 0.3]);
xticks(ax, 1:12);
ax.XTickLabelRotation = 0;
yticks(ax, [-0.25 0 0.25]);
set(ax, 'FontSize', tickFS, 'LineWidth', 1);

lg = legend(ax, {'Lower SI (C-SSRS)', 'Higher SI (C-SSRS)'}, ...
    'Location', 'northeast');
set(lg, 'FontSize', legendFS, 'Box', 'off');

set(gcf, 'PaperPositionMode', 'auto');
print(gcf, 'Figure_2_F.svg', '-dsvg', '-painters');

disp('Saved Figure_2_F.svg');

% ===================== Robustness shuffle analysis for Figure 2G =====================
out = Freud_Autocorr_Advanced(X, G, ...
    'MaxLagBlock', 12, ...
    'RhythmMaxLag', 8, ...
    'NumPerm', 10000, ...
    'NumShufflePreserve', 10000, ...
    'NumShuffleDestroy', 10000, ...
    'ShuffleDestroyMode', 'permuteBlocks', ...
    'FigureNamePrefix', 'MyTask', ...
    'BlockSummary', 'mean', ...
    'BlockLen', b_len, ...
    'NumBlocks', b_num);

% Export Figure 2G and print the participant-level RhythmIndex t-test.
camera_ready_export_figure_2g_and_ttest(out, G, 'Figure_2_G.svg');

function camera_ready_export_figure_2g_and_ttest(out, G, outFile)
%CAMERA_READY_EXPORT_FIGURE_2G_AND_TTEST Export Figure 2G and print RI t-test.
%
% Inputs:
%   out     : output structure from Freud_Autocorr_Advanced
%   G       : binary group labels, one per participant
%   outFile : output filename for Figure 2G
%
% Output:
%   Prints Welch two-sample t-test on participant-level RhythmIndex.
%   Saves Figure 2G: destroy-ABAB null distribution for group RhythmIndex.

labelFS  = 24;
tickFS   = 20;
legendFS = 20;

%% ===================== Participant-level t-test =====================
ri = out.rhythmIndex(:);
G  = G(:);

keep = isfinite(ri) & isfinite(G);
ri = ri(keep);
G  = G(keep);

g0 = ri(G == 0);
g1 = ri(G == 1);

[~, p_t, ci_t, stats_t] = ttest2(g0, g1, 'Vartype', 'unequal');

fprintf('\n[Robustness dataset: two-sample t-test on participant-level RhythmIndex]\n');
fprintf('  Mean RI (without active SI) = %.4f\n', mean(g0, 'omitnan'));
fprintf('  Mean RI (with active SI)    = %.4f\n', mean(g1, 'omitnan'));
fprintf('  t = %.4f, df = %.2f, p = %.5f\n', stats_t.tstat, stats_t.df, p_t);
fprintf('  95%% CI of difference = [%.4f, %.4f]\n', ci_t(1), ci_t(2));

%% ===================== Figure 2G: Destroy-ABAB null =====================
nullValsB = out.tests.shuffleDestroyAB.perm.groupDiff;
obsValB   = out.tests.shuffleDestroyAB.obs.groupDiff;

nullValsB = nullValsB(isfinite(nullValsB));
muNullB   = mean(nullValsB, 'omitnan');
sdNullB   = std(nullValsB, 'omitnan');

fB = figure('Color', 'w', 'Position', [100 100 900 520]);
set(fB, 'Renderer', 'painters');

axB = axes('Parent', fB);
hold(axB, 'on');
grid(axB, 'off');
box(axB, 'off');

hHistB = histogram(axB, nullValsB, 40, 'Normalization', 'pdf');

ylB = ylim(axB);
hObsB = plot(axB, [obsValB obsValB], ylB, 'r-', 'LineWidth', 3);
plot(axB, [muNullB muNullB], ylB, 'k--', 'LineWidth', 2);

xlabel(axB, 'RhythmIndex group difference (SI+ - SI-)', 'FontSize', labelFS);
ylabel(axB, 'Density', 'FontSize', labelFS);
set(axB, 'FontSize', tickFS, 'LineWidth', 1);
title(axB, '');

xl = xlim(axB);
xlim(axB, [-0.15 xl(2)]);

legStrB = sprintf('Permuted (mean = %.3f, std = %.3f)', muNullB, sdNullB);
lgB = legend(axB, [hObsB hHistB], {'Observed', legStrB}, ...
    'Location', 'northoutside', 'Orientation', 'horizontal');
set(lgB, 'FontSize', legendFS, 'Box', 'off');

set(fB, 'PaperPositionMode', 'auto');
print(fB, outFile, '-dsvg', '-painters');
close(fB);

disp(['Saved Figure_2_G: ' outFile]);
end