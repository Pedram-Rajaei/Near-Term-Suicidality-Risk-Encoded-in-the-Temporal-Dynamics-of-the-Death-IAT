%% Freud_Main_Block_Analysis.m
% Main block-level autocorrelation and rhythm analysis.
%
% This script generates the block-level Figure 2 panels from the processed
% BDIAT data. It computes block-level autocorrelation functions, RhythmIndex
% statistics, group-label permutation tests, and within-subject ABAB shuffle
% controls.
%
% Required input:
%   Freud_Processed_BDIAT.mat
%
% Required helper:
%   Freud_Autocorr_Advanced.m
%
% Outputs:
%   Figure_2_D.svg
%   Figure_2_E.svg
%   Freud_Main_Block_Analysis_Results.mat

clear; close all; clc;

%% ----------------------- Load data -----------------------
load('Freud_Processed_BDIAT.mat');

X = exp(XF);              % Convert processed log RT back to RT scale
G = active_score(:);      % 0 = SI-, 1 = SI+

%% ----------------------- Run block ACF and shuffle analyses -----------------------
outJan = Freud_Autocorr_Advanced(X, G, ...
    'MaxLagBlock', 8, ...
    'RhythmMaxLag', 8, ...
    'NumPerm', 10000, ...
    'NumShufflePreserve', 10000, ...
    'NumShuffleDestroy', 10000, ...
    'ShuffleDestroyMode', 'permuteBlocks', ...
    'FigureNamePrefix', 'MyTask', ...
    'BlockSummary', 'mean', ...
    'UseLog', false, ...
    'MakePlots', true);

% Freud_Autocorr_Advanced creates the block ACF figure. This script
% explicitly retrieves that figure handle before camera-ready export.
fig2d_handle = findobj('Type', 'figure', 'Name', 'MyTask - Block ACF');
assert(~isempty(fig2d_handle), 'Could not find Block ACF figure.');
fig2d_handle = fig2d_handle(1);

%% ----------------------- Participant-level null comparisons -----------------------
destroyNull = outJan.tests.shuffleDestroyAB.perSubject.nullRI;
destroyObs  = outJan.tests.shuffleDestroyAB.perSubject.obsRI;

preserveNull = outJan.tests.shufflePreserveAB.perSubject.nullRI;
preserveObs  = outJan.tests.shufflePreserveAB.perSubject.obsRI;

mean_destroy = mean(destroyNull, 2, 'omitnan');
mode_destroy = estimate_row_modes(destroyNull, 50);

mean_preserve = mean(preserveNull, 2, 'omitnan');
mode_preserve = estimate_row_modes(preserveNull, 50);

diff_mean_destroy = destroyObs - mean_destroy;
diff_mode_destroy = destroyObs - mode_destroy;

diff_mean_preserve = preserveObs - mean_preserve;
diff_mode_preserve = preserveObs - mode_preserve;

%% ----------------------- Summary statistics -----------------------
[~, p_md,  ~, stats_md]  = ttest(diff_mean_destroy);
[~, p_mod, ~, stats_mod] = ttest(diff_mode_destroy);

[~, p_mp,  ~, stats_mp]  = ttest(diff_mean_preserve);
[~, p_mop, ~, stats_mop] = ttest(diff_mode_preserve);

fprintf('\n===== Within-subject null comparison summaries =====\n');

fprintf('\nDestroy-ABAB null:\n');
fprintf('  Observed - null mean: t = %.4f, p = %.4g\n', stats_md.tstat, p_md);
fprintf('  Observed - null mode: t = %.4f, p = %.4g\n', stats_mod.tstat, p_mod);

fprintf('\nPreserve-ABAB null:\n');
fprintf('  Observed - null mean: t = %.4f, p = %.4g\n', stats_mp.tstat, p_mp);
fprintf('  Observed - null mode: t = %.4f, p = %.4g\n', stats_mop.tstat, p_mop);

%% ----------------------- Group comparison on RhythmIndex -----------------------
RI = outJan.rhythmIndex;
G  = active_score(:);

valid = isfinite(RI) & isfinite(G);

RI0 = RI(valid & (G == 0));   % SI-
RI1 = RI(valid & (G == 1));   % SI+

m0 = mean(RI0, 'omitnan');
m1 = mean(RI1, 'omitnan');

[~, p_t, ~, stats_ri] = ttest2(RI1, RI0, 'Vartype', 'unequal');

fprintf('\n===== Two-sample t-test on participant-level RhythmIndex =====\n');
fprintf('  Mean RI (SI-) = %.4f\n', m0);
fprintf('  Mean RI (SI+) = %.4f\n', m1);
fprintf('  t = %.4f, df = %.1f, p = %.4g\n', stats_ri.tstat, stats_ri.df, p_t);

%% ----------------------- Export Figure 2 panels only -----------------------
export_figure_2_panels(fig2d_handle, outJan.tests.shuffleDestroyAB, 'Figure_2_');

%% ----------------------- Save numerical results -----------------------
BlockResults = struct();

BlockResults.outJan = outJan;

BlockResults.destroy.meanNull = mean_destroy;
BlockResults.destroy.modeNull = mode_destroy;
BlockResults.destroy.diffMean = diff_mean_destroy;
BlockResults.destroy.diffMode = diff_mode_destroy;
BlockResults.destroy.ttestMean.p = p_md;
BlockResults.destroy.ttestMean.stats = stats_md;
BlockResults.destroy.ttestMode.p = p_mod;
BlockResults.destroy.ttestMode.stats = stats_mod;

BlockResults.preserve.meanNull = mean_preserve;
BlockResults.preserve.modeNull = mode_preserve;
BlockResults.preserve.diffMean = diff_mean_preserve;
BlockResults.preserve.diffMode = diff_mode_preserve;
BlockResults.preserve.ttestMean.p = p_mp;
BlockResults.preserve.ttestMean.stats = stats_mp;
BlockResults.preserve.ttestMode.p = p_mop;
BlockResults.preserve.ttestMode.stats = stats_mop;

BlockResults.rhythmIndex.RI0 = RI0;
BlockResults.rhythmIndex.RI1 = RI1;
BlockResults.rhythmIndex.meanRI0 = m0;
BlockResults.rhythmIndex.meanRI1 = m1;
BlockResults.rhythmIndex.ttest.p = p_t;
BlockResults.rhythmIndex.ttest.stats = stats_ri;

save('Freud_Main_Block_Analysis_Results.mat', 'BlockResults');

disp('Saved Figure_2_D.svg');
disp('Saved Figure_2_E.svg');
disp('Saved Freud_Main_Block_Analysis_Results.mat');

%% ========================================================================
%% Local functions
%% ========================================================================

function rowModes = estimate_row_modes(M, nBins)
%ESTIMATE_ROW_MODES Estimate histogram-based mode for each row.
%
% Inputs:
%   M     : matrix whose rows contain null samples
%   nBins : number of histogram bins
%
% Output:
%   rowModes : row-wise histogram-mode estimates

    n = size(M, 1);
    rowModes = nan(n, 1);

    for i = 1:n
        vals = M(i, :);
        vals = vals(isfinite(vals));

        if isempty(vals)
            continue;
        end

        [counts, edges] = histcounts(vals, nBins);
        [~, idx] = max(counts);
        rowModes(i) = mean(edges(idx:idx + 1));
    end
end

function export_figure_2_panels(fig2d_handle, destroy_stats, outPrefix)
%EXPORT_FIGURE_2_PANELS Export camera-ready Figure 2D and Figure 2E.
%
% Inputs:
%   fig2d_handle  : handle to the block-level ACF figure
%   destroy_stats : out.tests.shuffleDestroyAB structure
%   outPrefix     : output prefix, e.g., 'Figure_2_'

    labelFS  = 24;
    tickFS   = 20;
    legendFS = 20;

    %% -------------------- Figure 2D: Block ACF --------------------
    figure(fig2d_handle);
    set(gcf, 'Color', 'w');
    set(gcf, 'Renderer', 'painters');

    ax = gca;
    grid(ax, 'off');
    box(ax, 'off');

    title(ax, '');
    xlim(ax, [1 8]);
    ylim(ax, [-0.4 0.3]);

    xlabel(ax, 'Lag (blocks)', 'FontSize', labelFS);
    ylabel(ax, 'Autocorrelation', 'FontSize', labelFS);

    set(ax, 'FontSize', tickFS, 'LineWidth', 1);
    yticks(ax, [-0.5 -0.25 0 0.25 0.5]);

    lg = legend(ax, {'SI-', 'SI+'}, 'Location', 'northeast');
    set(lg, 'FontSize', legendFS, 'Box', 'off');

    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, [outPrefix 'D.svg'], '-dsvg', '-painters');

    %% -------------------- Figure 2E: Destroy-ABAB null --------------------
    nullVals = destroy_stats.perm.groupDiff;
    obsVal   = destroy_stats.obs.groupDiff;

    muNull = mean(nullVals, 'omitnan');
    sdNull = std(nullVals, 'omitnan');

    f = figure('Color', 'w', 'Position', [100 100 900 520]);
    set(f, 'Renderer', 'painters');

    ax2 = axes('Parent', f);
    hold(ax2, 'on');
    grid(ax2, 'off');
    box(ax2, 'off');

    hHist = histogram(ax2, nullVals, 40, 'Normalization', 'pdf');

    yl = ylim(ax2);

    hObs = plot(ax2, [obsVal obsVal], yl, 'r-', 'LineWidth', 3);
    plot(ax2, [muNull muNull], yl, 'k--', 'LineWidth', 2);

    xlabel(ax2, 'RhythmIndex group difference (SI+ - SI-)', ...
        'FontSize', labelFS);
    ylabel(ax2, 'Density', 'FontSize', labelFS);

    set(ax2, 'FontSize', tickFS, 'LineWidth', 1);

    title(ax2, '');
    axis(ax2, 'tight');

    xl = xlim(ax2);
    xlim(ax2, [-0.3 xl(2)]);

    legStrPerm = sprintf('Permuted (mean=%.3f, std=%.3f)', muNull, sdNull);

    lg2 = legend(ax2, [hObs hHist], {'Observed', legStrPerm}, ...
        'Location', 'northoutside', ...
        'Orientation', 'horizontal');
    set(lg2, 'FontSize', legendFS, 'Box', 'off');

    print(f, [outPrefix 'E.svg'], '-dsvg', '-painters');
    close(f);
end