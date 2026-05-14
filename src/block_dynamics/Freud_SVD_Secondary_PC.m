%% Freud_SVD_Secondary_PC.m
% Diagnostic analysis of secondary SVD components in trial-level dynamics.
%
% This script computes the first two SVD-derived trial-position components
% for Death + Me and Life + Me trial conditions. It generates diagnostic
% plots comparing SI groups for PC1 and PC2, including bootstrap confidence
% intervals and per-trial t-test summaries.
%
% This script is exploratory/diagnostic only. It does not save or export
% any figures or data files.
%
% Required input:
%   Freud_Processed_BDIAT.mat
%
% Expected variables:
%   XF           : subjects x 360 processed log-reaction-time matrix
%   active_score : subjects x 1 group label, where 0 = SI- and 1 = SI+

clear; close all; clc;

%% ----------------------- User settings -----------------------
alpha_svd = 1.1;
alphaCI   = 0.05;
nBoot     = 2000;
rng(1);

labelFS  = 24;
tickFS   = 20;
legendFS = 12;

%% ----------------------- Load data -----------------------
load('Freud_Processed_BDIAT.mat');

active_score = active_score(:);
classes      = [0 1];
blockLen     = 20;

startSets = {1:40:321, 21:40:341};
setNames  = {'Death + Me', 'Life + Me'};

S = struct();

%% ----------------------- Main loop over conditions -----------------------
for sIdx = 1:numel(startSets)

    blockStarts = startSets{sIdx};
    setLabel    = setNames{sIdx};

    fprintf('\nProcessing %s\n', setLabel);

    blocksIdx = [];
    for s = blockStarts
        blocksIdx = [blocksIdx, s:(s + blockLen - 1)];
    end

    nBlocks = numel(blockStarts);
    X = XF(:, blocksIdx);
    [nRows, ~] = size(X);

    rowMean = zeros(nRows, blockLen);
    rowVar  = zeros(nRows, blockLen);
    rowEV   = zeros(nRows, blockLen, 2);

    for i = 1:nRows
        Xi = X(i, :);

        Xi_blocks = reshape(Xi, blockLen, nBlocks).';

        rowMean(i, :) = mean(Xi_blocks, 1);
        rowVar(i,  :) = var(Xi_blocks, 0, 1);

        Y_row = exp(-alpha_svd * Xi_blocks);
        C_row = Y_row' * Y_row;

        [~, ~, V] = svd(C_row);

        rowEV(i, :, 1) = V(:, 1).';
        rowEV(i, :, 2) = V(:, 2).';
    end

    M_class   = cell(1, numel(classes));
    V_class   = cell(1, numel(classes));
    EV1_class = cell(1, numel(classes));
    EV2_class = cell(1, numel(classes));

    for ci = 1:numel(classes)
        idxC = (active_score == classes(ci));

        M_class{ci}   = rowMean(idxC, :);
        V_class{ci}   = rowVar(idxC, :);
        EV1_class{ci} = rowEV(idxC, :, 1);
        EV2_class{ci} = rowEV(idxC, :, 2);
    end

    xAxis = 1:blockLen;

    n0 = size(M_class{1}, 1);
    n1 = size(M_class{2}, 1);

    [m0, lo0, hi0] = bootMeanCI(M_class{1}, alphaCI, nBoot);
    [m1, lo1, hi1] = bootMeanCI(M_class{2}, alphaCI, nBoot);

    [v0, vlo0, vhi0] = bootMeanCI(V_class{1}, alphaCI, nBoot);
    [v1, vlo1, vhi1] = bootMeanCI(V_class{2}, alphaCI, nBoot);

    [e0, elo0, ehi0] = bootMeanCI(EV1_class{1}, alphaCI, nBoot);
    [e1, elo1, ehi1] = bootMeanCI(EV1_class{2}, alphaCI, nBoot);

    [e20, e2lo0, e2hi0] = bootMeanCI(EV2_class{1}, alphaCI, nBoot);
    [e21, e2lo1, e2hi1] = bootMeanCI(EV2_class{2}, alphaCI, nBoot);

    %% Diagnostic bootstrap CI figure
    figure('Name', ['Secondary-PC bootstrap diagnostics — ', setLabel], 'Color', 'w');

    subplot(4, 1, 1); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [lo0 fliplr(hi0)], ...
        'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([xAxis fliplr(xAxis)], [lo1 fliplr(hi1)], ...
        'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(xAxis, m0, '-o', 'LineWidth', 1.5);
    plot(xAxis, m1, '-s', 'LineWidth', 1.5);
    xlabel('Trial position');
    ylabel('Mean');
    title([setLabel, ': mean trial-position response']);
    legend(sprintf('SI- mean (n=%d)', n0), sprintf('SI+ mean (n=%d)', n1), ...
        'Location', 'best');

    subplot(4, 1, 2); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [vlo0 fliplr(vhi0)], ...
        'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([xAxis fliplr(xAxis)], [vlo1 fliplr(vhi1)], ...
        'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(xAxis, v0, '-o', 'LineWidth', 1.5);
    plot(xAxis, v1, '-s', 'LineWidth', 1.5);
    xlabel('Trial position');
    ylabel('Variance');
    title([setLabel, ': trial-position variance']);
    legend('SI- mean', 'SI+ mean', 'Location', 'best');

    subplot(4, 1, 3); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [elo0 fliplr(ehi0)], ...
        'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([xAxis fliplr(xAxis)], [elo1 fliplr(ehi1)], ...
        'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(xAxis, e0, '-o', 'LineWidth', 1.5);
    plot(xAxis, e1, '-s', 'LineWidth', 1.5);
    xlabel('Trial position');
    ylabel('PC1 entry');
    title([setLabel, ': first SVD component']);
    legend('SI- mean', 'SI+ mean', 'Location', 'best');

    subplot(4, 1, 4); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [e2lo0 fliplr(e2hi0)], ...
        'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([xAxis fliplr(xAxis)], [e2lo1 fliplr(e2hi1)], ...
        'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(xAxis, e20, '-o', 'LineWidth', 1.5);
    plot(xAxis, e21, '-s', 'LineWidth', 1.5);
    xlabel('Trial position');
    ylabel('PC2 entry');
    title([setLabel, ': second SVD component']);
    legend('SI- mean', 'SI+ mean', 'Location', 'best');

    %% Per-trial t-test diagnostics
    p_mean = nan(1, blockLen);
    p_var  = nan(1, blockLen);
    p_ev1  = nan(1, blockLen);
    p_ev2  = nan(1, blockLen);

    for k = 1:blockLen
        [~, p_mean(k)] = ttest2(M_class{1}(:, k),   M_class{2}(:, k),   'Vartype', 'unequal');
        [~, p_var(k)]  = ttest2(V_class{1}(:, k),   V_class{2}(:, k),   'Vartype', 'unequal');
        [~, p_ev1(k)]  = ttest2(EV1_class{1}(:, k), EV1_class{2}(:, k), 'Vartype', 'unequal');
        [~, p_ev2(k)]  = ttest2(EV2_class{1}(:, k), EV2_class{2}(:, k), 'Vartype', 'unequal');
    end

    alpha      = 0.05;
    bonf_alpha = alpha / blockLen;

    q_mean = fdr_bh(p_mean);
    q_var  = fdr_bh(p_var);
    q_ev1  = fdr_bh(p_ev1);
    q_ev2  = fdr_bh(p_ev2);

    figure('Name', ['Secondary-PC t-test diagnostics — ', setLabel], 'Color', 'w');

    subplot(4, 1, 1);
    semilogy(xAxis, p_mean, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_mean, '-o', 'LineWidth', 1.0);
    yline(alpha, '--', 'alpha=0.05');
    yline(bonf_alpha, ':', 'Bonferroni');
    grid on;
    xlabel('Trial position');
    ylabel('p / q-value');
    title([setLabel, ': group test on mean']);
    legend('p', 'q_{FDR}', '0.05', 'Bonferroni', 'Location', 'best');

    subplot(4, 1, 2);
    semilogy(xAxis, p_var, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_var, '-o', 'LineWidth', 1.0);
    yline(alpha, '--', 'alpha=0.05');
    yline(bonf_alpha, ':', 'Bonferroni');
    grid on;
    xlabel('Trial position');
    ylabel('p / q-value');
    title([setLabel, ': group test on variance']);
    legend('p', 'q_{FDR}', '0.05', 'Bonferroni', 'Location', 'best');

    subplot(4, 1, 3);
    semilogy(xAxis, p_ev1, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_ev1, '-o', 'LineWidth', 1.0);
    yline(alpha, '--', 'alpha=0.05');
    yline(bonf_alpha, ':', 'Bonferroni');
    grid on;
    xlabel('Trial position');
    ylabel('p / q-value');
    title([setLabel, ': group test on PC1']);
    legend('p', 'q_{FDR}', '0.05', 'Bonferroni', 'Location', 'best');

    subplot(4, 1, 4);
    semilogy(xAxis, p_ev2, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_ev2, '-o', 'LineWidth', 1.0);
    yline(alpha, '--', 'alpha=0.05');
    yline(bonf_alpha, ':', 'Bonferroni');
    grid on;
    xlabel('Trial position');
    ylabel('p / q-value');
    title([setLabel, ': group test on PC2']);
    legend('p', 'q_{FDR}', '0.05', 'Bonferroni', 'Location', 'best');

    %% Mixed-effects model diagnostic
    [SubIdx, TrialIdx] = ndgrid(1:nRows, 1:blockLen);

    tbl_mean = table( ...
        categorical(SubIdx(:)), ...
        categorical(active_score(SubIdx(:))), ...
        TrialIdx(:), ...
        rowMean(:), ...
        'VariableNames', {'Subject', 'Group', 'Trial', 'ScaleRT'});

    lme_mean = fitlme(tbl_mean, 'ScaleRT ~ Group*Trial + (1|Subject)');

    fprintf('\nMixed-effects model for %s:\n', setLabel);
    disp(anova(lme_mean));

    plannedTrials = 1:6;
    contrast_pvals = zeros(numel(plannedTrials), 1);

    for tt = 1:numel(plannedTrials)
        tval = plannedTrials(tt);
        H = [0 1 0 tval];
        contrast_pvals(tt) = coefTest(lme_mean, H);
    end

    contrast_qvals = fdr_bh(contrast_pvals);

    fprintf('\nPlanned contrasts for %s, trials 1-6:\n', setLabel);
    disp(table(plannedTrials(:), contrast_pvals, contrast_qvals, ...
        'VariableNames', {'Trial', 'pValue', 'qValue'}));

    %% Store quantities for summary diagnostic figures
    if strcmp(setLabel, 'Death + Me')
        S.Death.xAxis = xAxis;
        S.Death.m0 = m0; S.Death.lo0 = lo0; S.Death.hi0 = hi0;
        S.Death.m1 = m1; S.Death.lo1 = lo1; S.Death.hi1 = hi1;

        S.Death.e0 = e0; S.Death.elo0 = elo0; S.Death.ehi0 = ehi0;
        S.Death.e1 = e1; S.Death.elo1 = elo1; S.Death.ehi1 = ehi1;

        S.Death.e20 = e20; S.Death.e2lo0 = e2lo0; S.Death.e2hi0 = e2hi0;
        S.Death.e21 = e21; S.Death.e2lo1 = e2lo1; S.Death.e2hi1 = e2hi1;

        S.Death.n0 = n0; S.Death.n1 = n1;

    elseif strcmp(setLabel, 'Life + Me')
        S.Life.xAxis = xAxis;
        S.Life.m0 = m0; S.Life.lo0 = lo0; S.Life.hi0 = hi0;
        S.Life.m1 = m1; S.Life.lo1 = lo1; S.Life.hi1 = hi1;
        S.Life.n0 = n0; S.Life.n1 = n1;
    end
end

%% ----------------------- Summary diagnostic figure: PC1 -----------------------
assert(isfield(S, 'Death') && isfield(S, 'Life'), ...
    'Summary figures require both Death + Me and Life + Me results.');

xAxis = S.Death.xAxis;

figure('Name', 'Diagnostic summary: PC1', 'Color', 'w', 'Position', [100 100 900 900]);

subplot(3, 1, 1); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial position', 'FontSize', labelFS);
ylabel('Mean response', 'FontSize', labelFS);
set(gca, 'FontSize', tickFS);
legend({sprintf('SI- mean (n=%d)', S.Death.n0), sprintf('SI+ mean (n=%d)', S.Death.n1)}, ...
    'FontSize', legendFS, 'Location', 'best');
title('');

subplot(3, 1, 2); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial position', 'FontSize', labelFS);
ylabel('Mean response', 'FontSize', labelFS);
set(gca, 'FontSize', tickFS);
legend({sprintf('SI- mean (n=%d)', S.Life.n0), sprintf('SI+ mean (n=%d)', S.Life.n1)}, ...
    'FontSize', legendFS, 'Location', 'best');
title('');

subplot(3, 1, 3); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.elo0 fliplr(S.Death.ehi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([xAxis fliplr(xAxis)], [S.Death.elo1 fliplr(S.Death.ehi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(xAxis, S.Death.e0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.e1, '-s', 'LineWidth', 1.5);
xlabel('Trial position', 'FontSize', labelFS);
ylabel('PC1 entry', 'FontSize', labelFS);
set(gca, 'FontSize', tickFS);
legend({sprintf('SI- mean (n=%d)', S.Death.n0), sprintf('SI+ mean (n=%d)', S.Death.n1)}, ...
    'FontSize', legendFS, 'Location', 'best');
title('');

%% ----------------------- Summary diagnostic figure: PC2 -----------------------
figure('Name', 'Diagnostic summary: PC2', 'Color', 'w', 'Position', [120 120 900 900]);

subplot(3, 1, 1); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial position', 'FontSize', labelFS);
ylabel('Mean response', 'FontSize', labelFS);
set(gca, 'FontSize', tickFS);
legend({sprintf('SI- mean (n=%d)', S.Death.n0), sprintf('SI+ mean (n=%d)', S.Death.n1)}, ...
    'FontSize', legendFS, 'Location', 'best');
title('');

subplot(3, 1, 2); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial position', 'FontSize', labelFS);
ylabel('Mean response', 'FontSize', labelFS);
set(gca, 'FontSize', tickFS);
legend({sprintf('SI- mean (n=%d)', S.Life.n0), sprintf('SI+ mean (n=%d)', S.Life.n1)}, ...
    'FontSize', legendFS, 'Location', 'best');
title('');

subplot(3, 1, 3); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.e2lo0 fliplr(S.Death.e2hi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([xAxis fliplr(xAxis)], [S.Death.e2lo1 fliplr(S.Death.e2hi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(xAxis, S.Death.e20, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.e21, '-s', 'LineWidth', 1.5);
xlabel('Trial position', 'FontSize', labelFS);
ylabel('PC2 entry', 'FontSize', labelFS);
set(gca, 'FontSize', tickFS);
legend({sprintf('SI- mean (n=%d)', S.Death.n0), sprintf('SI+ mean (n=%d)', S.Death.n1)}, ...
    'FontSize', legendFS, 'Location', 'best');
title('');

disp('Generated secondary-PC diagnostic plots. No files were saved.');

%% ========================================================================
%% Local functions
%% ========================================================================

function [m, lo, hi] = bootMeanCI(A, alphaCI, nBoot)
%BOOTMEANCI Bootstrap percentile confidence interval for column means.
%
% Inputs:
%   A       : subjects x trial-position matrix
%   alphaCI : alpha level for confidence interval
%   nBoot   : number of bootstrap resamples
%
% Outputs:
%   m  : empirical column mean
%   lo : lower bootstrap percentile bound
%   hi : upper bootstrap percentile bound

    A = A(~any(isnan(A), 2), :);
    [n, T] = size(A);

    m = mean(A, 1);

    bootMeans = zeros(nBoot, T);
    for b = 1:nBoot
        idx = randi(n, [n, 1]);
        bootMeans(b, :) = mean(A(idx, :), 1);
    end

    lo = prctile(bootMeans, 100 * (alphaCI / 2), 1);
    hi = prctile(bootMeans, 100 * (1 - alphaCI / 2), 1);
end

function q = fdr_bh(p)
%FDR_BH Benjamini-Hochberg FDR correction.
%
% Input:
%   p : vector of p-values
%
% Output:
%   q : vector of FDR-adjusted q-values

    p = p(:);
    m = numel(p);

    [p_sorted, sortIdx] = sort(p);
    ranks = (1:m)';

    q_sorted = p_sorted .* m ./ ranks;

    for i = m-1:-1:1
        q_sorted(i) = min(q_sorted(i), q_sorted(i + 1));
    end

    q = zeros(size(p));
    q(sortIdx) = q_sorted;
end