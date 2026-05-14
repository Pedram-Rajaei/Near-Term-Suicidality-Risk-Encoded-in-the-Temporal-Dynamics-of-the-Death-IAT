%% Freud_PCA_Trial_Dynamics.m
% Trial-level reaction-time dynamics for Figure 3.
%
% This script:
%   1. Loads processed BDIAT trial-level data.
%   2. Computes trial-position summaries for:
%        - Death + Me trials
%        - Life + Me trials
%   3. Estimates bootstrap confidence intervals for SI- and SI+ groups.
%   4. Computes the first PC entry from the trial-position covariance
%      structure for each subject.
%   5. Exports the four Figure 3 panels separately:
%
%        Figure_3_A.svg
%        Figure_3_B.svg
%        Figure_3_C.svg
%        Figure_3_D.svg
%
% Required input:
%   Freud_Processed_BDIAT.mat
%
% Expected variables:
%   XF           : subjects x 360 processed log-reaction-time matrix
%   active_score : subjects x 1 group label, where 0 = SI- and 1 = SI+
%
% Outputs:
%   Figure_3_A.svg : Death + Me RT-scale dynamics
%   Figure_3_B.svg : Life + Me RT-scale dynamics
%   Figure_3_C.svg : Death + Me PC1 dynamics
%   Figure_3_D.svg : Life + Me PC1 dynamics

clear; close all; clc;

%% ----------------------- User settings -----------------------
alpha_svd = 1.1;
alphaCI   = 0.05;
nBoot     = 2000;
rng(1);

labelFS  = 24;
tickFS   = 20;
legendFS = 20;

%% ----------------------- Load data -----------------------
load('Freud_Processed_BDIAT.mat');

active_score = active_score(:);
classes      = [0 1];
blockLen     = 20;

% Trial-condition definitions.
startSets = {1:40:321, 21:40:341};
setNames  = {'Death + Me', 'Life + Me'};

S = struct();

%% ----------------------- Main analysis loop -----------------------
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
    rowEV   = zeros(nRows, blockLen);

    for i = 1:nRows
        Xi = X(i, :);

        Xi_blocks    = reshape(Xi, blockLen, nBlocks).';
        Xi_blocks_rt = exp(Xi_blocks);

        rowMean(i, :) = mean(Xi_blocks_rt, 1);

        Y_row = exp(-alpha_svd * Xi_blocks);
        C_row = Y_row' * Y_row;
        [~, ~, V] = svd(C_row);
        rowEV(i, :) = V(:, 1).';
    end

    M_class  = cell(1, numel(classes));
    EV_class = cell(1, numel(classes));

    for ci = 1:numel(classes)
        idxC = (active_score == classes(ci));
        M_class{ci}  = rowMean(idxC, :);
        EV_class{ci} = rowEV(idxC, :);
    end

    xAxis = 1:blockLen;

    [m0, lo0, hi0] = bootMeanCI(M_class{1}, alphaCI, nBoot);
    [m1, lo1, hi1] = bootMeanCI(M_class{2}, alphaCI, nBoot);

    [e0, elo0, ehi0] = bootMeanCI(EV_class{1}, alphaCI, nBoot);
    [e1, elo1, ehi1] = bootMeanCI(EV_class{2}, alphaCI, nBoot);

    n0 = size(M_class{1}, 1);
    n1 = size(M_class{2}, 1);

    if strcmp(setLabel, 'Death + Me')
        S.Death.xAxis = xAxis;
        S.Death.m0 = m0; S.Death.lo0 = lo0; S.Death.hi0 = hi0;
        S.Death.m1 = m1; S.Death.lo1 = lo1; S.Death.hi1 = hi1;
        S.Death.e0 = e0; S.Death.elo0 = elo0; S.Death.ehi0 = ehi0;
        S.Death.e1 = e1; S.Death.elo1 = elo1; S.Death.ehi1 = ehi1;
        S.Death.n0 = n0; S.Death.n1 = n1;

    elseif strcmp(setLabel, 'Life + Me')
        S.Life.xAxis = xAxis;
        S.Life.m0 = m0; S.Life.lo0 = lo0; S.Life.hi0 = hi0;
        S.Life.m1 = m1; S.Life.lo1 = lo1; S.Life.hi1 = hi1;
        S.Life.e0 = e0; S.Life.elo0 = elo0; S.Life.ehi0 = ehi0;
        S.Life.e1 = e1; S.Life.elo1 = elo1; S.Life.ehi1 = ehi1;
        S.Life.n0 = n0; S.Life.n1 = n1;
    end
end

%% ----------------------- Export Figure 3 panels separately -----------------------
assert(isfield(S, 'Death') && isfield(S, 'Life'), ...
    'Figure 3 requires both Death + Me and Life + Me results.');

xAxis = S.Death.xAxis;

%% ---------------- Figure_3_A: Death + Me RT-scale ----------------
figA = figure('Color', 'w', 'Position', [100 100 700 520]);
axA = axes('Parent', figA); hold(axA, 'on');

fill(axA, [xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill(axA, [xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

plot(axA, xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(axA, xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);

xlabel(axA, 'Trial index', 'FontSize', labelFS);
ylabel(axA, 'RT-scale', 'FontSize', labelFS);
set(axA, 'FontSize', tickFS);

ylA = ylim(axA);
yticks(axA, linspace(ylA(1), ylA(2), 6));
ytickformat(axA, '%.2f');

patch(axA, [3.5 6 6 3.5], [ylA(1) ylA(1) ylA(2) ylA(2)], ...
    [0.5 0.5 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

set(figA, 'PaperPositionMode', 'auto');
print(figA, 'Figure_3_A.svg', '-dsvg');
close(figA);

%% ---------------- Figure_3_B: Life + Me RT-scale ----------------
figB = figure('Color', 'w', 'Position', [100 100 700 520]);
axB = axes('Parent', figB); hold(axB, 'on');

fill(axB, [xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill(axB, [xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

plot(axB, xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(axB, xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);

xlabel(axB, 'Trial index', 'FontSize', labelFS);
ylabel(axB, 'RT-scale', 'FontSize', labelFS);
set(axB, 'FontSize', tickFS);

ylB = ylim(axB);
yticks(axB, linspace(ylB(1), ylB(2), 6));
ytickformat(axB, '%.2f');

patch(axB, [3.5 6 6 3.5], [ylB(1) ylB(1) ylB(2) ylB(2)], ...
    [0.5 0.5 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

lgd = legend(axB, {sprintf('SI- (n=%d)', S.Death.n0), ...
                   sprintf('SI+ (n=%d)', S.Death.n1)}, ...
    'FontSize', legendFS, 'Location', 'northeast');
lgd.Box = 'off';

set(figB, 'PaperPositionMode', 'auto');
print(figB, 'Figure_3_B.svg', '-dsvg');
close(figB);

%% ---------------- Figure_3_C: Death + Me PC1 ----------------
figC = figure('Color', 'w', 'Position', [100 100 700 520]);
axC = axes('Parent', figC); hold(axC, 'on');

fill(axC, [xAxis fliplr(xAxis)], [S.Death.elo0 fliplr(S.Death.ehi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill(axC, [xAxis fliplr(xAxis)], [S.Death.elo1 fliplr(S.Death.ehi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

plot(axC, xAxis, S.Death.e0, '-o', 'LineWidth', 1.5);
plot(axC, xAxis, S.Death.e1, '-s', 'LineWidth', 1.5);

xlabel(axC, 'Trial index', 'FontSize', labelFS);
ylabel(axC, 'PC1', 'FontSize', labelFS);
set(axC, 'FontSize', tickFS);

ylim(axC, [-0.26 -0.16]);
ylC = ylim(axC);
yticks(axC, linspace(ylC(1), ylC(2), 6));
ytickformat(axC, '%.2f');

patch(axC, [3.5 6 6 3.5], [ylC(1) ylC(1) ylC(2) ylC(2)], ...
    [0.5 0.5 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

add_qvalue_stars_only(axC, 4, 1);

set(figC, 'PaperPositionMode', 'auto');
print(figC, 'Figure_3_C.svg', '-dsvg');
close(figC);

%% ---------------- Figure_3_D: Life + Me PC1 ----------------
figD = figure('Color', 'w', 'Position', [100 100 700 520]);
axD = axes('Parent', figD); hold(axD, 'on');

fill(axD, [xAxis fliplr(xAxis)], [S.Life.elo0 fliplr(S.Life.ehi0)], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill(axD, [xAxis fliplr(xAxis)], [S.Life.elo1 fliplr(S.Life.ehi1)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

plot(axD, xAxis, S.Life.e0, '-o', 'LineWidth', 1.5);
plot(axD, xAxis, S.Life.e1, '-s', 'LineWidth', 1.5);

xlabel(axD, 'Trial index', 'FontSize', labelFS);
ylabel(axD, 'PC1', 'FontSize', labelFS);
set(axD, 'FontSize', tickFS);

ylim(axD, [-0.26 -0.16]);
ylD = ylim(axD);
yticks(axD, linspace(ylD(1), ylD(2), 6));
ytickformat(axD, '%.2f');

patch(axD, [3.5 6 6 3.5], [ylD(1) ylD(1) ylD(2) ylD(2)], ...
    [0.5 0.5 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

set(figD, 'PaperPositionMode', 'auto');
print(figD, 'Figure_3_D.svg', '-dsvg');
close(figD);

disp('Saved Figure_3_A.svg, Figure_3_B.svg, Figure_3_C.svg, and Figure_3_D.svg');

%% ----------------------- Local functions -----------------------
function [m, lo, hi] = bootMeanCI(A, alphaCI, nBoot)
%BOOTMEANCI Bootstrap percentile confidence interval for column means.

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

function add_qvalue_stars_only(ax, xQ, nQ)
%ADD_QVALUE_STARS_ONLY Add q-value significance stars to a target axis.

    if nargin < 3 || isempty(xQ)
        return;
    end

    yt = yticks(ax);
    if numel(yt) < 6
        yl = ylim(ax);
        yticks(ax, linspace(yl(1), yl(2), 6));
        yt = yticks(ax);
    end

    yQ = yt(5);

    for i = 1:numel(xQ)
        text(ax, xQ(i), yQ, repmat('*', 1, nQ(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 32, ...
            'FontWeight', 'bold', ...
            'Clipping', 'off');
    end
end