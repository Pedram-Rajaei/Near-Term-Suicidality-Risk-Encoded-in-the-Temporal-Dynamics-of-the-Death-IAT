%% Freud_PCA_Loading_Tests_S4.m
% Trial-wise statistical tests of PC1 and PC2 loading entries for Figure S4.
%
% This supplementary analysis computes per-subject SVD/PCA loading vectors
% within Death + Me blocks and performs trial-wise Welch two-sample tests
% between SI groups.
%
% The exported panels report Benjamini-Hochberg FDR-adjusted q-values:
%
%   Figure_S4_A.svg : q-values for PC1 loading entries
%   Figure_S4_B.svg : q-values for PC2 loading entries
%
% Required input:
%   Freud_Processed_BDIAT.mat
%
% Expected variables:
%   XF           : subjects x 360 processed reaction-time matrix
%   active_score : binary SI group label, where 0 = SI- and 1 = SI+clear all; close all; clc;

%% ----------------------- USER SETTINGS -----------------------
alpha_svd = 1.1;      % used in exp(-alpha_svd * Xi_blocks)
alphaCI   = 0.05;     % 95% bootstrap CI
nBoot     = 2000;     % bootstrap resamples
rng(1);               % reproducible bootstrap

%% ----------------------- LOAD DATA ---------------------------
load('Freud_Processed_BDIAT');     % expects XF (n x 360) and active_score in workspace
active_score = active_score(:);    % ensure column
classes      = [0 1];
blockLen     = 20;

% Two sets of column-block starts:
%   Set 1: 1:20, 41:60, ..., 321:340
%   Set 2: 21:40, 61:80, ..., 341:360
startSets = {1:40:321, 21:40:341};

% BDIAT condition labels.
setNames  = {'Death + Me', 'Life + Me'};

for sIdx = 1 %1:numel(startSets)

    blockStarts = startSets{sIdx};
    setLabel    = setNames{sIdx};

    fprintf('\nProcessing %s\n', setLabel);

    %% ---- Build column indices for this set of blocks ----
    blocksIdx = [];
    for s = blockStarts
        blocksIdx = [blocksIdx, s:(s + blockLen - 1)];
    end
    nBlocks = numel(blockStarts);          % should be 9

    % Extract data for these columns
    X = XF(:, blocksIdx);                  % n x (nBlocks*blockLen)
    [nRows, ~] = size(X);

    %% Compute subject-level trial-position summaries
    rowMean = zeros(nRows, blockLen);
    rowVar  = zeros(nRows, blockLen);
    rowEV   = zeros(nRows, blockLen);

    for i = 1:nRows
        Xi = X(i, :);                                      % 1 x (nBlocks*blockLen)

        % Reshape to nBlocks x blockLen (blocks x trial index)
        Xi_blocks = reshape(Xi, blockLen, nBlocks).';      % nBlocks x blockLen

        % Per-row mean & variance across blocks
        rowMean(i, :) = mean(Xi_blocks, 1);                % 1 x 20
        rowVar(i,  :) = var(Xi_blocks, 0, 1);              % 1 x 20

        % SVD-based first PC on Y = exp(-alpha_svd*X)
        Y_row = exp(-alpha_svd * Xi_blocks);               % nBlocks x 20
        C_row = Y_row' * Y_row;                            % 20 x 20
        [~, ~, V] = svd(C_row);
        v1 = V(:, 1);
        if v1(1) > 0
            v1 = -v1;
        end
        rowEV(i, :) = v1.';                                % 1 x 20
    end

    %% --------- Split these PER-ROW quantities by class ---------
    M_class  = cell(1, numel(classes));   % rowMean
    V_class  = cell(1, numel(classes));   % rowVar
    EV_class = cell(1, numel(classes));   % rowEV

    for ci = 1:numel(classes)
        c = classes(ci);
        idxC = (active_score == c);

        M_class{ci}  = rowMean(idxC, :);   % (n_c x 20)
        V_class{ci}  = rowVar(idxC,  :);   % (n_c x 20)
        EV_class{ci} = rowEV(idxC,  :);    % (n_c x 20)
    end

    xAxis = 1:blockLen;

    %% Trial-wise tests of PC1 and PC2 loading entries
    % Welch two-sample tests are performed at each trial position, followed by
    % Benjamini-Hochberg FDR correction across the 20 trial positions.

    % ---- Recompute per-subject PC1 and PC2 loading vectors (EXP space) ----
    rowEV1 = nan(nRows, blockLen);   % PC1 loadings (subjects x 20)
    rowEV2 = nan(nRows, blockLen);   % PC2 loadings (subjects x 20)

    for i = 1:nRows
        Xi = X(i, :);
        Xi_blocks = reshape(Xi, blockLen, nBlocks).';      % nBlocks x 20

        Y_row = exp(-alpha_svd * Xi_blocks);               % nBlocks x 20
        C_row = Y_row' * Y_row;                            % 20 x 20
        [~, ~, V] = svd(C_row);

        % PC1
        v1 = V(:,1);
        if v1(1) > 0, v1 = -v1; end
        rowEV1(i,:) = v1.';

        % PC2
        v2 = V(:,2);
        if v2(1) > 0, v2 = -v2; end
        rowEV2(i,:) = v2.';
    end

    % ---- Split PC1/PC2 loading entries by class ----
    idx0 = (active_score == 0);
    idx1 = (active_score == 1);

    EV1_g0 = rowEV1(idx0, :);   EV1_g1 = rowEV1(idx1, :);
    EV2_g0 = rowEV2(idx0, :);   EV2_g1 = rowEV2(idx1, :);

    % ---- Trial-wise Welch t-tests (PC1 + PC2) ----
    p_pc1 = nan(1, blockLen);
    p_pc2 = nan(1, blockLen);

    for k = 1:blockLen
        [~, p_pc1(k)] = ttest2(EV1_g0(:,k), EV1_g1(:,k), 'Vartype', 'unequal');
        [~, p_pc2(k)] = ttest2(EV2_g0(:,k), EV2_g1(:,k), 'Vartype', 'unequal');
    end

    % ---- Multiple-comparison corrections ----
    alpha = 0.05;
    q_pc1 = fdr_bh(p_pc1);
    q_pc2 = fdr_bh(p_pc2);

    % Diagnostic two-panel preview.
    fsTitle  = 24;
    fsLabel  = 20;
    fsTicks  = 14;
    fsLegend = 14;

    figure('Name', ['t-tests on PCA loadings (PC1/PC2) — ', setLabel], 'Color', 'w');

    subplot(2,1,1);
    semilogy(xAxis, q_pc1, '-o', 'LineWidth', 2.0); hold on;
    yline(alpha, '--', 'alpha = 0.05', 'LineWidth', 1.8, 'Color', [0 0 0]);
    grid off;
    xlabel('Trial (1..20)', 'FontSize', fsLabel);
    ylabel('q-value', 'FontSize', fsLabel);
    title([setLabel, ': t-test on PC1 loading entries'], ...
          'FontSize', fsTitle, 'FontWeight', 'bold');
    set(gca, 'FontSize', fsTicks);
    legend('q_{FDR}(PC1)', 'Location', 'southwest', 'FontSize', fsLegend);

    subplot(2,1,2);
    semilogy(xAxis, q_pc2, '-o', 'LineWidth', 2.0); hold on;
    yline(alpha, '--', 'alpha = 0.05', 'LineWidth', 1.8, 'Color', [0 0 0]);
    grid off;
    xlabel('Trial (1..20)', 'FontSize', fsLabel);
    ylabel('q-value', 'FontSize', fsLabel);
    title([setLabel, ': t-test on PC2 loading entries'], ...
          'FontSize', fsTitle, 'FontWeight', 'bold');
    set(gca, 'FontSize', fsTicks);
    legend('q_{FDR}(PC2)', 'Location', 'southwest', 'FontSize', fsLegend);

end


%% ======================= Local function: FDR (BH) =======================
function q = fdr_bh(p)
% Benjamini–Hochberg FDR correction for a vector of p-values.
    p = p(:);
    m = numel(p);
    [p_sorted, sortIdx] = sort(p);
    ranks = (1:m)';
    q_sorted = p_sorted .* m ./ ranks;

    % enforce monotone non-increasing q-values from end to start
    for i = m-1:-1:1
        q_sorted(i) = min(q_sorted(i), q_sorted(i+1));
    end

    q = zeros(size(p));
    q(sortIdx) = q_sorted;
    q = reshape(q, size(p));   % back to original shape
end

%% Export Figure S4 panels
% Export q-value panels only, using the final supplementary figure style.
% - q-values only
% - no Bonferroni line
% - grid off
% - visible alpha dashed line
% - extra space below alpha
% - legend positioned lower near alpha line

labelFS  = 24;
tickFS   = 20;
legendFS = 20;

W = 900; H = 520;

if ~exist('blockLen','var') || isempty(blockLen)
    blockLen = 20;
end
xAxis = 1:blockLen;

if ~exist('alpha','var') || isempty(alpha)
    alpha = 0.05;
end

% avoid log(0)
epsVal = 1e-6;
q_pc1 = max(q_pc1, epsVal);
q_pc2 = max(q_pc2, epsVal);

% Disable TeX/LaTeX interpreters for stable SVG export.
oldTextInt   = get(groot,'defaultTextInterpreter');
oldTickInt   = get(groot,'defaultAxesTickLabelInterpreter');
oldLegendInt = get(groot,'defaultLegendInterpreter');

set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

cleanupObj = onCleanup(@() restoreInterpreters(groot, oldTextInt, oldTickInt, oldLegendInt));

% ---- export PC1 (Figure_S4_A.svg) ----
exportS4panel('Figure_S4_A.svg', xAxis, q_pc1, alpha, W, H, ...
    'Trial index', 'q-value', {'qFDR(PC1)'}, labelFS, tickFS, legendFS);

% ---- export PC2 (Figure_S4_B.svg) ----
exportS4panel('Figure_S4_B.svg', xAxis, q_pc2, alpha, W, H, ...
    'Trial index', 'q-value', {'qFDR(PC2)'}, labelFS, tickFS, legendFS);

disp('Saved SVGs: Figure_S4_A.svg and Figure_S4_B.svg');

% ---------------- local helpers ----------------
function exportS4panel(outSvg, xAxis, qVals, yl_alpha, W, H, xlab, ylab, legTxt, labelFS, tickFS, legendFS)
    f = figure('Color','w','Position',[100 100 W H]);
    ax = axes('Parent', f); hold(ax,'on'); grid(ax,'off');

    % Use normal plot + explicit log scale
    plot(ax, xAxis, qVals, '-o', 'LineWidth', 1.0);
    set(ax, 'YScale','log');

    % Give extra space below alpha so dashed line does not touch x-axis
    qPos = qVals(qVals > 0 & isfinite(qVals));
    if isempty(qPos)
        yLow = yl_alpha / 10;
    else
        yLow = min([qPos(:); yl_alpha]) / 4;
    end
    yHigh = max([qPos(:); yl_alpha]) * 1.25;
    ylim(ax, [yLow, yHigh]);

    % Alpha reference line
    h1 = yline(ax, yl_alpha, '--', 'alpha=0.05', ...
        'LineWidth', 1.8, 'Color', [0 0 0]);
    set(h1, 'HandleVisibility','off');
    h1.LabelHorizontalAlignment = 'right';
    try
        h1.Label.FontSize = 15;
    catch
        set(h1, 'FontSize', 15);
    end

    try
        set(h1, 'Interpreter', 'none');
    catch
    end

    xlabel(ax, xlab);
    ylabel(ax, ylab);

    set(ax, 'FontSize', tickFS);
    ax.XLabel.FontSize = labelFS;
    ax.YLabel.FontSize = labelFS;

    xlim(ax, [1 numel(xAxis)]);
    xticks(ax, [1 5 10 15 20]);
    xticklabels(ax, {'0','5','10','15','20'});
    ax.XTickLabelRotation = 0;

    % Legend placed lower, near alpha level
    L = legend(ax, legTxt, 'Location','none');
    set(L, 'Units','normalized');
    L.FontSize = legendFS;
    try
        L.ItemTokenSize = [18 10];
    catch
    end
    L.Position(3) = 0.22;
    L.Position(1) = 0.90 - L.Position(3);

    yl = ylim(ax);
    alphaNorm = (log10(yl_alpha) - log10(yl(1))) / (log10(yl(2)) - log10(yl(1)));
    L.Position(2) = alphaNorm - 0.02 - L.Position(4)/2;
    L.Position(2) = max(0.02, min(0.98 - L.Position(4), L.Position(2)));

    L.Box = 'off';

    set(f, 'PaperPositionMode','auto');

    try
        exportgraphics(f, outSvg, 'ContentType','vector');
    catch
        print(f, outSvg, '-dsvg');
    end
    close(f);
end

function restoreInterpreters(grootHandle, tInt, tickInt, legInt)
    set(grootHandle,'defaultTextInterpreter',tInt);
    set(grootHandle,'defaultAxesTickLabelInterpreter',tickInt);
    set(grootHandle,'defaultLegendInterpreter',legInt);
end