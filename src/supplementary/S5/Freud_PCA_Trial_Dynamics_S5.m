%% Freud_Trial_Dynamics_TTest_S5.m
% Trial-wise statistical tests for Figure S5.
%
% This supplementary analysis repeats the trial-level dynamics pipeline and
% exports FDR-adjusted q-value panels for rowMean and PC1 loading entries
% across Death + Me and Life + Me blocks.
%
% For each subject and condition, the script computes:
%
%   rowMean : mean trial-position response across blocks
%   rowVar  : across-block variance at each trial position
%   rowEV   : first SVD/PC entry from exp(-alpha_svd * X)
%
% Welch two-sample tests are performed at each trial position, followed by
% Benjamini-Hochberg FDR correction across the 20 trial positions.
%
% Final exported panels:
%
%   Figure_S5_A.svg : Death + Me rowMean q-values
%   Figure_S5_B.svg : Life + Me rowMean q-values
%   Figure_S5_C.svg : Death + Me rowEV/PC1 q-values
%   Figure_S5_D.svg : Life + Me rowEV/PC1 q-values
%
% Required input:
%   Freud_Processed_BDIAT.mat
%
% Expected variables:
%   XF           : subjects x 360 processed reaction-time matrix
%   active_score : binary SI group label, where 0 = SI- and 1 = SI+

clear all; close all; clc;

%% ----------------------- USER SETTINGS -----------------------
alpha_svd = 1.1;       % used in exp(-alpha_svd * Xi_blocks)
alphaCI   = 0.05;      % 95% bootstrap CI
nBoot     = 2000;      % bootstrap resamples
rng(1);                % reproducible bootstrap

% Figure formatting for exported S5 panels.
labelFS  = 24;
tickFS   = 20;
legendFS = 20;

%% ----------------------- LOAD DATA ---------------------------
load('Freud_Processed_BDIAT.mat');         % expects XF (n x 360) and active_score
active_score = active_score(:);  % ensure column vector
classes      = [0 1];
blockLen     = 20;

% Two sets of column-block starts:
%   Set 1: 1:20, 41:60, ..., 321:340  (Death+Me)
%   Set 2: 21:40, 61:80, ..., 341:360 (Life+Me)
startSets = {1:40:321, 21:40:341};

% BDIAT condition labels.
setNames  = {'Death + Me', 'Life + Me'};

% Storage for condition-specific summaries and statistical tests.
S = struct();   % S.Death and S.Life will be filled inside the loop

%% ======================= MAIN LOOP OVER CONDITIONS =======================
for sIdx = 1:numel(startSets)

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
        v1 = V(:, 1);                                      % 20 x 1
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

    %% ===================== FIGURE: Bootstrap CI bands =====================
    n0 = size(M_class{1}, 1);
    n1 = size(M_class{2}, 1);

    % rowMean (ScaleRT)
    [m0, lo0, hi0] = bootMeanCI(M_class{1}, alphaCI, nBoot);
    [m1, lo1, hi1] = bootMeanCI(M_class{2}, alphaCI, nBoot);

    % rowVar
    [v0, vlo0, vhi0] = bootMeanCI(V_class{1}, alphaCI, nBoot);
    [v1, vlo1, vhi1] = bootMeanCI(V_class{2}, alphaCI, nBoot);

    % rowEV (v1 entry)
    [e0, elo0, ehi0] = bootMeanCI(EV_class{1}, alphaCI, nBoot);
    [e1, elo1, ehi1] = bootMeanCI(EV_class{2}, alphaCI, nBoot);

    figure('Name', ['RT dynamics with bootstrap CI — ', setLabel], 'Color', 'w');

    % ---- Panel 1: rowMean (ScaleRT) ----
    subplot(3,1,1); hold on; grid off;
    fill([xAxis fliplr(xAxis)], [lo0 fliplr(hi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    fill([xAxis fliplr(xAxis)], [lo1 fliplr(hi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    plot(xAxis, m0, '-o', 'LineWidth', 1.5);
    plot(xAxis, m1, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('ScaleRT (rowMean)');
    title([setLabel, ': Mean ScaleRT \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend(sprintf('Inactive SI mean (n=%d)', n0), sprintf('Active SI mean (n=%d)', n1), 'Location','best');

    % ---- Panel 2: rowVar ----
    subplot(3,1,2); hold on; grid off;
    fill([xAxis fliplr(xAxis)], [vlo0 fliplr(vhi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    fill([xAxis fliplr(xAxis)], [vlo1 fliplr(vhi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    plot(xAxis, v0, '-o', 'LineWidth', 1.5);
    plot(xAxis, v1, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('rowVar');
    title([setLabel, ': Mean rowVar \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend('Inactive SI mean','Active SI mean', 'Location','best');

    % ---- Panel 3: rowEV ----
    subplot(3,1,3); hold on; grid off;
    fill([xAxis fliplr(xAxis)], [elo0 fliplr(ehi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    fill([xAxis fliplr(xAxis)], [elo1 fliplr(ehi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    plot(xAxis, e0, '-o', 'LineWidth', 1.5);
    plot(xAxis, e1, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('rowEV (v_1 entry)');
    title([setLabel, ': Mean v_1 entry \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend('Inactive SI mean','Active SI mean', 'Location','best');

    %% ==================  t-tests on rowMean, rowVar, rowEV  ==================
    p_mean = nan(1, blockLen);
    p_var  = nan(1, blockLen);
    p_ev   = nan(1, blockLen);

    for k = 1:blockLen
        [~, p_mean(k)] = ttest2(M_class{1}(:, k), M_class{2}(:, k), 'Vartype', 'unequal');
        [~, p_var(k)]  = ttest2(V_class{1}(:, k), V_class{2}(:, k), 'Vartype', 'unequal');
        [~, p_ev(k)]   = ttest2(EV_class{1}(:, k), EV_class{2}(:, k), 'Vartype', 'unequal');
    end

    % Multiple-comparison corrections: FDR (BH)
    alpha = 0.05;

    q_mean = fdr_bh(p_mean);
    q_var  = fdr_bh(p_var);
    q_ev   = fdr_bh(p_ev);

    % Store FDR-adjusted q-values for Figure S5 export.
    if strcmp(setLabel, 'Death + Me')
        S.Death_ttest.xAxis  = xAxis;
        S.Death_ttest.q_mean = q_mean;
        S.Death_ttest.q_ev   = q_ev;
        S.Death_ttest.alpha  = alpha;
    
    elseif strcmp(setLabel, 'Life + Me')
        S.Life_ttest.xAxis  = xAxis;
        S.Life_ttest.q_mean = q_mean;
        S.Life_ttest.q_ev   = q_ev;
        S.Life_ttest.alpha  = alpha;
    end

    figure('Name', ['t-tests on per-row measures — ', setLabel], 'Color', 'w');

    subplot(3,1,1);
    semilogy(xAxis, q_mean, '-o', 'LineWidth', 1.0); hold on;
    yline(alpha, '--', 'alpha=0.05');
    grid off;
    xlabel('Trial (1..20)');
    ylabel('q-value');
    title([setLabel, ': t-test on rowMean']);
    legend('q_{FDR}(rowMean)', '0.05', 'Location', 'best');

    subplot(3,1,2);
    semilogy(xAxis, q_var, '-o', 'LineWidth', 1.0); hold on;
    yline(alpha, '--', 'alpha=0.05');
    grid off;
    xlabel('Trial (1..20)');
    ylabel('q-value');
    title([setLabel, ': t-test on rowVar']);
    legend('q_{FDR}(rowVar)', '0.05', 'Location', 'best');

    subplot(3,1,3);
    semilogy(xAxis, q_ev, '-o', 'LineWidth', 1.0); hold on;
    yline(alpha, '--', 'alpha=0.05');
    grid off;
    xlabel('Trial (1..20)');
    ylabel('q-value');
    title([setLabel, ': t-test on rowEV (v_1 entries)']);
    legend('q_{FDR}(rowEV)', '0.05', 'Location', 'best');

    %% ================== Mixed-effects model on rowMean ==================
    [SubIdx, TrialIdx] = ndgrid(1:nRows, 1:blockLen);
    SubIdx   = SubIdx(:);
    TrialIdx = TrialIdx(:);

    Y_mean = rowMean(:);
    Group  = active_score(SubIdx);

    tbl_mean = table( ...
        categorical(SubIdx), ...
        categorical(Group), ...
        TrialIdx, ...
        Y_mean, ...
        'VariableNames', {'Subject','Group','Trial','ScaleRT'});

    lme_mean = fitlme(tbl_mean, 'ScaleRT ~ Group*Trial + (1|Subject)');

    fprintf('\nMixed-effects model for %s (rowMean / ScaleRT):\n', setLabel);
    disp(anova(lme_mean));

    %% =============== Planned contrasts for trials 1..6 ===============
    plannedTrials   = [1 2 3 4 5 6];
    contrast_pvals  = zeros(size(plannedTrials));

    for tt = 1:numel(plannedTrials)
        tval = plannedTrials(tt);
        % [Intercept, Group, Trial, Group*Trial]
        H = [0 1 0 tval];
        contrast_pvals(tt) = coefTest(lme_mean, H);
    end

    contrast_qvals = fdr_bh(contrast_pvals);

    fprintf('\nPlanned contrasts (Trials 1..6):\n');
    disp(table(plannedTrials.', contrast_pvals.', contrast_qvals, ...
        'VariableNames', {'Trial','pValue','qValue'}));

    %% ================== STORE ARRAYS FOR FINAL 3-PANEL FIGURE ==================
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

        % --- ALSO store the Life+Me V-entry (rowEV) ---
        S.Life.e0 = e0; S.Life.elo0 = elo0; S.Life.ehi0 = ehi0;
        S.Life.e1 = e1; S.Life.elo1 = elo1; S.Life.ehi1 = ehi1;

        S.Life.n0 = n0; S.Life.n1 = n1;
    end

end % loop over conditions


%% Diagnostic condition-summary preview
assert(isfield(S,'Death') && isfield(S,'Life'), ...
    'Final figure requires both Death and Life results. Check setNames/startSets.');

xAxis = S.Death.xAxis;

figFinal = figure('Color','w','Position',[100 100 900 1150]);

% -------- Panel 1: Death + Me Mean ScaleRT --------
ax1 = subplot(4,1,1); hold on; grid off;
fill([xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill([xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial index','FontSize',labelFS);
ylabel('Log scaled reaction time','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Without active SI mean (n=%d)', S.Death.n0), ...
        sprintf('With active SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim(ax1, [-0.15 0.25]);
yl = ylim(ax1);
yticks(ax1, linspace(yl(1), yl(2), 6));
ytickformat(ax1, '%.2f');

% -------- Panel 2: Life + Me ScaleRT --------
ax2 = subplot(4,1,2); hold on; grid off;
fill([xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill([xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial index','FontSize',labelFS);
ylabel('Log scaled reaction time','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Without active SI mean (n=%d)', S.Life.n0), ...
        sprintf('With active SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim(ax2, [-0.15 0.25]);
yl = ylim(ax2);
yticks(ax2, linspace(yl(1), yl(2), 6));
ytickformat(ax2, '%.2f');

% -------- Panel 3: Death + Me Mean V entry --------
ax3 = subplot(4,1,3); hold on; grid off;
fill([xAxis fliplr(xAxis)], [S.Death.elo0 fliplr(S.Death.ehi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill([xAxis fliplr(xAxis)], [S.Death.elo1 fliplr(S.Death.ehi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(xAxis, S.Death.e0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.e1, '-s', 'LineWidth', 1.5);
xlabel('Trial index','FontSize',labelFS);
ylabel('PC1','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Without active SI mean (n=%d)', S.Death.n0), ...
        sprintf('With active SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim(ax3, [-0.26 -0.16]);
yl = ylim(ax3);
yticks(ax3, linspace(yl(1), yl(2), 6));
ytickformat(ax3, '%.2f');

% -------- Panel 4: Life + Me Mean V entry --------
ax4 = subplot(4,1,4); hold on; grid off;
fill([xAxis fliplr(xAxis)], [S.Life.elo0 fliplr(S.Life.ehi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill([xAxis fliplr(xAxis)], [S.Life.elo1 fliplr(S.Life.ehi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(xAxis, S.Life.e0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Life.e1, '-s', 'LineWidth', 1.5);

xlabel('Trial index','FontSize',labelFS);
ylabel('PC1','FontSize',labelFS);
set(gca,'FontSize',tickFS);

legend({sprintf('Without active SI mean (n=%d)', S.Life.n0), ...
        sprintf('With active SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');

ylim(ax4, [-0.26 -0.16]);
yl = ylim(ax4);
yticks(ax4, linspace(yl(1), yl(2), 6));
ytickformat(ax4, '%.2f');

%% Diagnostic q-value preview
assert(isfield(S,'Death_ttest') && isfield(S,'Life_ttest'), ...
    'Figure 6 requires stored t-test results for both Death and Life.');

xAxis = S.Death_ttest.xAxis;

fig6 = figure('Color','w','Position',[100 100 1100 800]);

yl_alpha = 0.05;

% ---------------- (1) Death + Me: t-test on rowMean ----------------
ax1 = subplot(2,2,1); hold(ax1,'on'); grid(ax1,'off');
semilogy(ax1, xAxis, S.Death_ttest.q_mean, '-o', 'LineWidth', 1.0);

h1 = yline(ax1, yl_alpha, '--', 'alpha=0.05', ...
    'LineWidth', 1.8, 'Color', [0 0 0]);
set(h1, 'HandleVisibility','off');
h1.LabelHorizontalAlignment = 'right';

xlabel(ax1, 'Trial index','FontSize',labelFS);
ylabel(ax1, 'q-value','FontSize',labelFS);
set(ax1,'FontSize',tickFS);
title(ax1, 'Death + Me: t-test on rowMean','FontSize',labelFS);

legend(ax1, {'q_{FDR}(rowMean)'}, 'Location','best', 'FontSize',legendFS);

% ---------------- (2) Life + Me: t-test on rowMean ----------------
ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'off');
semilogy(ax2, xAxis, S.Life_ttest.q_mean, '-o', 'LineWidth', 1.0);

h1 = yline(ax2, yl_alpha, '--', 'alpha=0.05', ...
    'LineWidth', 1.8, 'Color', [0 0 0]);
set(h1, 'HandleVisibility','off');
h1.LabelHorizontalAlignment = 'right';

xlabel(ax2, 'Trial index','FontSize',labelFS);
ylabel(ax2, 'q-value','FontSize',labelFS);
set(ax2,'FontSize',tickFS);
title(ax2, 'Life + Me: t-test on rowMean','FontSize',labelFS);

legend(ax2, {'q_{FDR}(rowMean)'}, 'Location','best', 'FontSize',legendFS);

% ---------------- (3) Death + Me: t-test on rowEV ----------------
ax3 = subplot(2,2,3); hold(ax3,'on'); grid(ax3,'off');
semilogy(ax3, xAxis, S.Death_ttest.q_ev, '-o', 'LineWidth', 1.0);

h1 = yline(ax3, yl_alpha, '--', 'alpha=0.05', ...
    'LineWidth', 1.8, 'Color', [0 0 0]);
set(h1, 'HandleVisibility','off');
h1.LabelHorizontalAlignment = 'right';

xlabel(ax3, 'Trial index','FontSize',labelFS);
ylabel(ax3, 'q-value','FontSize',labelFS);
set(ax3,'FontSize',tickFS);
title(ax3, 'Death + Me: t-test on rowEV','FontSize',labelFS);

legend(ax3, {'q_{FDR}(rowEV)'}, 'Location','best', 'FontSize',legendFS);

% ---------------- (4) Life + Me: t-test on rowEV ----------------
ax4 = subplot(2,2,4); hold(ax4,'on'); grid(ax4,'off');
semilogy(ax4, xAxis, S.Life_ttest.q_ev, '-o', 'LineWidth', 1.0);

h1 = yline(ax4, yl_alpha, '--', 'alpha=0.05', ...
    'LineWidth', 1.8, 'Color', [0 0 0]);
set(h1, 'HandleVisibility','off');
h1.LabelHorizontalAlignment = 'right';

xlabel(ax4, 'Trial index','FontSize',labelFS);
ylabel(ax4, 'q-value','FontSize',labelFS);
set(ax4,'FontSize',tickFS);
title(ax4, 'Life + Me: t-test on rowEV','FontSize',labelFS);

legend(ax4, {'q_{FDR}(rowEV)'}, 'Location','best', 'FontSize',legendFS);

drawnow;

%% Export Figure S5 panels
assert(isfield(S,'Death_ttest') && isfield(S,'Life_ttest'), ...
    'Need S.Death_ttest and S.Life_ttest stored before exporting Figure_S5 SVGs.');

xAxis = S.Death_ttest.xAxis;

% canvas
W = 900; H = 520;

% thresholds
yl_alpha = 0.05;

% Disable TeX/LaTeX interpreters for stable SVG export.
oldTextInt   = get(groot,'defaultTextInterpreter');
oldTickInt   = get(groot,'defaultAxesTickLabelInterpreter');
oldLegendInt = get(groot,'defaultLegendInterpreter');

set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

cleanupObj = onCleanup(@() restoreInterpreters(groot, oldTextInt, oldTickInt, oldLegendInt));

% helper: one exporter
exportS5panel('Figure_S5_A.svg', xAxis, S.Death_ttest.q_mean, yl_alpha, W, H, ...
    'Trial index', 'q-value', {'qFDR(rowMean)'}, labelFS, tickFS, legendFS);

exportS5panel('Figure_S5_B.svg', xAxis, S.Life_ttest.q_mean, yl_alpha, W, H, ...
    'Trial index', 'q-value', {'qFDR(rowMean)'}, labelFS, tickFS, legendFS);

exportS5panel('Figure_S5_C.svg', xAxis, S.Death_ttest.q_ev, yl_alpha, W, H, ...
    'Trial index', 'q-value', {'qFDR(rowEV)'}, labelFS, tickFS, legendFS);

exportS5panel('Figure_S5_D.svg', xAxis, S.Life_ttest.q_ev, yl_alpha, W, H, ...
    'Trial index', 'q-value', {'qFDR(rowEV)'}, labelFS, tickFS, legendFS);

disp('Saved t-test SVGs: Figure_S5_A.svg, Figure_S5_B.svg, Figure_S5_C.svg, Figure_S5_D.svg');

%% ---------------- local helper functions ----------------
function exportS5panel(outSvg, xAxis, qVals, yl_alpha, W, H, xlab, ylab, legTxt, labelFS, tickFS, legendFS)
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

    % Only alpha reference line
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

    % Legend placement: slightly above the alpha label, tied to axis scale
    L = legend(ax, legTxt, 'Location','none');
    set(L, 'Units','normalized');
    L.FontSize = legendFS;
    L.ItemTokenSize = [18 10];
    L.Position(3) = 0.22;
    L.Position(1) = 0.90 - L.Position(3);

    yl = ylim(ax);
    alphaNorm = (log10(yl_alpha) - log10(yl(1))) / (log10(yl(2)) - log10(yl(1)));
    L.Position(2) = alphaNorm + 0.02 - L.Position(4)/2;   % slightly above alpha text
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


%% ======================= Local function: Bootstrap mean CI =======================
function [m, lo, hi] = bootMeanCI(A, alphaCI, nBoot)
% Bootstrap percentile CI for the mean at each column (trial index).
% A: nSubjects x T
    A = A(~any(isnan(A),2), :);  % drop NaN rows (optional)
    [n, T] = size(A);

    m = mean(A, 1);

    bootMeans = zeros(nBoot, T);
    for b = 1:nBoot
        idx = randi(n, [n, 1]);          % resample subjects with replacement
        bootMeans(b,:) = mean(A(idx,:), 1);
    end

    lo = prctile(bootMeans, 100*(alphaCI/2), 1);
    hi = prctile(bootMeans, 100*(1-alphaCI/2), 1);
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