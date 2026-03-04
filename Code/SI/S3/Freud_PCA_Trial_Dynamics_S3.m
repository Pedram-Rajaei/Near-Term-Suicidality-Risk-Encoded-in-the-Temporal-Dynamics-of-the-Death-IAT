%% ======================= FULL SCRIPT (Bootstrap CI + final 3-panel export) =======================
% What this script does:
%   - Computes per-subject, per-trial-position measures across 9 blocks (blockLen=20)
%       rowMean = mean RT across blocks (per trial index within block)
%       rowVar  = variance across blocks
%       rowEV   = first eigenvector entry from SVD on exp(-alpha_svd * X)
%   - Produces the original figures (bootstrap CI figure + t-test p/q figure) for each condition
%   - Stores the specific panels you requested
%   - After the loop, creates ONE final figure with 3 panels:
%       1) "Death + Me: Mean ScaleRT"   (rowMean for Death+Me)
%       2) "Life + Me Scale RT"        (rowMean for Life+Me)
%       3) "Death + Me Mean V entry"   (rowEV for Death+Me)
%   - Applies formatting: label font=18, ticks=15, legends=15, x-label="Trial position"
%   - Saves final figure as SVG/PDF/PNG

clear all; close all; clc;

%% ----------------------- USER SETTINGS -----------------------
alpha_svd = 1.1;       % used in exp(-alpha_svd * Xi_blocks)
alphaCI   = 0.05;      % 95% bootstrap CI
nBoot     = 2000;      % bootstrap resamples
rng(1);                % reproducible bootstrap

% Figure formatting (final figure)
labelFS  = 24;
tickFS   = 20;
legendFS = 20;

%% ----------------------- LOAD DATA ---------------------------
load('Freud_Processed_BDIAT.mat');
active_score = active_score(:);  % ensure column vector
classes      = [0 1];
blockLen     = 20;

% Two sets of column-block starts:
%   Set 1: 1:20, 41:60, ..., 321:340  (Death+Me)
%   Set 2: 21:40, 61:80, ..., 341:360 (Life+Me)
startSets = {1:40:321, 21:40:341};

% Condition labels (edit if reversed)
setNames  = {'Death + Me', 'Life + Me'};

% ================== STORAGE FOR FINAL 3-PANEL FIGURE ==================
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

    %% --------- PER-ROW statistics: rowMean, rowVar, rowEV ---------
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
    subplot(3,1,1); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [lo0 fliplr(hi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    fill([xAxis fliplr(xAxis)], [lo1 fliplr(hi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    plot(xAxis, m0, '-o', 'LineWidth', 1.5);
    plot(xAxis, m1, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('ScaleRT (rowMean)');
    title([setLabel, ': Mean ScaleRT \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend(sprintf('Inactive SI mean (n=%d)', n0), sprintf('Active SI mean (n=%d)', n1), 'Location','best');

    % ---- Panel 2: rowVar ----
    subplot(3,1,2); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [vlo0 fliplr(vhi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    fill([xAxis fliplr(xAxis)], [vlo1 fliplr(vhi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
    plot(xAxis, v0, '-o', 'LineWidth', 1.5);
    plot(xAxis, v1, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('rowVar');
    title([setLabel, ': Mean rowVar \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend('Inactive SI mean','Active SI mean', 'Location','best');

    % ---- Panel 3: rowEV ----
    subplot(3,1,3); hold on; grid on;
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

    % Multiple-comparison corrections: Bonferroni + FDR (BH)
    alpha      = 0.05;
    bonf_alpha = alpha / blockLen;

    q_mean = fdr_bh(p_mean);
    q_var  = fdr_bh(p_var);
    q_ev   = fdr_bh(p_ev);

    % ================== STORE T-TEST ARRAYS FOR FIGURE 6 ==================
    if strcmp(setLabel, 'Death + Me')
        S.Death_ttest.xAxis      = xAxis;
        S.Death_ttest.p_mean     = p_mean;
        S.Death_ttest.q_mean     = q_mean;
        S.Death_ttest.p_ev       = p_ev;
        S.Death_ttest.q_ev       = q_ev;
        S.Death_ttest.alpha      = alpha;
        S.Death_ttest.bonf_alpha = bonf_alpha;
    
    elseif strcmp(setLabel, 'Life + Me')
        S.Life_ttest.xAxis      = xAxis;
        S.Life_ttest.p_mean     = p_mean;
        S.Life_ttest.q_mean     = q_mean;
        S.Life_ttest.p_ev       = p_ev;
        S.Life_ttest.q_ev       = q_ev;
        S.Life_ttest.alpha      = alpha;
        S.Life_ttest.bonf_alpha = bonf_alpha;
    end

    figure('Name', ['t-tests on per-row measures — ', setLabel], 'Color', 'w');

    subplot(3,1,1);
    semilogy(xAxis, p_mean, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_mean, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Trial (1..20)');
    ylabel('p / q-value');
    title([setLabel, ': t-test on rowMean']);
    legend('p(rowMean)', 'q_{FDR}(rowMean)', '0.05', 'Bonferroni', 'Location', 'best');

    subplot(3,1,2);
    semilogy(xAxis, p_var, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_var, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Trial (1..20)');
    ylabel('p / q-value');
    title([setLabel, ': t-test on rowVar']);
    legend('p(rowVar)', 'q_{FDR}(rowVar)', '0.05', 'Bonferroni', 'Location', 'best');

    subplot(3,1,3);
    semilogy(xAxis, p_ev, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_ev, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Trial (1..20)');
    ylabel('p / q-value');
    title([setLabel, ': t-test on rowEV (v_1 entries)']);
    legend('p(rowEV)', 'q_{FDR}(rowEV)', '0.05', 'Bonferroni', 'Location', 'best');

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
    % We need:
    %   Panel 1: Death+Me rowMean (ScaleRT)
    %   Panel 2: Life+Me rowMean (ScaleRT)
    %   Panel 3: Death+Me rowEV (v1 entry)
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


%% ======================= FINAL 3-PANEL FIGURE =======================
% Titles must be:
%   1) "Death + Me: Mean ScaleRT"
%   2) "Life + Me Scale RT"
%   3) "Death + Me Mean V entry"
%
% Formatting:
%   - X-axis label: Trial position (all)
%   - label font: 18
%   - ticks: 15
%   - legend: 13

%% ======================= FIGURE (Figure 5) + export each subplot =======================
assert(isfield(S,'Death') && isfield(S,'Life'), ...
    'Final figure requires both Death and Life results. Check setNames/startSets.');

xAxis = S.Death.xAxis;

figFinal = figure('Color','w','Position',[100 100 900 1150]);

% -------- Panel 1: Death + Me Mean ScaleRT --------
ax1 = subplot(4,1,1); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill([xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial index','FontSize',labelFS);
ylabel('Scaled reaction time','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive SI mean (n=%d)', S.Death.n0), ...
        sprintf('Active SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim(ax1, [-0.15 0.25]);
yl = ylim(ax1);
yticks(ax1, linspace(yl(1), yl(2), 6));
ytickformat(ax1, '%.2f');

% -------- Panel 2: Life + Me ScaleRT --------
ax2 = subplot(4,1,2); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill([xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial index','FontSize',labelFS);
ylabel('Scaled reaction time','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive SI mean (n=%d)', S.Life.n0), ...
        sprintf('Active SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim(ax2, [-0.15 0.30]);
yl = ylim(ax2);
yticks(ax2, linspace(yl(1), yl(2), 6));
ytickformat(ax2, '%.2f');

% -------- Panel 3: Death + Me Mean V entry --------
ax3 = subplot(4,1,3); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.elo0 fliplr(S.Death.ehi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill([xAxis fliplr(xAxis)], [S.Death.elo1 fliplr(S.Death.ehi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(xAxis, S.Death.e0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.e1, '-s', 'LineWidth', 1.5);
xlabel('Trial index','FontSize',labelFS);
ylabel('PC1','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive SI mean (n=%d)', S.Death.n0), ...
        sprintf('Active SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim(ax3, [-0.26 -0.16]);
yl = ylim(ax3);
yticks(ax3, linspace(yl(1), yl(2), 6));
ytickformat(ax3, '%.2f');

% -------- Panel 4: Life + Me Mean V entry --------
ax4 = subplot(4,1,4); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Life.elo0 fliplr(S.Life.ehi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill([xAxis fliplr(xAxis)], [S.Life.elo1 fliplr(S.Life.ehi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(xAxis, S.Life.e0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Life.e1, '-s', 'LineWidth', 1.5);

xlabel('Trial index','FontSize',labelFS);
ylabel('PC1','FontSize',labelFS);
set(gca,'FontSize',tickFS);

legend({sprintf('Inactive SI mean (n=%d)', S.Life.n0), ...
        sprintf('Active SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');

ylim(ax4, [-0.26 -0.16]);
yl = ylim(ax4);
yticks(ax4, linspace(yl(1), yl(2), 6));
ytickformat(ax4, '%.2f');

%% ======================= FIGURE 6 (2x2): t-test panels from Fig2 & Fig4 =======================
assert(isfield(S,'Death_ttest') && isfield(S,'Life_ttest'), ...
    'Figure 6 requires stored t-test results for both Death and Life.');

xAxis = S.Death_ttest.xAxis;

fig6 = figure('Color','w','Position',[100 100 1100 800]);

yl_alpha   = 0.05;
bonf_alpha = 0.05 / numel(xAxis);

% ---------------- (1) Death + Me: t-test on rowMean ----------------
ax1 = subplot(2,2,1); hold(ax1,'on'); grid(ax1,'on');
semilogy(ax1, xAxis, S.Death_ttest.p_mean, '-x', 'LineWidth', 1.5);
semilogy(ax1, xAxis, S.Death_ttest.q_mean, '-o', 'LineWidth', 1.0);

h1 = yline(ax1, yl_alpha,   '--', 'alpha=0.05');
h2 = yline(ax1, bonf_alpha, ':',  'Bonferroni');
set([h1 h2], 'HandleVisibility','off');
h1.LabelHorizontalAlignment = 'right';
h2.LabelHorizontalAlignment = 'right';

xlabel(ax1, 'Trial index','FontSize',labelFS);
ylabel(ax1, 'p / q-value','FontSize',labelFS);
set(ax1,'FontSize',tickFS);
title(ax1, 'Death + Me: t-test on rowMean','FontSize',labelFS);

ylim(ax1, [1e-3 1]);
yticks(ax1, [1e-3 1e-2 1e-1 1]);

legend(ax1, {'p(rowMean)','q_{FDR}(rowMean)'}, 'Location','best', 'FontSize',legendFS);

% ---------------- (2) Life + Me: t-test on rowMean ----------------
ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'on');
semilogy(ax2, xAxis, S.Life_ttest.p_mean, '-x', 'LineWidth', 1.5);
semilogy(ax2, xAxis, S.Life_ttest.q_mean, '-o', 'LineWidth', 1.0);

h1 = yline(ax2, yl_alpha,   '--', 'alpha=0.05');
h2 = yline(ax2, bonf_alpha, ':',  'Bonferroni');
set([h1 h2], 'HandleVisibility','off');
h1.LabelHorizontalAlignment = 'right';
h2.LabelHorizontalAlignment = 'right';

xlabel(ax2, 'Trial index','FontSize',labelFS);
ylabel(ax2, 'p / q-value','FontSize',labelFS);
set(ax2,'FontSize',tickFS);
title(ax2, 'Life + Me: t-test on rowMean','FontSize',labelFS);

ylim(ax2, [1e-3 1]);
yticks(ax2, [1e-3 1e-2 1e-1 1]);

legend(ax2, {'p(rowMean)','q_{FDR}(rowMean)'}, 'Location','best', 'FontSize',legendFS);

% ---------------- (3) Death + Me: t-test on rowEV ----------------
ax3 = subplot(2,2,3); hold(ax3,'on'); grid(ax3,'on');
semilogy(ax3, xAxis, S.Death_ttest.p_ev, '-x', 'LineWidth', 1.5);
semilogy(ax3, xAxis, S.Death_ttest.q_ev, '-o', 'LineWidth', 1.0);

h1 = yline(ax3, yl_alpha,   '--', 'alpha=0.05');
h2 = yline(ax3, bonf_alpha, ':',  'Bonferroni');
set([h1 h2], 'HandleVisibility','off');
h1.LabelHorizontalAlignment = 'right';
h2.LabelHorizontalAlignment = 'right';

xlabel(ax3, 'Trial index','FontSize',labelFS);
ylabel(ax3, 'p / q-value','FontSize',labelFS);
set(ax3,'FontSize',tickFS);
title(ax3, 'Death + Me: t-test on rowEV','FontSize',labelFS);

ylim(ax3, [1e-3 1]);
yticks(ax3, [1e-3 1e-2 1e-1 1]);

legend(ax3, {'p(rowEV)','q_{FDR}(rowEV)'}, 'Location','best', 'FontSize',legendFS);

% ---------------- (4) Life + Me: t-test on rowEV ----------------
ax4 = subplot(2,2,4); hold(ax4,'on'); grid(ax4,'on');
semilogy(ax4, xAxis, S.Life_ttest.p_ev, '-x', 'LineWidth', 1.5);
semilogy(ax4, xAxis, S.Life_ttest.q_ev, '-o', 'LineWidth', 1.0);

h1 = yline(ax4, yl_alpha,   '--', 'alpha=0.05');
h2 = yline(ax4, bonf_alpha, ':',  'Bonferroni');
set([h1 h2], 'HandleVisibility','off');
h1.LabelHorizontalAlignment = 'right';
h2.LabelHorizontalAlignment = 'right';

xlabel(ax4, 'Trial index','FontSize',labelFS);
ylabel(ax4, 'p / q-value','FontSize',labelFS);
set(ax4,'FontSize',tickFS);
title(ax4, 'Life + Me: t-test on rowEV','FontSize',labelFS);

ylim(ax4, [1e-3 1]);
yticks(ax4, [1e-3 1e-2 1e-1 1]);

legend(ax4, {'p(rowEV)','q_{FDR}(rowEV)'}, 'Location','best', 'FontSize',legendFS);

drawnow;

%% -------- Clean export: re-plot each panel into its own tall figure and save SVG --------
drawnow;

% Use a taller canvas to increase vertical separation between curves
W = 900; H = 520;   % increase H for more vertical space (try 650 if you want even more)

% ---------- (A) Death + Me Mean ScaleRT ----------
fA = figure('Color','w','Position',[100 100 W H]);
axA = axes('Parent', fA); hold(axA,'on'); grid(axA,'on');

fill(axA, [xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], 'b', 'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill(axA, [xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], 'r', 'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(axA, xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(axA, xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);

xlabel(axA, 'Trial index','FontSize',labelFS);
ylabel(axA, 'Scaled reaction time','FontSize',labelFS);
set(axA,'FontSize',tickFS);

ylim(axA, [-0.15 0.25]);
yl = ylim(axA);
yticks(axA, linspace(yl(1), yl(2), 6));
ytickformat(axA, '%.2f');

legend(axA, {sprintf('Inactive SI mean (n=%d)', S.Death.n0), ...
             sprintf('Active SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');

set(fA, 'PaperPositionMode','auto');
% --- P-value line + stars (A) ---
add_pvalue_line_and_stars(axA, [4 14], [1 1]);

print(fA, 'Figure_3_A.svg', '-dsvg');
close(fA);

% ---------- (B) Life + Me Mean ScaleRT ----------
fB = figure('Color','w','Position',[100 100 W H]);
axB = axes('Parent', fB); hold(axB,'on'); grid(axB,'on');

fill(axB, [xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], 'b', 'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill(axB, [xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], 'r', 'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(axB, xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(axB, xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);

xlabel(axB, 'Trial index','FontSize',labelFS);
ylabel(axB, 'Scaled reaction time','FontSize',labelFS);
set(axB,'FontSize',tickFS);

ylim(axB, [-0.15 0.30]);
yl = ylim(axB);
yticks(axB, linspace(yl(1), yl(2), 6));
ytickformat(axB, '%.2f');

legend(axB, {sprintf('Inactive SI mean (n=%d)', S.Life.n0), ...
             sprintf('Active SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');

set(fB, 'PaperPositionMode','auto');
print(fB, 'Figure_3_B.svg', '-dsvg');
close(fB);

% ---------- (C) Death + Me Mean V entry ----------
fC = figure('Color','w','Position',[100 100 W H]);
axC = axes('Parent', fC); hold(axC,'on'); grid(axC,'on');

fill(axC, [xAxis fliplr(xAxis)], [S.Death.elo0 fliplr(S.Death.ehi0)], 'b', 'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill(axC, [xAxis fliplr(xAxis)], [S.Death.elo1 fliplr(S.Death.ehi1)], 'r', 'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(axC, xAxis, S.Death.e0, '-o', 'LineWidth', 1.5);
plot(axC, xAxis, S.Death.e1, '-s', 'LineWidth', 1.5);

xlabel(axC, 'Trial index','FontSize',labelFS);
ylabel(axC, 'PC1','FontSize',labelFS);
set(axC,'FontSize',tickFS);

ylim(axC, [-0.26 -0.16]);
yl = ylim(axC);
yticks(axC, linspace(yl(1), yl(2), 6));
ytickformat(axC, '%.2f');

legend(axC, {sprintf('Inactive SI mean (n=%d)', S.Death.n0), ...
             sprintf('Active SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');

set(fC, 'PaperPositionMode','auto');
% --- P-value line + stars (C) ---
% index 4 -> TWO stars, index 19 -> one star
add_pvalue_line_and_stars(axC, [4 19], [2 1], [4], [1]);

print(fC, 'Figure_3_C.svg', '-dsvg');
close(fC);

% ---------- (D) Life + Me Mean V entry ----------
fD = figure('Color','w','Position',[100 100 W H]);
axD = axes('Parent', fD); hold(axD,'on'); grid(axD,'on');

fill(axD, [xAxis fliplr(xAxis)], [S.Life.elo0 fliplr(S.Life.ehi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
fill(axD, [xAxis fliplr(xAxis)], [S.Life.elo1 fliplr(S.Life.ehi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none', 'HandleVisibility','off');
plot(axD, xAxis, S.Life.e0, '-o', 'LineWidth', 1.5);
plot(axD, xAxis, S.Life.e1, '-s', 'LineWidth', 1.5);

xlabel(axD, 'Trial index','FontSize',labelFS);
ylabel(axD, 'PC1','FontSize',labelFS);
set(axD,'FontSize',tickFS);

ylim(axD, [-0.26 -0.16]);
yl = ylim(axD);
yticks(axD, linspace(yl(1), yl(2), 6));
ytickformat(axD, '%.2f');

legend(axD, {sprintf('Inactive SI mean (n=%d)', S.Life.n0), ...
             sprintf('Active SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');

set(fD, 'PaperPositionMode','auto');
% --- P-value line + stars (D) ---
add_pvalue_line_and_stars(axD, [3], [1]);

print(fD, 'Figure_3_D.svg', '-dsvg');
close(fD);

disp('Saved clean subplot SVGs: Figure_3_A.svg, Figure_3_B.svg, Figure_3_C.svg, Figure_3_D.svg');

%% ======================= EXPORT: Figure S5 (A–D) t-test panels as SVG (ROBUST) =======================
assert(isfield(S,'Death_ttest') && isfield(S,'Life_ttest'), ...
    'Need S.Death_ttest and S.Life_ttest stored before exporting Figure_S5 SVGs.');

xAxis = S.Death_ttest.xAxis;

% canvas
W = 900; H = 520;

% thresholds
yl_alpha   = 0.05;
bonf_alpha = 0.05 / numel(xAxis);

% ---- IMPORTANT: Disable TeX/LaTeX in ALL text for SVG stability ----
oldTextInt   = get(groot,'defaultTextInterpreter');
oldTickInt   = get(groot,'defaultAxesTickLabelInterpreter');
oldLegendInt = get(groot,'defaultLegendInterpreter');

set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

cleanupObj = onCleanup(@() restoreInterpreters(groot, oldTextInt, oldTickInt, oldLegendInt));

% helper: one exporter
exportS5panel('Figure_S5_A.svg', xAxis, S.Death_ttest.p_mean, S.Death_ttest.q_mean, yl_alpha, bonf_alpha, W, H, ...
    'Trial index', 'p / q-value', {'p(rowMean)','qFDR(rowMean)'}, labelFS, tickFS, legendFS);

exportS5panel('Figure_S5_B.svg', xAxis, S.Life_ttest.p_mean,  S.Life_ttest.q_mean,  yl_alpha, bonf_alpha, W, H, ...
    'Trial index', 'p / q-value', {'p(rowMean)','qFDR(rowMean)'}, labelFS, tickFS, legendFS);

exportS5panel('Figure_S5_C.svg', xAxis, S.Death_ttest.p_ev,   S.Death_ttest.q_ev,   yl_alpha, bonf_alpha, W, H, ...
    'Trial index', 'p / q-value', {'p(rowEV)','qFDR(rowEV)'}, labelFS, tickFS, legendFS);

exportS5panel('Figure_S5_D.svg', xAxis, S.Life_ttest.p_ev,    S.Life_ttest.q_ev,    yl_alpha, bonf_alpha, W, H, ...
    'Trial index', 'p / q-value', {'p(rowEV)','qFDR(rowEV)'}, labelFS, tickFS, legendFS);


disp('Saved t-test SVGs: Figure_S5_A.svg, Figure_S5_B.svg, Figure_S5_C.svg, Figure_S5_D.svg');

%% ---------------- local helper functions ----------------
function exportS5panel(outSvg, xAxis, pVals, qVals, yl_alpha, bonf_alpha, W, H, xlab, ylab, legTxt, labelFS, tickFS, legendFS)
    f = figure('Color','w','Position',[100 100 W H]);
    ax = axes('Parent', f); hold(ax,'on'); grid(ax,'on');

    % Use normal plot + explicit log scale (avoids semilogy text quirks in SVG)
    plot(ax, xAxis, pVals, '-x', 'LineWidth', 1.5);
    plot(ax, xAxis, qVals, '-o', 'LineWidth', 1.0);
    set(ax, 'YScale','log');

    % reference lines (hide from legend)
    h1 = yline(ax, yl_alpha,   '--', 'alpha=0.05');
    h2 = yline(ax, bonf_alpha, ':',  'Bonferroni');
    set([h1 h2], 'HandleVisibility','off');
    h1.LabelHorizontalAlignment = 'right';
    h2.LabelHorizontalAlignment = 'right';
    try
        % newer versions (if supported)
        h1.Label.FontSize = 15;
        h2.Label.FontSize = 15;
    catch
        % older versions
        set([h1 h2], 'FontSize', 15);
    end
    
    % optional: avoid TeX rendering issues in SVG
    try
        set([h1 h2], 'Interpreter', 'none');
    catch
    end
    
    xlabel(ax, xlab);
    ylabel(ax, ylab);
    % -------- FONT SETTINGS (ADD THIS) --------
    set(ax, 'FontSize', tickFS);                 % ticks
    ax.XLabel.FontSize = labelFS;                % x label
    ax.YLabel.FontSize = labelFS;                % y label

    % fixed y-range + PLAIN tick labels (no TeX)
    ylim(ax, [1e-3 1]);
    yticks(ax, [1e-3 1e-2 1e-1 1]);
    yticklabels(ax, {'1e-3','1e-2','1e-1','1'});

    % ---- Legend between alpha and Bonferroni lines on the right ----
    L = legend(ax, legTxt, 'Location','none');
    set(L, 'Units','normalized');
    L.FontSize = legendFS;                       % legend text size

    % Reduce padding inside legend (huge improvement)
    L.ItemTokenSize = [18 10];   % smaller line samples
    
    % Force legend width (IMPORTANT)
    L.Position(3) = 0.22;        % try 0.18–0.25
    
    % Compute midpoint in LOG space
    yMid = 10.^((log10(yl_alpha) + log10(bonf_alpha))/2);
    
    yl = ylim(ax);
    yMidNorm = (log10(yMid) - log10(yl(1))) / (log10(yl(2)) - log10(yl(1)));
    
    % Vertical center
    L.Position(2) = yMidNorm - L.Position(4)/2;
    L.Position(2) = L.Position(2) + 0.1;   % move legend up a bit
    L.Position(2) = max(0.02, min(0.98 - L.Position(4), L.Position(2)));
    
    % Clamp vertically
    L.Position(2) = max(0.02, min(0.98 - L.Position(4), L.Position(2)));
    
    % -------- Anchor RIGHT EDGE instead of left --------
    rightMargin = 0.90;          % how close to the border
    L.Position(1) = rightMargin - L.Position(3);
    L.Box = 'off';      % ← removes the rectangle

    set(f, 'PaperPositionMode','auto');

    % export
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
function add_pvalue_line_and_stars(ax, xP, nP, xQ, nQ)
% Usage:
%   add_pvalue_line_and_stars(ax, xP, nP)                 % p-row only
%   add_pvalue_line_and_stars(ax, xP, nP, xQ, nQ)         % p-row + q-row
%
% Places:
%   - p-row at yt(4) (3rd tick from top, with 6 ticks)
%   - q-row midway between yt(4) and yt(5)
% Labels are outside axes to the right. No horizontal line.

    % ----- handle optional args -----
    if nargin < 3
        error('add_pvalue_line_and_stars:NeedArgs', ...
              'Need at least (ax, xP, nP).');
    end
    if nargin < 4 || isempty(xQ)
        xQ = [];
    end
    if nargin < 5 || isempty(nQ)
        nQ = [];
    end

    % Ensure 6 ticks exist
    yt = yticks(ax);
    if numel(yt) < 6
        yl = ylim(ax);
        yticks(ax, linspace(yl(1), yl(2), 6));
        yt = yticks(ax);
    end

    % p-value row (3rd tick from top)
    yP = yt(4);

    % q-value row (between 4th and 5th tick)
    yQ = (yt(4) + yt(5)) / 2;

    % -------- Stars for p --------
    for i = 1:numel(xP)
        text(ax, xP(i), yP, repmat('*', 1, nP(i)), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',18, ...
            'FontWeight','bold', ...
            'Clipping','off');
    end

    % -------- Stars for q (optional) --------
    for i = 1:numel(xQ)
        text(ax, xQ(i), yQ, repmat('*', 1, nQ(i)), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',18, ...
            'FontWeight','bold', ...
            'Clipping','off');
    end

    % -------- Labels outside axes --------
    yl = ylim(ax);
    yPnorm = (yP - yl(1)) / (yl(2) - yl(1));
    yQnorm = (yQ - yl(1)) / (yl(2) - yl(1));

    % put labels just outside the plot area
    xLabel = 1.015;   % increase if you want farther right

    text(ax, xLabel, yPnorm, 'p-value', ...
        'Units','normalized', ...
        'HorizontalAlignment','left', ...
        'VerticalAlignment','middle', ...
        'FontSize',16, ...
        'FontWeight','bold', ...
        'Clipping','off');

    % Only write q-value label if you actually use q-row
    if ~isempty(xQ)
        text(ax, xLabel, yQnorm, 'q-value', ...
            'Units','normalized', ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','middle', ...
            'FontSize',16, ...
            'FontWeight','bold', ...
            'Clipping','off');
    end
end
