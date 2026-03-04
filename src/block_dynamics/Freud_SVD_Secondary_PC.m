%% ======================= FULL SCRIPT (2 PCs for EV, keeps old plots, adds PC2 panels) =======================
% Same as your script, but:
%   - rowEV stores 2 PCs (PC1 and PC2)
%   - all old plots remain
%   - adds 4th subplot for PC2 in the bootstrap CI and t-test figures
%   - final 3-panel figure remains the same (uses PC1)
%   - exports an extra clean SVG for Death+Me PC2 (Figure_3_D.svg)

clear all; close all; clc;

%% ----------------------- USER SETTINGS -----------------------
alpha_svd = 1.1;       % used in exp(-alpha_svd * Xi_blocks)
alphaCI   = 0.05;      % 95% bootstrap CI
nBoot     = 2000;      % bootstrap resamples
rng(1);                % reproducible bootstrap

% Figure formatting (final figure)
labelFS  = 24;
tickFS   = 20;
legendFS = 2;

%% ----------------------- LOAD DATA ---------------------------
load('Freud_Processed_BDIAT.mat');   % expects XF (n x 360) and active_score
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

    %% --------- PER-ROW statistics: rowMean, rowVar, rowEV (NOW 2 PCs) ---------
    rowMean = zeros(nRows, blockLen);
    rowVar  = zeros(nRows, blockLen);
    rowEV   = zeros(nRows, blockLen, 2);   % <-- PC1 and PC2

    for i = 1:nRows
        Xi = X(i, :);                                      % 1 x (nBlocks*blockLen)

        % Reshape to nBlocks x blockLen (blocks x trial index)
        Xi_blocks = reshape(Xi, blockLen, nBlocks).';      % nBlocks x blockLen

        % Per-row mean & variance across blocks
        rowMean(i, :) = mean(Xi_blocks, 1);                % 1 x 20
        rowVar(i,  :) = var(Xi_blocks, 0, 1);              % 1 x 20

        % SVD-based PCs on Y = exp(-alpha_svd * X)
        Y_row = exp(-alpha_svd * Xi_blocks);               % nBlocks x 20
        C_row = Y_row' * Y_row;                            % 20 x 20
        [~, ~, V] = svd(C_row);

        % Store BOTH PC1 and PC2 eigenvector entries (length 20 each)
        rowEV(i,:,1) = V(:,1).';   % PC1 (same as your old v1)
        rowEV(i,:,2) = V(:,2).';   % PC2
    end

    %% --------- Split these PER-ROW quantities by class ---------
    M_class   = cell(1, numel(classes));   % rowMean
    V_class   = cell(1, numel(classes));   % rowVar
    EV1_class = cell(1, numel(classes));   % rowEV PC1 (n_c x 20)
    EV2_class = cell(1, numel(classes));   % rowEV PC2 (n_c x 20)

    for ci = 1:numel(classes)
        c = classes(ci);
        idxC = (active_score == c);

        M_class{ci}   = rowMean(idxC, :);
        V_class{ci}   = rowVar(idxC,  :);
        EV1_class{ci} = rowEV(idxC, :, 1);
        EV2_class{ci} = rowEV(idxC, :, 2);
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

    % rowEV PC1 (old behavior)
    [e0, elo0, ehi0] = bootMeanCI(EV1_class{1}, alphaCI, nBoot);
    [e1, elo1, ehi1] = bootMeanCI(EV1_class{2}, alphaCI, nBoot);

    % rowEV PC2 (new)
    [e20, e2lo0, e2hi0] = bootMeanCI(EV2_class{1}, alphaCI, nBoot);
    [e21, e2lo1, e2hi1] = bootMeanCI(EV2_class{2}, alphaCI, nBoot);

    figure('Name', ['RT dynamics with bootstrap CI — ', setLabel], 'Color', 'w');

    % ---- Panel 1: rowMean (ScaleRT) ----
    subplot(4,1,1); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [lo0 fliplr(hi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    fill([xAxis fliplr(xAxis)], [lo1 fliplr(hi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(xAxis, m0, '-o', 'LineWidth', 1.5);
    plot(xAxis, m1, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('ScaleRT (rowMean)');
    title([setLabel, ': Mean ScaleRT \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend(sprintf('Inactive-SI CI (n=%d)', n0), sprintf('Active-SI CI (n=%d)', n1), ...
           sprintf('Inactive-SI mean (n=%d)', n0), sprintf('Active-SI mean (n=%d)', n1), ...
           'Location','best');

    % ---- Panel 2: rowVar ----
    subplot(4,1,2); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [vlo0 fliplr(vhi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    fill([xAxis fliplr(xAxis)], [vlo1 fliplr(vhi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(xAxis, v0, '-o', 'LineWidth', 1.5);
    plot(xAxis, v1, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('rowVar');
    title([setLabel, ': Mean rowVar \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend('Inactive-SI CI','Active-SI CI','Inactive-SI mean','Active-SI mean', 'Location','best');

    % ---- Panel 3: rowEV PC1 (UNCHANGED from your old script) ----
    subplot(4,1,3); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [elo0 fliplr(ehi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    fill([xAxis fliplr(xAxis)], [elo1 fliplr(ehi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(xAxis, e0, '-o', 'LineWidth', 1.5);
    plot(xAxis, e1, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('rowEV (PC1 entry)');
    title([setLabel, ': Mean PC1 entry \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend('Inactive-SI CI','Active-SI CI','Inactive-SI mean','Active-SI mean', 'Location','best');

    % ---- Panel 4: rowEV PC2 (NEW) ----
    subplot(4,1,4); hold on; grid on;
    fill([xAxis fliplr(xAxis)], [e2lo0 fliplr(e2hi0)], 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    fill([xAxis fliplr(xAxis)], [e2lo1 fliplr(e2hi1)], 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(xAxis, e20, '-o', 'LineWidth', 1.5);
    plot(xAxis, e21, '-s', 'LineWidth', 1.5);
    xlabel('Trial (1..20)');
    ylabel('rowEV (PC2 entry)');
    title([setLabel, ': Mean PC2 entry \pm bootstrap ', num2str((1-alphaCI)*100), '% CI']);
    legend('Inactive-SI CI','Active-SI CI','Inactive-SI mean','Active-SI mean', 'Location','best');

    %% ==================  t-tests on rowMean, rowVar, rowEV PC1/PC2  ==================
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

    % Multiple-comparison corrections: Bonferroni + FDR (BH)
    alpha      = 0.05;
    bonf_alpha = alpha / blockLen;

    q_mean = fdr_bh(p_mean);
    q_var  = fdr_bh(p_var);
    q_ev1  = fdr_bh(p_ev1);
    q_ev2  = fdr_bh(p_ev2);

    figure('Name', ['t-tests on per-row measures — ', setLabel], 'Color', 'w');

    subplot(4,1,1);
    semilogy(xAxis, p_mean, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_mean, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Trial (1..20)');
    ylabel('p / q-value');
    title([setLabel, ': t-test on rowMean']);
    legend('p(rowMean)', 'q_{FDR}(rowMean)', '0.05', 'Bonferroni', 'Location', 'best');

    subplot(4,1,2);
    semilogy(xAxis, p_var, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_var, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Trial (1..20)');
    ylabel('p / q-value');
    title([setLabel, ': t-test on rowVar']);
    legend('p(rowVar)', 'q_{FDR}(rowVar)', '0.05', 'Bonferroni', 'Location', 'best');

    subplot(4,1,3);
    semilogy(xAxis, p_ev1, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_ev1, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Trial (1..20)');
    ylabel('p / q-value');
    title([setLabel, ': t-test on rowEV PC1 (entries)']);
    legend('p(PC1)', 'q_{FDR}(PC1)', '0.05', 'Bonferroni', 'Location', 'best');

    subplot(4,1,4);
    semilogy(xAxis, p_ev2, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_ev2, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Trial (1..20)');
    ylabel('p / q-value');
    title([setLabel, ': t-test on rowEV PC2 (entries)']);
    legend('p(PC2)', 'q_{FDR}(PC2)', '0.05', 'Bonferroni', 'Location', 'best');

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
    % Keep EXACT SAME final 3-panel figure behavior: it uses PC1 (e0/e1) just like before.
    if strcmp(setLabel, 'Death + Me')
        S.Death.xAxis = xAxis;
        S.Death.m0 = m0; S.Death.lo0 = lo0; S.Death.hi0 = hi0;
        S.Death.m1 = m1; S.Death.lo1 = lo1; S.Death.hi1 = hi1;

        % PC1 stored exactly as before
        S.Death.e0 = e0; S.Death.elo0 = elo0; S.Death.ehi0 = ehi0;
        S.Death.e1 = e1; S.Death.elo1 = elo1; S.Death.ehi1 = ehi1;

        % Also store PC2 for optional export
        S.Death.e20 = e20; S.Death.e2lo0 = e2lo0; S.Death.e2hi0 = e2hi0;
        S.Death.e21 = e21; S.Death.e2lo1 = e2lo1; S.Death.e2hi1 = e2hi1;

        S.Death.n0 = n0; S.Death.n1 = n1;

    elseif strcmp(setLabel, 'Life + Me')
        S.Life.xAxis = xAxis;
        S.Life.m0 = m0; S.Life.lo0 = lo0; S.Life.hi0 = hi0;
        S.Life.m1 = m1; S.Life.lo1 = lo1; S.Life.hi1 = hi1;
        S.Life.n0 = n0; S.Life.n1 = n1;
    end

end % loop over conditions

%% ======================= FINAL 3-PANEL FIGURE (UNCHANGED: uses PC1) =======================
assert(isfield(S,'Death') && isfield(S,'Life'), ...
    'Final figure requires both Death and Life results. Check setNames/startSets.');

xAxis = S.Death.xAxis;

figFinal = figure('Color','w','Position',[100 100 900 900]);

% -------- Panel 1: Death + Me Mean ScaleRT --------
ax1 = subplot(3,1,1); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
fill([xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
plot(xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial position','FontSize',labelFS);
ylabel('Scaled reaction time','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive-SI CI (n=%d)', S.Death.n0), sprintf('Active-SI CI (n=%d)', S.Death.n1), ...
        sprintf('Inactive-SI mean (n=%d)', S.Death.n0), sprintf('Active-SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim([-0.15 0.25]);
yticks(-0.15:0.05:0.25);

% -------- Panel 2: Life + Me ScaleRT --------
ax2 = subplot(3,1,2); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
fill([xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
plot(xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial position','FontSize',labelFS);
ylabel('Scaled reaction time','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive-SI CI (n=%d)', S.Life.n0), sprintf('Active-SI CI (n=%d)', S.Life.n1), ...
        sprintf('Inactive-SI mean (n=%d)', S.Life.n0), sprintf('Active-SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim([-0.15 0.30]);
yticks(-0.15:0.05:0.30);

% -------- Panel 3: Death + Me Mean V entry (PC1, unchanged) --------
ax3 = subplot(3,1,3); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.elo0 fliplr(S.Death.ehi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
fill([xAxis fliplr(xAxis)], [S.Death.elo1 fliplr(S.Death.ehi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
plot(xAxis, S.Death.e0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.e1, '-s', 'LineWidth', 1.5);
xlabel('Trial position','FontSize',labelFS);
ylabel('First eigenvector entry','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive-SI CI (n=%d)', S.Death.n0), sprintf('Active-SI CI (n=%d)', S.Death.n1), ...
        sprintf('Inactive-SI mean (n=%d)', S.Death.n0), sprintf('Active-SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim([-0.26 -0.16]);
yticks(-0.26:0.02:-0.16);

%% -------- Clean export: re-plot each panel into its own tall figure and save SVG --------
drawnow;

W = 900; H = 520;

% ---------- (A) Death + Me Mean ScaleRT ----------
fA = figure('Color','w','Position',[100 100 W H]);
axA = axes('Parent', fA); hold(axA,'on'); grid(axA,'on');
fill(axA, [xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], 'b', 'FaceAlpha',0.15, 'EdgeColor','none');
fill(axA, [xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], 'r', 'FaceAlpha',0.15, 'EdgeColor','none');
plot(axA, xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(axA, xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);
xlabel(axA, 'Trial position','FontSize',labelFS);
ylabel(axA, 'Scaled reaction time','FontSize',labelFS);
set(axA,'FontSize',tickFS);
ylim(axA, [-0.15 0.25]); yticks(axA, -0.15:0.05:0.25);
legend(axA, {sprintf('Inactive-SI CI (n=%d)', S.Death.n0), sprintf('Active-SI CI (n=%d)', S.Death.n1), ...
             sprintf('Inactive-SI mean (n=%d)', S.Death.n0), sprintf('Active-SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
set(fA, 'PaperPositionMode','auto');
print(fA, 'Figure_3_A.svg', '-dsvg');
close(fA);

% ---------- (B) Life + Me Mean ScaleRT ----------
fB = figure('Color','w','Position',[100 100 W H]);
axB = axes('Parent', fB); hold(axB,'on'); grid(axB,'on');
fill(axB, [xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], 'b', 'FaceAlpha',0.15, 'EdgeColor','none');
fill(axB, [xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], 'r', 'FaceAlpha',0.15, 'EdgeColor','none');
plot(axB, xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(axB, xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);
xlabel(axB, 'Trial position','FontSize',labelFS);
ylabel(axB, 'Scaled reaction time','FontSize',labelFS);
set(axB,'FontSize',tickFS);
ylim(axB, [-0.15 0.30]); yticks(axB, -0.15:0.05:0.30);
legend(axB, {sprintf('Inactive-SI CI (n=%d)', S.Life.n0), sprintf('Active-SI CI (n=%d)', S.Life.n1), ...
             sprintf('Inactive-SI mean (n=%d)', S.Life.n0), sprintf('Active-SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');
set(fB, 'PaperPositionMode','auto');
print(fB, 'Figure_3_B.svg', '-dsvg');
close(fB);

% ---------- (C) Death + Me Mean V entry (PC1, same as before) ----------
fC = figure('Color','w','Position',[100 100 W H]);
axC = axes('Parent', fC); hold(axC,'on'); grid(axC,'on');
fill(axC, [xAxis fliplr(xAxis)], [S.Death.elo0 fliplr(S.Death.ehi0)], 'b', 'FaceAlpha',0.15, 'EdgeColor','none');
fill(axC, [xAxis fliplr(xAxis)], [S.Death.elo1 fliplr(S.Death.ehi1)], 'r', 'FaceAlpha',0.15, 'EdgeColor','none');
plot(axC, xAxis, S.Death.e0, '-o', 'LineWidth', 1.5);
plot(axC, xAxis, S.Death.e1, '-s', 'LineWidth', 1.5);
xlabel(axC, 'Trial position','FontSize',labelFS);
ylabel(axC, 'First eigenvector entry','FontSize',labelFS);
set(axC,'FontSize',tickFS);
ylim(axC, [-0.26 -0.16]); yticks(axC, -0.26:0.02:-0.16);
legend(axC, {sprintf('Inactive-SI CI (n=%d)', S.Death.n0), sprintf('Active-SI CI (n=%d)', S.Death.n1), ...
             sprintf('Inactive-SI mean (n=%d)', S.Death.n0), sprintf('Active-SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
set(fC, 'PaperPositionMode','auto');
print(fC, 'Figure_3_C.svg', '-dsvg');
close(fC);

% ---------- (D) Death + Me Mean V entry (PC2, NEW export) ----------
fD = figure('Color','w','Position',[100 100 W H]);
axD = axes('Parent', fD); hold(axD,'on'); grid(axD,'on');
fill(axD, [xAxis fliplr(xAxis)], [S.Death.e2lo0 fliplr(S.Death.e2hi0)], 'b', 'FaceAlpha',0.15, 'EdgeColor','none');
fill(axD, [xAxis fliplr(xAxis)], [S.Death.e2lo1 fliplr(S.Death.e2hi1)], 'r', 'FaceAlpha',0.15, 'EdgeColor','none');
plot(axD, xAxis, S.Death.e20, '-o', 'LineWidth', 1.5);
plot(axD, xAxis, S.Death.e21, '-s', 'LineWidth', 1.5);
xlabel(axD, 'Trial position','FontSize',labelFS);
ylabel(axD, 'Second eigenvector entry','FontSize',labelFS);
set(axD,'FontSize',tickFS);
legend(axD, {sprintf('Inactive-SI CI (n=%d)', S.Death.n0), sprintf('Active-SI CI (n=%d)', S.Death.n1), ...
             sprintf('Inactive-SI mean (n=%d)', S.Death.n0), sprintf('Active-SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
set(fD, 'PaperPositionMode','auto');
print(fD, 'Figure_3_D.svg', '-dsvg');
close(fD);

disp('Saved clean subplot SVGs: Figure_3_A.svg, Figure_3_B.svg, Figure_3_C.svg, Figure_3_D.svg');
%% ======================= FIGURE 6 (Duplicate of Figure 5, but PC2 instead of PC1) =======================
% Figure 5 uses PC1 (S.Death.e0/e1)
% Figure 6 uses PC2 (S.Death.e20/e21)

assert(isfield(S,'Death') && isfield(S,'Life'), ...
    'Figure 6 requires both Death and Life results. Check setNames/startSets.');

% Make sure PC2 was stored (from the 2-PC modifications)
assert(isfield(S.Death,'e20') && isfield(S.Death,'e21'), ...
    'PC2 stats not found in S.Death. Make sure you stored PC2 (e20/e21, e2lo/e2hi).');

xAxis = S.Death.xAxis;

fig6 = figure('Color','w','Position',[120 120 900 900]);

% -------- Panel 1: Death + Me Mean ScaleRT (same as Fig 5) --------
subplot(3,1,1); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.lo0 fliplr(S.Death.hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
fill([xAxis fliplr(xAxis)], [S.Death.lo1 fliplr(S.Death.hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
plot(xAxis, S.Death.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial position','FontSize',labelFS);
ylabel('Scaled reaction time','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive-SI CI (n=%d)', S.Death.n0), sprintf('Active-SI CI (n=%d)', S.Death.n1), ...
        sprintf('Inactive-SI mean (n=%d)', S.Death.n0), sprintf('Active-SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim([-0.15 0.25]);
yticks(-0.15:0.05:0.25);

% -------- Panel 2: Life + Me ScaleRT (same as Fig 5) --------
subplot(3,1,2); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Life.lo0 fliplr(S.Life.hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
fill([xAxis fliplr(xAxis)], [S.Life.lo1 fliplr(S.Life.hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
plot(xAxis, S.Life.m0, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Life.m1, '-s', 'LineWidth', 1.5);
xlabel('Trial position','FontSize',labelFS);
ylabel('Scaled reaction time','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive-SI CI (n=%d)', S.Life.n0), sprintf('Active-SI CI (n=%d)', S.Life.n1), ...
        sprintf('Inactive-SI mean (n=%d)', S.Life.n0), sprintf('Active-SI mean (n=%d)', S.Life.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');
ylim([-0.15 0.30]);
yticks(-0.15:0.05:0.30);

% -------- Panel 3: Death + Me Mean V entry (PC2 instead of PC1) --------
subplot(3,1,3); hold on; grid on;
fill([xAxis fliplr(xAxis)], [S.Death.e2lo0 fliplr(S.Death.e2hi0)], 'b', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
fill([xAxis fliplr(xAxis)], [S.Death.e2lo1 fliplr(S.Death.e2hi1)], 'r', ...
    'FaceAlpha',0.15, 'EdgeColor','none');
plot(xAxis, S.Death.e20, '-o', 'LineWidth', 1.5);
plot(xAxis, S.Death.e21, '-s', 'LineWidth', 1.5);
xlabel('Trial position','FontSize',labelFS);
ylabel('Second eigenvector entry','FontSize',labelFS);
set(gca,'FontSize',tickFS);
legend({sprintf('Inactive-SI CI (n=%d)', S.Death.n0), sprintf('Active-SI CI (n=%d)', S.Death.n1), ...
        sprintf('Inactive-SI mean (n=%d)', S.Death.n0), sprintf('Active-SI mean (n=%d)', S.Death.n1)}, ...
       'FontSize',legendFS,'Location','best');
title('');

% NOTE: PC2 y-limits can differ a lot from PC1.
% If you want SAME y-limits as Fig 5's panel 3, uncomment:
% ylim([-0.26 -0.16]); yticks(-0.26:0.02:-0.16);

disp('Figure 6 created: duplicate of Figure 5 but using PC2 in panel 3.');

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
