clear all;
close all;
alpha = 0.99;

%% Assumes XF (n x 360) and active_score (n x 1) are already in workspace
load('Freud_Processed_BDIAT.mat');
%% Assumes XF (n x 360) and active_score (n x 1 or 1 x n) are in workspace
active_score = active_score(:);           % ensure column
classes      = [0 1];
blockLen     = 20;



% Two sets of column-block starts:
%   Set 1: 1:20, 41:60, 81:100, ..., 321:340
%   Set 2: 21:40, 61:80, 101:120, ..., 341:360
startSets = {1:40:321, 21:40:341};
setNames  = {'Blocks 1:20,41:60,...,321:340', ...
             'Blocks 21:40,61:80,...,341:360'};

for sIdx = 1:numel(startSets)

    blockStarts = startSets{sIdx};
    setLabel    = setNames{sIdx};

    fprintf('\nProcessing %s\n', setLabel);

    % ---- Build column indices for this set of blocks ----
    blocksIdx = [];
    for s = blockStarts
        blocksIdx = [blocksIdx, s:(s + blockLen - 1)];
    end
    nBlocks = numel(blockStarts);          % should be 9

    % Extract data for these columns
    X = XF(:, blocksIdx);                  % n x (nBlocks*blockLen)
    [nRows, ~] = size(X);

    % --------- PER-ROW statistics: rowMean, rowVar, rowEV ---------
    %
    % For each row i:
    %   Xi_blocks : 9 x 20 (blocks x index)
    %   rowMean(i,:) : 1 x 20 = mean over 9 blocks
    %   rowVar(i,:)  : 1 x 20 = var  over 9 blocks
    %   rowEV(i,:)   : 1 x 20 = first eigenvector of Y'*Y, Y=exp(-Xi_blocks)

    rowMean = zeros(nRows, blockLen);
    rowVar  = zeros(nRows, blockLen);
    rowEV   = zeros(nRows, blockLen);

    for i = 1:nRows
        Xi = X(i, :);                                      % 1 x (nBlocks*blockLen)

        % Reshape to 9 x 20 (blocks x index-within-block)
        Xi_blocks = reshape(Xi, blockLen, nBlocks).';      % nBlocks x blockLen

        % Per-row mean & variance across blocks
        rowMean(i, :) = mean(Xi_blocks, 1);           % 1 x 20
        rowVar(i,  :) = var(Xi_blocks, 0, 1);         % 1 x 20

        % SVD-based first PC on Y = exp(-X)
        % c= -0.1;
        % alpah = 10;
        % Y_row = 1 ./ (1 + exp(-alpha * (Xi_blocks - c)));

        Y_row = exp(-alpha*Xi_blocks);                      % nBlocks x 20
        C_row = Y_row' * Y_row;                            % 20 x 20

        [~, ~, V] = svd(C_row);
        v1 = V(:, 1);                                      % 20 x 1
        rowEV(i, :) = v1.';                                % 1 x 20
    end

    % --------- Split these PER-ROW quantities by class ---------
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

    %% =====================  Descriptive stats for X (per row) =====================

    % Mean across rows of rowMean / rowVar, per class
    mean_of_means = zeros(numel(classes), blockLen);
    mean_of_vars  = zeros(numel(classes), blockLen);

    for ci = 1:numel(classes)
        mean_of_means(ci, :) = mean(M_class{ci}, 1);   % 1 x 20
        mean_of_vars(ci,  :) = mean(V_class{ci}, 1);   % 1 x 20
    end

    xAxis = 1:blockLen;

    figure('Name', ['X stats (per-row) for ', setLabel], 'Color', 'w');

    % Mean of per-row means
    subplot(2,1,1);
    plot(xAxis, mean_of_means(1,:), '-o', 'LineWidth', 1.5); hold on;
    plot(xAxis, mean_of_means(2,:), '-s', 'LineWidth', 1.5);
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('Mean of row-means');
    title(['Mean of row-level means (', setLabel, ')']);
    legend('Class 0', 'Class 1', 'Location', 'best');

    % Mean of per-row variances
    subplot(2,1,2);
    plot(xAxis, mean_of_vars(1,:), '-o', 'LineWidth', 1.5); hold on;
    plot(xAxis, mean_of_vars(2,:), '-s', 'LineWidth', 1.5);
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('Mean of row-variances');
    title(['Mean of row-level variances (', setLabel, ')']);
    legend('Class 0', 'Class 1', 'Location', 'best');


    %% ==================  Descriptive stats for SVD (per row)  ==================

    meanEV = zeros(numel(classes), blockLen);
    varEV  = zeros(numel(classes), blockLen);

    for ci = 1:numel(classes)
        EVc = EV_class{ci};
        meanEV(ci, :) = mean(EVc, 1);            % 1 x 20
        varEV(ci,  :) = var(EVc, 0, 1);          % 1 x 20
    end

    figure('Name', ['SVD eigenvector stats (per-row) for ', setLabel], 'Color', 'w');

    % Mean of first eigenvector entries
    subplot(2,1,1);
    plot(xAxis, meanEV(1,:), '-o', 'LineWidth', 1.5); hold on;
    plot(xAxis, meanEV(2,:), '-s', 'LineWidth', 1.5);
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('Mean of v_1 entries');
    title(['Mean of first eigenvector entries (per-row, ', setLabel, ')']);
    legend('Class 0', 'Class 1', 'Location', 'best');

    % Variance of first eigenvector entries (across rows)
    subplot(2,1,2);
    plot(xAxis, varEV(1,:), '-o', 'LineWidth', 1.5); hold on;
    plot(xAxis, varEV(2,:), '-s', 'LineWidth', 1.5);
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('Variance of v_1 entries');
    title(['Variance of first eigenvector entries (per-row, ', setLabel, ')']);
    legend('Class 0', 'Class 1', 'Location', 'best');


    %% ==================  t-tests on rowMean, rowVar, rowEV  ==================

    % Separate t-test for EACH measure:
    %   p_mean(k): t-test on rowMean(:,k) between classes
    %   p_var(k) : t-test on rowVar(:,k)  between classes
    %   p_ev(k)  : t-test on rowEV(:,k)   between classes

    p_mean = nan(1, blockLen);
    p_var  = nan(1, blockLen);
    p_ev   = nan(1, blockLen);

    for k = 1:blockLen
        % rowMean
        [~, p_mean(k)] = ttest2(M_class{1}(:, k), M_class{2}(:, k), ...
                                'Vartype', 'unequal');

        % rowVar
        [~, p_var(k)]  = ttest2(V_class{1}(:, k), V_class{2}(:, k), ...
                                'Vartype', 'unequal');

        % rowEV (first eigenvector entries)
        [~, p_ev(k)]   = ttest2(EV_class{1}(:, k), EV_class{2}(:, k), ...
                                'Vartype', 'unequal');
    end

    %%% -------- Multiple-comparison corrections: Bonferroni + FDR (BH) --------
    alpha      = 0.05;
    bonf_alpha = alpha / blockLen;

    % Bonferroni-adjusted p-values
    p_mean_bonf = min(p_mean * blockLen, 1);
    p_var_bonf  = min(p_var  * blockLen, 1);
    p_ev_bonf   = min(p_ev   * blockLen, 1);

    % FDR-adjusted q-values (Benjamini–Hochberg)
    q_mean = fdr_bh(p_mean);
    q_var  = fdr_bh(p_var);
    q_ev   = fdr_bh(p_ev);

    % ---- Plot t-tests: one graph per measure (3 subplots) ----
    figure('Name', ['t-tests on per-row measures for ', setLabel], 'Color', 'w');

    % rowMean
    subplot(3,1,1);
    semilogy(xAxis, p_mean, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_mean,  '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('p / q-value');
    title(['t-test on rowMean (', setLabel, ')']);
    legend('p(rowMean)', 'q_F_D_R(rowMean)', '0.05', 'Bonferroni', ...
           'Location', 'best');

    % rowVar
    subplot(3,1,2);
    semilogy(xAxis, p_var, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_var, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('p / q-value');
    title(['t-test on rowVar (', setLabel, ')']);
    legend('p(rowVar)', 'q_F_D_R(rowVar)', '0.05', 'Bonferroni', ...
           'Location', 'best');

    % rowEV
    subplot(3,1,3);
    semilogy(xAxis, p_ev, '-x', 'LineWidth', 1.5); hold on;
    semilogy(xAxis, q_ev, '-o', 'LineWidth', 1.0);
    yline(alpha,      '--', 'alpha=0.05');
    yline(bonf_alpha, ':',  'Bonferroni');
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('p / q-value');
    title(['t-test on rowEV (first eigenvector, ', setLabel, ')']);
    legend('p(rowEV)', 'q_F_D_R(rowEV)', '0.05', 'Bonferroni', ...
           'Location', 'best');


    %%% ================== Mixed-effects model on rowMean ==================
    % Long-format table: each row = (subject, trial, group, value)
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

    % Mixed-effects model: ScaleRT ~ Group * Trial + (1 | Subject)
    % Requires Statistics and Machine Learning Toolbox
    lme_mean = fitlme(tbl_mean, 'ScaleRT ~ Group*Trial + (1|Subject)');

    fprintf('\nMixed-effects model for %s (rowMean / ScaleRT):\n', setLabel);
    disp(anova(lme_mean));

    %%% =============== Planned contrasts for trials 1, 4, 5, 6 ===============

    plannedTrials = [1 2 3 4 5 6];
    contrast_pvals = zeros(size(plannedTrials));
    
    for tt = 1:numel(plannedTrials)
        tval = plannedTrials(tt);
    
        % contrast: Group + Trial*Group:Trial = 0
        % Build hypothesis vector:
        % [Intercept, Group, Trial, Group*Trial]
        H = [0 1 0 tval];  
        contrast_pvals(tt) = coefTest(lme_mean, H);
    end
    
    % Apply FDR correction to the 4 planned contrasts
    contrast_qvals = fdr_bh(contrast_pvals);
    
    fprintf('\nPlanned contrasts (Trials 1,4,5,6):\n');
    disp(table(plannedTrials.', contrast_pvals.', contrast_qvals, ...
          'VariableNames', {'Trial','pValue','qValue'}));


end

%%% ======================= Local function: FDR (BH) =======================
function q = fdr_bh(p)
% q = fdr_bh(p)
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
