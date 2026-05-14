clear all;
close all;
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
        rowMean(i, :) = mean(exp(Xi_blocks), 1);                % 1 x 20
        rowVar(i,  :) = var(exp(Xi_blocks), 0, 1);              % 1 x 20

        % SVD-based first PC on Y = exp(-X)
        Y_row = exp(-1.0*exp(Xi_blocks));                           % nBlocks x 20
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

    % ---- Plot t-tests: one graph per measure (3 subplots) ----
    figure('Name', ['t-tests on per-row measures for ', setLabel], 'Color', 'w');

    % rowMean
    subplot(3,1,1);
    semilogy(xAxis, p_mean, '-x', 'LineWidth', 1.5); hold on;
    yline(0.05, '--');
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('p-value');
    title(['t-test on rowMean (', setLabel, ')']);
    legend('p(rowMean)', '0.05', 'Location', 'best');

    % rowVar
    subplot(3,1,2);
    semilogy(xAxis, p_var, '-x', 'LineWidth', 1.5); hold on;
    yline(0.05, '--');
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('p-value');
    title(['t-test on rowVar (', setLabel, ')']);
    legend('p(rowVar)', '0.05', 'Location', 'best');

    % rowEV
    subplot(3,1,3);
    semilogy(xAxis, p_ev, '-x', 'LineWidth', 1.5); hold on;
    yline(0.05, '--');
    grid on;
    xlabel('Index within block (1..20)');
    ylabel('p-value');
    title(['t-test on rowEV (first eigenvector, ', setLabel, ')']);
    legend('p(rowEV)', '0.05', 'Location', 'best');

end
