%% LOOCV for joint model: p_i = sigmoid( b0 + sum_j B_j' * (X_i' * v_j) )
% - Xi: 9x20 (blocks x time)
% - v_j: 9x1 (unit, block-space), learned
% - B_j: 20x1 (time weights), learned with L1 sparsity (lassoglm)
% - LOOCV evaluation + probability distribution plots

clear; clc; close all;

%% --- Load & prepare (your data layout) ---
m = 9;  p = 20;                                  % 9 blocks, 20 time columns
load('Freud_Processed_BDIAT.mat');                     % expects XF (S360) and active_score
load(filename);
S360   = XF;
labels = active_score(:);

% Slice indices: 9 windows of length 20 -> total 180 columns per sample
starts = 1:40:321;
idx = cell2mat(arrayfun(@(s) s:(s+19), starts, 'UniformOutput', false));
N = size(S360,1);

% Build sample matrices (9 x 20)
Xcells = cell(N,1);
for ii = 1:N
    x  = S360(ii, idx);          % 1x180
    Xi = reshape(x, [p, m])';    % (20x9) -> (9x20)
    Xcells{ii} = exp(-0.99*Xi);
end

% Binary labels {0,1} (map larger label to 1)
classes = unique(labels(:));
assert(numel(classes)==2, 'Need exactly two classes.');
cA = classes(1); cB = classes(2);
y  = double(labels == cB);

%% --- Config --------------------------------------------------------------
cfg = struct();
cfg.J               = 2;        % number of (v_j, B_j) pairs: 1 or 2
cfg.rng_seed        = 42;

% Alternating scheme
cfg.altIters        = 6;       % outer alternations (B-step, then v-step)
cfg.vGradSteps      = 12;       % gradient steps on v per alternation
cfg.vStepSize       = 0.15;     % gradient step size on v
cfg.orthOn          = true;     % Gram-Schmidt orthogonalize v's each step

% L1-Logistic (lassoglm) for sparse B
cfg.cvFolds         = 5;        % k-fold CV inside lassoglm
cfg.useOneSE        = true;     % pick 1SE (sparser) or MinDeviance (denser)
cfg.balanceWeights  = true;     % class imbalance weights
cfg.standardize     = true;    % if true, Z is z-scored using TRAIN stats

rng(cfg.rng_seed,'twister');

%% --- Run LOOCV -----------------------------------------------------------
results = run_loocv_alt(Xcells, y, cfg);

%% --- Summary -------------------------------------------------------------
fprintf('\n===== Joint (B sparse, v learned) — LOOCV =====\n');
fprintf('Samples / folds: %d\n', numel(y));
fprintf('LOOCV Accuracy:        %.3f\n', mean(results.yhat_all == y));
% Balanced accuracy:
rec0 = mean(results.yhat_all(y==0) == 0);
rec1 = mean(results.yhat_all(y==1) == 1);
fprintf('LOOCV Balanced Acc:    %.3f  (rec0=%.3f, rec1=%.3f)\n', mean([rec0,rec1]), rec0, rec1);

disp('LOOCV Confusion (rows=true [0,1], cols=pred [0,1]):');
C = zeros(2,2);
for n=1:numel(y), C(y(n)+1, results.yhat_all(n)+1) = C(y(n)+1, results.yhat_all(n)+1) + 1; end
disp(array2table(C, 'VariableNames', {'pred_0','pred_1'}, 'RowNames', {'true_0','true_1'}));

%% --- Probability distribution plots --------------------------------------
figure('Name','LOOCV test probabilities by true class');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
histogram(results.p_all(y==0), 20, 'Normalization','pdf'); grid on;
xlabel('p (predicted P[class=1])'); ylabel('pdf'); title('True class = 0');

nexttile;
histogram(results.p_all(y==1), 20, 'Normalization','pdf'); grid on;
xlabel('p (predicted P[class=1])'); ylabel('pdf'); title('True class = 1');

%% --- ROC curve (LOOCV predictions) ---------------------------------------
if isfield(results,'p_all') && ~isempty(results.p_all)
    scores = results.p_all(:);   % predicted P(class=1)
    labels = y(:);               % true labels

    % Try perfcurve (Stats & ML Toolbox); otherwise manual fallback
    try
        [fpr,tpr,~,auc] = perfcurve(labels, scores, 1);
        figure('Name','ROC curve — LOOCV');
        plot(fpr, tpr, 'LineWidth', 1.8); grid on;
        xlabel('False Positive Rate'); ylabel('True Positive Rate');
        title(sprintf('ROC Curve (AUC = %.3f)', auc));
        xlim([0 1]); ylim([0 1]);
    catch
        th = linspace(0,1,200);
        tpr = zeros(size(th)); fpr = zeros(size(th));
        P = sum(labels==1); N0 = sum(labels==0);
        for k=1:numel(th)
            yhat = scores >= th(k);
            tp = sum(yhat==1 & labels==1);
            fp = sum(yhat==1 & labels==0);
            tpr(k) = tp / max(P,1);
            fpr(k) = fp / max(N0,1);
        end
        [fpr_sorted,ord] = sort(fpr); tpr_sorted = tpr(ord);
        auc = trapz(fpr_sorted, tpr_sorted);
        figure('Name','ROC curve — LOOCV');
        plot(fpr_sorted, tpr_sorted, 'LineWidth', 1.8); grid on;
        xlabel('False Positive Rate'); ylabel('True Positive Rate');
        title(sprintf('ROC Curve (AUC ≈ %.3f) [manual]', auc));
        xlim([0 1]); ylim([0 1]);
    end
else
    warning('No probabilities in results.p_all — cannot draw ROC.');
end

%% ========================================================================
%%                               FUNCTIONS
%% ========================================================================

function results = run_loocv_alt(Xcells, y, cfg)
    N = numel(Xcells);
    p_all    = nan(N,1);
    yhat_all = nan(N,1);

    % TIP: If you have the Parallel Toolbox, you can parfor the LOOCV loop
    for i = 1:N
        i
        trIdx = true(N,1); trIdx(i) = false;
        trIdx = find(trIdx);
        teIdx = i;

        % Train on N-1
        model = train_alt_joint(Xcells, y, trIdx, cfg);
        % Predict on the held-out sample
        [p_te, yhat_te] = predict_alt_joint(Xcells, teIdx, model);
        p_all(i)    = p_te;
        yhat_all(i) = yhat_te;
    end

    results.p_all    = p_all;
    results.yhat_all = yhat_all;
end

function model = train_alt_joint(Xcells, y, idxTrain, cfg)
% TRAIN_ALT_JOINT
% Alternating training for the model:
%   p_i = sigmoid( b0 + sum_{j=1..J} B_j' * (X_i' * v_j) )
% where:
%   - X_i is 9x20,
%   - v_j is 9x1 (unit, learned),
%   - B_j is 20x1 (sparse, learned via L1-logistic),
%   - features are standardized (train-only) before the L1 step.
%
% This version:
%   - Uses class weights (inverse-frequency) in BOTH the B-step and v-step
%   - Applies chain rule through standardization in the v-gradient:
%       Btilde_j = B_j ./ sigZ_block_j
%
% Inputs:
%   Xcells   : N x 1 cell of 9x20 matrices (each Xi)
%   y        : N x 1 binary labels in {0,1}
%   idxTrain : vector of train indices
%   cfg      : struct with fields:
%                J, altIters, vGradSteps, vStepSize, orthOn,
%                cvFolds, useOneSE, balanceWeights, standardize
%
% Output:
%   model : struct with fields:
%             J
%             v_list    : 1xJ cell, each 9x1
%             B_concat  : (20*J) x 1
%             b0        : scalar intercept
%             muZ, sigZ : 1 x (20*J) standardization stats

    J   = cfg.J;
    ytr = y(idxTrain);

    % ---- init v_j from pooled block PCs (9D)
    v_list = init_v_from_block_pcs(Xcells, idxTrain, J);

    % ---- class weights (imbalance)
    if cfg.balanceWeights
        nPos = sum(ytr==1); nNeg = sum(ytr==0);
        if nPos==0 || nNeg==0
            w_tr = ones(size(ytr));
        else
            w_pos = 0.5 * numel(ytr) / nPos;
            w_neg = 0.5 * numel(ytr) / nNeg;
            w_tr = zeros(size(ytr));
            w_tr(ytr==1) = w_pos; 
            w_tr(ytr==0) = w_neg;
        end
    else
        w_tr = ones(size(ytr));
    end

    % ---- Alternating loop
    b0 = 0; 
    B_concat = zeros(J*size(Xcells{1},2),1);  % (20J)x1 placeholder
    muZ = []; sigZ = [];

    for it = 1:cfg.altIters
        
        % ----- (A) Sparse B step (lassoglm on standardized Z)
        Ztr = build_Z(Xcells, idxTrain, v_list);              % Ntr x (20J)
        [ZtrZ, muZ, sigZ] = zscore_train(Ztr, cfg.standardize);

        [B_concat, b0] = fit_l1_logistic(ZtrZ, ytr, w_tr, cfg.cvFolds, cfg.useOneSE);

        % Pre-split B into J blocks, and pick corresponding sigZ blocks
        P  = size(Ztr,2) / J;                                 % P = 20
        B_list = cell(J,1);
        sig_blocks = cell(J,1);
        for j = 1:J
            cols = (j-1)*P + (1:P);
            B_list{j}     = B_concat(cols);
            sb            = sigZ(cols);
            sig_blocks{j} = sb(:);                            % column P x 1
        end

        % ----- (B) Gradient steps on v given B (keep b0, B fixed)
        % Weighted residuals with chain rule through standardization:
        ZtrZ = (Ztr - muZ) ./ sigZ;
        lin  = b0 + ZtrZ * B_concat;
        p_i  = 1 ./ (1 + exp(-lin));
        r_i  = w_tr .* (ytr - p_i);                           % Ntr x 1

        for gs = 1:cfg.vGradSteps
            for j = 1:J
                Btilde_j = B_list{j} ./ sig_blocks{j};        % 20x1
                g = zeros(9,1);
                for t = 1:numel(idxTrain)
                    Xi = Xcells{idxTrain(t)};                 % 9x20
                    g  = g + r_i(t) * (Xi * Btilde_j);        % 9x1
                end

                % gradient step + renormalize
                v_list{j} = v_list{j} + cfg.vStepSize * g;
                nv = norm(v_list{j}); 
                if nv > 0, v_list{j} = v_list{j} / nv; end
            end

            % Optional orthogonalization (for J>=2)
            if cfg.orthOn && J>=2
                v_list{1} = v_list{1} / max(norm(v_list{1}), eps);
                v_list{2} = v_list{2} - (v_list{1}'*v_list{2}) * v_list{1};
                v_list{2} = v_list{2} / max(norm(v_list{2}), eps);
            end

            % Refresh residuals with updated v’s
            Ztr  = build_Z(Xcells, idxTrain, v_list);
            ZtrZ = (Ztr - muZ) ./ sigZ;
            lin  = b0 + ZtrZ * B_concat;
            p_i  = 1 ./ (1 + exp(-lin));
            r_i  = w_tr .* (ytr - p_i);
        end
    end

    % ---- Pack model
    model.J        = J;
    model.v_list   = v_list;           % 1xJ cell, each 9x1
    model.B_concat = B_concat(:);      % (20J)x1
    model.b0       = b0;
    model.muZ      = muZ;              % 1 x (20J)
    model.sigZ     = sigZ;             % 1 x (20J)
end

function [p, yhat] = predict_alt_joint(Xcells, idx, model)
    % Build Z, standardize with TRAIN stats, and predict
    Z = build_Z(Xcells, idx, model.v_list);
    Zz = (Z - model.muZ) ./ model.sigZ;
    lin = model.b0 + Zz * model.B_concat;
    p = 1 ./ (1 + exp(-lin));
    yhat = (p >= 0.5);
    % If idx is scalar, p and yhat will be scalar; else vectors.
end

% =========================== Helper functions ============================

function v_list = init_v_from_block_pcs(Xcells, idxTrain, J)
    % Initialize v_j as top J block-space PCs from sum_i Xi*Xi'
    C = zeros(9,9);
    for t = 1:numel(idxTrain)
        Xi = Xcells{idxTrain(t)};
        C  = C + Xi*Xi.';
    end
    C = C / max(1,numel(idxTrain));
    [U,~,~] = svd(C,'econ');

    v_list = cell(J,1);
    v_list{1} = U(:,1) / max(norm(U(:,1)), eps);
    if J >= 2
        v2 = U(:,2) - (v_list{1}'*U(:,2))*v_list{1};
        v_list{2} = v2 / max(norm(v2), eps);
    end
end

function Z = build_Z(Xcells, idx, v_list)
    % Z(i,:) = [Xi'v_1; Xi'v_2; ...]'     (concatenate 20-d blocks)
    J = numel(v_list);
    P = size(Xcells{1},2);   % 20
    Z = zeros(numel(idx), J*P);
    for t = 1:numel(idx)
        Xi = Xcells{idx(t)}; % 9x20
        row = [];
        for j = 1:J
            row = [row; Xi' * v_list{j}]; %#ok<AGROW>
        end
        Z(t,:) = row(:).';
    end
end

function [Zz, muZ, sigZ] = zscore_train(Z, doStd)
    if ~doStd
        Zz  = Z; 
        muZ = zeros(1,size(Z,2)); 
        sigZ= ones(1,size(Z,2));
        return;
    end
    muZ  = mean(Z,1);
    sigZ = std(Z,0,1); 
    sigZ(sigZ==0) = 1;
    Zz = (Z - muZ) ./ sigZ;
end

function [B_concat, b0] = fit_l1_logistic(ZtrZ, ytr, w_tr, cvFolds, useOneSE)
  % More forgiving path + higher iteration budget on the proper knobs
    numLambda   = 60;     % length of lambda path
    lambdaRatio = 1e-3;   % don’t go too close to 0 (helps convergence)
    reltol      = 1e-4;   % convergence tolerance for coord-descent
    maxiter     = 5000;   % iteration cap for coord-descent

    [B,FitInfo] = lassoglm( ...
        ZtrZ, ytr, 'binomial', ...
        'Alpha', 1, ...                    % L1
        'CV', cvFolds, ...
        'Weights', w_tr, ...
        'Standardize', false, ...          % we already standardized ZtrZ (or left as-is)
        'NumLambda', numLambda, ...
        'LambdaRatio', lambdaRatio, ...
        'RelTol', reltol, ...
        'MaxIter', maxiter);

    idx = FitInfo.IndexMinDeviance;
    if useOneSE, idx = FitInfo.Index1SE; end

    B_concat = B(:,idx);
    if ~any(B_concat) && useOneSE
        % Fallback to MinDeviance if 1SE gives all zeros
        idx = FitInfo.IndexMinDeviance;
        B_concat = B(:,idx);
    end
    b0 = FitInfo.Intercept(idx);
end
