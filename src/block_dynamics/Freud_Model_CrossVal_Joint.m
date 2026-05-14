function results = Freud_Model_CrossVal_Joint(Xcells, y, cfg)
%FREUD_MODEL_CROSSVAL_JOINT Leave-one-out cross-validation for joint classifier.
%
% This function evaluates a joint projection-and-classification model using
% LOOCV. For each held-out subject, block-space projection vectors and sparse
% logistic-regression weights are learned from the training subjects only.
%
% Model:
%   p_i = sigmoid(b0 + sum_j B_j' * (X_i' * v_j))
%
% Inputs:
%   Xcells : N x 1 cell array, where each cell contains a 9 x 20 matrix
%   y      : N x 1 binary labels in {0,1}
%   cfg    : configuration structure with fields controlling the alternating
%            optimization, cross-validation, class weighting, and scaling.
%
% Output:
%   results : structure containing LOOCV probabilities, predicted labels,
%             fold-specific models, learned projection vectors, learned
%             logistic weights, and a full-data model for visualization.

    if ~isfield(cfg,'rng_seed'), cfg.rng_seed = 42; end
    rng(cfg.rng_seed,'twister');

    N = numel(Xcells);
    p_all    = nan(N,1);
    yhat_all = nan(N,1);

    % Store per-fold models for downstream visualization.
    fold_models = cell(N,1);

    % Store learned v and B parameters across LOOCV folds.
    m = size(Xcells{1},1);   % 9
    p = size(Xcells{1},2);   % 20
    J = cfg.J;
    V_all = nan(m, J, N);
    B_all = nan(p, J, N);

    for i = 1:N
        trMask = true(N,1); trMask(i) = false;
        trIdx  = find(trMask);
        teIdx  = i;

        % Train on N-1 subjects.
        model = train_alt_joint(Xcells, y, trIdx, cfg);

        % Predict the held-out subject.
        [p_te, yhat_te] = predict_alt_joint(Xcells, teIdx, model);

        p_all(i)    = p_te;
        yhat_all(i) = yhat_te;

        % Store model parameters for fold-level summaries.
        fold_models{i} = model;

        % v vectors
        for j = 1:J
            V_all(:,j,i) = model.v_list{j}(:);
        end

        % B blocks: model.B_concat is (p*J) x 1 = [B1; B2; ...]
        for j = 1:J
            cols = (j-1)*p + (1:p);
            B_all(:,j,i) = model.B_concat(cols);
        end
    end

    results.p_all       = p_all;
    results.yhat_all    = yhat_all;

    % Additional outputs for downstream visualization.
    results.fold_models = fold_models;
    results.V_all       = V_all;
    results.B_all       = B_all;

    % Train one full-data model for embedding/visualization.
    idxAll = (1:N).';
    results.Theta_full = train_alt_joint(Xcells, y, idxAll, cfg);
end

% ========================================================================
%                              TRAIN / PRED
% ========================================================================

function model = train_alt_joint(Xcells, y, idxTrain, cfg)
% TRAIN_ALT_JOINT
% Alternating training for:
%   p_i = sigmoid( b0 + Z_i * B_concat )
% where Z_i = [Xi'v_1; ...; Xi'v_J]'  (concatenate 20-d blocks)

    J   = cfg.J;
    ytr = y(idxTrain);

    % ---- init v_j from pooled block PCs (explicit orthogonalization)
    v_list = init_v_from_block_pcs(Xcells, idxTrain, J);

    % ---- class weights (imbalance)
    if isfield(cfg,'balanceWeights') && cfg.balanceWeights
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

    % ---- placeholders
    b0 = 0;
    P  = size(Xcells{1},2);                % 20
    B_concat = zeros(J*P,1);
    muZ = [];
    sigZ = [];

    altIters   = getfield_default(cfg,'altIters',6);
    vGradSteps = getfield_default(cfg,'vGradSteps',12);
    vStepSize  = getfield_default(cfg,'vStepSize',0.15);
    orthOn     = getfield_default(cfg,'orthOn',true);

    cvFolds    = getfield_default(cfg,'cvFolds',5);
    useOneSE   = getfield_default(cfg,'useOneSE',true);
    standardize= getfield_default(cfg,'standardize',true);

    for it = 1:altIters

        % ----- (A) Sparse B step (lassoglm on standardized Z)
        Ztr = build_Z(Xcells, idxTrain, v_list);           % Ntr x (20J)
        [ZtrZ, muZ, sigZ] = zscore_train(Ztr, standardize);

        [B_concat, b0] = fit_l1_logistic(ZtrZ, ytr, w_tr, cvFolds, useOneSE);

        % Split B into blocks and grab sigZ blocks
        B_list = cell(J,1);
        sig_blocks = cell(J,1);
        for j = 1:J
            cols = (j-1)*P + (1:P);
            B_list{j}     = B_concat(cols);
            sig_blocks{j} = sigZ(cols).';      % row -> make column below
            sig_blocks{j} = sig_blocks{j}(:);  % P x 1
        end

        % ----- (B) Gradient steps on v given B (keep b0,B fixed)
        ZtrZ = (Ztr - muZ) ./ sigZ;
        lin  = b0 + ZtrZ * B_concat;
        p_i  = 1 ./ (1 + exp(-lin));
        r_i  = w_tr .* (ytr - p_i);            % Ntr x 1

        for gs = 1:vGradSteps

            for j = 1:J
                % chain rule through standardization:
                % d/dv uses Btilde = B ./ sigZ_block
                Btilde_j = B_list{j} ./ sig_blocks{j};  % P x 1

                g = zeros(size(v_list{j}));
                for t = 1:numel(idxTrain)
                    Xi = Xcells{idxTrain(t)};           % 9x20
                    g  = g + r_i(t) * (Xi * Btilde_j);  % 9x1
                end

                v_list{j} = v_list{j} + vStepSize * g;
                nv = norm(v_list{j});
                if nv > 0
                    v_list{j} = v_list{j} / nv;
                end
            end

            % Optional Gram–Schmidt orthogonalization each step
            if orthOn && J >= 2
                v_list = gramschmidt_vlist(v_list);
            end

            % Refresh residuals after updating v
            Ztr  = build_Z(Xcells, idxTrain, v_list);
            ZtrZ = (Ztr - muZ) ./ sigZ;
            lin  = b0 + ZtrZ * B_concat;
            p_i  = 1 ./ (1 + exp(-lin));
            r_i  = w_tr .* (ytr - p_i);
        end
    end

    model.J        = J;
    model.v_list   = v_list;
    model.B_concat = B_concat(:);
    model.b0       = b0;
    model.muZ      = muZ;
    model.sigZ     = sigZ;
end

function [p, yhat] = predict_alt_joint(Xcells, idx, model)
% PREDICT_ALT_JOINT
% Standardize with TRAIN stats and predict

    Z  = build_Z(Xcells, idx, model.v_list);
    Zz = (Z - model.muZ) ./ model.sigZ;
    lin = model.b0 + Zz * model.B_concat;
    p = 1 ./ (1 + exp(-lin));
    yhat = (p >= 0.5);
end

% ========================================================================
%                                HELPERS
% ========================================================================

function v_list = init_v_from_block_pcs(Xcells, idxTrain, J)
% INIT_V_FROM_BLOCK_PCS
% Initialize v_j as top J PCs of C = mean_i Xi*Xi' in block-space (9x9),
% Projection vectors are explicitly orthogonalized for reproducibility.
    m = size(Xcells{1},1); % 9
    C = zeros(m,m);
    for t = 1:numel(idxTrain)
        Xi = Xcells{idxTrain(t)};
        C  = C + Xi*Xi.';
    end
    C = C / max(1,numel(idxTrain));

    [U,~,~] = svd(C,'econ');

    v_list = cell(J,1);

    % v1
    v1 = U(:,1);
    v1 = v1 / max(norm(v1), eps);
    v_list{1} = v1;

    % v2..vJ with explicit orthogonalization against previous
    for j = 2:J
        v = U(:,j);
        for k = 1:(j-1)
            v = v - (v_list{k}'*v) * v_list{k};
        end
        v = v / max(norm(v), eps);
        v_list{j} = v;
    end
end

function Z = build_Z(Xcells, idx, v_list)
% BUILD_Z
% Z(i,:) = [Xi'v1; Xi'v2; ...]' (concatenate 20-d blocks)
    J = numel(v_list);
    P = size(Xcells{1},2);   % 20
    Z = zeros(numel(idx), J*P);
    for t = 1:numel(idx)
        Xi = Xcells{idx(t)};
        row = zeros(J*P,1);
        for j = 1:J
            row((j-1)*P + (1:P)) = Xi' * v_list{j};
        end
        Z(t,:) = row(:).';
    end
end

function [Zz, muZ, sigZ] = zscore_train(Z, doStd)
% ZSCORE_TRAIN
% Standardize using TRAIN-only stats
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
% FIT_L1_LOGISTIC
% L1 logistic via lassoglm, selecting 1SE (or MinDeviance)
    numLambda   = 60;
    lambdaRatio = 1e-3;
    reltol      = 1e-4;
    maxiter     = 5000;

    [B,FitInfo] = lassoglm( ...
        ZtrZ, ytr, 'binomial', ...
        'Alpha', 1, ...
        'CV', cvFolds, ...
        'Weights', w_tr, ...
        'Standardize', false, ...
        'NumLambda', numLambda, ...
        'LambdaRatio', lambdaRatio, ...
        'RelTol', reltol, ...
        'MaxIter', maxiter);

    idx = FitInfo.IndexMinDeviance;
    if useOneSE
        idx = FitInfo.Index1SE;
    end

    B_concat = B(:,idx);

    if ~any(B_concat) && useOneSE
        % Use the minimum-deviance model if the 1SE solution is fully sparse.
        idx = FitInfo.IndexMinDeviance;
        B_concat = B(:,idx);
    end

    b0 = FitInfo.Intercept(idx);
end

function v_list = gramschmidt_vlist(v_list)
% GRAMSCHMIDT_VLIST
% Classical Gram–Schmidt on the list of v's (each 9x1)
    J = numel(v_list);
    for j = 1:J
        v = v_list{j};
        for k = 1:(j-1)
            v = v - (v_list{k}'*v) * v_list{k};
        end
        v = v / max(norm(v), eps);
        v_list{j} = v;
    end
end

function val = getfield_default(s, name, def)
% GETFIELD_DEFAULT
    if isfield(s, name)
        val = s.(name);
    else
        val = def;
    end
end
