function results = Freud_Model_CrossVal_Fixed(Xcells, y, cfg)
% FREUD_MODEL_CROSSVAL_FIXED
% LOOCV for the model:
%   p_i = sigmoid( b0 + sum_j B_j' * (X_i' * v_j) )
% where v_j are FIXED from PCs of sum_i (Xi * Xi') (per fold).

    if isfield(cfg,'rng_seed') && ~isempty(cfg.rng_seed)
        rng(cfg.rng_seed,'twister');   % <-- makes fixed results independent of ALT
    end

    N = numel(Xcells);
    p_all    = nan(N,1);
    yhat_all = nan(N,1);

    for i = 1:N
        trMask    = true(N,1);
        trMask(i) = false;
        idxTrain  = find(trMask);
        idxTest   = i;

        model = train_fixedV_joint(Xcells, y, idxTrain, cfg);
        [p_te, yhat_te] = predict_alt_joint(Xcells, idxTest, model);

        p_all(i)    = p_te;
        yhat_all(i) = yhat_te;
    end

    results.p_all    = p_all;
    results.yhat_all = yhat_all;
end

function model = train_fixedV_joint(Xcells, y, idxTrain, cfg)
% TRAIN_FIXEDV_JOINT
% Fixed-v version of the joint model:
%   p_i = sigmoid( b0 + sum_{j=1..J} B_j' * (X_i' * v_j) )
% where:
%   - X_i is 9x20
%   - v_j is 9x1, obtained from top J PCs of sum_i (X_i X_i')
%   - B_j is 20x1, learned via L1-logistic (lassoglm)
%   - Features can be standardized (train-only) before L1 step.
%
% Inputs:
%   Xcells   : N x 1 cell of 9x20 matrices (each Xi)
%   y        : N x 1 binary labels in {0,1}
%   idxTrain : vector of train indices
%   cfg      : struct with fields:
%                J, cvFolds, useOneSE, balanceWeights, standardize
%
% Output:
%   model : struct with fields:
%             J
%             v_list    : 1xJ cell, each 9x1 (fixed)
%             B_concat  : (20*J) x 1
%             b0        : scalar intercept
%             muZ, sigZ : 1 x (20*J) standardization stats

    J   = cfg.J;
    ytr = y(idxTrain);

    % ---- (1) Initialize v_j from block PCs (fixed for this fold)
    v_list = init_v_from_block_pcs(Xcells, idxTrain, J);

    % ---- (2) Class weights for imbalance
    if cfg.balanceWeights
        nPos = sum(ytr==1);
        nNeg = sum(ytr==0);
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

    % ---- (3) Build Z using fixed v_j
    % Z(i,:) = [Xi'v_1; Xi'v_2; ...]' (concatenated 20-d blocks)
    Ztr = build_Z(Xcells, idxTrain, v_list);   % Ntr x (20*J)

    % ---- (4) Standardize features (train-only)
    [ZtrZ, muZ, sigZ] = zscore_train(Ztr, cfg.standardize);

    % ---- (5) Fit sparse logistic (lassoglm) on ZtrZ
    [B_concat, b0] = fit_l1_logistic(ZtrZ, ytr, w_tr, ...
                                     cfg.cvFolds, cfg.useOneSE);

    % ---- (6) Pack model
    model.J        = J;
    model.v_list   = v_list;           % 1xJ cell, each 9x1
    model.B_concat = B_concat(:);      % (20J)x1
    model.b0       = b0;
    model.muZ      = muZ;              % 1 x (20J)
    model.sigZ     = sigZ;             % 1 x (20J)
end

function [p, yhat] = predict_alt_joint(Xcells, idx, model)
% PREDICT_ALT_JOINT
% Prediction for:
%   p_i = sigmoid( b0 + sum_j B_j' * (X_i' * v_j) )
% using stored v_list, B_concat, b0, and standardization stats.

    Z = build_Z(Xcells, idx, model.v_list);        % nTest x (20J)
    Zz = (Z - model.muZ) ./ model.sigZ;            % same transform as train
    lin = model.b0 + Zz * model.B_concat;
    p = 1 ./ (1 + exp(-lin));
    yhat = (p >= 0.5);
end

% =========================== Helper functions ============================

function v_list = init_v_from_block_pcs(Xcells, idxTrain, J)
% INIT_V_FROM_BLOCK_PCS
% Initialize v_j as top J block-space PCs from:
%   C = (1/Ntr) * sum_i (Xi * Xi')

    C = zeros(9,9);
    for t = 1:numel(idxTrain)
        Xi = Xcells{idxTrain(t)};      % 9x20
        C  = C + Xi*Xi.';
    end
    C = C / max(1,numel(idxTrain));

    [U,~,~] = svd(C,'econ');

    v_list = cell(J,1);

    % v1
    v_list{1} = U(:,1) / max(norm(U(:,1)), eps);

    % v2..vJ (same “deflation” style as your v2 line, generalized)
    for j = 2:J
        vj = U(:,j);
        for k = 1:(j-1)
            vj = vj - (v_list{k}' * vj) * v_list{k};
        end
        v_list{j} = vj / max(norm(vj), eps);
    end
end

function Z = build_Z(Xcells, idx, v_list)
% BUILD_Z
% Z(i,:) = [Xi'v_1; Xi'v_2; ...]' (concatenated 20-d blocks; row vector)

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
% ZSCORE_TRAIN
% Standardize Z (train-only). If doStd=false, returns identity transform.

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
% Thin wrapper around lassoglm for L1-penalized logistic regression.

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
    if useOneSE
        idx = FitInfo.Index1SE;
    end

    B_concat = B(:,idx);
    if ~any(B_concat) && useOneSE
        % Fallback to MinDeviance if 1SE gives all zeros
        idx = FitInfo.IndexMinDeviance;
        B_concat = B(:,idx);
    end
    b0 = FitInfo.Intercept(idx);
end
