%% Freud_Audit_Task_Switch.m
% Linear mixed-effects audit of post-switch trial adjustment.
%
% This diagnostic script tests whether SI group status interacts with the
% first K trials after each task switch. It does not generate figures.
%
% Required input:
%   Freud_Processed_BDIAT.mat
%
% Expected variables:
%   XF           : subjects x 360 processed log-reaction-time matrix
%   active_score : subjects x 1 group label, where 0 = SI- and 1 = SI+

clear; close all; clc;

%% ----------------------- User settings -----------------------
K = 6;  % number of early post-switch trials to test

%% ----------------------- Load data ---------------------------
load('Freud_Processed_BDIAT.mat');

X = exp(XF);                 % convert processed log RT back to RT scale
G = active_score(:);         % group label: 0 = SI-, 1 = SI+

[n, T] = size(X);
assert(T == 360, 'Expected 360 trials: 18 blocks x 20 trials.');
assert(numel(G) == n, 'active_score must match the number of subjects.');

%% ----------------------- Build trial/block predictors -----------------------
trial = (1:T)';
block = ceil(trial / 20);
trialInBlock = mod(trial - 1, 20) + 1;

% Odd/even block type for ABAB task structure.
blockType = mod(block, 2);              % 1 = odd block, 0 = even block

% Every block after the first is a switch block.
switchBlock = double(block > 1);

% Switch direction: +1 or -1 for alternating block transitions.
prevType = [blockType(1); blockType(1:end-1)];
switchDir = zeros(size(blockType));
switchDir(block > 1) = blockType(block > 1) - prevType(block > 1);

%% ----------------------- Convert to long format -----------------------
Subj = repelem((1:n)', T, 1);
Group = repelem(G, T, 1);

Trial = repmat(trial, n, 1);
Block = repmat(block, n, 1);
TrialInBlock = repmat(trialInBlock, n, 1);
BlockType = repmat(blockType, n, 1);
Switch = repmat(switchBlock, n, 1);
SwitchDir = repmat(switchDir, n, 1);

RT = X';
RT = RT(:);

% Remove invalid values before model fitting.
ok = isfinite(RT) & RT > 0 & isfinite(Group);

RT = RT(ok);
Subj = Subj(ok);
Group = Group(ok);
Trial = Trial(ok);
Block = Block(ok);
TrialInBlock = TrialInBlock(ok);
BlockType = BlockType(ok);
Switch = Switch(ok);
SwitchDir = SwitchDir(ok);

logRT = log(RT);

tbl = table( ...
    categorical(Subj), ...
    categorical(Group), ...
    Trial, ...
    Block, ...
    TrialInBlock, ...
    BlockType, ...
    Switch, ...
    SwitchDir, ...
    logRT, ...
    RT, ...
    'VariableNames', {'Subj','Group','Trial','Block','TrialInBlock', ...
                      'BlockType','Switch','SwitchDir','logRT','RT'});

%% ----------------------- Post-switch indicators -----------------------
Post = zeros(height(tbl), 1);
Post(tbl.Switch == 1) = tbl.TrialInBlock(tbl.Switch == 1);

for k = 1:K
    tbl.(sprintf('PS%d', k)) = double(Post == k);
end

% Center nuisance covariates.
tbl.BlockC = tbl.Block - mean(tbl.Block);
tbl.TrialInBlockC = tbl.TrialInBlock - mean(tbl.TrialInBlock);

%% ----------------------- Fit mixed-effects model -----------------------
psTerms = strjoin(arrayfun(@(k) sprintf('PS%d', k), 1:K, ...
    'UniformOutput', false), ' + ');

psIntTerms = strjoin(arrayfun(@(k) sprintf('Group:PS%d', k), 1:K, ...
    'UniformOutput', false), ' + ');

formula_full = sprintf(['logRT ~ 1 + Group + BlockType + BlockC + TrialInBlockC + ' ...
                        '%s + %s + (1 + %s | Subj)'], ...
                        psTerms, psIntTerms, psTerms);

lme_full = fitlme(tbl, formula_full, 'FitMethod', 'REML');

fprintf('\n===== Full post-switch LME model =====\n');
disp(lme_full);

%% ----------------------- Individual Group x PS tests -----------------------
coefNames = lme_full.CoefficientNames;
nCoef = numel(coefNames);

pIndividual = nan(K, 1);

fprintf('\n===== Individual Group x post-switch trial tests =====\n');

for k = 1:K
    targetName = sprintf('Group_1:PS%d', k);
    idx = strcmp(coefNames, targetName);

    if ~any(idx)
        warning('Coefficient %s was not found in the fitted model.', targetName);
        continue;
    end

    H = zeros(1, nCoef);
    H(idx) = 1;

    pIndividual(k) = coefTest(lme_full, H);

    fprintf('Group x PS%d: p = %.4g\n', k, pIndividual(k));
end

%% ----------------------- Joint Group x PS test -----------------------
idxJoint = false(1, nCoef);

for k = 1:K
    idxJoint = idxJoint | strcmp(coefNames, sprintf('Group_1:PS%d', k));
end

C = zeros(nnz(idxJoint), nCoef);
C(:, idxJoint) = eye(nnz(idxJoint));

pJoint = coefTest(lme_full, C);

fprintf('\nJoint p-value for all Group x PS terms: %.4g\n', pJoint);

%% ----------------------- Save audit results -----------------------
TaskSwitchAudit = struct();
TaskSwitchAudit.model = lme_full;
TaskSwitchAudit.formula = formula_full;
TaskSwitchAudit.pIndividual = pIndividual;
TaskSwitchAudit.pJoint = pJoint;
TaskSwitchAudit.K = K;

save('Freud_Audit_Task_Switch_Results.mat', 'TaskSwitchAudit');

disp('Saved Freud_Audit_Task_Switch_Results.mat');