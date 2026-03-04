% Load data
load('Freud_Processed_BDIAT.mat');
X = XF;
G = active_score;

%% Prepare Data
% X: n x 360 RT
% G: n x 1 (0/1)
[n, T] = size(X);
assert(T == 360, 'Expected 360 trials (18 blocks x 20).');

% Indices
trial = (1:T)';                              % 1..360
block = ceil(trial/20);                      % 1..18
trialInBlock = mod(trial-1, 20) + 1;         % 1..20

% Block type: A for odd blocks, B for even blocks
blockType = mod(block,2);                    % 1 for odd (A), 0 for even (B)
% (you can flip if you want: define A=0,B=1 etc)

% Switch indicator: since types alternate, every block after the first is a switch.
switchBlock = double(block > 1);             % 0 for block 1, 1 for blocks 2..18

% (Optional) switch direction (A->B vs B->A), for blocks 2..18
% direction = +1 for A->B, -1 for B->A (or 0 for block 1)
prevType = [blockType(1); blockType(1:end-1)];
switchDir = zeros(size(blockType));
switchDir(block>1) = (blockType(block>1) - prevType(block>1));  % will be +/-1

% Repeat predictors for each participant
Subj = repelem((1:n)', T, 1);
Group = repelem(G(:), T, 1);

Trial = repmat(trial, n, 1);
Block = repmat(block, n, 1);
TrialInBlock = repmat(trialInBlock, n, 1);
BlockType = repmat(blockType, n, 1);
Switch = repmat(switchBlock, n, 1);
SwitchDir = repmat(switchDir, n, 1);

% Response in long format
RT = X'; RT = RT(:);                         % column vector length n*T

% Basic cleaning: remove NaNs, <=0 RTs, etc.
ok = isfinite(RT) & RT > 0;
RT = RT(ok);
Subj = Subj(ok);
Group = Group(ok);
Trial = Trial(ok);
Block = Block(ok);
TrialInBlock = TrialInBlock(ok);
BlockType = BlockType(ok);
Switch = Switch(ok);
SwitchDir = SwitchDir(ok);

% Log-transform RT (recommended)
logRT = log(RT);

tbl = table( ...
    categorical(Subj), categorical(Group), ...
    Trial, Block, TrialInBlock, BlockType, Switch, SwitchDir, ...
    logRT, RT, ...
    'VariableNames', {'Subj','Group','Trial','Block','TrialInBlock','BlockType','Switch','SwitchDir','logRT','RT'});

%% Post Switch Adjustment
K = 6;  % number of early trials you consider "switch adjustment"

% Post-switch trial index: 1..20 for switched blocks, 0 for block 1
Post = zeros(height(tbl),1);
Post(tbl.Switch == 1) = tbl.TrialInBlock(tbl.Switch == 1);

% FIR indicators PS1..PSK (1 if that post-switch trial number matches)
PS = zeros(height(tbl), K);
for k = 1:K
    PS(:,k) = double(Post == k);
end

% Add to table
for k = 1:K
    tbl.(sprintf('PS%d',k)) = PS(:,k);
end

%% test the model
% Center/scaling options (helpful)
tbl.BlockC = tbl.Block - mean(tbl.Block);              % practice/fatigue proxy
tbl.TrialInBlockC = tbl.TrialInBlock - mean(tbl.TrialInBlock);

% Build formula string
psTerms = strjoin(arrayfun(@(k)sprintf('PS%d',k), 1:K, 'UniformOutput', false), ' + ');
psIntTerms = strjoin(arrayfun(@(k)sprintf('Group:PS%d',k), 1:K, 'UniformOutput', false), ' + ');

% Fixed effects:
% - Group main effect (overall speed difference)
% - BlockType (steady-state difference between types)
% - BlockC (learning/fatigue)
% - TrialInBlockC (warmup within blocks)
% - PS1..PSK capture early post-switch shape
% - Group:PSk is the key test: group difference specific to switch adjustment
formula_full = sprintf(['logRT ~ 1 + Group + BlockType + BlockC + TrialInBlockC + ' ...
                        '%s + %s + (1 + %s | Subj)'], ...
                        psTerms, psIntTerms, psTerms);

lme_full = fitlme(tbl, formula_full, 'FitMethod','REML');
disp(lme_full);

%% test Difference in first couple of trials
% Construct hypothesis: Group:PS1 = 0, ..., Group:PSK = 0
H = cell(K,1);
for k = 1:K
    H{k} = sprintf('Group_1:PS%d = 0', k);  % Group is categorical; Group_1 corresponds to value '1'
end

% You can test individually:
for k = 1:K
    fprintf('--- Testing Group x PS%d ---\n', k);
    disp(coefTest(lme_full, H{k}));
end

% Or joint test by building a contrast matrix:
coefNames = lme_full.CoefficientNames;
idx = false(size(coefNames));
for k = 1:K
    idx = idx | strcmp(coefNames, sprintf('Group_1:PS%d',k));
end
C = zeros(nnz(idx), numel(coefNames));
C(:, idx) = eye(nnz(idx));
pJoint = coefTest(lme_full, C);  % returns p-value
fprintf('Joint p-value for all Group×PS terms: %.4g\n', pJoint);
