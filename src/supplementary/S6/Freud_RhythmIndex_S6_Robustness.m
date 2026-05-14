%% Freud_RhythmIndex_S6_Robustness.m
% Rhythm-index robustness analysis for Figure S6.
%
% This supplementary analysis evaluates whether the observed block-level
% rhythm index exceeds participant-specific null expectations under two
% shuffle controls:
%
%   1. Destroy-ABAB shuffle:
%      destroys block-order structure by permuting blocks.
%
%   2. Preserve-ABAB shuffle:
%      preserves ABAB task alternation while testing whether the observed
%      rhythm index remains elevated relative to matched null structure.
%
% For each participant, the observed rhythm index is compared with the mode
% of that participant's null rhythm-index distribution. One-sided Wilcoxon
% signed-rank tests are then used to test whether observed-minus-null
% differences are greater than zero.
% Datasets:
%   - Main BD-IAT dataset   : Freud_Processed_BDIAT.mat
%                             20 trials/block x 18 blocks
%   - Short/online dataset  : Freud_Processed_BDIAT_Short.mat
%                             10 trials/block x 36 blocks

clear; close all; clc;

%% --------------------------- User settings ------------------------------
numShuffleDestroy  = 10000;
numShufflePreserve = 10000;

numMainPreserveOutliersToRemove = 4;
nModeBins = 50;

seedMainPerm       = 1;
seedMainPreserve   = 2;
seedMainDestroy    = 3;

seedOnlinePerm     = 11;
seedOnlinePreserve = 12;
seedOnlineDestroy  = 13;

outDir = pwd;

%% --------------------------- Main dataset ------------------------------
main = run_dataset_analysis( ...
    'dataFile', 'Freud_Processed_BDIAT.mat', ...
    'blockLen', 20, ...
    'numBlocks', 18, ...
    'maxLagBlock', 8, ...
    'rhythmMaxLag', 8, ...
    'numShuffleDestroy', numShuffleDestroy, ...
    'numShufflePreserve', numShufflePreserve, ...
    'numPreserveOutliersToRemove', numMainPreserveOutliersToRemove, ...
    'numModeBins', nModeBins, ...
    'blockSummary', 'mean', ...
    'useLog', false, ...
    'detrendBlocks', true, ...
    'seedPerm', seedMainPerm, ...
    'seedPreserve', seedMainPreserve, ...
    'seedDestroy', seedMainDestroy, ...
    'figurePrefix', 'MAIN');

%% --------------------------- Online dataset ----------------------------
online = run_dataset_analysis( ...
    'dataFile', 'Freud_Processed_BDIAT_Short.mat', ...
    'blockLen', 10, ...
    'numBlocks', 36, ...
    'maxLagBlock', 12, ...
    'rhythmMaxLag', 8, ...
    'numShuffleDestroy', numShuffleDestroy, ...
    'numShufflePreserve', numShufflePreserve, ...
    'numPreserveOutliersToRemove', 0, ...
    'numModeBins', nModeBins, ...
    'blockSummary', 'mean', ...
    'useLog', false, ...
    'detrendBlocks', true, ...
    'seedPerm', seedOnlinePerm, ...
    'seedPreserve', seedOnlinePreserve, ...
    'seedDestroy', seedOnlineDestroy, ...
    'figurePrefix', 'ONLINE');

%% --------------------------- Export figures ----------------------------
export_s6_panel(main,   'destroy',  fullfile(outDir,'Figure_S6_A.svg'));
export_s6_panel(main,   'preserve', fullfile(outDir,'Figure_S6_B.svg'));
export_s6_panel(online, 'destroy',  fullfile(outDir,'Figure_S6_C.svg'));
export_s6_panel(online, 'preserve', fullfile(outDir,'Figure_S6_D.svg'));

%% --------------------------- Save results ------------------------------
save(fullfile(outDir,'rhythm_s6_results.mat'), 'main', 'online');

fprintf('\nSaved results to: %s\n', fullfile(outDir,'rhythm_s6_results.mat'));
fprintf('Saved figures:\n');
fprintf('  %s\n', fullfile(outDir,'Figure_S6_A.svg'));
fprintf('  %s\n', fullfile(outDir,'Figure_S6_B.svg'));
fprintf('  %s\n', fullfile(outDir,'Figure_S6_C.svg'));
fprintf('  %s\n', fullfile(outDir,'Figure_S6_D.svg'));

%% ========================== Local analysis functions =============================
function S = run_dataset_analysis(varargin)

p = inputParser;
p.addParameter('dataFile', '', @(s)ischar(s)||isstring(s));
p.addParameter('blockLen', 20, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('numBlocks', 18, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('maxLagBlock', 8, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
p.addParameter('rhythmMaxLag', 8, @(x)isnumeric(x)&&isscalar(x)&&x>=2);
p.addParameter('numShuffleDestroy', 10000, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('numShufflePreserve', 10000, @(x)isnumeric(x)&&isscalar(x)&&x>0);
p.addParameter('numPreserveOutliersToRemove', 0, @(x)isnumeric(x)&&isscalar(x)&&x>=0);
p.addParameter('numModeBins', 50, @(x)isnumeric(x)&&isscalar(x)&&x>=5);
p.addParameter('blockSummary', 'mean', @(s)ischar(s)||isstring(s));
p.addParameter('useLog', false, @(b)islogical(b)&&isscalar(b));
p.addParameter('detrendBlocks', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('seedPerm', 1, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('seedPreserve', 2, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('seedDestroy', 3, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('figurePrefix', 'DATA', @(s)ischar(s)||isstring(s));
p.parse(varargin{:});
R = p.Results;

% ---------- Load data ----------
D = load(R.dataFile);
assert(isfield(D,'XF') && isfield(D,'active_score'), ...
    'Expected XF and active_score in %s.', R.dataFile);

X = exp(D.XF);
G = double(D.active_score(:));

% ---------- Core rhythm analysis ----------
out = Freud_Autocorr_Advanced(X, G, ...
    'DoTrialACF', false, ...
    'DoBlockACF', true, ...
    'MaxLagBlock', R.maxLagBlock, ...
    'RhythmMaxLag', R.rhythmMaxLag, ...
    'NumPerm', 5000, ...
    'NumShufflePreserve', R.numShufflePreserve, ...
    'NumShuffleDestroy', R.numShuffleDestroy, ...
    'ShuffleDestroyMode', 'permuteBlocks', ...
    'FigureNamePrefix', char(R.figurePrefix), ...
    'BlockSummary', char(R.blockSummary), ...
    'UseLog', R.useLog, ...
    'DetrendBlocks', R.detrendBlocks, ...
    'BlockLen', R.blockLen, ...
    'NumBlocks', R.numBlocks, ...
    'Seed', R.seedPerm, ...
    'ShufflePreserveSeed', R.seedPreserve, ...
    'ShuffleDestroySeed', R.seedDestroy, ...
    'MakePlots', false);

ri_obs = out.rhythmIndex(:);

% ---------- Destroy-ABAB null comparison ----------
nullRI_destroy = out.tests.shuffleDestroyAB.perSubject.nullRI;
ri_destroy_mode = compute_hist_mode_per_subject(nullRI_destroy, R.numModeBins);
diff_destroy = ri_obs - ri_destroy_mode;
destroy_stats = signedrank_right_test(diff_destroy);

% ---------- Preserve-ABAB null comparison ----------
nullRI_preserve = out.tests.shufflePreserveAB.perSubject.nullRI;
ri_preserve_mode = compute_hist_mode_per_subject(nullRI_preserve, R.numModeBins);
diff_preserve = ri_obs - ri_preserve_mode;

preserve_full = signedrank_right_test(diff_preserve);

% Main-dataset-only sensitivity analysis after removing the largest
% positive preserve-null differences.
preserve_trimmed = struct();
preserve_trimmed.applied = false;
preserve_trimmed.removed_indices = [];
preserve_trimmed.removed_values = [];
preserve_trimmed.remaining_values = diff_preserve;
preserve_trimmed.n_remaining = sum(isfinite(diff_preserve));
preserve_trimmed.wilcoxon = [];

if R.numPreserveOutliersToRemove > 0
    preserve_trimmed.applied = true;

    x = diff_preserve(:);
    validIdx = find(isfinite(x));
    validVals = x(validIdx);

    [~, ord] = sort(validVals, 'descend');
    nRemove = min(R.numPreserveOutliersToRemove, numel(ord));
    rmLocal = ord(1:nRemove);
    rmIdx   = validIdx(rmLocal);

    keepMask = true(size(x));
    keepMask(rmIdx) = false;

    preserve_trimmed.removed_indices = rmIdx(:);
    preserve_trimmed.removed_values  = x(rmIdx);
    preserve_trimmed.remaining_values = x(keepMask);
    preserve_trimmed.n_remaining = sum(isfinite(preserve_trimmed.remaining_values));
    preserve_trimmed.wilcoxon = signedrank_right_test(preserve_trimmed.remaining_values);
end

% ---------- Package ----------
S = struct();
S.meta = struct();
S.meta.dataFile = char(R.dataFile);
S.meta.blockLen = R.blockLen;
S.meta.numBlocks = R.numBlocks;
S.meta.maxLagBlock = R.maxLagBlock;
S.meta.rhythmMaxLag = R.rhythmMaxLag;
S.meta.nParticipants = numel(ri_obs);
S.meta.numShuffleDestroy = R.numShuffleDestroy;
S.meta.numShufflePreserve = R.numShufflePreserve;
S.meta.numPreserveOutliersToRemove = R.numPreserveOutliersToRemove;
S.meta.numModeBins = R.numModeBins;
S.meta.figurePrefix = char(R.figurePrefix);

S.G = G;
S.ri_obs = ri_obs;
S.ri_destroy_mode = ri_destroy_mode;
S.ri_preserve_mode = ri_preserve_mode;
S.diff_destroy = diff_destroy;
S.diff_preserve = diff_preserve;
S.out = out;
S.destroy = destroy_stats;
S.preserve.full = preserve_full;
S.preserve.trimmed = preserve_trimmed;

% ---------- Terminal output ----------
fprintf('\n============================================================\n');
fprintf('%s\n', char(R.figurePrefix));
fprintf('File: %s\n', char(R.dataFile));
fprintf('BlockLen = %d | NumBlocks = %d | MaxLagBlock = %d | RhythmMaxLag = %d\n', ...
    R.blockLen, R.numBlocks, R.maxLagBlock, R.rhythmMaxLag);
fprintf('n = %d participants\n', S.meta.nParticipants);

fprintf('\n[Destroy-ABAB paired superiority | mode-based]\n');
fprintf('Data sizes:\n');
fprintf('  Observed RI values               : %d\n', numel(ri_obs));
fprintf('  Destroy null matrix size         : %d x %d\n', ...
    size(nullRI_destroy,1), size(nullRI_destroy,2));
fprintf('  Participant-level destroy modes  : %d\n', numel(ri_destroy_mode));
fprintf('  Paired differences               : %d\n', numel(diff_destroy));

fprintf('Statistics:\n');
fprintf('  Mean diff    = %.4f\n', destroy_stats.mean_diff);
fprintf('  SD diff      = %.4f\n', destroy_stats.sd_diff);
fprintf('  Median diff  = %.4f\n', destroy_stats.median_diff);
fprintf('  Signed-rank  = %.4f\n', destroy_stats.signedrank);
fprintf('  z            = %.4f\n', destroy_stats.zval);
fprintf('  One-sided p  = %.6g\n', destroy_stats.p_one_sided);
fprintf('  Positives    = %d\n', destroy_stats.n_positive);
fprintf('  Negatives    = %d\n', destroy_stats.n_negative);
fprintf('  Zeros        = %d\n', destroy_stats.n_zero);

fprintf('\n[Preserve-ABAB paired analysis | mode-based]\n');
fprintf('Data sizes:\n');
fprintf('  Observed RI values               : %d\n', numel(ri_obs));
fprintf('  Preserve null matrix size        : %d x %d\n', ...
    size(nullRI_preserve,1), size(nullRI_preserve,2));
fprintf('  Participant-level preserve modes : %d\n', numel(ri_preserve_mode));
fprintf('  Paired differences               : %d\n', numel(diff_preserve));

fprintf('\nFull-data Wilcoxon signed-rank:\n');
fprintf('  Mean diff    = %.4f\n', preserve_full.mean_diff);
fprintf('  SD diff      = %.4f\n', preserve_full.sd_diff);
fprintf('  Median diff  = %.4f\n', preserve_full.median_diff);
fprintf('  Signed-rank  = %.4f\n', preserve_full.signedrank);
fprintf('  z            = %.4f\n', preserve_full.zval);
fprintf('  One-sided p  = %.6g\n', preserve_full.p_one_sided);
fprintf('  Positives    = %d\n', preserve_full.n_positive);
fprintf('  Negatives    = %d\n', preserve_full.n_negative);
fprintf('  Zeros        = %d\n', preserve_full.n_zero);

if preserve_trimmed.applied
    fprintf('\nTrimmed Wilcoxon (top %d positive outliers removed):\n', R.numPreserveOutliersToRemove);
    fprintf('  Removed indices : ');
    fprintf('%d ', preserve_trimmed.removed_indices);
    fprintf('\n');
    fprintf('  Removed values  : ');
    fprintf('%.4f ', preserve_trimmed.removed_values);
    fprintf('\n');
    fprintf('  Remaining n     = %d\n', preserve_trimmed.n_remaining);
    fprintf('  Mean diff       = %.4f\n', preserve_trimmed.wilcoxon.mean_diff);
    fprintf('  SD diff         = %.4f\n', preserve_trimmed.wilcoxon.sd_diff);
    fprintf('  Median diff     = %.4f\n', preserve_trimmed.wilcoxon.median_diff);
    fprintf('  Signed-rank     = %.4f\n', preserve_trimmed.wilcoxon.signedrank);
    fprintf('  z               = %.4f\n', preserve_trimmed.wilcoxon.zval);
    fprintf('  One-sided p     = %.6g\n', preserve_trimmed.wilcoxon.p_one_sided);
    fprintf('  Positives       = %d\n', preserve_trimmed.wilcoxon.n_positive);
    fprintf('  Negatives       = %d\n', preserve_trimmed.wilcoxon.n_negative);
    fprintf('  Zeros           = %d\n', preserve_trimmed.wilcoxon.n_zero);
end

fprintf('============================================================\n');
end

function modeVals = compute_hist_mode_per_subject(nullRI, nBins)
modeVals = zeros(size(nullRI,1),1);
for i = 1:size(nullRI,1)
    vals = nullRI(i,:);
    vals = vals(isfinite(vals));
    if isempty(vals)
        modeVals(i) = NaN;
    else
        [counts, edges] = histcounts(vals, nBins);
        [~, idx] = max(counts);
        modeVals(i) = mean(edges(idx:idx+1));
    end
end
end

function stats = signedrank_right_test(x)
x = x(:);
x = x(isfinite(x));
n = numel(x);
assert(n >= 2, 'Need at least 2 finite values.');

m   = mean(x);
sd  = std(x,0);
med = median(x);

n_positive = sum(x > 0);
n_negative = sum(x < 0);
n_zero     = sum(x == 0);

[p_right,~,sr_stats] = signrank(x, 0, 'tail', 'right');

stats = struct();
stats.n = n;
stats.mean_diff = m;
stats.sd_diff = sd;
stats.median_diff = med;
stats.p_one_sided = p_right;

if isfield(sr_stats,'signedrank')
    stats.signedrank = sr_stats.signedrank;
else
    stats.signedrank = NaN;
end

if isfield(sr_stats,'zval')
    stats.zval = sr_stats.zval;
else
    stats.zval = NaN;
end

stats.n_positive = n_positive;
stats.n_negative = n_negative;
stats.n_zero = n_zero;
stats.values = x;
end

function export_s6_panel(S, mode, outFile)

labelFS  = 24;
tickFS   = 20;
legendFS = 20;

f = figure('Color','w','Position',[100 100 900 520]);
set(f,'Renderer','painters');

ax = axes('Parent',f); 
hold(ax,'on');
grid(ax,'off');
box(ax,'off');

switch lower(mode)
    case 'destroy'
        xvals = S.diff_destroy;
        xvals = xvals(isfinite(xvals));
        stats = S.destroy;

        xLabelText = '$\mathrm{RI}_{\mathrm{obs}}-\mathrm{mode}(\mathrm{RI}_{\mathrm{destroy}})$';

    case 'preserve'
        xvals = S.diff_preserve;
        xvals = xvals(isfinite(xvals));
        stats = S.preserve.full;

        xLabelText = '$\mathrm{RI}_{\mathrm{obs}}-\mathrm{mode}(\mathrm{RI}_{\mathrm{preserve}})$';

    otherwise
        error('Mode must be ''destroy'' or ''preserve''.');
end

% ---------- Histogram ----------
hHist = histogram(ax, xvals, 40, 'Normalization','pdf');

% ---------- Labels ----------
xlabel(ax, xLabelText, 'FontSize', labelFS, 'Interpreter','latex');
ylabel(ax, 'Density', 'FontSize', labelFS);

set(ax, 'FontSize', tickFS, 'LineWidth', 1);
title(ax, '');

% ---------- Legend (ONLY histogram, with p-value) ----------
legendText = sprintf('Participant distribution (p = %.3g)', stats.p_one_sided);

lg = legend(ax, hHist, {legendText}, ...
    'Location','northoutside', ...
    'Orientation','horizontal');

set(lg, 'FontSize', legendFS, 'Box', 'off');

% ---------- Save ----------
set(f,'PaperPositionMode','auto');
print(f, outFile, '-dsvg', '-painters');
close(f);

fprintf('Saved %s\n', outFile);
end