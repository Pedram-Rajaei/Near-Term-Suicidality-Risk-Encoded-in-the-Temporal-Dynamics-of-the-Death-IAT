%% Multi-taper PSD by participant, group mean plots, and per-bin t-tests (Bonferroni)
% Inputs you already have:
%   active_score : [77 x 1] values are 0 or 1
%   XF           : [77 x 360] each row is one participant time series
%
% You MUST set fs (sampling rate, Hz) correctly for your data.

load('Freud_Processed_BDIAT.mat')

%% -------------------- USER SETTINGS --------------------
fs   = 1;     % <-- CHANGE THIS to your sampling rate (Hz)
NW   = 2.0;       % time-bandwidth product for pmtm (common: 2.5 to 4)
nfft = 360;    % FFT length (>= length of time series; can be larger for smoother freq grid)
alpha = 0.05;   % family-wise alpha for Bonferroni
useUnequalVar = true; % Welch's t-test if true
% --------------------------------------------------------

%% Basic checks
assert(size(XF,1) == numel(active_score), 'XF rows must match length(active_score).');
assert(all(ismember(active_score(:),[0 1])), 'active_score must contain only 0/1.');
XF = exp(double(XF));
active_score = active_score(:);

% Remove per-participant mean (often recommended for PSD comparisons)
XF = XF- mean(XF, 2);


Nsubj = size(XF,1);
T     = size(XF,2);

% Ensure nfft is at least T
nfft = max(nfft, T);

%% Compute multi-taper PSD for each participant
% pmtm returns one-sided PSD for real signals by default.
% Pxx: [Nfreq x Nsubj], f: [Nfreq x 1]
Pxx = [];
for i = 1:Nsubj
    [P_i, f] = pmtm(XF(i,:), NW, nfft, fs);  % P_i is [Nfreq x 1]
    if isempty(Pxx)
        Pxx = zeros(numel(P_i), Nsubj);
    end
    Pxx(:,i) = P_i;
end
f = f*nfft;
% Log-transform for stats
logPxx = log10(Pxx);   % log10 PSD

%% Group indices
g0 = (active_score == 0);
g1 = (active_score == 1);

assert(any(g0) && any(g1), 'Need at least 1 subject in each group (active_score 0 and 1).');

%% Group summaries (mean +/- SEM)
mean0 = mean(Pxx(:,g0), 2, 'omitnan');
mean1 = mean(Pxx(:,g1), 2, 'omitnan');
sem0  = std(Pxx(:,g0), 0, 2, 'omitnan') ./ sqrt(sum(g0));
sem1  = std(Pxx(:,g1), 0, 2, 'omitnan') ./ sqrt(sum(g1));

meanLog0 = mean(logPxx(:,g0), 2, 'omitnan');
meanLog1 = mean(logPxx(:,g1), 2, 'omitnan');
semLog0  = std(logPxx(:,g0), 0, 2, 'omitnan') ./ sqrt(sum(g0));
semLog1  = std(logPxx(:,g1), 0, 2, 'omitnan') ./ sqrt(sum(g1));

%% Plot: Group mean PSD (linear) with SEM shading
figure('Color','w'); hold on;

% Shaded error (manual, no toolbox required)
fill([f; flipud(f)], [mean0-sem0; flipud(mean0+sem0)], ...
     1, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(f, mean0, 'LineWidth', 2);

fill([f; flipud(f)], [mean1-sem1; flipud(mean1+sem1)], ...
     1, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(f, mean1, 'LineWidth', 2);

set(gca,'YScale','log'); % PSD is often displayed on log y-scale
xlabel('Frequency (Hz)');
ylabel('PSD (units^2/Hz)');
title('Multi-taper PSD: Group Means (active\_score = 0 vs 1)');
legend({'Group 0 SEM','Group 0 mean','Group 1 SEM','Group 1 mean'}, 'Location','best');
grid on;

%% Plot: Group mean log10(PSD) with SEM shading (often easier to interpret)
figure('Color','w'); hold on;

fill([f; flipud(f)], [meanLog0-semLog0; flipud(meanLog0+semLog0)], ...
     1, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(f, meanLog0, 'LineWidth', 2);

fill([f; flipud(f)], [meanLog1-semLog1; flipud(meanLog1+semLog1)], ...
     1, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(f, meanLog1, 'LineWidth', 2);

xlabel('Frequency (Hz)');
ylabel('log_{10}(PSD)');
title('Multi-taper log_{10}(PSD): Group Means (active\_score = 0 vs 1)');
legend({'Group 0 SEM','Group 0 mean','Group 1 SEM','Group 1 mean'}, 'Location','best');
grid on;

%% Per-frequency-bin t-tests on log10(PSD), Bonferroni adjustment
pvals = nan(numel(f),1);
tstat = nan(numel(f),1);

X0 = logPxx(:, g0);  % [Nfreq x N0]
X1 = logPxx(:, g1);  % [Nfreq x N1]

for k = 1:numel(f)
    if useUnequalVar
        [~, p, ~, stats] = ttest2(X0(k,:), X1(k,:), 'Vartype','unequal');
    else
        [~, p, ~, stats] = ttest2(X0(k,:), X1(k,:), 'Vartype','equal');
    end
    pvals(k) = p;
    tstat(k) = stats.tstat;
end

% Bonferroni
m = numel(f);
p_bonf = min(pvals * m, 1);
sig_bonf = p_bonf < alpha;

% Optional: report significant frequency bins
sigFreqs = f(sig_bonf);
fprintf('Bonferroni alpha = %.4g over %d bins\n', alpha, m);
fprintf('Significant bins (Bonferroni) count: %d\n', numel(sigFreqs));
if ~isempty(sigFreqs)
    fprintf('Significant frequencies (Hz):\n');
    disp(sigFreqs(:)');
end

%% Plot p-values (raw and Bonferroni-adjusted) + significance
figure('Color','w'); hold on;
plot(f, pvals, 'LineWidth', 1.5);
plot(f, p_bonf, 'LineWidth', 1.5);

yline(alpha, '--', 'alpha (0.05)');
xlabel('Frequency (Hz)');
ylabel('p-value');
title('Per-bin t-tests on log_{10}(PSD): Raw vs Bonferroni-adjusted');
legend({'Raw p','Bonferroni-adjusted p','alpha'}, 'Location','best');
grid on;
set(gca,'YScale','log'); % helps visualize small p-values

%% Plot t-statistics and mark Bonferroni-significant bins
figure('Color','w'); hold on;
plot(f, tstat, 'LineWidth', 1.5);
plot(f(sig_bonf), tstat(sig_bonf), 'o', 'MarkerSize', 5, 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('t-statistic');
title('t-statistics by frequency (markers = Bonferroni significant)');
legend({'t-stat','Bonferroni significant'}, 'Location','best');
grid on;

%% (Optional) Save outputs to struct
results = struct();
results.f = f;
results.Pxx = Pxx;
results.logPxx = logPxx;
results.group0_idx = g0;
results.group1_idx = g1;
results.p_raw = pvals;
results.p_bonf = p_bonf;
results.sig_bonf = sig_bonf;
results.tstat = tstat;
