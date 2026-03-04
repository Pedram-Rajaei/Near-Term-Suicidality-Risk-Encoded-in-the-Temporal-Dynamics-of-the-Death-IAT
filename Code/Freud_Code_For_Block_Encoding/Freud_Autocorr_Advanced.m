function out = Freud_Autocorr_Advanced(X, G, varargin)
%FREUD_AUTOCORR_ADVANCED
%  Block-level ACF analysis for ABAB block-switch tasks, with:
%   1) Block summaries (median or mean of log RT) -> 18-point series per subject
%   2) Subject-wise detrend + mean-center
%   3) Block-level ACF per subject
%   4) Plots:
%       - Group mean ACF curves (with SEM ribbons)
%       - Group difference curve (G1-G0) with between-subject permutation 95% band
%   5) Hypothesis tests:
%       A) Multi-lag RhythmIndex (even lags minus odd lags) + group-label permutation test
%       B) Within-subject shuffles:
%           (i) Preserve-ABAB: permute odd blocks among odd positions, even among even
%           (ii) Destroy-ABAB: permute all blocks among all positions (breaks ABAB)
%
%  IMPORTANT: Rhythm is a multi-lag pattern. Primary inferential statistic here is:
%     RhythmIndex(K) = mean(ACF even lags 2..K) - mean(ACF odd lags 1..K)
%
%  ADDED (per your request):
%   - Three figures showing the null distribution of RhythmIndex under:
%       (A) group-label permutation (group difference)
%       (B1) within-subject Preserve-ABAB shuffle (subject mean + group diff)
%       (B2) within-subject Destroy-ABAB shuffle (subject mean + group diff)
%     with the observed statistic overlaid.

% ---------------- Parse inputs ----------------
p = inputParser;

p.addParameter('UseLog', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('BlockLen', 20, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('NumBlocks', 18, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('BlockSummary', 'median', @(s)ischar(s)||isstring(s));
p.addParameter('DetrendBlocks', true, @(b)islogical(b)&&isscalar(b));

p.addParameter('DoTrialACF', false, @(b)islogical(b)&&isscalar(b));
p.addParameter('MaxLagTrial', 80, @(v)isnumeric(v)&&isscalar(v)&&v>=0);

p.addParameter('DoBlockACF', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('MaxLagBlock', 8, @(v)isnumeric(v)&&isscalar(v)&&v>=0);

p.addParameter('RhythmMaxLag', 8, @(v)isnumeric(v)&&isscalar(v)&&v>=2);

p.addParameter('MakePlots', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('PlotSEM', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('PlotDiffWithPermBand', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('FigureNamePrefix', 'ACF', @(s)ischar(s)||isstring(s));

p.addParameter('DoGroupPermTest', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('NumPerm', 5000, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('Seed', 1, @(v)isnumeric(v)&&isscalar(v));

p.addParameter('DoShufflePreserveAB', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('NumShufflePreserve', 5000, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('ShufflePreserveSeed', 2, @(v)isnumeric(v)&&isscalar(v));

p.addParameter('DoShuffleDestroyAB', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('NumShuffleDestroy', 5000, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('ShuffleDestroySeed', 3, @(v)isnumeric(v)&&isscalar(v));
p.addParameter('ShuffleDestroyMode', 'permuteBlocks', @(s)ischar(s)||isstring(s));

p.parse(varargin{:});

useLog    = p.Results.UseLog;
L         = p.Results.BlockLen;
B         = p.Results.NumBlocks;
sumfun    = lower(string(p.Results.BlockSummary));
doDetrend = p.Results.DetrendBlocks;

doTrial   = p.Results.DoTrialACF;
maxLagT   = p.Results.MaxLagTrial;

doBlock   = p.Results.DoBlockACF;
maxLagB   = p.Results.MaxLagBlock;

rhythmK   = p.Results.RhythmMaxLag;

makePlots = p.Results.MakePlots;
plotSEM   = p.Results.PlotSEM;
plotDiff  = p.Results.PlotDiffWithPermBand;
figPrefix = string(p.Results.FigureNamePrefix);

doPerm    = p.Results.DoGroupPermTest;
nPerm     = p.Results.NumPerm;
seed      = p.Results.Seed;

doShufPres= p.Results.DoShufflePreserveAB;
nShufPres = p.Results.NumShufflePreserve;
seedPres  = p.Results.ShufflePreserveSeed;

doShufDes = p.Results.DoShuffleDestroyAB;
nShufDes  = p.Results.NumShuffleDestroy;
seedDes   = p.Results.ShuffleDestroySeed;
destroyMode = lower(strtrim(string(p.Results.ShuffleDestroyMode)));

% ---------------- Validate inputs ----------------
[n,T] = size(X);
assert(T == L*B, 'Expected X to have %d columns (BlockLen*NumBlocks).', L*B);
G = double(G(:));
assert(numel(G)==n, 'G must match number of rows in X.');
assert(all(ismember(unique(G(~isnan(G))), [0 1])), 'G must contain only 0/1 (or NaN).');

maxLagB = min(maxLagB, B-1);
rhythmK = min(rhythmK, maxLagB);
lagsBlock = (0:maxLagB)';

% ---------------- Optional: trial-level ACF (not primary) ----------------
acfTrial = [];
lagsTrial = [];
if doTrial
    lagsTrial = (0:maxLagT)';
    acfTrial = nan(n, numel(lagsTrial));
    for s = 1:n
        x = X(s,:)';
        ok = isfinite(x) & x > 0;
        x = x(ok);
        if numel(x) < maxLagT+2, continue; end
        if useLog, x = log(x); end
        x = x - mean(x);
        denom = var(x,1);
        if denom <= 0, continue; end
        for k = 0:maxLagT
            x1 = x(1:end-k);
            x2 = x(1+k:end);
            acfTrial(s,k+1) = (x1'*x2) / (numel(x1) * denom);
        end
    end
end

% ---------------- Block summaries ----------------
blockSeries = nan(n,B);
for s = 1:n
    for b = 1:B
        idx = (b-1)*L + (1:L);
        xb = X(s,idx);
        xb = xb(isfinite(xb) & xb > 0);
        if isempty(xb), blockSeries(s,b) = NaN; continue; end
        if useLog, xb = log(xb); end
        switch sumfun
            case "median"
                blockSeries(s,b) = median(xb,'omitnan');
            case "mean"
                blockSeries(s,b) = mean(xb,'omitnan');
            otherwise
                error('BlockSummary must be ''median'' or ''mean''.');
        end
    end
end

% ---------------- Detrend + center ----------------
blockSeriesDT = detrend_and_center(blockSeries, doDetrend);

% ---------------- Block-level ACF ----------------
acfBlock = compute_acf_rows(blockSeriesDT, maxLagB);

% ---------------- RhythmIndex (multi-lag) per subject ----------------
oddLags  = 1:2:rhythmK;
evenLags = 2:2:rhythmK;

oddCols  = oddLags + 1;   % lag k is col k+1
evenCols = evenLags + 1;

rhythmIndex = mean(acfBlock(:, evenCols), 2, 'omitnan') ...
            - mean(acfBlock(:, oddCols),  2, 'omitnan');

% ---------------- Group means/SEMs ----------------
g0 = (G==0); g1 = (G==1);
mean_sem = @(M, idx) deal(mean(M(idx,:),1,'omitnan'), ...
                         std(M(idx,:),0,1,'omitnan') ./ sqrt(sum(isfinite(M(idx,:)),1)));

[m0B, s0B] = mean_sem(acfBlock, g0);
[m1B, s1B] = mean_sem(acfBlock, g1);

% ---------------- Output base ----------------
out = struct();
out.acfBlock = acfBlock;
out.lagsBlock = lagsBlock;
out.blockSeries = blockSeries;
out.blockSeriesDT = blockSeriesDT;
out.rhythmIndex = rhythmIndex;
out.rhythmIndexSettings = struct('RhythmMaxLag', rhythmK, 'OddLags', oddLags, 'EvenLags', evenLags);

if doTrial
    out.acfTrial = acfTrial;
    out.lagsTrial = lagsTrial;
end

out.groupMeanBlockACF0 = m0B; out.groupSEMBlockACF0 = s0B;
out.groupMeanBlockACF1 = m1B; out.groupSEMBlockACF1 = s1B;

% ---------------- Plots: group mean ACF ----------------
if makePlots && doBlock
    figure('Name', figPrefix + " - Block ACF");
    hold on;

    h0 = plot(lagsBlock, m0B, '-o', 'DisplayName','Group 0');
    h1 = plot(lagsBlock, m1B, '-o', 'DisplayName','Group 1');

    if plotSEM
        x = lagsBlock(:);
        fill([x; flipud(x)], [m0B(:)-s0B(:); flipud(m0B(:)+s0B(:))], h0.Color, ...
            'FaceAlpha', 0.10, 'EdgeColor','none', 'HandleVisibility','off');
        fill([x; flipud(x)], [m1B(:)-s1B(:); flipud(m1B(:)+s1B(:))], h1.Color, ...
            'FaceAlpha', 0.10, 'EdgeColor','none', 'HandleVisibility','off');
    end

    yline(0,'--','HandleVisibility','off');
    xlabel('Lag (blocks)'); ylabel('Autocorrelation');
    title(sprintf('Block-level ACF (K=%d for RhythmIndex)', rhythmK));
    legend('Location','best');
    hold off;
end

% ---------------- Tests ----------------
out.tests = struct();
out.tests.groupPerm = struct();
out.tests.shufflePreserveAB = struct();
out.tests.shuffleDestroyAB = struct();

% ===== A) Between-subject group-label permutation tests =====
permDiffAll = [];
if doPerm
    rng(seed);

    % Observed group difference in RhythmIndex
    obs = struct();
    obs.rhythmIndex = mean(rhythmIndex(G==1),'omitnan') - mean(rhythmIndex(G==0),'omitnan');

    % Also store the observed ACF difference curve for plotting
    obs.acfDiffCurve = mean(acfBlock(G==1,:),1,'omitnan') - mean(acfBlock(G==0,:),1,'omitnan');

    % Permute group labels among valid subjects
    valid = isfinite(G) & isfinite(rhythmIndex);
    idx = find(valid);

    perm = struct();
    perm.rhythmIndex = nan(nPerm,1);
    permDiffAll = nan(nPerm, numel(lagsBlock));

    for pp = 1:nPerm
        Gp = G;
        Gp(idx) = G(idx(randperm(numel(idx))));

        perm.rhythmIndex(pp) = mean(rhythmIndex(Gp==1),'omitnan') - mean(rhythmIndex(Gp==0),'omitnan');
        permDiffAll(pp,:) = mean(acfBlock(Gp==1,:),1,'omitnan') - mean(acfBlock(Gp==0,:),1,'omitnan');
    end

    pval = struct();
    pval.rhythmIndex = mean(abs(perm.rhythmIndex) >= abs(obs.rhythmIndex));

    out.tests.groupPerm.obs = obs;
    out.tests.groupPerm.perm = perm;
    out.tests.groupPerm.p = pval;
    out.tests.groupPerm.settings = struct('NumPerm', nPerm, 'Seed', seed);

    % Plot: ACF difference curve with permutation band
    if makePlots && plotDiff
        bandLo = prctile(permDiffAll, 2.5, 1);
        bandHi = prctile(permDiffAll, 97.5, 1);

        figure('Name', figPrefix + " - ACF Diff (G1-G0) w/ Perm Band");
        hold on;
        x = lagsBlock(:);
        fill([x; flipud(x)], [bandLo(:); flipud(bandHi(:))], 'k', ...
            'FaceAlpha', 0.15, 'EdgeColor','none', 'DisplayName','Perm 95% band');
        plot(lagsBlock, obs.acfDiffCurve, '-o', 'LineWidth', 1.5, 'DisplayName','Observed (G1-G0)');
        yline(0,'--','HandleVisibility','off');
        xlabel('Lag (blocks)'); ylabel('ACF difference (Group1 - Group0)');
        title('Group difference in ACF across lags (permutation band)');
        legend('Location','best');
        hold off;
    end

    % --- ADDED FIGURE (A): Null distribution of RhythmIndex (group diff) ---
    if makePlots
        figName = figPrefix + " - Null Dist (A) GroupPerm RhythmIndex";
        figure('Name', figName);
        ax = axes; %#ok<LAXES>
        plot_null_hist(ax, perm.rhythmIndex, obs.rhythmIndex, ...
            'RhythmIndex group difference (G1 - G0)', ...
            sprintf('Group-label permutation null (K=%d), p=%.4g', rhythmK, pval.rhythmIndex));
    end

    fprintf('\n[Between-subject group-label permutation]\n');
    fprintf('  RhythmIndex(K=%d) group diff p = %.4g\n', rhythmK, pval.rhythmIndex);
end

% ===== B1) Within-subject shuffle that PRESERVES ABAB =====
if doShufPres
    rng(seedPres);

    % Observed overall rhythm (subject mean) and observed group diff
    obsPres = struct();
    obsPres.subjectMean = mean(rhythmIndex,'omitnan');
    obsPres.groupDiff = mean(rhythmIndex(G==1),'omitnan') - mean(rhythmIndex(G==0),'omitnan');

    % Shuffle distribution (compute BOTH subjectMean and groupDiff)
    statMean = nan(nShufPres,1);
    statDiff = nan(nShufPres,1);

    oddIdx  = 1:2:B;
    evenIdx = 2:2:B;

    for ss = 1:nShufPres
        blkShuf = blockSeries;
        for s = 1:n
            y = blkShuf(s,:);
            if all(isnan(y)), continue; end
            yo = y(oddIdx);  yo = yo(randperm(numel(yo)));
            ye = y(evenIdx); ye = ye(randperm(numel(ye)));
            y2 = y; y2(oddIdx)=yo; y2(evenIdx)=ye;
            blkShuf(s,:) = y2;
        end

        blkShufDT = detrend_and_center(blkShuf, doDetrend);
        acfShuf = compute_acf_rows(blkShufDT, maxLagB);

        rIdx = mean(acfShuf(:, evenCols), 2, 'omitnan') - mean(acfShuf(:, oddCols), 2, 'omitnan');

        statMean(ss) = mean(rIdx,'omitnan');
        statDiff(ss) = mean(rIdx(G==1),'omitnan') - mean(rIdx(G==0),'omitnan');
    end

    pMean = mean(abs(statMean) >= abs(obsPres.subjectMean));
    pDiff = mean(abs(statDiff) >= abs(obsPres.groupDiff));

    out.tests.shufflePreserveAB.obs = obsPres;
    out.tests.shufflePreserveAB.perm.subjectMean = statMean;
    out.tests.shufflePreserveAB.perm.groupDiff = statDiff;
    out.tests.shufflePreserveAB.p.subjectMean = pMean;
    out.tests.shufflePreserveAB.p.groupDiff = pDiff;
    out.tests.shufflePreserveAB.settings = struct('NumShuffle', nShufPres, 'Seed', seedPres);

    % --- ADDED FIGURE (B1): Preserve-ABAB null distributions (2 panels, 1 figure) ---
    if makePlots
        figName = figPrefix + " - Null Dist (B1) PreserveAB RhythmIndex";
        figure('Name', figName);
        t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

        ax1 = nexttile(t,1);
        plot_null_hist(ax1, statMean, obsPres.subjectMean, ...
            'Mean RhythmIndex across subjects', ...
            sprintf('Preserve-ABAB null: subject mean (K=%d), p=%.4g', rhythmK, pMean));

        ax2 = nexttile(t,2);
        plot_null_hist(ax2, statDiff, obsPres.groupDiff, ...
            'RhythmIndex group difference (G1 - G0)', ...
            sprintf('Preserve-ABAB null: group diff (K=%d), p=%.4g', rhythmK, pDiff));
    end

    fprintf('[Within-subject shuffle PRESERVE ABAB]\n');
    fprintf('  RhythmIndex(K=%d): subject-mean p = %.4g | group-diff p = %.4g\n', rhythmK, pMean, pDiff);
end

% ===== B2) Within-subject RHYTHM-DESTROYING shuffle =====
if doShufDes
    rng(seedDes);

    obsDes = struct();
    obsDes.subjectMean = mean(rhythmIndex,'omitnan');
    obsDes.groupDiff = mean(rhythmIndex(G==1),'omitnan') - mean(rhythmIndex(G==0),'omitnan');

    statMean = nan(nShufDes,1);
    statDiff = nan(nShufDes,1);

    for ss = 1:nShufDes
        blkShuf = blockSeries;
        for s = 1:n
            y = blkShuf(s,:);
            if all(isnan(y)), continue; end

            switch destroyMode
                case "permuteblocks"
                    y2 = y(randperm(B));
                case "circularshift"
                    sh = randi([0 B-1],1,1);
                    y2 = circshift(y, [0 sh]);
                otherwise
                    error('ShuffleDestroyMode must be ''permuteBlocks'' or ''circularShift''.');
            end

            blkShuf(s,:) = y2;
        end

        blkShufDT = detrend_and_center(blkShuf, doDetrend);
        acfShuf = compute_acf_rows(blkShufDT, maxLagB);

        rIdx = mean(acfShuf(:, evenCols), 2, 'omitnan') - mean(acfShuf(:, oddCols), 2, 'omitnan');

        statMean(ss) = mean(rIdx,'omitnan');
        statDiff(ss) = mean(rIdx(G==1),'omitnan') - mean(rIdx(G==0),'omitnan');
    end

    pMean = mean(abs(statMean) >= abs(obsDes.subjectMean));
    pDiff = mean(abs(statDiff) >= abs(obsDes.groupDiff));

    out.tests.shuffleDestroyAB.obs = obsDes;
    out.tests.shuffleDestroyAB.mode = destroyMode;
    out.tests.shuffleDestroyAB.perm.subjectMean = statMean;
    out.tests.shuffleDestroyAB.perm.groupDiff = statDiff;
    out.tests.shuffleDestroyAB.p.subjectMean = pMean;
    out.tests.shuffleDestroyAB.p.groupDiff = pDiff;
    out.tests.shuffleDestroyAB.settings = struct('NumShuffle', nShufDes, 'Seed', seedDes);

    % --- ADDED FIGURE (B2): Destroy-ABAB null distributions (2 panels, 1 figure) ---
    if makePlots
        figName = figPrefix + " - Null Dist (B2) DestroyAB RhythmIndex";
        figure('Name', figName);
        t = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

        ax1 = nexttile(t,1);
        plot_null_hist(ax1, statMean, obsDes.subjectMean, ...
            'Mean RhythmIndex across subjects', ...
            sprintf('Destroy-ABAB null (%s): subject mean (K=%d), p=%.4g', destroyMode, rhythmK, pMean));

        ax2 = nexttile(t,2);
        plot_null_hist(ax2, statDiff, obsDes.groupDiff, ...
            'RhythmIndex group difference (G1 - G0)', ...
            sprintf('Destroy-ABAB null (%s): group diff (K=%d), p=%.4g', destroyMode, rhythmK, pDiff));
    end

    fprintf('[Within-subject shuffle DESTROY ABAB; %s]\n', destroyMode);
    fprintf('  RhythmIndex(K=%d): subject-mean p = %.4g | group-diff p = %.4g\n', rhythmK, pMean, pDiff);
end

end

% ===================== Helper functions =====================

function Y = detrend_and_center(Yin, doDetrend)
Y = Yin;
[n,B] = size(Y);
for s = 1:n
    y = Y(s,:);
    if all(isnan(y)), continue; end
    if doDetrend
        t = 1:B;
        ok = isfinite(y);
        if nnz(ok) >= 3
            tt = t(ok)'; yy = y(ok)';
            beta = [ones(size(tt)) tt] \ yy;
            trend = beta(1) + beta(2)*t;
            y = y - trend;
        end
    end
    y = y - mean(y,'omitnan');
    Y(s,:) = y;
end
end

function acfM = compute_acf_rows(M, maxLag)
[n,~] = size(M);
acfM = nan(n, maxLag+1);
for s = 1:n
    x = M(s,:)';
    ok = isfinite(x);
    x = x(ok);
    if numel(x) < maxLag+2, continue; end
    x = x - mean(x);
    denom = var(x,1);
    if denom <= 0, continue; end
    for k = 0:maxLag
        x1 = x(1:end-k);
        x2 = x(1+k:end);
        acfM(s,k+1) = (x1'*x2) / (numel(x1) * denom);
    end
end
end

% --- ADDED: histogram of null + observed line (single axis) ---
function plot_null_hist(ax, nullVals, obsVal, xlab, ttl)
    axes(ax); %#ok<LAXES>
    hold on;
    histogram(nullVals, 'Normalization','pdf');
    yl = ylim;
    plot([obsVal obsVal], yl, '-', 'LineWidth', 2, 'DisplayName','Observed');
    xlabel(xlab);
    ylabel('Density');
    title(ttl);
    legend('Location','best');
    hold off;
end
