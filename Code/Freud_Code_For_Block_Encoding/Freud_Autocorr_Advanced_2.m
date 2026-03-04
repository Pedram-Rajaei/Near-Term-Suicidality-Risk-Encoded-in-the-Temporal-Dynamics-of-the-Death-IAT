function out = Freud_Autocorr_Advanced_2(X, G, varargin)
%ACF_TRIAL_VS_BLOCK_WITH_PLOTS  Compute trial/block ACFs and plot group-mean ACF curves.
%
% Same functionality as before, plus plots:
%   - Group mean ACF (with SEM bands) for trial-level ACF (optional)
%   - Group mean ACF (with SEM bands) for block-level ACF (recommended)
%
% Usage:
%   out = acf_trial_vs_block_with_plots(X,G,'DoTrialACF',false,'DoBlockACF',true,'MaxLagBlock',8);
%
% Notes:
%   - Uses log(RT) by default.
%   - Block ACF is computed on per-subject block summaries, detrended + mean-centered.
%
% See options below.

p = inputParser;
p.addParameter('UseLog', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('MaxLagTrial', 80, @(v)isnumeric(v)&&isscalar(v)&&v>=0);
p.addParameter('MaxLagBlock', 8, @(v)isnumeric(v)&&isscalar(v)&&v>=0);
p.addParameter('BlockLen', 20, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('NumBlocks', 18, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('BlockSummary', 'median', @(s)ischar(s)||isstring(s));
p.addParameter('DetrendBlocks', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('DoTrialACF', false, @(b)islogical(b)&&isscalar(b)); % default off (often noisy)
p.addParameter('DoBlockACF', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('MakePlots', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('PlotSEM', true, @(b)islogical(b)&&isscalar(b));
p.addParameter('FigureNamePrefix', 'ACF', @(s)ischar(s)||isstring(s));
p.parse(varargin{:});

useLog = p.Results.UseLog;
maxLagT = p.Results.MaxLagTrial;
maxLagB = p.Results.MaxLagBlock;
L = p.Results.BlockLen;
B = p.Results.NumBlocks;
sumfun = lower(string(p.Results.BlockSummary));
doDetrend = p.Results.DetrendBlocks;
doTrial = p.Results.DoTrialACF;
doBlock = p.Results.DoBlockACF;
makePlots = p.Results.MakePlots;
plotSEM = p.Results.PlotSEM;
figPrefix = string(p.Results.FigureNamePrefix);

[n,T] = size(X);
assert(T == L*B, 'Expected X to have %d columns (BlockLen*NumBlocks).', L*B);
G = double(G(:));
assert(numel(G)==n, 'G must match number of rows in X.');
assert(all(ismember(unique(G(~isnan(G))), [0 1])), 'G must contain only 0/1 (or NaN).');

lagsTrial = (0:maxLagT)';
lagsBlock = (0:maxLagB)';

acfTrial = nan(n, numel(lagsTrial));
acfBlock = nan(n, numel(lagsBlock));

% ---------------- TRIAL-LEVEL ACF ----------------
if doTrial
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

% ---------------- BLOCK-LEVEL SERIES + ACF ----------------
blockSeries = nan(n,B);
blockSeriesDT = nan(n,B);

if doBlock
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

    % detrend + mean-center per subject
    blockSeriesDT = blockSeries;
    for s = 1:n
        y = blockSeriesDT(s,:);
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
        blockSeriesDT(s,:) = y;
    end

    % ACF on block series
    for s = 1:n
        x = blockSeriesDT(s,:)';
        ok = isfinite(x);
        x = x(ok);
        if numel(x) < maxLagB+2, continue; end
        x = x - mean(x);
        denom = var(x,1);
        if denom <= 0, continue; end
        for k = 0:maxLagB
            x1 = x(1:end-k);
            x2 = x(1+k:end);
            acfBlock(s,k+1) = (x1'*x2) / (numel(x1) * denom);
        end
    end
end

% ---------------- group stats helpers ----------------
g0 = (G==0);
g1 = (G==1);

% mean + SEM across subjects (ignoring NaNs)
mean_sem = @(M, idx) deal(mean(M(idx,:),1,'omitnan'), ...
                         std(M(idx,:),0,1,'omitnan') ./ sqrt(sum(isfinite(M(idx,:)),1)));

out = struct();
out.acfTrial = acfTrial;
out.acfBlock = acfBlock;
out.lagsTrial = lagsTrial;
out.lagsBlock = lagsBlock;
out.blockSeries = blockSeries;
out.blockSeriesDT = blockSeriesDT;

if doTrial
    [m0, s0] = mean_sem(acfTrial, g0);
    [m1, s1] = mean_sem(acfTrial, g1);
    out.groupMeanTrialACF0 = m0; out.groupSEMTrialACF0 = s0;
    out.groupMeanTrialACF1 = m1; out.groupSEMTrialACF1 = s1;
end

if doBlock
    [m0, s0] = mean_sem(acfBlock, g0);
    [m1, s1] = mean_sem(acfBlock, g1);
    out.groupMeanBlockACF0 = m0; out.groupSEMBlockACF0 = s0;
    out.groupMeanBlockACF1 = m1; out.groupSEMBlockACF1 = s1;
end

% ---------------- PLOTS ----------------
if makePlots
    if doBlock
        figure('Name', figPrefix + " - Block ACF");
        hold on;
        plot(lagsBlock, out.groupMeanBlockACF0, '-o', 'DisplayName','Inactive SI');
        plot(lagsBlock, out.groupMeanBlockACF1, '-o', 'DisplayName','Active SI');

        if plotSEM
            % add simple SEM ribbons using fill
            x = lagsBlock(:);
            m0 = out.groupMeanBlockACF0(:); s0 = out.groupSEMBlockACF0(:);
            m1 = out.groupMeanBlockACF1(:); s1 = out.groupSEMBlockACF1(:);

            fill([x; flipud(x)], [m0-s0; flipud(m0+s0)], 'b', ...
                'FaceAlpha', 0.20, 'EdgeColor','none', 'HandleVisibility','off');
            fill([x; flipud(x)], [m1-s1; flipud(m1+s1)], 'r', ...
                'FaceAlpha', 0.20, 'EdgeColor','none', 'HandleVisibility','off');
        end

        yline(0,'--','HandleVisibility','off');
        xlabel('Lag (blocks)');
        ylabel('Autocorrelation');
        title('Block-level ACF (detrended, mean-centered per subject)');
        legend('Location','best');
        hold off;
    end

    if doTrial
        figure('Name', figPrefix + " - Trial ACF");
        hold on;
        plot(lagsTrial, out.groupMeanTrialACF0, 'DisplayName','Inactive SI');
        plot(lagsTrial, out.groupMeanTrialACF1, 'DisplayName','Active SI');

        if plotSEM
            x = lagsTrial(:);
            m0 = out.groupMeanTrialACF0(:); s0 = out.groupSEMTrialACF0(:);
            m1 = out.groupMeanTrialACF1(:); s1 = out.groupSEMTrialACF1(:);

            fill([x; flipud(x)], [m0-s0; flipud(m0+s0)], 'b', ...
                'FaceAlpha', 0.20, 'EdgeColor','none', 'HandleVisibility','off');
            fill([x; flipud(x)], [m1-s1; flipud(m1+s1)], 'r', ...
                'FaceAlpha', 0.20, 'EdgeColor','none', 'HandleVisibility','off');
        end

        yline(0,'--','HandleVisibility','off');
        xlabel('Lag (trials)');
        ylabel('Autocorrelation');
        title('Trial-level ACF (mean-centered per subject)');
        legend('Location','best');
        hold off;
    end
end

end
