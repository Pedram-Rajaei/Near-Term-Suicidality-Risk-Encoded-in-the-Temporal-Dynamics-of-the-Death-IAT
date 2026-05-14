function [x_filled, mu_global] = Freud_Impute_RT_Series(x, isClipped, thr, win, doPreDemean)
%FREUD_IMPUTE_RT_SERIES Impute missing and censored reaction-time series.
%
% This helper function completes a single subject-level reaction-time series
% in log space. Random missing values are interpolated using shape-preserving
% cubic interpolation, while censored thresholded samples are imputed using a
% local truncated-normal expectation.
%
% Inputs:
%   x            : time-series vector. Missing values should be NaN.
%   isClipped    : logical vector with TRUE for threshold-censored samples.
%   thr          : censoring threshold in the same scale as x.
%   win          : optional local half-window size. Default is 20 samples.
%   doPreDemean  : optional flag for preliminary mean removal. Default false.
%
% Outputs:
%   x_filled     : completed time series after interpolation/imputation.
%   mu_global    : mean of the completed time series.
%
% Example:
%   [x_filled, mu] = Freud_Impute_RT_Series(x, isClipped, thr, 30);

    if nargin < 4 || isempty(win)
        win = 20;
    end
    if nargin < 5 || isempty(doPreDemean)
        doPreDemean = false;
    end

    x = x(:);
    isClipped = logical(isClipped(:));
    N = numel(x);

    assert(numel(isClipped) == N, ...
        'x and isClipped must have the same number of elements.');

    %% Optional preliminary de-meaning
    prelim_mean = 0;
    if doPreDemean
        obsMask = ~isnan(x) & ~isClipped;
        if any(obsMask)
            prelim_mean = mean(x(obsMask));
            x = x - prelim_mean;
        end
    end

    %% Interpolate random missing values
    isNaN = isnan(x);
    isRandMissing = isNaN & ~isClipped;
    t = (1:N)';

    x_tmp = x;
    x_tmp(isClipped) = NaN;

    interp_ok = ~isnan(x_tmp);
    x_interp = x_tmp;

    if any(isRandMissing)
        if nnz(interp_ok) >= 2
            x_interp(isRandMissing) = interp1( ...
                t(interp_ok), x_tmp(interp_ok), ...
                t(isRandMissing), 'pchip', 'extrap');
        else
            x_interp(isRandMissing) = thr;
        end
    end

    x_filled = x_interp;

    %% Impute threshold-censored samples
    idxClipped = find(isClipped);

    for k = 1:numel(idxClipped)
        i = idxClipped(k);

        i0 = max(1, i - win);
        i1 = min(N, i + win);

        neigh = x_filled(i0:i1);
        neighClip = isClipped(i0:i1);

        neighValid = neigh(~isnan(neigh) & ~neighClip);

        if numel(neighValid) < 5
            x_filled(i) = thr;
            continue;
        end

        mu_loc = mean(neighValid);
        sigma_loc = std(neighValid);

        if sigma_loc <= 0
            x_filled(i) = max(mu_loc, thr);
            continue;
        end

        % Conditional expectation under X > threshold.
        a = (thr - mu_loc) / sigma_loc;
        phi = exp(-0.5 * a.^2) / sqrt(2*pi);
        Phi = 0.5 * (1 + erf(a / sqrt(2)));

        if Phi >= 1
            x_filled(i) = max(mu_loc, thr);
        else
            x_filled(i) = mu_loc + sigma_loc * (phi / (1 - Phi));
        end
    end

    %% Restore mean and return completed-series mean
    if doPreDemean
        x_filled = x_filled + prelim_mean;
    end

    mu_global = mean(x_filled, 'omitnan');
end