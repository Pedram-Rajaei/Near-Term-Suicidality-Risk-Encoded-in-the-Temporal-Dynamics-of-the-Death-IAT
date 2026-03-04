function [x_filled, mu_global] = Freud_Impute_RT_Series(...)
    x, isClipped, thr, win, doPreDemean)
%IMPUTE_TS_THRESHOLD_MEAN  Impute missing/censored time series + handle mean.
%
%   [x_filled, mu_global] = impute_ts_threshold_mean(x, isClipped, thr, win, doPreDemean)
%
%   Inputs:
%       x          - time series (vector). Missing values should be NaN.
%       isClipped  - logical vector, same size as x. TRUE where samples
%                    are missing because they exceeded the threshold.
%       thr        - scalar threshold value.
%       win        - (optional) half-window length in samples for local
%                    statistics, default = 20.
%       doPreDemean- (optional) logical flag:
%                    * false (default): no preliminary de-meaning
%                    * true:  subtract a preliminary mean before imputation
%
%   Outputs:
%       x_filled   - time series with imputed values.
%       mu_global  - refined global mean of the completed (imputed) series.
%
%   Typical usage:
%       [x_filled, mu] = impute_ts_threshold_mean(x, isClipped, thr);
%       x_centered = x_filled - mu;  % mean-removed series for analysis
%
%   NOTE:
%   - If doPreDemean is true, a preliminary mean is estimated from observed
%     non-censored values and subtracted before imputation, then added back.
%     Finally, mu_global is computed from the completed series.

    if nargin < 4 || isempty(win)
        win = 20;
    end
    if nargin < 5 || isempty(doPreDemean)
        doPreDemean = false;
    end

    x = x(:);                     % ensure column
    isClipped = logical(isClipped(:));
    N = numel(x);

    %----------------------------------------------------------------------
    % STEP 0: Optional preliminary de-meaning (on observed, non-censored data)
    %----------------------------------------------------------------------
    prelim_mean = 0;
    if doPreDemean
        obsMask = ~isnan(x) & ~isClipped;     % observed & not censored
        if any(obsMask)
            prelim_mean = mean(x(obsMask));
            x = x - prelim_mean;
        end
    end

    %----------------------------------------------------------------------
    % STEP 1: Fill random missing values (NaNs that are NOT clipped)
    %----------------------------------------------------------------------
    isNaN = isnan(x);
    isRandMissing = isNaN & ~isClipped;   % missing at random
    t = (1:N)';

    % For interpolation, treat clipped positions as missing too
    x_tmp = x;
    x_tmp(isClipped) = NaN;

    interp_ok = ~isnan(x_tmp);   % locations with usable neighbors

    x_interp = x_tmp;
    if any(isRandMissing)
        x_interp(isRandMissing) = interp1( ...
            t(interp_ok), x_tmp(interp_ok), ...
            t(isRandMissing), 'pchip', 'extrap');  % can be 'spline'/'linear'
    end

    % Now:
    % - observed values are in x_interp
    % - random-missing filled
    % - clipped positions still NaN
    x_filled = x_interp;

    %----------------------------------------------------------------------
    % STEP 2: Handle clipped (thresholded) samples via truncated normal
    %----------------------------------------------------------------------
    idxClipped = find(isClipped);

    for k = 1:numel(idxClipped)
        i = idxClipped(k);

        % Local window around point i
        i0 = max(1, i - win);
        i1 = min(N, i + win);

        neigh     = x_filled(i0:i1);
        neighClip = isClipped(i0:i1);

        % Use only valid, non-clipped neighbors
        neighValid = neigh(~isnan(neigh) & ~neighClip);

        if numel(neighValid) < 5
            % Not enough local data: simple fallback
            x_filled(i) = thr;          % in de-meaned domain if doPreDemean
            continue;
        end

        mu_loc = mean(neighValid);
        sigma_loc = std(neighValid);

        if sigma_loc <= 0
            % Almost constant neighborhood
            x_filled(i) = max(mu_loc, thr);
            continue;
        end

        % Truncated normal expectation E[X | X > thr]
        a = (thr - mu_loc) / sigma_loc;

        % Standard normal pdf and cdf
        phi = exp(-0.5 * a.^2) / sqrt(2*pi);
        Phi = 0.5 * (1 + erf(a / sqrt(2)));

        if Phi >= 1   % avoid division by zero
            x_filled(i) = max(mu_loc, thr);
        else
            Ex = mu_loc + sigma_loc * (phi / (1 - Phi));
            x_filled(i) = Ex;
        end
    end

    %----------------------------------------------------------------------
    % STEP 3: Add back preliminary mean (if used) and compute global mean
    %----------------------------------------------------------------------
    if doPreDemean
        x_filled = x_filled + prelim_mean;
    end

    % Refined global mean on the completed (imputed) series
    mu_global = mean(x_filled, 'omitnan');
end
