%% Freud_Param_Sensitivity_Sweep.m
% Sensitivity analysis for the SVD exponential scale parameter.
%
% This diagnostic script evaluates how the trial-level PC1 separation changes
% as a function of the exponential scale parameter alpha. The analysis is run
% for a selected trial index using the Death + Me trial condition.
%
% This script is intended for parameter checking and documentation. It is not
% part of the main Figure 3 generation pipeline.
%
% Required input:
%   Freud_Processed_BDIAT.mat
%
% Expected variables:
%   XF           : subjects x 360 processed log-reaction-time matrix
%   active_score : subjects x 1 group label, where 0 = SI- and 1 = SI+
%
% Output:
%   Figure_Param_Sensitivity_Sweep.svg

clear; close all; clc;

%% ----------------------- User settings -----------------------
trialIdx  = 4;                    % Trial index to evaluate, from 1 to 20
alphaVals = linspace(0.01, 15, 100);

%% ----------------------- Load data ---------------------------
load('Freud_Processed_BDIAT.mat');   % expects XF and active_score

active_score = active_score(:);
classes = [0 1];

assert(all(ismember(active_score, classes)), ...
    'active_score must contain only 0 and 1 labels.');

%% ----------------------- Select trial condition -----------------------
% Death + Me trial condition:
% columns 1:20, 41:60, ..., 321:340
blockLen    = 20;
blockStarts = 1:40:321;

blocksIdx = [];
for s = blockStarts
    blocksIdx = [blocksIdx, s:(s + blockLen - 1)];
end

X = XF(:, blocksIdx);
[nRows, ~] = size(X);

idxC0 = (active_score == 0);
idxC1 = (active_score == 1);

assert(any(idxC0) && any(idxC1), ...
    'Both SI- and SI+ groups must contain at least one subject.');

%% ----------------------- Alpha sensitivity sweep -----------------------
nAlpha = numel(alphaVals);

meanDiff = zeros(1, nAlpha);
avgVar   = zeros(1, nAlpha);
ratioVal = zeros(1, nAlpha);

for a = 1:nAlpha
    alpha = alphaVals(a);

    rowEV = zeros(nRows, blockLen);

    for i = 1:nRows
        Xi = X(i, :);

        % Reshape subject-level vector into blocks x trial-position matrix.
        Xi_blocks = reshape(Xi, blockLen, []).';

        % Construct transformed matrix and extract first PC direction.
        Y = exp(-alpha * exp(Xi_blocks));
        C = Y' * Y;

        [~, ~, V] = svd(C);
        rowEV(i, :) = V(:, 1).';
    end

    EV0 = rowEV(idxC0, trialIdx);
    EV1 = rowEV(idxC1, trialIdx);

    m0 = mean(EV0);
    m1 = mean(EV1);

    v0 = var(EV0);
    v1 = var(EV1);

    meanDiff(a) = abs(m1 - m0);
    avgVar(a)   = 0.5 * (v0 + v1);

    if avgVar(a) > 0
        ratioVal(a) = meanDiff(a) / sqrt(avgVar(a));
    else
        ratioVal(a) = NaN;
    end
end

%% ----------------------- Report best alpha -----------------------
[bestRatio, bestIdx] = max(ratioVal, [], 'omitnan');
bestAlpha = alphaVals(bestIdx);

fprintf('\nParameter sensitivity summary\n');
fprintf('Trial index: %d\n', trialIdx);
fprintf('Best alpha by mean-difference / sqrt(avg variance): %.4f\n', bestAlpha);
fprintf('Best ratio value: %.4f\n\n', bestRatio);

%% ----------------------- Plot diagnostic figure -----------------------
fig = figure('Color', 'w', 'Position', [100 100 900 850]);

subplot(3, 1, 1);
plot(alphaVals, meanDiff, 'LineWidth', 2);
xline(bestAlpha, '--', 'LineWidth', 1.5);
xlabel('\alpha');
ylabel('Mean difference');
title(sprintf('PC1 group mean difference at trial %d', trialIdx));
grid on;

subplot(3, 1, 2);
plot(alphaVals, avgVar, 'LineWidth', 2);
xline(bestAlpha, '--', 'LineWidth', 1.5);
xlabel('\alpha');
ylabel('Average variance');
title(sprintf('Average within-group variance at trial %d', trialIdx));
grid on;

subplot(3, 1, 3);
plot(alphaVals, ratioVal, 'LineWidth', 2);
xline(bestAlpha, '--', 'LineWidth', 1.5);
xlabel('\alpha');
ylabel('Separation ratio');
title('Mean difference / sqrt(average variance)');
grid on;

set(fig, 'PaperPositionMode', 'auto');
print(fig, 'Figure_Param_Sensitivity_Sweep.svg', '-dsvg');
close(fig);

disp('Saved Figure_Param_Sensitivity_Sweep.svg');