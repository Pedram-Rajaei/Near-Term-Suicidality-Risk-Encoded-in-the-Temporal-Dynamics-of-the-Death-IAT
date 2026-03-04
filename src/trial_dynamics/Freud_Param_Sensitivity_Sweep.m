clear all;
close all;

%% Load data (assumes XF and active_score)
load('Freud_Processed_BDIAT.mat');
active_score = active_score(:);
classes      = [0 1];

%% Choose the trial index you want to analyze (1–20)
trialIdx = 4;      % <--- YOU ENTER THIS

%% Block settings (using first block set: 1:20,41:60,...)
blockLen     = 20;
blockStarts  = 1:40:321;         % 9 blocks
blocksIdx = [];
for s = blockStarts
    blocksIdx = [blocksIdx, s:(s+blockLen-1)];
end

% Extract only the needed columns
X = XF(:, blocksIdx);
[nRows, ~] = size(X);

% Class masks
idxC0 = (active_score==0);
idxC1 = (active_score==1);

%% Alpha sweep
alphaVals = linspace(0.01, 15, 100);
nAlpha    = numel(alphaVals);

meanDiff = zeros(1, nAlpha);
avgVar   = zeros(1, nAlpha);
ratioVal = zeros(1, nAlpha);

%% Loop over alpha
for a = 1:nAlpha
    alpha = alphaVals(a);

    % compute rowEV for this alpha
    rowEV = zeros(nRows, blockLen);

    for i = 1:nRows
        Xi = X(i,:);
        Xi_blocks = reshape(Xi, blockLen, []).';    % 9×20

        % did not give the best result
        % mu = mean(Xi_blocks, 1);
        % sd = std(Xi_blocks, 0, 1);
        % Z_row = (Xi_blocks - mu) ./ sd;
        % 
        % Y = tanh(alpha * Z_row);    % <-- try this
       
        % gamma = 0.5;
        %Y = sign(Xi_blocks) .* abs(Xi_blocks).^gamma;
    
     
        %   Y = log(1 + alpha * exp(Xi_blocks));
        % c= -0.1;
        % Y = 1 ./ (1 + exp(-alpha * (Xi_blocks - c)));

        Y = exp(-alpha*exp(Xi_blocks));           % 9×20
        C = Y' * Y;

        [~,~,V] = svd(C);
        rowEV(i,:) = V(:,1).';
    end

    % class splits
    EV0 = rowEV(idxC0, trialIdx);
    EV1 = rowEV(idxC1, trialIdx);

    m0 = mean(EV0);           v0 = var(EV0);
    m1 = mean(EV1);           v1 = var(EV1);

    meanDiff(a) = abs(m1 - m0);
    avgVar(a)   = 0.5 * (v0 + v1);
    ratioVal(a) = abs(meanDiff(a)) / sqrt(avgVar(a));
end

%% ======================= Plot results =======================

figure('Color', 'w');

subplot(3,1,1);
plot(alphaVals, meanDiff, 'LineWidth', 2);
xlabel('\alpha'); ylabel('Mean difference');
title(['Mean difference at trial ', num2str(trialIdx)]);
grid on;

subplot(3,1,2);
plot(alphaVals, avgVar, 'LineWidth', 2);
xlabel('\alpha'); ylabel('Average variance');
title(['Average variance at trial ', num2str(trialIdx)]);
grid on;

subplot(3,1,3);
plot(alphaVals, ratioVal, 'LineWidth', 2);
xlabel('\alpha'); ylabel('Ratio');
title(['MeanDiff / sqrt(AvgVar) at trial ', num2str(trialIdx)]);
grid on;
