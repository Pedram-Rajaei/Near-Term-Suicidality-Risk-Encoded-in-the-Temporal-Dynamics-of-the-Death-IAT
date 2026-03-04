%% ===================== FIGURE 2A + 2B: D-score boxplot + ROC =====================
% Inputs expected in data_bdiat.mat:
%   XF           : nSubjects x 360 (RT-related matrix; you decide raw vs exp)
%   active_score : nSubjects x 1 (0=Inactive-SI, 1=Active-SI)

clear; close all; clc;

%% -------------------- USER SETTINGS --------------------
USE_EXP = true;     % true -> use X = exp(XF); false -> use X = XF

labelFS  = 24;
tickFS   = 20;
legendFS = 20;

outPrefix = 'Figure_2_';

%% -------------------- LOAD DATA --------------------
load('Freud_Processed_BDIAT.mat');   % XF, active_score
active_score = active_score(:);

if USE_EXP
    X = exp(XF);
else
    X = XF;
end

[nSub, nT] = size(X);
assert(nT == 360, 'Expected XF to have 360 columns.');

%% -------------------- DEFINE TRIAL INDICES FOR CONDITIONS --------------------
% Based on your earlier convention:
% Death+Me blocks: 1:20, 41:60, ..., 321:340  (9 blocks * 20 = 180)
% Life+Me blocks : 21:40, 61:80, ..., 341:360 (9 blocks * 20 = 180)
blockLen = 20;
deathStarts = 1:40:321;
lifeStarts  = 21:40:341;

deathIdx = [];
lifeIdx  = [];

for s = deathStarts
    deathIdx = [deathIdx, s:(s+blockLen-1)];
end
for s = lifeStarts
    lifeIdx  = [lifeIdx,  s:(s+blockLen-1)];
end

deathIdx = deathIdx(:);
lifeIdx  = lifeIdx(:);

%% -------------------- COMPUTE D-SCORE PER SUBJECT --------------------
muDeath = mean(X(:, deathIdx), 2, 'omitnan');
muLife  = mean(X(:, lifeIdx),  2, 'omitnan');

% Standard deviation over ALL trials used (Death+Me + Life+Me)
sigmaAll = std(X(:, [deathIdx; lifeIdx]), 0, 2, 'omitnan');

% Avoid divide-by-zero
sigmaAll(sigmaAll < eps) = NaN;

D = (muDeath - muLife) ./ sigmaAll;

% Group labels
isActive = (active_score == 1);
isInact  = (active_score == 0);

D_inact = D(isInact);
D_act   = D(isActive);

%% ===================== FIGURE 2A: Proper boxplot (camera ready) =====================
figA = figure('Color','w','Position',[100 100 700 520]);
axA = axes('Parent', figA); hold(axA,'on'); grid(axA,'on'); box(axA,'off');

% Prepare boxplot input
group = nan(numel(D),1);
group(isInact)  = 1;
group(isActive) = 2;

boxplot(axA, D, group, ...
    'Labels', {'Without active SI','With active SI'}, ...
    'Symbol', 'k.', ...
    'Whisker', 1.5);

ylabel(axA, 'D-score', 'FontSize', labelFS);
set(axA,'FontSize',tickFS,'LineWidth',1);

yticks(axA, [-0.5  0 0.5]);

boxLW = 2.5;  % thickness of box/whiskers/median/caps 
set(findobj(figA,'Tag','Box'),     'LineWidth', boxLW);
set(findobj(figA,'Tag','Whisker'), 'LineWidth', boxLW);
set(findobj(figA,'Tag','Median'),  'LineWidth', boxLW);
set(findobj(figA,'Tag','Cap'),     'LineWidth', boxLW);

x1 = 1 + 0.08*(rand(sum(isInact),1)-0.5);
x2 = 2 + 0.08*(rand(sum(isActive),1)-0.5);

plot(axA, x1, D_inact, 'k.', 'MarkerSize', 18); 
plot(axA, x2, D_act,   'k.', 'MarkerSize', 18); 

title(axA,'');  % caption in paper

set(figA,'PaperPositionMode','auto');
print(figA, [outPrefix 'A.svg'], '-dsvg', '-painters');

%% ===================== FIGURE 2B: ROC curve (FPR/TPR, equal axes) =====================
% Use MATLAB perfcurve if available; otherwise fall back to manual.
yTrue = isActive;     % positive class = Active-SI
scores = D;           % higher D => more likely Active-SI (adjust if needed)

% Remove NaNs
keep = isfinite(scores) & ~isnan(yTrue);
yTrue = yTrue(keep);
scores = scores(keep);

figB = figure('Color','w','Position',[100 100 600 600]);
axB = axes('Parent', figB); hold(axB,'on'); grid(axB,'on'); box(axB,'off');

havePerfcurve = exist('perfcurve','file') == 2;

if havePerfcurve
    [FPR, TPR, ~, AUC] = perfcurve(yTrue, scores, true);
else
    % Manual ROC (no toolbox)
    [FPR, TPR, AUC] = manual_roc(yTrue, scores);
end

% ---- CHANGED: thicker ROC curve ----
hROC = plot(axB, FPR, TPR, 'LineWidth', 4);  % was 2

% Chance line (do NOT show in legend)
hChance = plot(axB, [0 1], [0 1], '--', 'LineWidth', 1);
hChance.Annotation.LegendInformation.IconDisplayStyle = 'off';  % hide from legend

% ---- CHANGED: legend text shows AUC only (no "D-score") ----
legend(axB, hROC, sprintf('AUC = %.3f', AUC), ...
    'FontSize', legendFS, 'Location', 'southeast');

xlabel(axB, 'False Positive Rate (FPR)', 'FontSize', labelFS);
ylabel(axB, 'True Positive Rate (TPR)',  'FontSize', labelFS);

xlim(axB, [0 1]);
ylim(axB, [0 1]);
axis(axB, 'square');   % <-- same scale on x and y

set(axB,'FontSize',tickFS,'LineWidth',1);

title(axB,'');  % caption in paper

set(figB,'PaperPositionMode','auto');
print(figB, [outPrefix 'B.svg'], '-dsvg', '-painters');

disp('Saved camera-ready Figure 2A and 2B exports.');

%% ===================== Helper: Manual ROC (no toolbox) =====================
function [FPR, TPR, AUC] = manual_roc(yTrue, scores)
    % yTrue: logical (1=positive)
    % scores: higher => more positive
    [scoresSorted, idx] = sort(scores, 'descend');
    y = yTrue(idx);

    P = sum(y);
    N = sum(~y);

    TP = cumsum(y);
    FP = cumsum(~y);

    TPR = TP / max(P,1);
    FPR = FP / max(N,1);

    % Add endpoints
    TPR = [0; TPR; 1];
    FPR = [0; FPR; 1];

    % Trapezoidal AUC
    AUC = trapz(FPR, TPR);
end
