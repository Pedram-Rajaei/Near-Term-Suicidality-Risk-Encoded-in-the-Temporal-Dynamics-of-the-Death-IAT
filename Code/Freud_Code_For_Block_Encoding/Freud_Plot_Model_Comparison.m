%% ======================= Figure 4A: ONE ROC plot with 6 curves (camera-ready) =======================
% Produces Fig 4A as a SINGLE ROC panel overlaying 6 ROC curves:
%   - Alt-learned v (J=1,2,3)
%   - Fixed-PC v    (J=1,2,3)
%
% Prints (TERMINAL ONLY) for each of the 6 models:
%   AUC, Sensitivity (TPR*), Specificity (TNR*), Balanced Accuracy (at Youden J threshold)
%
% Also saves ROC + LOOCV results for:
%   Alt-learned v with J=2 (NOT fixed PC) -> used for Fig4B/4C/4D scripts
%
% Requirements:
%   - data_bdiat.mat contains: XF, active_score
%   - run_loocv_alt.m, run_loocv_fixedV.m are on MATLAB path
%   - perfcurve available (Stats Toolbox) OR manual ROC fallback works

clear; close all; clc;

%% ----------------------- Camera-ready formatting -----------------------
labelFS  = 24;
tickFS   = 20;
legendFS = 18;

outPrefix = "Figure_4_A";

%% ----------------------- Verify dependencies ---------------------------
needFns = ["Freud_Model_CrossVal_Joint","Freud_Model_CrossVal_Fixed"];
for k = 1:numel(needFns)
    if exist(needFns(k), 'file') ~= 2
        error('Missing function "%s". Put it on MATLAB path.', needFns(k));
    end
end

%% ----------------------- Load data once --------------------------------
m = 9; p = 20;                         % blocks x time
load('Freud_Processed_BDIAT.mat');                % expects XF (n x 360) and active_score
S360   = XF;
labels = active_score(:);

% Slice indices: Death+Me blocks (9 windows of length 20 -> total 180 columns)
starts = 1:40:321;
idx = cell2mat(arrayfun(@(s) s:(s+19), starts, 'UniformOutput', false));
N = size(S360,1);

% Binary labels {0,1}: map larger label to 1
classes = unique(labels(:));
assert(numel(classes)==2, 'Need exactly two classes in active_score.');
cA = classes(1); cB = classes(2);
y  = double(labels == cB);

fprintf('\n===== DATA SUMMARY =====\n');
fprintf('N=%d | active_score classes=[%g,%g] -> positive class is %g\n', N, cA, cB, cB);
fprintf('Counts: y=0: %d | y=1: %d\n\n', sum(y==0), sum(y==1));

%% ----------------------- Build Xcells for BOTH variants ----------------
alpha_alt   = 0.99;   % learned-v script uses exp(-0.99*Xi)
alpha_fixed = 0.985;  % fixed-v script uses exp(-0.985*Xi)

Xcells_alt   = cell(N,1);
Xcells_fixed = cell(N,1);

for ii = 1:N
    x  = S360(ii, idx);           % 1x180
    Xi = reshape(x, [p, m])';     % 9x20
    Xcells_alt{ii}   = exp(-alpha_alt   * Xi);
    Xcells_fixed{ii} = exp(-alpha_fixed * Xi);
end

%% ----------------------- Base cfg templates ----------------------------
cfg_alt = struct();
cfg_alt.rng_seed        = 42;
cfg_alt.altIters        = 6;
cfg_alt.vGradSteps      = 12;
cfg_alt.vStepSize       = 0.15;
cfg_alt.orthOn          = true;
cfg_alt.cvFolds         = 5;
cfg_alt.useOneSE        = true;
cfg_alt.balanceWeights  = true;
cfg_alt.standardize     = true;

cfg_fixed = struct();
cfg_fixed.rng_seed        = 42;
cfg_fixed.cvFolds         = 5;
cfg_fixed.useOneSE        = true;
cfg_fixed.balanceWeights  = true;
cfg_fixed.standardize     = true;

rng(cfg_alt.rng_seed,'twister');

%% ----------------------- Run 6 LOOCV + compute ROC + metrics -----------

J_list = 1:3;

ROC = struct();
ROC.alt = repmat(struct('fpr',[],'tpr',[],'auc',[],'scores',[],'labels',[],'results',[], ...
                        'sens',[],'spec',[],'balacc',[],'thr',[]), 1, 3);
ROC.fix = repmat(struct('fpr',[],'tpr',[],'auc',[],'scores',[],'labels',[],'results',[], ...
                        'sens',[],'spec',[],'balacc',[],'thr',[]), 1, 3);

cfg_alt_J2_used = [];

fprintf('===== LOOCV METRICS (printed only) =====\n');
for J = J_list

    % ---------- ALT (learned v) ----------
    cfg_alt.J = J;
    fprintf('Running ALT (learned v) LOOCV, J=%d...\n', J);
    res_alt = Freud_Model_CrossVal_Joint(Xcells_alt, y, cfg_alt);


    if J == 2
        cfg_alt_J2_used = cfg_alt;
    end

    [fprA,tprA,aucA, thrA, sensA, specA, balA] = compute_roc_and_metrics(res_alt.p_all(:), y(:));

    ROC.alt(J).fpr     = fprA;
    ROC.alt(J).tpr     = tprA;
    ROC.alt(J).auc     = aucA;
    ROC.alt(J).thr     = thrA;
    ROC.alt(J).sens    = sensA;
    ROC.alt(J).spec    = specA;
    ROC.alt(J).balacc  = balA;
    ROC.alt(J).scores  = res_alt.p_all(:);
    ROC.alt(J).labels  = y(:);
    ROC.alt(J).results = res_alt;

    fprintf('  ALT  J=%d | AUC=%.3f | Sens=%.3f | Spec=%.3f | BalAcc=%.3f | Thr*=%.3f\n', ...
        J, aucA, sensA, specA, balA, thrA);

    % ---------- FIXED (PC v) ----------
    cfg_fixed.J = J;
    fprintf('Running FIXED (PC v) LOOCV, J=%d...\n', J);
    res_fix = Freud_Model_CrossVal_Fixed(Xcells_fixed, y, cfg_fixed);

    [fprF,tprF,aucF, thrF, sensF, specF, balF] = compute_roc_and_metrics(res_fix.p_all(:), y(:));

    ROC.fix(J).fpr     = fprF;
    ROC.fix(J).tpr     = tprF;
    ROC.fix(J).auc     = aucF;
    ROC.fix(J).thr     = thrF;
    ROC.fix(J).sens    = sensF;
    ROC.fix(J).spec    = specF;
    ROC.fix(J).balacc  = balF;
    ROC.fix(J).scores  = res_fix.p_all(:);
    ROC.fix(J).labels  = y(:);
    ROC.fix(J).results = res_fix;

    fprintf('  FIX  J=%d | AUC=%.3f | Sens=%.3f | Spec=%.3f | BalAcc=%.3f | Thr*=%.3f\n\n', ...
        J, aucF, sensF, specF, balF, thrF);
end

%% ----------------------- Save J=2 ALT-learned-v package for Fig4B/C/D ---
saveFile = 'Freud_Model_J2_Latents.mat';
Jsave = 2;

ALT_J2 = struct();
ALT_J2.cfg     = cfg_alt_J2_used;       % exact cfg used for J=2
ALT_J2.roc_fpr = ROC.alt(Jsave).fpr;
ALT_J2.roc_tpr = ROC.alt(Jsave).tpr;
ALT_J2.auc     = ROC.alt(Jsave).auc;
ALT_J2.thrStar = ROC.alt(Jsave).thr;
ALT_J2.sens    = ROC.alt(Jsave).sens;
ALT_J2.spec    = ROC.alt(Jsave).spec;
ALT_J2.balacc  = ROC.alt(Jsave).balacc;
ALT_J2.scores  = ROC.alt(Jsave).scores; % p_all
ALT_J2.labels  = ROC.alt(Jsave).labels; % y
ALT_J2.results = ROC.alt(Jsave).results;

save(saveFile, 'ALT_J2');
fprintf('\nSaved J=2 PC learned package for Fig4B/C/D to:\n  %s\n', saveFile);

%% ----------------------- Plot Figure 4A (ONE panel, 6 curves) ----------
fig = figure('Color','w','Position',[100 100 900 720]);
set(fig,'Renderer','painters');
ax = axes('Parent',fig); 
hold(ax,'on'); 
grid(ax,'on');

% Chance line (diagonal) - keep line but hide from legend
plot(ax, [0 1], [0 1], '--', ...
    'LineWidth', 1.2, ...
    'HandleVisibility','off');


% --- Plot ALT curves first, keep handles ---
hAlt = gobjects(1,3);
for J = 1:3
    hAlt(J) = plot(ax, ROC.alt(J).fpr, ROC.alt(J).tpr, ...
        'LineWidth', 2.2, ...
        'DisplayName', sprintf('PC learned, J=%d (AUC=%.3f)', J, ROC.alt(J).auc));
end

% --- Plot FIXED curves, keep handles ---
hFix = gobjects(1,3);
for J = 1:3
    hFix(J) = plot(ax, ROC.fix(J).fpr, ROC.fix(J).tpr, ...
        'LineWidth', 2.2, ...
        'DisplayName', sprintf('PC fixed, J=%d (AUC=%.3f)', J, ROC.fix(J).auc));
end

% ==========================================================
% Highlight ALT J=2  (the main model)
% ==========================================================
set(hAlt(2), 'Color', [0.85 0.1 0.1]);   % strong red
set(hAlt(2), 'LineWidth', 4.0);          % thicker


axis(ax,'square');
xlim(ax,[0 1]); ylim(ax,[0 1]);

xlabel(ax,'False Positive Rate','FontSize',labelFS);
ylabel(ax,'True Positive Rate','FontSize',labelFS);
set(ax,'FontSize',tickFS,'LineWidth',1);
title(ax,'');

lg = legend(ax,'Location','southeast');
set(lg,'FontSize',legendFS,'Box','off');

% Export
set(fig,'PaperPositionMode','auto');
print(fig, outPrefix + ".svg", "-dsvg", "-painters");

fprintf('\nSaved Figure 4A as:\n  %s.svg\n  %s.pdf\n  %s.png\n', outPrefix, outPrefix, outPrefix);

%% ----------------------- Save all ROC data package ----------------------
save('Freud_ROC_Comparison_Data.mat', 'ROC', 'cfg_alt', 'cfg_fixed', 'alpha_alt', 'alpha_fixed');
fprintf('Saved all ROC curves/data to: Fig4A_all_ROC_data.mat\n');

%% =======================================================================
%%                               HELPERS
%% =======================================================================

function [fpr,tpr,auc,thrStar,sensStar,specStar,balAccStar] = compute_roc_and_metrics(scores, labels_bin)
% - ROC + AUC (perfcurve if available, else manual ROC)
% - Metrics computed at thrStar chosen by Youden J = max(TPR - FPR)

    scores = scores(:);
    labels_bin = labels_bin(:);

    if any(isnan(scores))
        keep = ~isnan(scores);
        scores = scores(keep);
        labels_bin = labels_bin(keep);
    end

    usedPerfcurve = false;
    try
        [fpr,tpr,thr,auc] = perfcurve(labels_bin, scores, 1);
        usedPerfcurve = true;
    catch
        thr = linspace(0,1,401);
        tpr = zeros(size(thr));
        fpr = zeros(size(thr));
        P = sum(labels_bin==1);
        N = sum(labels_bin==0);
        for k = 1:numel(thr)
            yhat = scores >= thr(k);
            tp = sum(yhat==1 & labels_bin==1);
            fp = sum(yhat==1 & labels_bin==0);
            tpr(k) = tp / max(P,1);
            fpr(k) = fp / max(N,1);
        end
        [fpr,ord] = sort(fpr);
        tpr = tpr(ord);
        thr = thr(ord);
        auc = trapz(fpr, tpr);
    end

    [~,kStar] = max(tpr - fpr);
    thrStar  = thr(kStar);

    if usedPerfcurve && ~isfinite(thrStar)
        thrStar = 0.5;
    end

    yhatStar = scores >= thrStar;
    tp = sum(yhatStar==1 & labels_bin==1);
    fn = sum(yhatStar==0 & labels_bin==1);
    tn = sum(yhatStar==0 & labels_bin==0);
    fp = sum(yhatStar==1 & labels_bin==0);

    sensStar   = tp / max(tp+fn,1);   % TPR
    specStar   = tn / max(tn+fp,1);   % TNR
    balAccStar = 0.5*(sensStar + specStar);
end
