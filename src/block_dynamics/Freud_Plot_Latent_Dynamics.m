%% Freud_Plot_Latent_Dynamics.m
% Generate latent-dynamics panels for Figure 4.
%
% This script loads the saved J=2 learned-projection classifier package and
% visualizes the full-data embedding, learned block-space projections, and
% learned trial-position weights.
%
% Required inputs:
%   Freud_Model_J2_Latents.mat
%   Freud_Processed_BDIAT.mat
%
% Expected variables:
%   ALT_J2 in Freud_Model_J2_Latents.mat
%   XF, active_score, mdd_ctrl in Freud_Processed_BDIAT.mat
%
% Outputs:
%   Figure_4_C.svg : full-data two-dimensional embedding
%   Figure_4_E.svg : learned trial-position weights
%   Figure_4_F.svg : learned block-space projection vectors

clear; close all; clc;

%% ---------------- PNAS camera-ready formatting ----------------
labelFS  = 34;
tickFS   = 26;
legendFS = 34;
lwMean   = 3.5;     % bold mean
alphaStd = 0.22;    % subtle band
ms       = 70;      % marker size (scatter)

% Colorblind-safe-ish (blue/orange)
col1 = [0.00 0.35 0.80];   % blue
col2 = [0.85 0.10 0.10];  % orange

% Export settings
outDir = pwd;

%% ---------------- Files ----------------
pkgFile  = 'Freud_Model_J2_Latents.mat';
dataFile = 'Freud_Processed_BDIAT.mat';

assert(exist(pkgFile,'file')==2, 'Missing %s', pkgFile);
assert(exist(dataFile,'file')==2, 'Missing %s', dataFile);

S = load(pkgFile, 'ALT_J2');
ALT_J2 = S.ALT_J2;

D = load(dataFile);  % expects XF, active_score, mdd_ctrl at least
assert(isfield(D,'XF'), 'data_bdiat.mat must contain XF');
assert(isfield(D,'active_score'), 'data_bdiat.mat must contain active_score');
assert(isfield(D,'mdd_ctrl'), 'data_bdiat.mat must contain mdd_ctrl');

S360     = D.XF;
labels   = D.active_score(:); %#ok<NASGU>
mdd_ctrl = D.mdd_ctrl(:);

%% ---------------- Rebuild Xcells (ALT uses exp(-0.99*Xi)) ----------------
m = 9; p = 20;
starts = 1:40:321;
idx = cell2mat(arrayfun(@(s) s:(s+19), starts, 'UniformOutput', false));
N = size(S360,1);

Xcells = cell(N,1);
for ii = 1:N
    x  = S360(ii, idx);          % 1x180
    Xi = reshape(x, [p, m])';    % 9x20
    Xcells{ii} = exp(-0.99 * Xi);
end

%% ---------------- Load full-data Theta ----------------
res = ALT_J2.results;
assert(isfield(res,'Theta_full') && ~isempty(res.Theta_full), ...
    'ALT_J2.results.Theta_full not found. Re-run Fig4A AFTER updating run_loocv_alt.m');
Theta_full = res.Theta_full;

% Validate J=2
assert(isfield(Theta_full,'v_list') && numel(Theta_full.v_list) >= 2, 'Theta_full.v_list missing / not J=2');
assert(isfield(Theta_full,'B_concat') && numel(Theta_full.B_concat) >= 40, 'Theta_full.B_concat missing / not J=2');

%% ================================================================
% Figure 4C: Full-data embedding
%% ================================================================
B_concat = Theta_full.B_concat(:);
B1 = B_concat(1:20);
B2 = B_concat(21:40);
v1 = Theta_full.v_list{1}(:);
v2 = Theta_full.v_list{2}(:);

useStd = isfield(Theta_full,'muZ') && isfield(Theta_full,'sigZ') && ...
         ~isempty(Theta_full.muZ) && ~isempty(Theta_full.sigZ);

S2 = zeros(N,2);

if useStd
    Zraw = zeros(N,40);
    for ii = 1:N
        Xi = Xcells{ii};
        z1 = Xi' * v1; % 20x1
        z2 = Xi' * v2; % 20x1
        Zraw(ii,:) = [z1(:); z2(:)].';
    end
    muZ  = Theta_full.muZ(:).';
    sigZ = Theta_full.sigZ(:).'; sigZ(sigZ==0) = 1;
    Zstd = (Zraw - muZ) ./ sigZ;

    S2(:,1) = Zstd(:,1:20)  * B1;
    S2(:,2) = Zstd(:,21:40) * B2;
else
    for ii = 1:N
        Xi = Xcells{ii};
        S2(ii,1) = v1' * Xi * B1;
        S2(ii,2) = v2' * Xi * B2;
    end
end

u = unique(mdd_ctrl(~isnan(mdd_ctrl)));
assert(numel(u)==2, 'mdd_ctrl must have exactly 2 unique values.');
gCTRL = u(1);
gMDD  = u(2);
idxCTRL = find(mdd_ctrl == gCTRL);
idxMDD  = find(mdd_ctrl == gMDD);

fig1 = figure('Color','w','Position',[120 120 900 720]);
set(fig1,'Renderer','painters');
ax1 = axes('Parent',fig1); hold(ax1,'on'); grid(ax1,'off');

scatter(ax1, S2(idxCTRL,1), S2(idxCTRL,2), ms, 'o', ...
    'MarkerEdgeColor','k', 'MarkerFaceColor', col1, 'LineWidth', 0.8);
scatter(ax1, S2(idxMDD,1),  S2(idxMDD,2),  ms, 'o', ...
    'MarkerEdgeColor','k', 'MarkerFaceColor', col2, 'LineWidth', 0.8);

% ==========================================================
% Add the actual model decision boundary
% Model: logit(p_i) = b0 + x + y
% where x = S2(:,1) and y = S2(:,2)
% so the decision boundary is:
%   b0 + x + y = 0   <=>   y = -x - b0
% ==========================================================

assert(isfield(Theta_full,'b0') && ~isempty(Theta_full.b0), ...
    'Theta_full.b0 not found.');

b0 = Theta_full.b0;

% ---- Force same numeric span on x and y axes ----
xData = S2(:,1);
yData = S2(:,2);

xmin = min(xData); xmax = max(xData);
ymin = min(yData); ymax = max(yData);

xmid = (xmin + xmax)/2;
ymid = (ymin + ymax)/2;

halfSpan = 0.5 * max([xmax - xmin, ymax - ymin]);
pad = 0.05 * max(2*halfSpan, eps);

xlim(ax1, [xmid - halfSpan - pad, xmid + halfSpan + pad]);
ylim(ax1, [ymid - halfSpan - pad, ymid + halfSpan + pad]);

axis(ax1,'equal');

% ---- Draw boundary over full visible x-range ----
xl = xlim(ax1);
xLine = linspace(xl(1), xl(2), 400);
yLine = -xLine - b0;

plot(ax1, xLine, yLine, 'k-', 'LineWidth', 2.2, 'HandleVisibility','off');

fprintf('\n[Model decision boundary]\n');
fprintf('  Boundary: x + y + b0 = 0\n');
fprintf('  Using b0 = %.4f\n', b0);

xlabel(ax1, ...
    '$\mathbf{b}_1^{\top}\mathbf{Z}_i\mathbf{v}_1$', ...
    'FontSize', labelFS, ...
    'Interpreter', 'latex', ...
    'FontWeight', 'bold');
ylabel(ax1, ...
    '$\mathbf{b}_2^{\top}\mathbf{Z}_i\mathbf{v}_2$', ...
    'FontSize', labelFS, ...
    'Interpreter', 'latex', ...
    'FontWeight', 'bold');set(ax1,'FontSize',tickFS,'LineWidth',1.1);
title(ax1,'');

lg1 = legend(ax1, {'SI-','SI+'}, 'Location','northeast');
set(lg1,'FontSize',legendFS,'Box','off');

set(fig1,'Renderer','painters');
print(fig1, fullfile(outDir,'Figure_4_C.svg'), '-dsvg');


%% ================================================================
% Extract folds: V_all (9x2xK), B_all (20x2xK)
%% ================================================================
V_all = [];
B_all = [];

candV = {'V_all','v_all','Vfolds','v_folds','V_folds'};
for k = 1:numel(candV)
    if isfield(res, candV{k}), V_all = res.(candV{k}); break; end
end
candB = {'B_all','b_all','Bfolds','b_folds','B_folds'};
for k = 1:numel(candB)
    if isfield(res, candB{k}), B_all = res.(candB{k}); break; end
end

if isempty(V_all) || isempty(B_all)
    foldModels = [];
    if isfield(res,'fold_models'), foldModels = res.fold_models; end
    if isempty(foldModels) && isfield(res,'models'), foldModels = res.models; end
    if ~isempty(foldModels)
        K = numel(foldModels);
        V_all = zeros(m,2,K);
        B_all = zeros(p,2,K);
        for k = 1:K
            Tk = foldModels{k};
            V_all(:,1,k) = Tk.v_list{1}(:);
            V_all(:,2,k) = Tk.v_list{2}(:);
            bc = Tk.B_concat(:);
            B_all(:,1,k) = bc(1:20);
            B_all(:,2,k) = bc(21:40);
        end
    end
end

assert(~isempty(V_all) && ~isempty(B_all), 'Could not find fold-wise V_all/B_all in ALT_J2.results.');
assert(all(size(V_all,1:2)==[m 2]), 'V_all must be 9x2xK');
assert(all(size(B_all,1:2)==[p 2]), 'B_all must be 20x2xK');

Kfold = size(V_all,3);

%% ---------------- Sign-align folds to full-data solution ----------------
Vref = zeros(m,2);
Bref = zeros(p,2);
Vref(:,1) = Theta_full.v_list{1}(:);
Vref(:,2) = Theta_full.v_list{2}(:);
bc_full = Theta_full.B_concat(:);
Bref(:,1) = bc_full(1:20);
Bref(:,2) = bc_full(21:40);

for k = 1:Kfold
    for j = 1:2
        if dot(V_all(:,j,k), Vref(:,j)) < 0
            V_all(:,j,k) = -V_all(:,j,k);
            B_all(:,j,k) = -B_all(:,j,k);
        end
    end
end

%% ================================================================
% Figure 4F: Block-space projection vectors across folds
%% ================================================================
xV = 1:m;

V1_all = squeeze(V_all(:,1,:)); % m x K
V2_all = squeeze(V_all(:,2,:)); % m x K
muV1 = -mean(V1_all,2); sdV1 = std(V1_all,0,2);
muV2 = mean(V2_all,2); sdV2 = std(V2_all,0,2);

fig2 = figure('Color','w','Position',[140 140 1100 460]);
set(fig2,'Renderer','painters');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% V1
nexttile; hold on; grid off;
hStd = fill([xV fliplr(xV)], [ (muV1+sdV1).' fliplr((muV1-sdV1).') ], col1, ...
    'FaceAlpha', alphaStd, 'EdgeColor','none');
hStd.Annotation.LegendInformation.IconDisplayStyle = 'off';
hMean = plot(xV, muV1, '-', 'Color', col1, 'LineWidth', lwMean);
xlabel('Block index','FontSize',labelFS);
ylabel('$\mathbf{b}_1$', ...
    'Interpreter','latex', ...
    'FontSize',labelFS, ...
    'FontWeight','bold');
set(gca,'FontSize',tickFS,'LineWidth',1.1);
yl = ylim(gca);
yticks(linspace(yl(1), yl(2), 5));
ytickformat('%.2f');

title('');
hold off;

% V2
nexttile; hold on; grid off;
hStd = fill([xV fliplr(xV)], [ (muV2+sdV2).' fliplr((muV2-sdV2).') ], col2, ...
    'FaceAlpha', alphaStd, 'EdgeColor','none');
hStd.Annotation.LegendInformation.IconDisplayStyle = 'off';
hMean = plot(xV, muV2, '-', 'Color', col2, 'LineWidth', lwMean);
xlabel('Block index','FontSize',labelFS);
ylabel('$\mathbf{b}_2$', ...
    'Interpreter','latex', ...
    'FontSize',labelFS, ...
    'FontWeight','bold');set(gca,'FontSize',tickFS,'LineWidth',1.1);
yl = ylim(gca);
yticks(linspace(yl(1), yl(2), 5));
ytickformat('%.2f');

title('');
hold off;

set(fig2,'Renderer','painters');
print(fig2, fullfile(outDir,'Figure_4_F.svg'), '-dsvg');

%% ================================================================
% Figure 4E: Trial-position weights across folds
%% ================================================================
xB = 1:p;

B1_all = squeeze(B_all(:,1,:)); % p x K
B2_all = squeeze(B_all(:,2,:)); % p x K
muB1 = -mean(B1_all,2); sdB1 = std(B1_all,0,2);
muB2 = mean(B2_all,2); sdB2 = std(B2_all,0,2);

fig3 = figure('Color','w','Position',[160 160 1100 460]);
set(fig3,'Renderer','painters');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% B1
nexttile; hold on; grid off;
hStd = fill([xB fliplr(xB)], [ (muB1+sdB1).' fliplr((muB1-sdB1).') ], col1, ...
    'FaceAlpha', alphaStd, 'EdgeColor','none');
hStd.Annotation.LegendInformation.IconDisplayStyle = 'off';
hMean = plot(xB, muB1, '-', 'Color', col1, 'LineWidth', lwMean);
xlabel('Trial index','FontSize',labelFS);
ylabel('$\mathbf{v}_1$', ...
    'Interpreter','latex', ...
    'FontSize',labelFS, ...
    'FontWeight','bold');
set(gca,'FontSize',tickFS,'LineWidth',1.1);
yl = ylim(gca);
yticks(linspace(yl(1), yl(2), 5));
ytickformat('%.2f');

title('');
hold off;

% B2
nexttile; hold on; grid off;
hStd = fill([xB fliplr(xB)], [ (muB2+sdB2).' fliplr((muB2-sdB2).') ], col2, ...
    'FaceAlpha', alphaStd, 'EdgeColor','none');
hStd.Annotation.LegendInformation.IconDisplayStyle = 'off';
hMean = plot(xB, muB2, '-', 'Color', col2, 'LineWidth', lwMean);
xlabel('Trial index','FontSize',labelFS);

ylabel('$\mathbf{v}_2$', ...
    'Interpreter','latex', ...
    'FontSize',labelFS, ...
    'FontWeight','bold');
set(gca,'FontSize',tickFS,'LineWidth',1.1);
yl = ylim(gca);
yticks(linspace(yl(1), yl(2), 5));
ytickformat('%.2f');

title('');
hold off;

set(fig3,'Renderer','painters');
print(fig3, fullfile(outDir,'Figure_4_E.svg'), '-dsvg');
fprintf('\nFigure 4 latent-dynamics exports saved to:\n  %s\n', outDir);
