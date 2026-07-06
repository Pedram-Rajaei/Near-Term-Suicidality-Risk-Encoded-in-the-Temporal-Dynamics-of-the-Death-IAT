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
labelFS  = 40;
tickFS   = 34;
legendFS = 40;
lwMean   = 3.5;     % bold mean
alphaStd = 0.22;    % subtle band
ms       = 150;      % marker size (scatter)

% Colorblind-safe-ish (blue/orange)
col1 = [0.00 0.50 0.15];   % dark green
col2 = [0.55 0.10 0.55];   % deep magenta

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

fig1 = figure('Color','w','Position',[80 80 1600 1300]);
set(fig1,'Renderer','painters');
ax1 = axes('Parent',fig1); hold(ax1,'on'); grid(ax1,'off');

scatter(ax1, S2(idxCTRL,1), S2(idxCTRL,2), ms, 'o', ...
    'MarkerEdgeColor','k', 'MarkerFaceColor', [0.21 0.49 0.72], 'LineWidth', 0.8);
scatter(ax1, S2(idxMDD,1),  S2(idxMDD,2),  ms, 'o', ...
    'MarkerEdgeColor','k', 'MarkerFaceColor', [0.89 0.29 0.20], 'LineWidth', 0.8);

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

xlim(ax1, [-4 4]);
ylim(ax1, [-4 4]);
axis(ax1, 'equal');
xlim(ax1, [-4 4]);
ylim(ax1, [-4 4]);

% ---- Draw boundary clipped to axis limits ----
xl = xlim(ax1);
yl = ylim(ax1);
xLine = linspace(xl(1), xl(2), 400);
yLine = -xLine - b0;

valid = yLine >= yl(1) & yLine <= yl(2);
xLine = xLine(valid);
yLine = yLine(valid);

hBound = plot(ax1, xLine, yLine, 'k-', 'LineWidth', 2.2, 'HandleVisibility','off');
set(hBound, 'Clipping', 'on');
ax1.Clipping = 'on';
drawnow;
xlim(ax1, [-4 4]);
ylim(ax1, [-4 4]);

fprintf('\n[Model decision boundary]\n');
fprintf('  Boundary: x + y + b0 = 0\n');
fprintf('  Using b0 = %.4f\n', b0);
%% ================================================================
% Figure 4C — Marginal PDFs: read ax1 position AFTER axis equal
% Insert AFTER decision boundary plot, BEFORE xlabel/ylabel/legend
%% ================================================================

% Force the figure to finish rendering so ax1 Position is finalised
drawnow;

% Read ax1's ACTUAL position in normalized units (axis equal may have
% already shrunk it — we must read, not write)
ax1.Units = 'normalized';
ap = ax1.Position;   % [left, bottom, width, height] as-rendered

% =====================================================================
%  MANUAL PDF-STRIP GEOMETRY CONTROLS  (set these to taste)
%  -------------------------------------------------------------------
%  TOP strip   = X-axis marginal density.
%  RIGHT strip = Y-axis marginal density  -> unchanged ("fine")
%
%  NOTE: with axis(ax1,'equal') the square data region is letter-boxed
%  inside the (wider) axes rectangle, so the real plot is NARROWER than
%  the axes box. The AUTO top strip is pinned to that true data x-extent
%  (computed below from getpixelposition), so the density now starts at
%  the Y-axis and stops at the right data edge instead of running out to
%  the image border.
% =====================================================================
topPDF_width     = [];      % [] = auto (span the true data x-extent).
                            %  Set a normalized number (0..1) to fix the
                            %  X-axis PDF width manually, e.g. 0.45
topPDF_align     = 'left';  % anchor used only when topPDF_width is set:
                            %  'left' | 'center' | 'right'
topPDF_thickness = 0.10;    % height (thickness) of the top strip

rightPDF_thickness = 0.10;  % width of the right (Y-axis) strip
gap = 0.003;                % gap between scatter edge and strips

% ---- Right (Y-axis) strip placement ----
legPos = [0.61 0.3 0.12 0.25];  % legend box [l b w h] (also used by legend())
rightPDF_dockToLegend = true;   % true : sit just right of the legend
                                % false: original far-right placement
rightPDF_gap    = 0.005;        % horizontal gap from legend to the strip
rightPDF_height = [];           % [] = full Y-axis length (data y-range);
                                %  set a normalized number to shorten it
                                %  (top stays aligned to the data top)

% Provisional placement (final placement is computed after the ax1 shrink,
% using the true data area). Base = full axes box for now.
[tL, tW] = topStripLW(ap(1), ap(3), topPDF_width, topPDF_align);

topPos   = [tL, ap(2)+ap(4)+gap, tW, topPDF_thickness];
rightPos = [ap(1)+ap(3)+gap*10, ap(2), rightPDF_thickness, ap(4)];

ax_top   = axes('Parent', fig1, 'Units', 'normalized', 'Position', topPos);
ax_right = axes('Parent', fig1, 'Units', 'normalized', 'Position', rightPos);

% Colors matching scatter dots exactly
cCTRL = [0.21 0.49 0.72];
cMDD  = [0.89 0.29 0.20];
bw_factor = 1.0;

x_ctrl = S2(idxCTRL, 1);   y_ctrl = S2(idxCTRL, 2);
x_mdd  = S2(idxMDD,  1);   y_mdd  = S2(idxMDD,  2);

xl_main = xlim(ax1);
yl_main = ylim(ax1);
nGrid   = 400;
xGrid   = linspace(xl_main(1), xl_main(2), nGrid);
yGrid   = linspace(yl_main(1), yl_main(2), nGrid);

kde = @(data, grid, bw) ...
    mean(normpdf(grid, data(:), bw * 1.06 * std(data) * numel(data)^(-0.2)), 1);

% --- Top marginal ---
axes(ax_top); hold(ax_top, 'on');
pdf_x_ctrl = kde(x_ctrl, xGrid, bw_factor);
pdf_x_mdd  = kde(x_mdd,  xGrid, bw_factor);

hf_ctrl = fill(ax_top, [xGrid, fliplr(xGrid)], [pdf_x_ctrl, zeros(1,nGrid)], ...
     cCTRL, 'FaceAlpha', 0.40, 'EdgeColor', 'none');
hf_mdd  = fill(ax_top, [xGrid, fliplr(xGrid)], [pdf_x_mdd, zeros(1,nGrid)], ...
     cMDD,  'FaceAlpha', 0.40, 'EdgeColor', 'none');
plot(ax_top, xGrid, pdf_x_ctrl, '-', 'Color', cCTRL, 'LineWidth', 2.2, ...
     'HandleVisibility', 'off');
plot(ax_top, xGrid, pdf_x_mdd,  '-', 'Color', cMDD,  'LineWidth', 2.2, ...
     'HandleVisibility', 'off');

xlim(ax_top, xl_main);
ylim(ax_top, [0, max([pdf_x_ctrl, pdf_x_mdd]) * 1.05]);
set(ax_top, 'Clipping', 'on');
set(ax_top, 'XTickLabel', [], 'YTickLabel', [], 'Box', 'off', ...
    'XColor', 'none', 'YColor', 'none');

% --- Right marginal ---
axes(ax_right); hold(ax_right, 'on');
pdf_y_ctrl = kde(y_ctrl, yGrid, bw_factor);
pdf_y_mdd  = kde(y_mdd,  yGrid, bw_factor);

fill(ax_right, [pdf_y_ctrl, zeros(1,nGrid)], [yGrid, fliplr(yGrid)], ...
     cCTRL, 'FaceAlpha', 0.35, 'EdgeColor', 'none');
fill(ax_right, [pdf_y_mdd,  zeros(1,nGrid)], [yGrid, fliplr(yGrid)], ...
     cMDD,  'FaceAlpha', 0.35, 'EdgeColor', 'none');
plot(ax_right, pdf_y_ctrl, yGrid, '-', 'Color', cCTRL, 'LineWidth', 2.0);
plot(ax_right, pdf_y_mdd,  yGrid, '-', 'Color', cMDD,  'LineWidth', 2.0);

ylim(ax_right, yl_main);
xlim(ax_right, [0, max([pdf_y_ctrl, pdf_y_mdd]) * 1.05]);
set(ax_right, 'Clipping', 'on');
set(ax_right, 'YTickLabel', [], 'XTickLabel', [], 'Box', 'off', ...
    'YColor', 'none', 'XColor', 'none');

% --- Return focus to ax1 ---
axes(ax1);

xlabel(ax1, 'First Learned Component', 'FontSize', labelFS+6);
ylabel(ax1, 'Second Learned Component', 'FontSize', labelFS+6);
set(ax1,'FontSize',tickFS,'LineWidth',1.1);
xticks(ax1, -4:2:4);
yticks(ax1, -4:2:4);
    % ---------------------------------------
title(ax1,'');

% Expand outer margins so PDF strips are not clipped on top/right
ax1.Units   = 'normalized';
outerMargin = 0.06;   % extra breathing room for PDF strips
set(fig1, 'Units', 'normalized');
drawnow;   % ensure positions are settled before reading

% Push ax1 slightly inward (left and bottom only) to free up top+right space
ap = ax1.Position;
ax1.Position = [ap(1), ap(2), ap(3)*0.82, ap(4)*0.82];
drawnow;

% Re-anchor PDF strips. The TOP strip is pinned to the TRUE data area of
% ax1 (axis equal letter-boxes the square data region inside the wider axes
% box), so the X-density spans exactly the plotted x-range — starting at the
% Y-axis and stopping at the right data edge, not the image border.
% The right (Y-axis) strip behavior is unchanged.
ap2 = ax1.Position;
drawnow;   % make sure the rendered geometry is settled before measuring

% --- true data area of ax1 in figure-normalized units ---
axPx  = getpixelposition(ax1, true);   % [x y w h] px, relative to figure
figPx = getpixelposition(fig1);        % figure size in px
% axis equal + equal data spans => square data region of side min(w,h),
% centred inside the axes pixel box.
side  = min(axPx(3), axPx(4));
dxPx  = axPx(1) + (axPx(3) - side)/2;  % data-left   (px)
dyPx  = axPx(2) + (axPx(4) - side)/2;  % data-bottom (px)
dataL = dxPx / figPx(3);               % data-left   (norm)
dataW = side / figPx(3);               % data-width  (norm)
dataB = dyPx / figPx(4);               % data-bottom (norm)
dataH = side / figPx(4);               % data-height (norm)

% Top strip spans the true data x-extent (auto) or a manual width anchored
% within it, and sits just above the top data edge.
[tL, tW] = topStripLW(dataL, dataW, topPDF_width, topPDF_align);
ax_top.Position   = [tL, dataB + dataH + gap, tW, topPDF_thickness];

% Right (Y-axis) strip: spans the full Y-axis (data y-range), positioned
% horizontally next to the legend (dock) or out to the far right.
if isempty(rightPDF_height)
    rB = dataB;  rH = dataH;                     % full Y-axis length
else
    rH = rightPDF_height;  rB = dataB + dataH - rH;   % shorter, top-aligned to data top
end
if rightPDF_dockToLegend
    rLeft = legPos(1) + legPos(3) + rightPDF_gap; % just right of the legend
else
    rLeft = ap2(1) + ap2(3) + gap*10;             % original far-right placement
end
ax_right.Position = [rLeft, rB, rightPDF_thickness, rH];

% ---- Unified 4-entry legend on ax1 (scatter entries + PDF entries) ----
% Grab the two scatter plot handles (blue dots, red dots)
ax1_children = get(ax1, 'Children');
% Children are in reverse draw order: boundary line, red scatter, blue scatter
% Find scatter objects specifically
scatter_handles = findobj(ax1, 'Type', 'scatter');
% scatter_handles(1) = MDD (drawn second = red), scatter_handles(2) = CTRL (blue)
% Ensure correct order: CTRL first
if scatter_handles(1).CData(1) < scatter_handles(2).CData(1)
    h_sc_ctrl = scatter_handles(1);
    h_sc_mdd  = scatter_handles(2);
else
    h_sc_ctrl = scatter_handles(2);
    h_sc_mdd  = scatter_handles(1);
end

[lgd, icons] = legend(ax1, ...
    [h_sc_ctrl, h_sc_mdd], ...
    {'SI-', 'SI+'}, ...
    'Location', 'none', ...
    'Position', legPos, ...
    'FontSize', legendFS, ...
    'Box',      'off');

% Enlarge only the legend marker dots (not the scatter dots)
legMarkerSize = 18;   % <-- increase this to taste
icoMarkers = findobj(icons, 'Type', 'patch');   % scatter markers in legend
set(icoMarkers, 'MarkerSize', legMarkerSize);

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

fig2 = figure('Color','w','Position',[140 140 1500 460]);
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
yticks(linspace(yl(1), yl(2), 3));
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
yticks(linspace(yl(1), yl(2), 3));
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

fig3 = figure('Color','w','Position',[160 160 1500 460]);
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
yticks(linspace(yl(1), yl(2), 3));
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
yticks(linspace(yl(1), yl(2), 3));
ytickformat('%.2f');

title('');
hold off;

set(fig3,'Renderer','painters');
print(fig3, fullfile(outDir,'Figure_4_E.svg'), '-dsvg');
fprintf('\nFigure 4 latent-dynamics exports saved to:\n  %s\n', outDir);


%% ===================== local functions =====================
function [L, W] = topStripLW(baseL, baseW, manualWidth, align)
% Horizontal [left, width] of the top (X-axis) PDF strip.
%   baseL, baseW : left & width of the region to span (the true data area)
%   manualWidth  : [] -> auto (span baseW); else fixed normalized width
%   align        : 'left' | 'center' | 'right' (used only when manualWidth set)
if isempty(manualWidth)
    W = baseW;          % auto: span the supplied region (true data x-extent)
    L = baseL;
else
    W = manualWidth;    % manual override
    switch lower(align)
        case 'center', L = baseL + (baseW - W)/2;
        case 'right',  L = baseL + (baseW - W);
        otherwise,     L = baseL;     % 'left' / default
    end
end
end