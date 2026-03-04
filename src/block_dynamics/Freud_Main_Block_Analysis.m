clear all
close all

% Load data
load('Freud_Processed_BDIAT.mat');
X = exp(XF);
G = active_score;

% ===================== Figure 2C (Block ACF) =====================
outC  = Freud_Autocorr_Engine(X, G, ...
    'DoTrialACF', false, 'DoBlockACF', true, ...
    'MaxLagBlock', 8, 'BlockSummary', 'mean', 'DetrendBlocks', true,'UseLog',false);

% Grab handle for the most recent figure (assumes block ACF is last created here)
fig2c_handle = gcf;

% ===================== Figure 2D source (Destroy...) =====================
outJan  = Freud_Autocorr_Advanced(X, G, ...
    'MaxLagBlock', 8, ...
    'RhythmMaxLag', 8, ...
    'NumPerm', 10000, ...
    'NumShufflePreserve', 10000, ...
    'NumShuffleDestroy', 10000, ...
    'ShuffleDestroyMode', 'permuteBlocks', ...
    'FigureNamePrefix', 'MyTask','BlockSummary', 'mean');

% Export camera-ready panels
camera_ready_export(fig2c_handle, outJan.tests.shuffleDestroyAB, 'Figure_2_');



function camera_ready_export(fig2c_handle, destroy_stats, outPrefix)
% camera_ready_export(fig2c_handle, destroy_stats, outPrefix)
% - fig2c_handle: handle to the Block ACF figure (Figure 2C)
% - destroy_stats: out.tests.shuffleDestroyAB struct from acf_trial_vs_block_with_plots_new_jan
% - outPrefix: e.g., 'Fig2'

labelFS  = 24;
tickFS   = 20;
legendFS = 20;

%% -------------------- FIGURE 2C: Block ACF (camera-ready) --------------------
figure(fig2c_handle);
set(gcf,'Color','w');
set(gcf,'Renderer','painters');   % best for vector output

ax = gca;
grid(ax,'on');
box(ax,'off');

title(ax,''); % paper caption handles title
xlim(ax, [1 8]);
ylim(ax, [-0.5 0.5]);
xlabel(ax,'Lag (blocks)','FontSize',labelFS);
ylabel(ax,'Autocorrelation','FontSize',labelFS);
set(ax,'FontSize',tickFS,'LineWidth',1);
yticks(ax, [-0.5 -0.25 0 0.25 0.5]);
lg = legend(ax, {'Without active SI','With active SI'}, ...
            'Location','northeast');

set(lg,'FontSize',legendFS,'Box','off');

set(gcf,'PaperPositionMode','auto');
print(gcf, [outPrefix 'C.svg'], '-dsvg', '-painters');

%% -------------------- FIGURE 2D: Destroy null (RIGHT panel only) --------------------
% destroy_stats must contain:
%   destroy_stats.perm.groupDiff  (null values)
%   destroy_stats.obs.groupDiff   (observed value)
nullVals = destroy_stats.perm.groupDiff;
obsVal   = destroy_stats.obs.groupDiff;

% Compute permuted distribution stats
muNull = mean(nullVals,'omitnan');
sdNull = std(nullVals,'omitnan');

f = figure('Color','w','Position',[100 100 900 520]);
set(f,'Renderer','painters');
ax2 = axes('Parent',f); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'off');

% Camera-ready histogram
hHist = histogram(ax2, nullVals, 40, 'Normalization','pdf');

% y-limits AFTER histogram so lines span full height
yl = ylim(ax2);

% Observed value (red)
hObs = plot(ax2, [obsVal obsVal], yl, 'r-', 'LineWidth', 2);

% Permuted mean (black dashed) [optional but nice]
hMu  = plot(ax2, [muNull muNull], yl, 'k--', 'LineWidth', 2);
xlabel(ax2,'RhythmIndex group difference (With active SI - Without active SI)','FontSize',labelFS);
ylabel(ax2,'Density','FontSize',labelFS);
set(ax2,'FontSize',tickFS,'LineWidth',1);

title(ax2,''); % remove title
axis(ax2,'tight');
xlim(ax2, [-0.3 ax2.XLim(2)]);

% Legend: Observed + Permuted stats
legStrPerm = sprintf('Permuted (mean=%.3f, std=%.3f)', muNull, sdNull);
lg2 = legend(ax2, [hObs hHist], {'Observed', legStrPerm}, ...
    'Location','northoutside', ...
    'Orientation','horizontal');

set(lg2,'FontSize',legendFS,'Box','off');
print(f, [outPrefix 'D.svg'], '-dsvg', '-painters');

close(f);


disp('Saved camera-ready Fig2C and Fig2D exports.');
end
