clear all
close all

% Load data
load('Freud_Processed_BDIAT.mat');    b_len = 20; b_num = 18;
load('Freud_Processed_BDIAT_Short.mat'); b_len = 10; b_num = 36;
X = exp(XF);
G = active_score;

% ===================== Figure 2E source (Figure 1 from this call) =====================
out = Freud_Autocorr_Advanced_2(X, G, ...
    'DoTrialACF', false, 'DoBlockACF', true, ...
    'MaxLagBlock', 12, 'BlockSummary', 'mean', 'DetrendBlocks', true, ...
    'UseLog', false, 'BlockLen', b_len,'NumBlocks', b_num);

% ===================== Camera-ready formatting + export (Fig 2E) =====================
fig2e_handle = gcf;     % this is Figure 1 you are talking about

labelFS  = 24;
tickFS   = 20;
legendFS = 20;

figure(fig2e_handle);
set(gcf,'Color','w');
set(gcf,'Renderer','painters');   % vector export safe

ax = gca;
grid(ax,'on');
box(ax,'off');

% ---- KEY REQUEST: cut x-axis from 0.5 to end ----
xlim(ax, [0.5 12]);   % since MaxLagBlock=12; or use [0.5 max(ax.XLim)]

% (optional) keep y clean; comment out if you prefer auto
% ylim(ax, [-0.2 1.02]);

xlabel(ax,'Lag (blocks)','FontSize',labelFS);
ylabel(ax,'Autocorrelation','FontSize',labelFS);
xlim(ax, [1 12]);
ylim(ax, [-0.15 0.15]);
xticks(ax, 1:12);      % or 1:maxLag
ax.XTickLabelRotation = 0;
yticks(ax, -0.15:0.05:0.15);
set(ax,'FontSize',tickFS,'LineWidth',1);
yticks(ax, [-0.15 -0.075 0 0.075 0.15]);
lg = legend(ax, {'Without active SI','With active SI'}, ...
            'Location','northeast');

set(lg,'FontSize',legendFS,'Box','off');

title(ax,'');   % remove title for paper

set(gcf,'PaperPositionMode','auto');
print(gcf,'Figure_2_E.svg','-dsvg','-painters');

disp('Saved camera-ready Fig 2E (x >= 0.5)');

% % If you want to also see trial-level ACF (usually dominated by inertia)
% out2 = acf_trial_vs_block_with_plots(X, G, ...
%     'DoTrialACF', true, 'MaxLagTrial', 120, ...
%     'DoBlockACF', true, 'MaxLagBlock', 10);

% ===================== The rest of your code =====================
out = Freud_Autocorr_Advanced(X, G, ...
    'MaxLagBlock', 12, ...
    'RhythmMaxLag', 8, ...
    'NumPerm', 10000, ...
    'NumShufflePreserve', 10000, ...
    'NumShuffleDestroy', 10000, ...
    'ShuffleDestroyMode', 'permuteBlocks', ...
    'FigureNamePrefix', 'MyTask','BlockSummary', 'mean', ...
    'BlockLen', b_len,'NumBlocks', b_num);
