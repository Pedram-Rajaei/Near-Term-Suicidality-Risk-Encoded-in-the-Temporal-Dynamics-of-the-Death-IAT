%% Freud_Plot_Representative_RT_Traces.m
% Generate representative trial-level reaction-time traces for Figure 2C.
%
% Outputs:
%   Figure_2_C_1.svg
%   Figure_2_C_2.svg
%
% Notes:
%   - Uses raw RT values from Freud_Cohort_N80.xlsx
%   - Missing/nonpositive RT values are interpolated before log transform
%   - Figure_2_C_1 uses participant ID 367
%   - Figure_2_C_2 uses participant ID 341
clear;
close all;
clc;

%% Load raw cohort data
T = readtable('Freud_Cohort_N80.xlsx');
rtVars = strcat("RT_", string(1:360));
RT = T{:, rtVars};
ID = T.ID;

%% Basic preprocessing
RT(RT <= 0) = NaN;
for i = 1:size(RT,1)
    xi = RT(i,:);
    t  = 1:numel(xi);
    good = isfinite(xi);
    if sum(good) == 0
        continue;
    elseif sum(good) == 1
        xi(~good) = xi(good);
    else
        xi(~good) = interp1( ...
            t(good), xi(good), t(~good), ...
            'linear', 'extrap');
    end
    RT(i,:) = xi;
end
logRT = log(RT);

%% Figure settings
blockLen = 20;
smoothWin = 17;
rawColor_SIminus    = [0.39 0.72 0.81];
smoothColor_SIminus = [0.21 0.49 0.72];
rawColor_SIplus     = [1.00 0.55 0.55];
smoothColor_SIplus  = [0.89 0.29 0.20];
grayBlock   = [0.85 0.85 0.85];
yellowBlock = [1.00 0.98 0.66];
labelFS = 52;
tickFS  = 38;

%% Plot representative traces
plot_subject_trace( ...
    logRT, ID, 367, ...
    'Figure_2_C_1.svg', ...
    'SI-', ...
    blockLen, smoothWin, ...
    rawColor_SIminus, smoothColor_SIminus, ...
    grayBlock, yellowBlock, ...
    labelFS, tickFS);

plot_subject_trace( ...
    logRT, ID, 341, ...
    'Figure_2_C_2.svg', ...
    'SI+', ...
    blockLen, smoothWin, ...
    rawColor_SIplus, smoothColor_SIplus, ...
    grayBlock, yellowBlock, ...
    labelFS, tickFS);

fprintf('Saved Figure_2_C_1.svg and Figure_2_C_2.svg\n');

%% =======================================================================
function plot_subject_trace( ...
    logRT, ID, targetID, outFile, ...
    cornerLabel, ...
    blockLen, smoothWin, ...
    rawColor, smoothColor, ...
    grayBlock, yellowBlock, ...
    labelFS, tickFS)

    idx = find(ID == targetID, 1);
    if isempty(idx)
        error('Participant ID %d not found.', targetID);
    end

    y  = logRT(idx,:);
    ys = smoothdata(y, 'movmean', smoothWin);
    x  = 1:numel(y);

    fig = figure( ...
        'Color', 'w', ...
        'Position', [100 100 1500 480]);
    ax = axes(fig);
    hold(ax, 'on');

    %% Alternating task blocks
    nBlocks = numel(y) / blockLen;
    for b = 1:nBlocks
        x0 = (b - 1) * blockLen + 1;
        x1 = b * blockLen;
        if mod(b,2) == 1
            c = grayBlock;
        else
            c = yellowBlock;
        end
        patch( ...
            [x0 x1 x1 x0], ...
            [6.4 6.4 7.6 7.6], ...
            c, ...
            'FaceAlpha', 0.70, ...
            'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
    end

    %% Raw and smoothed traces
    plot(x, y, ...
        'Color', rawColor, ...
        'LineWidth', 2.0);
    plot(x, ys, ...
        'Color', smoothColor, ...
        'LineWidth', 5.0);

    %% Axes styling
    xlim([1 360]);
    ylim([6.4 7.6]);
    xlabel('Trial', ...
        'FontSize', labelFS);
    ylabel('log(RT)', ...
        'FontSize', labelFS);
    xticks(50:50:350);
    yticks([6.5 7.0 7.5]);
    yticklabels({'6.5','7','7.5'});
    set(ax, ...
        'FontSize', tickFS, ...
        'LineWidth', 2.0);
    box off;
    grid off;

    %% Corner label (top-right)
    text(ax, 360, 7.6, cornerLabel, ...
        'FontSize',           labelFS, ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment',  'top');

    set(fig, 'PaperPositionMode', 'auto');
    print(fig, outFile, '-dsvg');
    close(fig);
end