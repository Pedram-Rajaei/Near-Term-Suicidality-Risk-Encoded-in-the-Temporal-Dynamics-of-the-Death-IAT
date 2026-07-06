

%% --- Load Dataset ---
table_path = 'Freud_Cohort_N80.xlsx';
tx = readtable(table_path);
participant_ID = tx.ID;
reaction_time  = tx{:, 2:361};       % RT_1 to RT_360
active_score   = tx.SI_label;        % 0 = Low SI, 1 = High SI
si_score = tx.activeSI;
temp = strcmp(tx.Group, 'mdd');   % mdd = 1, ctl = 0
mdd_ctrl = double(temp); % ctrl = 0, mdd=1

rowsToRemove = [6, 48, 72];
%rowsToRemove = [6, 48, 72];
%rowsToRemove = [70, 57, 73,6, 48, 72];
%rowsToRemove = [45,66,70, 57, 73,6, 48, 72];


active_score(rowsToRemove)
active_score(rowsToRemove, :) = [];  % Remove these rows
reaction_time(rowsToRemove, :) = [];
si_score(rowsToRemove, :) = [];  % Remove these rows


n_trials = 360;
N = length(active_score);

XS  = nan(N, 360);
XF  = nan(N, 360);
VS  = nan(N, 360);
YT  = nan(N, 360);

INFO= [];

for i = 1:N
    i
    temp = reaction_time(i,:)';
    ind_valid   = find(temp < 350 & temp > 0);
    ind_exclude = find(temp == 0);
    [i numel(ind_exclude) numel(ind_valid) numel(ind_exclude)+numel(ind_valid) active_score(i)]

    INFO=[INFO;i numel(ind_exclude) numel(ind_valid) numel(ind_exclude)+numel(ind_valid) active_score(i)];

    Yk = log(temp);
    Yk(ind_valid) = nan;
    Yk(ind_exclude) = nan;
    
    % Suppose:
    %  - x has NaN where data is missing (either random or clipped).
    %  - You also know which missing points are because of clipping:
    isClipped = false(size(Yk));
    isClipped(ind_exclude) = true;   % idx_clipped = indices of clipped samples

    thr = log(2000);   % your threshold
    win = 30;     % half-window for local stats (optional)

    [x_filled,mu] = Freud_Impute_RT_Series(Yk, isClipped, thr, win);

    XF(i,:)=x_filled;
end
save('Freud_Processed_BDIAT.mat','active_score','si_score','mdd_ctrl','XS','XF');