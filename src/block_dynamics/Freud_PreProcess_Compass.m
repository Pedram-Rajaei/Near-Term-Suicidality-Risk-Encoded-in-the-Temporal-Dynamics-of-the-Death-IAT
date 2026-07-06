clear all;
close all;
clc;




%% --- Load Dataset ---
table_path = 'Freud_Cohort_N80.xlsx';
tx = readtable(table_path);
participant_ID = tx.ID;
reaction_time  = tx{:, 2:361};       % RT_1 to RT_360
active_score   = tx.SI_label;        % 0 = Low SI, 1 = High SI


rowsToRemove = [6, 48, 72];
active_score(rowsToRemove)
active_score(rowsToRemove, :) = [];  % Remove these rows
reaction_time(rowsToRemove, :) = [];

n_trials = 360;
N = length(active_score);

XS  = nan(N, 360);
XF  = nan(N, 360);

for i = 1:N
    temp = reaction_time(i,:)';
    ind_valid   = find(temp < 350 & temp > 0);
    ind_exclude = find(temp == 0);

    Yk = log(temp);
    In = ones(length(Yk),2);
    Un = zeros(length(Yk),1);
    valid = ones(length(Yk),1);
    valid(ind_valid) = 0;
    valid(ind_exclude) = 2;

    Param = compass_create_state_space(1,1,2,0,1,1,0,1,0);
    %Param = compass_create_state_space(3,1,3,0,eye(3,3),[1 2 3],[0 0 0],[1 2 3],[0 0 0]);
    Param.W0 = 10;
    Param.Wk = 0.16;
    Param.Vk = 0.05;
    temp_ind = find(valid==1);
    Param.Dk = [0 mean(Yk(temp_ind))];
    
    Param = compass_set_learning_param(Param,100,0,0,0,1,1,0,1,2,0);
    % 1,1 worked with smaller accuracy
    % 2,1 worked with balanced specificity and sensitivity
    % 2,2 more sensitive
    % 1, 2 works best
    % 2 or 1, followed by 1

    Param = compass_set_censor_threshold_proc_mode(Param,log(2000),2,1);

    ind_a = [1:20 41:60 81:100 121:140 161:180 201:220 241:260 281:300 321:340];
    
    [Xs,~,Param,Xf,~,~,Yp] = compass_em([1 0], Un, In, [], Yk, [], Param, valid);

    xf = zeros(360,1);
    for t=1:360
        temp = Xf{t};
        xf(t)= temp(1);
    end

    xs = zeros(360,1);
    for t=1:360
        temp = Xs{t};
        xs(t)= temp(1);
    end
    
    XS(i,:) = xs';
    XF(i,:) = xf';
end
save('Freud_Processed_BDIAT.mat','active_score','XS','XF');
