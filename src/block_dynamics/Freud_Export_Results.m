%% Freud_Export_Results.m
% Export raw cohort variables from Excel to a MATLAB data library.
%
% This script reads the cohort spreadsheet, extracts participant IDs,
% reaction-time trials, and available clinical/SI variables, and saves them
% to a reusable .mat file.
%
% Required input:
%   Freud_Cohort_N80.xlsx
%
% Required columns:
%   ID, RT_1 ... RT_360, SI_label
%
% Optional columns:
%   CSSRS, SI, activeSI, Group
%
% Output:
%   Freud_Data_Library.mat

clear; close all; clc;

%% ----------------------- User settings -----------------------
xlsxPath   = "Freud_Cohort_N80.xlsx";
outMatPath = "Freud_Data_Library.mat";
sheetName  = 1;

%% ----------------------- Read spreadsheet -----------------------
T = readtable(xlsxPath, "Sheet", sheetName);
varNames = string(T.Properties.VariableNames);

%% ----------------------- Extract participant IDs -----------------------
if ~ismember("ID", varNames)
    error("Missing required column: ID");
end

if iscellstr(T.ID) || isstring(T.ID) || iscategorical(T.ID)
    ID = string(T.ID);
else
    ID = T.ID;
end

%% ----------------------- Extract reaction-time matrix -----------------------
rtVars = "RT_" + string(1:360);

missingRT = setdiff(rtVars, varNames);
if ~isempty(missingRT)
    error("Missing RT columns: %s", strjoin(missingRT, ", "));
end

RT = T{:, rtVars};

%% ----------------------- Extract SI label -----------------------
if ~ismember("SI_label", varNames)
    error("Missing required column: SI_label");
end

SI_label = T.SI_label;

if iscellstr(SI_label) || isstring(SI_label) || iscategorical(SI_label)
    SI_label = string(SI_label);
end

%% ----------------------- Extract optional clinical/SI variables -----------------------
CSSRS = [];
SI = [];
activeSI = [];
Group = [];

if ismember("CSSRS", varNames)
    CSSRS = T.CSSRS;
else
    warning("No CSSRS column found. Saving CSSRS as empty.");
end

if ismember("SI", varNames)
    SI = T.SI;
elseif ismember("activeSI", varNames)
    SI = T.activeSI;
    activeSI = T.activeSI;
else
    warning("No SI or activeSI column found. Saving SI as empty.");
end

if isempty(activeSI) && ismember("activeSI", varNames)
    activeSI = T.activeSI;
end

if ismember("Group", varNames)
    if iscellstr(T.Group) || isstring(T.Group) || iscategorical(T.Group)
        Group = string(T.Group);
    else
        Group = T.Group;
    end
else
    warning("No Group column found. Saving Group as empty.");
end

%% ----------------------- Sanity checks -----------------------
n = height(T);

assert(size(RT, 1) == n, 'RT row count mismatch.');
assert(size(RT, 2) == 360, 'Expected 360 RT columns.');

%% ----------------------- Save MATLAB data library -----------------------
save(outMatPath, ...
    "ID", ...
    "RT", ...
    "CSSRS", ...
    "SI", ...
    "activeSI", ...
    "SI_label", ...
    "Group");

fprintf("Saved parsed variables to: %s\n", outMatPath);
fprintf("RT matrix size: %d x %d\n", size(RT, 1), size(RT, 2));
fprintf("Saved variables: ID, RT, CSSRS, SI, activeSI, SI_label, Group\n");