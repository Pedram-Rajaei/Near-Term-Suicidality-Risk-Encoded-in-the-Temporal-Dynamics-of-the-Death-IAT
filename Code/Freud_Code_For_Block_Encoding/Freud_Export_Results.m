%% load_excel_to_mat.m
% Loads an XLSX with columns:
% ID, RT_1 ... RT_360, CSSRS, SI, SI_label
% Creates arrays and saves to a .mat file.

% ---- USER SETTINGS ----
xlsxPath = "Freud_Cohort_N80.xlsx";
outMatPath = "Freud_Data_Library.mat";
sheetName = 1;                    % can be 1 or "Sheet1"
% ----------------------

% Read table
T = readtable(xlsxPath, "Sheet", sheetName);

% ---- Extract ID ----
% Supports numeric IDs or string IDs
if iscellstr(T.ID) || isstring(T.ID) || iscategorical(T.ID)
    ID = string(T.ID);
else
    ID = T.ID;
end

% ---- Extract RT matrix (RT_1 ... RT_360) ----
rtVars = "RT_" + string(1:360);

% Check that all expected RT columns exist
missing = setdiff(rtVars, string(T.Properties.VariableNames));
if ~isempty(missing)
    error("Missing RT columns: %s", strjoin(missing, ", "));
end

% Convert to numeric matrix (N x 360)
RT = T{:, rtVars};

% ---- Extract CSSRS, SI ----
CSSRS = T.CSSRS;
SI    = T.SI;

% ---- Extract SI_label (could be numeric or text) ----
if iscellstr(T.SI_label) || isstring(T.SI_label) || iscategorical(T.SI_label)
    SI_label = string(T.SI_label);
else
    SI_label = T.SI_label;
end

% ---- Optional: Basic sanity checks ----
n = height(T);
if size(RT,1) ~= n
    error("RT row count mismatch.");
end

% ---- Save ----
save(outMatPath, "ID", "RT", "CSSRS", "SI", "SI_label");

fprintf("Saved parsed variables to: %s\n", outMatPath);
fprintf("RT matrix size: %d x %d\n", size(RT,1), size(RT,2));
