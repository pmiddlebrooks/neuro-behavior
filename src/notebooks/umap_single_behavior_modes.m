%%                     Compare neuro-behavior in UMAP spaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
% cd '/Users/paulmiddlebrooks/Projects/toolboxes/umapFileExchange (4.4)/umap/'

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 1 * 60 * 60; % seconds
opts.frameSize = .1;

getDataType = 'all';
get_standard_data

%% Get the kinematics file to re-save for single behaviors
% bhvDataPath = 'E:/Projects/neuro-behavior/data/processed_behavior/';
animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';
if strcmp(sessionBhv, '112321_1')
    sessionSave = '112321';
end
    bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
    bhvFileName = '2021-11-23_13-19-58DLC_resnet50_bottomup_clearSep21shuffle1_700000.csv';

% Define the path to your CSV file
csvFilePath = [bhvDataPath, bhvFileName];

% Read the first 3 rows to get headers using readmatrix, assuming the CSV is not too large
% This step is primarily for obtaining the headers
headerData = readmatrix(csvFilePath, 'OutputType', 'string', 'Range', '1:3');

% Concatenate the strings from the 2nd and 3rd row for each column to form column names
concatenatedHeader = headerData(2,:) + "_" + headerData(3,:);

% Now, prepare to read the entire file with custom headers
% Initialize detectImportOptions again to specify the DataLines and VariableNames
opts = detectImportOptions(csvFilePath, 'NumHeaderLines', 3); % Skip the first 3 header lines
opts.VariableNames = matlab.lang.makeValidName(concatenatedHeader); % Make valid MATLAB variable names

% Assuming data starts from the 4th row, adjust VariableTypes if your data differs
for i = 1:length(opts.VariableTypes)
    opts = setvartype(opts, repmat({'double'}, 1, length(opts.VariableTypes))); % Assuming all data are type double
end

% Read the CSV data with the new settings
kinData = readtable(csvFilePath, opts);

% Display the first few rows of the table to check
disp(head(kinData));


%% Create spreasheets of the processed kinematics
bhvName = 'locomotion';
bhvCode = codes(strcmp(behaviors, bhvName));






