%%      Compare neural activity between mutliple modes of a single behavior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% Curate behavior labels
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);



%% Get the "original" (in 60Hz bins) behavior labels, to match the kinematics file
paths = get_paths;
    bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
    bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];


    opts.dataPath = bhvDataPath;
    opts.fileName = bhvFileName;
    dataFull = readtable([opts.dataPath, opts.fileName]);

        % Use a time window of recorded data
    getWindow = (1 + opts.fsBhv * opts.collectStart : opts.fsBhv * (opts.collectStart + opts.collectFor));
    dataWindow = dataFull(getWindow,:);
    dataWindow.Time = dataWindow.Time - dataWindow.Time(1);
    bhvID = dataWindow.Code;



%% Get the kinematics file to re-save for single behaviors
% bhvDataPath = 'E:/Projects/neuro-behavior/data/processed_behavior/';
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
optsImport = detectImportOptions(csvFilePath, 'NumHeaderLines', 3); % Skip the first 3 header lines
optsImport.VariableNames = matlab.lang.makeValidName(concatenatedHeader); % Make valid MATLAB variable names

% Assuming data starts from the 4th row, adjust VariableTypes if your data differs
for i = 1:length(optsImport.VariableTypes)
    optsImport = setvartype(optsImport, repmat({'double'}, 1, length(optsImport.VariableTypes))); % Assuming all data are type double
end

% Read the CSV data with the new settings
kinData = readtable(csvFilePath, optsImport);

% Display the first few rows of the table to check
disp(head(kinData));


%% Create spreasheets of the processed kinematics
bhvName = 'locomotion';
bhvCode = codes(strcmp(behaviors, bhvName));


bhvInd = bhvID == bhvCode;
dataSave = kinData(bhvInd,:);





