%%      Compare neural activity between mutliple modes of a single behavior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
% cd '/Users/paulmiddlebrooks/Projects/toolboxes/umapFileExchange (4.4)/umap/'

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 4 * 60 * 60; % seconds
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
% getWindow = (1 + opts.fsBhv * opts.collectStart : opts.fsBhv * (opts.collectStart + opts.collectFor));
% dataWindow = dataFull(getWindow,:);
% dataWindow.Time = dataWindow.Time - dataWindow.Time(1);
% bhvID = dataWindow.Code;

%%
% sessionBhv = '112321_2';
% bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
% bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];
%
%
% opts.dataPath = bhvDataPath;
% opts.fileName = bhvFileName;
% dataFull2 = readtable([opts.dataPath, opts.fileName]);
%


%% Get the kinematics file to re-save for single behaviors
% bhvDataPath = 'E:/Projects/neuro-behavior/data/processed_behavior/';
bhvDataPath = strcat(paths.bhvDataPath, 'animal_',animal,'/');
bhvFileName = '2021-11-23_13-19-58DLC_resnet50_bottomup_clearSep21shuffle1_700000.csv';

% Define the path to your CSV file
csvFile = [bhvDataPath, bhvFileName];

% Read the first 3 rows to get headers using readmatrix, assuming the CSV is not too large
% This step is primarily for obtaining the headers
headerData = readmatrix(csvFile, 'OutputType', 'string', 'Range', '1:3');

% Concatenate the strings from the 2nd and 3rd row for each column to form column names
concatenatedHeader = headerData(2,:) + "_" + headerData(3,:);

% Now, prepare to read the entire file with custom headers
% Initialize detectImportOptions again to specify the DataLines and VariableNames
optsImport = detectImportOptions(csvFile, 'NumHeaderLines', 3); % Skip the first 3 header lines
optsImport.VariableNames = matlab.lang.makeValidName(concatenatedHeader); % Make valid MATLAB variable names

% Assuming data starts from the 4th row, adjust VariableTypes if your data differs
for i = 1:length(optsImport.VariableTypes)
    optsImport = setvartype(optsImport, repmat({'double'}, 1, length(optsImport.VariableTypes))); % Assuming all data are type double
end

% Read the CSV data with the new settings
kinData = readtable(csvFile, optsImport);

% Display the first few rows of the table to check
disp(head(kinData));


%% Create spreasheets of the processed kinematics
bhvName = 'locomotion';
bhvCode = codes(strcmp(behaviors, bhvName));

% Create a new table with only kinematics during the behavior of interest
bhvInd = dataFull.Code == bhvCode;


% Take all the kinData rows of selected behavior
dataSave = kinData(bhvInd,:);


outputFileName = ['2021-11-23_13-19-58DLC_resnet50_bottomup_clearSep21shuffle1_700000_', bhvName, '.csv'];
outputFile = [bhvDataPath, outputFileName];

% Read the header data from the input CSV file
fileID = fopen(csvFile, 'r');
headerData = cell(3, 1);
for i = 1:3
    headerData{i} = fgetl(fileID);
end
fclose(fileID);

% Open the output file for writing
fileID = fopen(outputFile, 'w');

% Write the header data to the output file
for i = 1:3
    fprintf(fileID, '%s\n', headerData{i});
end

% Write the dataSave table to the output file
writetable(dataSave, outputFile, 'WriteMode', 'append', 'WriteVariableNames', false, 'FileType', 'text');

% Close the output file
fclose(fileID);


%% GO DO B-SOID TO GENERATE Labeled modes of the behavior, then come back here


%%
%
% Load newly labeled modes of the data
csvName = 'Aug-07-2024_mapping.csv';
% csvName = 'Aug-07-2024_assign_prob.csv';
csvName = 'BSOID/Aug-07-2024labels_pose_60Hz2021-11-23_13-19-58DLC_resnet50_bottomup_clearSep21shuffle1_700000_locomotion.csv';
workDir = 'C:/Users/yttri-lab/Desktop/B-SOID Project/SessionLocomotion/';
% workDir = 'C:/Users/yttri-lab/Desktop/B-SOID Project/Output/';
csvFile = [workDir, csvName];

% Use the 'readtable' function with 'HeaderLines' to skip the first 3 rows
csvOpts = detectImportOptions(csvFile);
csvOpts.DataLines = [4 Inf];  % Start reading from the 4th row until the end
% Specify the columns to read
csvOpts.SelectedVariableNames = csvOpts.VariableNames([2, 3]);  % Select only 2nd and 3rd columns

dataBhvCsv = readtable(csvFile, csvOpts);

% Rename the 3rd column to "Frame60Hz"
dataBhvCsv.Properties.VariableNames{2} = 'Time';
dataBhvCsv.Time = dataBhvCsv.Time / opts.fsBhv;
% bhvData.Frame = 1 + floor(bhvData.Time / opts.frameSize);

dataBhvCsv(dataBhvCsv.Time >= opts.collectFor, :) = [];

%% Plot bouts, each frame color-coded by b-soid label
boutFrame = [1; find(diff(dataBhvCsv.Time) > .017) + 1];
boutNumber = zeros(size(dataBhvCsv, 1), 1);
boutFrameEach = zeros(size(dataBhvCsv, 1), 1);
for i = 1 : length(boutFrame) - 1
    boutNumber(boutFrame(i): boutFrame(i+1)-1) = i;
    boutFrameEach(boutFrame(i): boutFrame(i+1)-1) = 1 : length(boutFrame(i): boutFrame(i+1)-1);
end
boutNumber(boutFrame(i+1) : end) = i + 1;
boutFrameEach(boutFrame(i+1) : end) = 1 : length(boutFrame(i+1):size(dataBhvCsv, 1));

colors = [0 0 1;...
    0 1 0;...
1 0 1;...
    0.8500 0.3250 0.0980];
colorsForPlot = arrayfun(@(x) colors(x,:), dataBhvCsv.B_SOiDLabels + 1, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix


figure(823); clf; hold on
scatter(boutFrameEach, boutNumber, 15, colorsForPlot, '.')
% Set the background color of the plot area (axes background)
ylabel('Bout Number')
xlabel('Frames (60Hz)')
title([bhvName, ' modes'])
ax = gca;  % Get the current axes
ax.Color = [0.5 0.5 0.5];  % Set the axes background color to gray


%% Create a bhvIDMatSingle of behavior labels and frames that matches the frame times of the neural matrix
changeBhv = [0; diff(dataBhvCsv.B_SOiDLabels)]; % nonzeros at all the indices when a new behavior begins
changeBhvIdx = find(changeBhv);

dataBhvSingle = table();
dataBhvSingle.ID = [dataBhvCsv.B_SOiDLabels(1); dataBhvCsv.B_SOiDLabels(changeBhvIdx)];
dataBhvSingle.StartTime = [dataBhvCsv.Time(1); dataBhvCsv.Time(changeBhvIdx)];
dataBhvSingle.StartFrame = 1 + floor(dataBhvSingle.StartTime / opts.frameSize);


    nFrame = ceil(opts.collectFor / opts.frameSize);
    % Use StartFrame and DurFrame method:
    bhvIDMatSingle = int8(zeros(nFrame, 1));
    for i = 1 : size(dataBhv, 1) - 1
        iInd = dataBhvSingle.StartFrame(i) : dataBhvSingle.StartFrame(i+1) - 1;
        bhvIDMatSingle(iInd) = dataBhv.ID(i);
    end
    bhvIDMatSingle(iInd(end) + 1 : end) = dataBhvSingle.ID(end);
disp('done')

%%

for i = 2 : length(switches)
    disp(bhvData(switches(i) - 9 : switches(i) + 10, :))
    pause
end












%% Analyses

minProp = .7; % Get bouts that have at least minProp proportion in one mode.

