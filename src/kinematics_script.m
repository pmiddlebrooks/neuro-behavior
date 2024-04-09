%%
opts = neuro_behavior_options;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 2 * 60 * 60; % seconds
opts.frameSize = 1/60;
opts.frameSize = .1;

getDataType = 'all';
get_standard_data
%%
bhvDataPath = 'E:/Projects/neuro-behavior/data/processed_behavior/';
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




%%
figure(43); clf; hold on;
% plot(sqrt(kinData.left_forepaw_x(1:500).^2 +kinData.left_forepaw_y(1:500).^2))
plot((kinData.left_forepaw_x(1:500) - kinData.snout_x(1:500)),  kinData.left_forepaw_y(1:500) - kinData.snout_y(1:500))
plot((kinData.right_forepaw_x(1:500) - kinData.snout_x(1:500)),  kinData.right_forepaw_y(1:500) - kinData.snout_y(1:500))







%%
idx = sort([2:3:size(kinData, 2), 3:3:size(kinData, 2)]);
kinPosition = table2array(kinData(:,idx));
kinVelocity = [];
for i = 1 : 2 : size(kinPosition, 2)
% Calculate differences in position (dx and dy) between consecutive samples
dx = diff(kinPosition(:,i)); % Differences in x
dy = diff(kinPosition(:,i+1)); % Differences in y
vel = sqrt(dx.^2 + dy.^2);

kinVelocity = [kinVelocity, [0; vel]];
end
%% Do PCA on the position and velocity data
[coeff, score, ~, ~, explained] = pca(zscore([kinPosition, kinVelocity], 1));
%% Plot the first few PCA components
nDim = 7;
window = 5 * 60 * 60 + (1 : 360);
figure(52); clf; hold on;
for d = 1 : nDim
plot(score(window, d));
end








%%
nosePos = [kinData.snout_x, kinData.snout_y];
tailPos = [kinData.tail_base_x, kinData.tail_base_y];
leftPaw = [kinData.left_forepaw_x, kinData.left_forepaw_y];
rightPaw = [kinData.right_forepaw_x, kinData.right_forepaw_y];

[leftPawY, leftPawX] = calculateSignedDistances(nosePos, tailPos, leftPaw);
[rightPawY, rightPawX] = calculateSignedDistances(nosePos, tailPos, rightPaw);


plotWindow = 1 : 300;
figure(43); clf; hold on;

yline(0, '--k')
plot(opts.frameSize * (0:length(plotWindow)-1), rightPawX(plotWindow), 'r')
plot(opts.frameSize * (0:length(plotWindow)-1), leftPawX(plotWindow), 'b')


function [distFromNose, distFromMidline] = calculateSignedDistances(nosePos, tailPos, thirdPoint)
    % Calculate the vector from the nose to each third point
    noseToThirdVec = thirdPoint - nosePos;
    
    % Calculate the unit vector for the line from the nose to the tail
    lineVec = tailPos - nosePos;
    unitLineVec = lineVec ./ sqrt(sum(lineVec.^2, 2));
    
    % Rotate the line vector by 90 degrees to get the perpendicular direction
    perpUnitLineVec = [-unitLineVec(:,2), unitLineVec(:,1)];
    
    % Calculate the signed distance from the nose along the line
    % This is the projection of the noseToThirdVec onto the line vector
    distFromNose = sum(noseToThirdVec .* unitLineVec, 2);
    
    % Calculate the signed perpendicular distance
    % This is the projection of the noseToThirdVec onto the perpendicular vector
    distFromMidline = sum(noseToThirdVec .* perpUnitLineVec, 2);
end