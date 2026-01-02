%%
% d2_per_binSize - Test how d2 changes as a function of binSize
%
% This script computes d2 for the first 60 seconds of a session
% across different binSize values, with and without circular shuffling.
%
% Variables:
%   sessionType - Session type ('reach', 'schall', 'hong', etc.)
%   sessionName - Session name
%   dataSource - 'spikes' or 'lfp' (default: 'spikes')
%
% Goal:
%   Compute d2 for binSize values from 0.005 to 0.1 seconds (step 0.005)
%   Plot raw d2, shuffled d2, and normalized d2 per binSize

% Add paths
basePath = fileparts(mfilename('fullpath'));  % criticality
srcPath = fullfile(basePath, '..');           % src

swDataPrepPath = fullfile(srcPath, 'sliding_window_prep', 'data_prep');
swUtilsPath = fullfile(srcPath, 'sliding_window_prep', 'utils');

if exist(swDataPrepPath, 'dir')
    addpath(swDataPrepPath);
end
if exist(swUtilsPath, 'dir')
    addpath(swUtilsPath);
end

% Configuration
if ~exist('sessionType', 'var') || ~exist('sessionName', 'var')
    error('sessionType and sessionName must be defined in workspace');
end

if ~exist('dataSource', 'var')
    dataSource = 'spikes';
end

% Analysis parameters
testDuration = 6;  % First 60 seconds
binSizes = 0.005:0.005:0.05;  % Bin sizes to test
pOrder = 10;
critType = 2;
nShuffles = 3;  % Number of circular shuffles for null model

% Load data
fprintf('\n=== Loading Data ===\n');
fprintf('Session: %s\n', sessionName);
fprintf('Session type: %s\n', sessionType);
fprintf('Data source: %s\n', dataSource);

opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = 5*60;  % Only load first 60 seconds
opts.minFiringRate = .05;
opts.maxFiringRate = 100;

dataStruct = load_sliding_window_data(sessionType, dataSource, ...
    'sessionName', sessionName, 'opts', opts);

areas = dataStruct.areas;
numAreas = length(areas);

if isfield(dataStruct, 'areasToTest')
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

fprintf('Number of areas: %d\n', numAreas);
fprintf('Testing %d bin sizes from %.3f to %.3f seconds\n', ...
    length(binSizes), min(binSizes), max(binSizes));
fprintf('\n');

% Calculate and print optimal bin/window sizes for each area
fprintf('=== Optimal Bin/Window Sizes (based on firing rate) ===\n');
minSpikesPerBin = 3;  % Same as used in run_criticality_ar.m
minBinsPerWindow = 1000;  % Same as used in run_criticality_ar.m

for a = areasToTest
    aID = dataStruct.idMatIdx{a};
    
    if isempty(aID)
        fprintf('Area %s: No neurons, skipping\n', areas{a});
        continue;
    end
    
    % Calculate firing rate for this area
    % Use the full dataMat (not just first 60 seconds) for accurate firing rate
    % But we'll use a representative bin size (0.02s) to calculate it
    tempBinSize = 0.02;
    tempDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), tempBinSize);
    totalSpikes = sum(tempDataMat(:));
    totalTime = size(dataStruct.dataMat, 1) / 1000;  % Convert ms to seconds
    firingRate = totalSpikes / totalTime;
    
    % Find optimal bin and window sizes
    [optimalBinSize, optimalWindowSize] = find_optimal_bin_and_window(...
        firingRate, minSpikesPerBin, minBinsPerWindow);
    
    fprintf('Area %s: firing rate = %.2f sp/s, optimal bin = %.3f s, optimal window = %.1f s\n', ...
        areas{a}, firingRate, optimalBinSize, optimalWindowSize);
end
fprintf('\n');

% Initialize results storage
% Results will be: [numAreas x numBinSizes]
d2Raw = nan(numAreas, length(binSizes));
d2Shuffled = nan(numAreas, length(binSizes));
d2Normalized = nan(numAreas, length(binSizes));

% Process each bin size
for b = 1:length(binSizes)
    binSize = binSizes(b);
    fprintf('Processing bin size %.3f s (%d/%d)...\n', binSize, b, length(binSizes));
    
    % Process each area
    for a = areasToTest
        aID = dataStruct.idMatIdx{a};
        
        if isempty(aID)
            continue;
        end
        
        % Bin the data for this area
        aDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), binSize);
        
        % Check if we have enough data points
        if size(aDataMat, 1) < 100
            fprintf('  Area %s: Insufficient data points (%d), skipping\n', ...
                areas{a}, size(aDataMat, 1));
            continue;
        end
        
        % Compute population activity (mean across neurons)
        popActivity = mean(aDataMat, 2);
        
        % Compute d2 for raw data
        try
            [varphi, ~] = myYuleWalker3(popActivity, pOrder);
            d2Raw(a, b) = getFixedPointDistance2(pOrder, critType, varphi);
        catch ME
            fprintf('  Area %s: d2 calculation failed: %s\n', areas{a}, ME.message);
            d2Raw(a, b) = nan;
        end
        
        % Compute shuffled d2 (circular permutation per neuron)
        d2ShuffValues = nan(1, nShuffles);
        for s = 1:nShuffles
            % Circularly shift each neuron's activity independently
            permutedDataMat = zeros(size(aDataMat));
            for n = 1:size(aDataMat, 2)
                shiftAmount = randi(size(aDataMat, 1));
                permutedDataMat(:, n) = circshift(aDataMat(:, n), shiftAmount);
            end
            
            % Compute population activity from shuffled data
            permutedPopActivity = mean(permutedDataMat, 2);
            
            % Compute d2 for shuffled data
            try
                [varphi, ~] = myYuleWalker3(permutedPopActivity, pOrder);
                d2ShuffValues(s) = getFixedPointDistance2(pOrder, critType, varphi);
            catch
                % Leave as NaN if calculation fails
            end
        end
        
        % Average shuffled d2
        d2Shuffled(a, b) = nanmean(d2ShuffValues);
        
        % Compute normalized d2
        if ~isnan(d2Raw(a, b)) && ~isnan(d2Shuffled(a, b)) && d2Shuffled(a, b) > 0
            d2Normalized(a, b) = d2Raw(a, b) / d2Shuffled(a, b);
        else
            d2Normalized(a, b) = nan;
        end
    end
end

fprintf('\n=== Analysis Complete ===\n');

% Create plots
fprintf('\n=== Creating Plots ===\n');

figure(1000); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [100, 100, 1200, 600]);

% Define colors for each area
areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};  % Red, Green, Blue, Magenta

% Subplot 1: Raw d2 and shuffled d2 (bar graph)
subplot(1, 2, 1);
hold on;

% Create grouped bar chart: each binSize gets bars for each area
% For each area, show raw and shuffled side by side
numValidBinSizes = length(binSizes);
xPos = 1:numValidBinSizes;

% Prepare data matrix for bar chart
% Format: [raw_area1, shuffled_area1, raw_area2, shuffled_area2, ...]
barData = [];
barLabels = {};

for aIdx = 1:length(areasToTest)
    a = areasToTest(aIdx);
    barData = [barData, d2Raw(a, :)', d2Shuffled(a, :)'];
    barLabels{end+1} = sprintf('%s (raw)', areas{a});
    barLabels{end+1} = sprintf('%s (shuffled)', areas{a});
end

% Create grouped bar chart
b = bar(xPos, barData, 'grouped');

% Color the bars by area (raw = solid, shuffled = semi-transparent)
for aIdx = 1:length(areasToTest)
    a = areasToTest(aIdx);
    areaColor = areaColors{min(a, length(areaColors))};
    
    % Raw bars (odd indices: 1, 3, 5, ...)
    barIdx = (aIdx - 1) * 2 + 1;
    if barIdx <= length(b)
        b(barIdx).FaceColor = areaColor;
        b(barIdx).EdgeColor = areaColor;
    end
    
    % Shuffled bars (even indices: 2, 4, 6, ...)
    barIdx = (aIdx - 1) * 2 + 2;
    if barIdx <= length(b)
        b(barIdx).FaceColor = areaColor;
        b(barIdx).EdgeColor = areaColor;
        b(barIdx).FaceAlpha = 0.5;
    end
end

xlabel('Bin Size (s)', 'FontSize', 12);
ylabel('d2', 'FontSize', 12);
title('Raw d2 and Shuffled d2 vs Bin Size', 'FontSize', 14);
set(gca, 'XTick', xPos, 'XTickLabel', arrayfun(@(x) sprintf('%.3f', x), binSizes, 'UniformOutput', false));
xtickangle(45);
legend(barLabels, 'Location', 'best', 'FontSize', 9);
grid on;
set(gca, 'FontSize', 11);

% Subplot 2: Normalized d2 (bar graph)
subplot(1, 2, 2);
hold on;

% Create grouped bar chart: each binSize gets bars for each area
barDataNorm = [];
barLabelsNorm = {};

for aIdx = 1:length(areasToTest)
    a = areasToTest(aIdx);
    barDataNorm = [barDataNorm, d2Normalized(a, :)'];
    barLabelsNorm{end+1} = areas{a};
end

% Create grouped bar chart
bNorm = bar(xPos, barDataNorm, 'grouped');

% Color the bars by area
for aIdx = 1:length(areasToTest)
    a = areasToTest(aIdx);
    areaColor = areaColors{min(a, length(areaColors))};
    
    if aIdx <= length(bNorm)
        bNorm(aIdx).FaceColor = areaColor;
        bNorm(aIdx).EdgeColor = areaColor;
    end
end

% Add reference line at 1.0
yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'DisplayName', 'Shuffled mean');

xlabel('Bin Size (s)', 'FontSize', 12);
ylabel('d2 (normalized)', 'FontSize', 12);
title('Normalized d2 vs Bin Size', 'FontSize', 14);
set(gca, 'XTick', xPos, 'XTickLabel', arrayfun(@(x) sprintf('%.3f', x), binSizes, 'UniformOutput', false));
xtickangle(45);
legend(barLabelsNorm, 'Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);

% Adjust y-axis to ensure 1.0 is visible
if any(~isnan(d2Normalized(:)))
    yRange = [min(d2Normalized(~isnan(d2Normalized))), max(d2Normalized(~isnan(d2Normalized)))];
    yRange(1) = min(1.0, yRange(1)) - 0.05 * (yRange(2) - yRange(1));
    yRange(2) = max(1.0, yRange(2)) + 0.05 * (yRange(2) - yRange(1));
    ylim(yRange);
end

% Super title
sgtitle(sprintf('d2 vs Bin Size - %s (%s, first %d s)', ...
    sessionName, sessionType, testDuration), 'FontSize', 16);

fprintf('Plots created.\n');
fprintf('\n');

