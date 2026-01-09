%%
% Criticality Reach vs Intertrial d2 Analysis
% Compares d2 criticality measures between reaches and intertrial intervals
% Performs sliding window d2 analysis around reach starts and intertrial midpoints
%
% Variables:
%   reachStart - Reach start times in seconds
%   intertrialMidpoints - Midpoint times between consecutive reaches
%   slidingWindowSize - Window size for d2 analysis (user-defined, same for all areas)
%   windowBuffer - Minimum distance from window edge to event/midpoint
%   beforeAlign - Start of sliding window range (seconds before alignment point)
%   afterAlign - End of sliding window range (seconds after alignment point)
%   stepSize - Step size for sliding window (seconds)
%   nShuffles - Number of circular permutations for d2 normalization
%   normalizeD2 - Set to true to normalize d2 by shuffled d2 values (default: true)
%
% This script uses load_sliding_window_data() to load data (as in choose_task_and_session.m)
% and finds optimal bin sizes per area (as in criticality_ar_analysis.m)
%
% Normalization: d2 values are normalized by circularly permuting each neuron's
% activity independently within each window, then computing d2 on the permuted
% population activity. The real d2 is divided by the mean of shuffled d2 values.

%% =============================    Data Loading    =============================
% Use load_sliding_window_data() to load data
% This requires sessionType, sessionName, and opts to be set in workspace
% (typically from choose_task_and_session.m)

fprintf('\n=== Loading Reach Data ===\n');

% Validate that required variables are set
if ~exist('sessionType', 'var')
    error('sessionType must be defined. Run choose_task_and_session.m first or set sessionType in workspace.');
end

if ~exist('sessionName', 'var')
    error('sessionName must be defined. Run choose_task_and_session.m first or set sessionName in workspace.');
end

    opts = neuro_behavior_options;
    opts.frameSize = .001;
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0;
    opts.collectEnd = [];
    opts.minFiringRate = .1;
    opts.maxFiringRate = 100;

% Load data using load_sliding_window_data
dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

% Extract reach start times from dataR if available
if ~isfield(dataStruct, 'dataR')
    error('dataR must be available in dataStruct for reach data. Check load_reach_data() implementation.');
end

dataR = dataStruct.dataR;
reachStart = dataR.R(:,1) / 1000; % Convert from ms to seconds
totalReaches = length(reachStart);

fprintf('Loaded %d reaches\n', totalReaches);

% Get areas and idMatIdx from dataStruct
areas = dataStruct.areas;
idMatIdx = dataStruct.idMatIdx;
numAreas = length(areas);

% Get areasToTest
if isfield(dataStruct, 'areasToTest')
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

% Define engagement segments for this session using reach_task_engagement
paths = get_paths;
reachDataFile = fullfile(paths.reachDataPath, [sessionName, '.mat']);
if ~exist(reachDataFile, 'file')
    % Try alternative path structure
    reachDataFile = fullfile(paths.reachDataPath, sessionName);
end

if exist(reachDataFile, 'file')
    segmentOpts = struct(); % use defaults unless overridden
    segmentWindowsEng = reach_task_engagement(reachDataFile, segmentOpts);
    segmentNames = {'Block1 Eng', 'Block1 Not', 'Block2 Eng', 'Block2 Not'};
    segmentWindowsList = {
        segmentWindowsEng.block1EngagedWindow;
        segmentWindowsEng.block1NotEngagedWindow;
        segmentWindowsEng.block2EngagedWindow;
        segmentWindowsEng.block2NotEngagedWindow;
    };
    nSegments = numel(segmentNames);
else
    warning('Reach data file not found for engagement analysis. Skipping segment analysis.');
    segmentNames = {};
    segmentWindowsList = {};
    nSegments = 0;
end

% Calculate intertrial midpoints (halfway between consecutive reaches)
intertrialMidpoints = nan(1, totalReaches - 1);
for i = 1:totalReaches - 1
    intertrialMidpoints(i) = (reachStart(i) + reachStart(i+1)) / 2;
end
fprintf('Calculated %d intertrial midpoints\n', length(intertrialMidpoints));

% =============================    Configuration    =============================
% Sliding window parameters
beforeAlign = -2;  % Start sliding from this many seconds before alignment point
afterAlign  =  2;  % End sliding at this many seconds after alignment point
slidingWindowSize = 5;  % Window size for d2 analysis (user-defined, same for all areas)
stepSize    = .25; % Step size for sliding window (seconds)
windowBuffer = .5; % Minimum distance from window edge to event/midpoint (seconds)
minWindowSize = slidingWindowSize;  % Minimum window size required (seconds)

% d2 analysis parameters
pOrder = 10;        % AR model order for d2 calculation
critType = 2;       % Criticality type for d2 calculation
minSpikesPerBin = 3;  % Minimum spikes per bin for optimal bin size calculation
minBinsPerWindow = 1000;  % Minimum bins per window (used for optimal bin size, but window size is user-defined)
nShuffles = 3;      % Number of circular permutations for d2 normalization
normalizeD2 = true;  % Set to true to normalize d2 by shuffled d2 values

% PCA options
pcaFlag = 0;           % Set to 1 to use PCA
pcaFirstFlag = 1;      % Use first nDim dimensions if 1, last nDim if 0
nDim = 4;              % Number of PCA dimensions to use

% Plotting options
loadResultsForPlotting = false;  % Set to true to load saved results for plotting
                                  % Set to false to use variables from workspace
resultsFileForPlotting = '';     % Path to results file (empty = auto-detect from saveDir)
makePlots = true;                 % Set to true to generate plots

% Get saveDir from dataStruct
saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% =============================    Find Optimal Bin Sizes Per Area    =============================
fprintf('\n=== Finding Optimal Bin Sizes Per Area ===\n');

% Add path to criticality functions
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

% Apply PCA to original data if requested
fprintf('\n--- PCA on original data if requested ---\n');
reconstructedDataMat = cell(1, numAreas);
for a = areasToTest
    aID = dataStruct.idMatIdx{a};
    thisDataMat = dataStruct.dataMat(:, aID);
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, min(6, forDim));
        nDim = 1:forDim;
        reconstructedDataMat{a} = score(:,nDim) * coeff(:,nDim)' + mu;
    else
        reconstructedDataMat{a} = thisDataMat;
    end
end

% Find optimal bin size per area
binSize = zeros(1, numAreas);
for a = areasToTest
    thisDataMat = reconstructedDataMat{a};
    thisFiringRate = sum(thisDataMat(:)) / (size(thisDataMat, 1)/1000);
    [binSize(a), ~] = find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: bin size = %.3f s, firing rate = %.2f spikes/s\n', ...
        areas{a}, binSize(a), thisFiringRate);
end

% =============================    Find Valid Reaches and Intertrial Midpoints    =============================
fprintf('\n=== Finding Valid Reaches and Intertrial Midpoints ===\n');

% Find maximum possible window duration that satisfies buffer constraints
% For reach-centered windows: edges must not overlap with windowBuffer sec after 
%   previous intertrial midpoint or windowBuffer sec before next intertrial midpoint
% For intertrial-centered windows: edges must not overlap with windowBuffer sec after 
%   previous reach or windowBuffer sec before next reach

% Calculate maximum window size for each reach and filter out those below minimum
validReachIndices = [];
maxWindowPerReach = nan(1, totalReaches);

for r = 1:totalReaches
    reachTime = reachStart(r);
    
    % Find previous and next intertrial midpoints
    maxWindowForThisReach = inf;
    
    if r > 1
        prevMidpoint = intertrialMidpoints(r-1);
        % Constraint: reachTime - slidingWindowSize/2 >= prevMidpoint + windowBuffer
        % Rearranging: slidingWindowSize <= 2 * (reachTime - prevMidpoint - windowBuffer)
        maxWindowFromPrev = 2 * (reachTime - prevMidpoint - windowBuffer);
        if maxWindowFromPrev < maxWindowForThisReach
            maxWindowForThisReach = maxWindowFromPrev;
        end
    end
    
    if r <= length(intertrialMidpoints)
        nextMidpoint = intertrialMidpoints(r);
        % Constraint: reachTime + slidingWindowSize/2 <= nextMidpoint - windowBuffer
        % Rearranging: slidingWindowSize <= 2 * (nextMidpoint - reachTime - windowBuffer)
        maxWindowFromNext = 2 * (nextMidpoint - reachTime - windowBuffer);
        if maxWindowFromNext < maxWindowForThisReach
            maxWindowForThisReach = maxWindowFromNext;
        end
    end
    
    maxWindowPerReach(r) = maxWindowForThisReach;
    
    % Keep this reach if its maximum window is >= minWindowSize
    if maxWindowForThisReach >= minWindowSize
        validReachIndices = [validReachIndices, r];
    end
end

% Calculate maximum window size for each intertrial midpoint and filter out those below minimum
validIntertrialIndices = [];
maxWindowPerIntertrial = nan(1, length(intertrialMidpoints));

for i = 1:length(intertrialMidpoints)
    midpointTime = intertrialMidpoints(i);
    
    % Find previous and next reaches
    % Intertrial midpoint i is between reach i and reach i+1
    prevReach = reachStart(i);
    nextReach = reachStart(i+1);
    
    maxWindowForThisIntertrial = inf;
    
    % Constraint: midpointTime - slidingWindowSize/2 >= prevReach + windowBuffer
    % Rearranging: slidingWindowSize <= 2 * (midpointTime - prevReach - windowBuffer)
    maxWindowFromPrev = 2 * (midpointTime - prevReach - windowBuffer);
    if maxWindowFromPrev < maxWindowForThisIntertrial
        maxWindowForThisIntertrial = maxWindowFromPrev;
    end
    
    % Constraint: midpointTime + slidingWindowSize/2 <= nextReach - windowBuffer
    % Rearranging: slidingWindowSize <= 2 * (nextReach - midpointTime - windowBuffer)
    maxWindowFromNext = 2 * (nextReach - midpointTime - windowBuffer);
    if maxWindowFromNext < maxWindowForThisIntertrial
        maxWindowForThisIntertrial = maxWindowFromNext;
    end
    
    maxWindowPerIntertrial(i) = maxWindowForThisIntertrial;
    
    % Keep this intertrial midpoint if its maximum window is >= minWindowSize
    if maxWindowForThisIntertrial >= minWindowSize
        validIntertrialIndices = [validIntertrialIndices, i];
    end
end

% Check if user-defined slidingWindowSize is valid
if isempty(validReachIndices) && isempty(validIntertrialIndices)
    error('No valid reaches or intertrial midpoints found with window size >= %.1f seconds. Try reducing windowBuffer or minWindowSize.', minWindowSize);
end

% Find maximum window from valid events
maxReachWindow = inf;
if ~isempty(validReachIndices)
    maxReachWindow = min(maxWindowPerReach(validReachIndices));
end

maxIntertrialWindow = inf;
if ~isempty(validIntertrialIndices)
    maxIntertrialWindow = min(maxWindowPerIntertrial(validIntertrialIndices));
end

% Use the smaller of the two to ensure both reach and intertrial windows work
maxAllowedWindow = min(maxReachWindow, maxIntertrialWindow);

% Also need to consider the sliding range
% maxWindowFromSliding = abs(beforeAlign) + abs(afterAlign);
% maxAllowedWindow = min(maxAllowedWindow, maxWindowFromSliding);

% Check if user-defined slidingWindowSize is valid
if slidingWindowSize > maxAllowedWindow
    warning('User-defined slidingWindowSize (%.2f s) exceeds maximum allowed (%.2f s). Using maximum allowed.', ...
        slidingWindowSize, maxAllowedWindow);
    slidingWindowSize = maxAllowedWindow;
end

% Ensure slidingWindowSize is at least minWindowSize
if slidingWindowSize < minWindowSize
    slidingWindowSize = minWindowSize;
end

% Ensure slidingWindowSize is positive and reasonable
if slidingWindowSize <= 0 || isnan(slidingWindowSize) || isinf(slidingWindowSize)
    error('Invalid slidingWindowSize. Try reducing windowBuffer or increasing inter-reach intervals.');
end

% Keep original arrays for reference (needed for finding adjacent events)
reachStartOriginal = reachStart;
intertrialMidpointsOriginal = intertrialMidpoints;

% Update reachStart to only include valid reaches
reachStart = reachStart(validReachIndices);
totalReaches = length(reachStart);

% Recalculate intertrial midpoints from filtered reachStart
% (since midpoints are between consecutive valid reaches)
% Only include midpoints that also passed the window size test
% Build as properly indexed array where index i corresponds to gap between reach i and i+1
intertrialMidpointsFiltered = nan(1, totalReaches - 1);
intertrialMidpointValid = false(1, totalReaches - 1);

for i = 1:totalReaches - 1
    % This midpoint is between reachStart(i) and reachStart(i+1) in filtered array
    midpointTime = (reachStart(i) + reachStart(i+1)) / 2;
    
    % Find the original reach indices that correspond to these filtered reaches
    origReachIdx1 = validReachIndices(i);
    origReachIdx2 = validReachIndices(i+1);
    
    % Only include this midpoint if the two original reaches are consecutive
    % (i.e., there was an original midpoint between them)
    % AND that original midpoint was in validIntertrialIndices
    if origReachIdx2 == origReachIdx1 + 1
        % Original midpoint index is origReachIdx1 (since midpoint i is between reach i and i+1)
        if ismember(origReachIdx1, validIntertrialIndices)
            intertrialMidpointsFiltered(i) = midpointTime;
            intertrialMidpointValid(i) = true;
        end
    end
end

% Store valid intertrial midpoints and their indices for processing
intertrialMidpoints = intertrialMidpointsFiltered(intertrialMidpointValid);
validIntertrialIndicesFiltered = find(intertrialMidpointValid);

fprintf('Using sliding window size: %.2f seconds\n', slidingWindowSize);
fprintf('  Minimum window size: %.2f seconds\n', minWindowSize);
fprintf('  Valid reaches: %d/%d (%.1f%%)\n', length(validReachIndices), length(maxWindowPerReach), 100*length(validReachIndices)/length(maxWindowPerReach));
fprintf('  Valid intertrial midpoints: %d/%d (%.1f%%)\n', length(intertrialMidpoints), length(maxWindowPerIntertrial), 100*length(intertrialMidpoints)/length(maxWindowPerIntertrial));
fprintf('  Window buffer: %.2f seconds\n', windowBuffer);

%% =============================    Analysis    =============================
fprintf('\n=== Processing Areas ===\n');

% Calculate sliding window positions
slidingPositions = beforeAlign:stepSize:afterAlign;
numSlidingPositions = length(slidingPositions);

% Initialize storage for d2 metrics (per sliding position, concatenated windows)
% Structure: {area}{eventType}[slidingPosition]
% eventType: 'reach' or 'intertrial'
d2Metrics = struct();
d2Metrics.reach = cell(1, numAreas);
d2Metrics.intertrial = cell(1, numAreas);
d2Metrics.reachNormalized = cell(1, numAreas);
d2Metrics.intertrialNormalized = cell(1, numAreas);

for a = areasToTest
    d2Metrics.reach{a} = nan(1, numSlidingPositions);
    d2Metrics.intertrial{a} = nan(1, numSlidingPositions);
    d2Metrics.reachNormalized{a} = nan(1, numSlidingPositions);
    d2Metrics.intertrialNormalized{a} = nan(1, numSlidingPositions);
end

% Process each area
for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    tic;
    
    aID = dataStruct.idMatIdx{a};
    
    % Bin the data using area-specific optimal bin size
    aDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), binSize(a));
    numTimePoints = size(aDataMat, 1);
    timePoints = (0:numTimePoints-1) * binSize(a); % Time axis in seconds
    
    % Apply PCA to binned data if requested
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, min(6, forDim));
        nDim = 1:forDim;
        aDataMat = score(:,nDim) * coeff(:,nDim)' + mu;
    end
    
    % Store binned data matrix for permutation analysis (needed for per-window permutations)
    % aDataMat is [time bins x neurons] - we'll need this for circular permutations
    % Note: We'll compute population activity per window, not globally
    
    % Initialize storage for collected windows and their center times
    % For each sliding position, collect all valid windows
    % Store as cell array indexed by area and position: collectedReachWindows{area, position}
    % Also store the window data matrices for permutation analysis and window center times
    if a == areasToTest(1)
        % Initialize on first area
        collectedReachWindows = cell(numAreas, numSlidingPositions);
        collectedIntertrialWindows = cell(numAreas, numSlidingPositions);
        collectedReachWindowData = cell(numAreas, numSlidingPositions);  % Store [time x neurons] matrices
        collectedIntertrialWindowData = cell(numAreas, numSlidingPositions);  % Store [time x neurons] matrices
        collectedReachWindowCenters = cell(numAreas, numSlidingPositions);  % Store window center times
        collectedIntertrialWindowCenters = cell(numAreas, numSlidingPositions);  % Store window center times
        collectedReachWindowCenterBins = cell(numAreas, numSlidingPositions);  % Store center bin values from each window
        collectedIntertrialWindowCenterBins = cell(numAreas, numSlidingPositions);  % Store center bin values from each window
        collectedReachWindowCenterBinsNormalized = cell(numAreas, numSlidingPositions);  % Store normalized center bin values
        collectedIntertrialWindowCenterBinsNormalized = cell(numAreas, numSlidingPositions);  % Store normalized center bin values
    end
    
    % Step 1: Collect all reach-aligned windows
    fprintf('  Collecting reach-aligned windows...\n');
    for posIdx = 1:numSlidingPositions
        offset = slidingPositions(posIdx);
        
        for r = 1:totalReaches
            reachTime = reachStart(r);
            
            % Find original reach index
            origReachIdx = validReachIndices(r);
            
            % Find previous and next intertrial midpoints using original arrays
            prevMidpoint = [];
            nextMidpoint = [];
            if origReachIdx > 1
                % Previous midpoint is between original reach (origReachIdx-1) and origReachIdx
                prevOrigMidpointIdx = origReachIdx - 1;
                if prevOrigMidpointIdx <= length(intertrialMidpointsOriginal)
                    prevMidpoint = intertrialMidpointsOriginal(prevOrigMidpointIdx);
                end
            end
            if origReachIdx <= length(intertrialMidpointsOriginal)
                % Next midpoint is between original reach origReachIdx and (origReachIdx+1)
                nextOrigMidpointIdx = origReachIdx;
                if nextOrigMidpointIdx <= length(intertrialMidpointsOriginal)
                    nextMidpoint = intertrialMidpointsOriginal(nextOrigMidpointIdx);
                end
            end
            
            windowCenter = reachTime + offset;
            
            % Calculate window bounds
            winStart = windowCenter - slidingWindowSize / 2;
            winEnd = windowCenter + slidingWindowSize / 2;
            
            % Check buffer constraints
            constraintViolated = false;
            if ~isempty(prevMidpoint)
                if winStart < prevMidpoint + windowBuffer
                    constraintViolated = true;
                end
            end
            if ~isempty(nextMidpoint)
                if winEnd > nextMidpoint - windowBuffer
                    constraintViolated = true;
                end
            end
            
            if constraintViolated
                continue; % Skip this window if buffer constraint violated
            end
            
            % Convert window center to indices for this area's binning
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                windowCenter, slidingWindowSize, binSize(a), numTimePoints);
            
            % Check if window is valid (within bounds)
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                % Extract window data [time bins x neurons] for permutation analysis
                wDataMat = aDataMat(startIdx:endIdx, :);
                wPopActivity = mean(wDataMat, 2);  % Population activity for this window
                
                % Extract center bin(s) from this window
                numBins = length(wPopActivity);
                centerBinIdx = ceil(numBins / 2);  % Middle bin index
                centerBinValue = wPopActivity(centerBinIdx);  % Single center bin value
                
                % Compute normalized center bin if requested
                centerBinValueNormalized = centerBinValue;
                if normalizeD2
                    numNeurons = size(wDataMat, 2);
                    numTimeBins = size(wDataMat, 1);
                    shuffledCenterBins = nan(1, nShuffles);
                    
                    for s = 1:nShuffles
                        % Circularly shift each neuron's activity independently
                        permutedDataMat = zeros(size(wDataMat));
                        for n = 1:numNeurons
                            shiftAmount = randi(numTimeBins);
                            permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                        end
                        
                        % Compute population activity from shuffled data
                        permutedPopActivity = mean(permutedDataMat, 2);
                        
                        % Extract center bin from shuffled data
                        shuffledCenterBins(s) = permutedPopActivity(centerBinIdx);
                    end
                    
                    % Normalize center bin by mean of shuffled center bins
                    meanShuffledCenterBin = nanmean(shuffledCenterBins);
                    if ~isnan(meanShuffledCenterBin) && meanShuffledCenterBin ~= 0
                        centerBinValueNormalized = centerBinValue / meanShuffledCenterBin;
                    else
                        centerBinValueNormalized = nan;
                    end
                end
                
                % Store this window's population activity (as a row vector)
                % Store as: {area, position}(reachIndex, :)
                % Also store window center time and center bin value (raw and normalized)
                if isempty(collectedReachWindows{a, posIdx})
                    collectedReachWindows{a, posIdx} = wPopActivity(:)';
                    collectedReachWindowData{a, posIdx} = {wDataMat};
                    collectedReachWindowCenters{a, posIdx} = windowCenter;
                    collectedReachWindowCenterBins{a, posIdx} = centerBinValue;
                    collectedReachWindowCenterBinsNormalized{a, posIdx} = centerBinValueNormalized;
                else
                    collectedReachWindows{a, posIdx} = [collectedReachWindows{a, posIdx}; wPopActivity(:)'];
                    collectedReachWindowData{a, posIdx}{end+1} = wDataMat;
                    collectedReachWindowCenters{a, posIdx} = [collectedReachWindowCenters{a, posIdx}, windowCenter];
                    collectedReachWindowCenterBins{a, posIdx} = [collectedReachWindowCenterBins{a, posIdx}, centerBinValue];
                    collectedReachWindowCenterBinsNormalized{a, posIdx} = [collectedReachWindowCenterBinsNormalized{a, posIdx}, centerBinValueNormalized];
                end
            end
        end
    end
    
    % Step 2: Collect all intertrial-aligned windows
    fprintf('  Collecting intertrial-aligned windows...\n');
    for posIdx = 1:numSlidingPositions
        offset = slidingPositions(posIdx);
        
        for idx = 1:length(intertrialMidpoints)
            midpointTime = intertrialMidpoints(idx);
            
            % Find the index in the filtered array that this midpoint corresponds to
            % validIntertrialIndicesFiltered(idx) gives the index in the filtered reach array
            i = validIntertrialIndicesFiltered(idx);
            
            % Find previous and next reaches for this intertrial midpoint
            % Filtered midpoint at index i is between filtered reach i and i+1
            prevReach = reachStart(i);
            nextReach = reachStart(i+1);
            
            windowCenter = midpointTime + offset;
            
            % Calculate window bounds
            winStart = windowCenter - slidingWindowSize / 2;
            winEnd = windowCenter + slidingWindowSize / 2;
            
            % Check buffer constraints
            constraintViolated = false;
            if winStart < prevReach + windowBuffer
                constraintViolated = true;
            end
            if winEnd > nextReach - windowBuffer
                constraintViolated = true;
            end
            
            if constraintViolated
                continue; % Skip this window if buffer constraint violated
            end
            
            % Convert window center to indices for this area's binning
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                windowCenter, slidingWindowSize, binSize(a), numTimePoints);
            
            % Check if window is valid (within bounds)
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                % Extract window data [time bins x neurons] for permutation analysis
                wDataMat = aDataMat(startIdx:endIdx, :);
                wPopActivity = mean(wDataMat, 2);  % Population activity for this window
                
                % Extract center bin(s) from this window
                numBins = length(wPopActivity);
                centerBinIdx = ceil(numBins / 2);  % Middle bin index
                centerBinValue = wPopActivity(centerBinIdx);  % Single center bin value
                
                % Compute normalized center bin if requested
                centerBinValueNormalized = centerBinValue;
                if normalizeD2
                    numNeurons = size(wDataMat, 2);
                    numTimeBins = size(wDataMat, 1);
                    shuffledCenterBins = nan(1, nShuffles);
                    
                    for s = 1:nShuffles
                        % Circularly shift each neuron's activity independently
                        permutedDataMat = zeros(size(wDataMat));
                        for n = 1:numNeurons
                            shiftAmount = randi(numTimeBins);
                            permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                        end
                        
                        % Compute population activity from shuffled data
                        permutedPopActivity = mean(permutedDataMat, 2);
                        
                        % Extract center bin from shuffled data
                        shuffledCenterBins(s) = permutedPopActivity(centerBinIdx);
                    end
                    
                    % Normalize center bin by mean of shuffled center bins
                    meanShuffledCenterBin = nanmean(shuffledCenterBins);
                    if ~isnan(meanShuffledCenterBin) && meanShuffledCenterBin ~= 0
                        centerBinValueNormalized = centerBinValue / meanShuffledCenterBin;
                    else
                        centerBinValueNormalized = nan;
                    end
                end
                
                % Store this window's population activity (as a row vector)
                % Store as: {area, position}(intertrialIndex, :)
                % Also store window center time and center bin value (raw and normalized)
                if isempty(collectedIntertrialWindows{a, posIdx})
                    collectedIntertrialWindows{a, posIdx} = wPopActivity(:)';
                    collectedIntertrialWindowData{a, posIdx} = {wDataMat};
                    collectedIntertrialWindowCenters{a, posIdx} = windowCenter;
                    collectedIntertrialWindowCenterBins{a, posIdx} = centerBinValue;
                    collectedIntertrialWindowCenterBinsNormalized{a, posIdx} = centerBinValueNormalized;
                else
                    collectedIntertrialWindows{a, posIdx} = [collectedIntertrialWindows{a, posIdx}; wPopActivity(:)'];
                    collectedIntertrialWindowData{a, posIdx}{end+1} = wDataMat;
                    collectedIntertrialWindowCenters{a, posIdx} = [collectedIntertrialWindowCenters{a, posIdx}, windowCenter];
                    collectedIntertrialWindowCenterBins{a, posIdx} = [collectedIntertrialWindowCenterBins{a, posIdx}, centerBinValue];
                    collectedIntertrialWindowCenterBinsNormalized{a, posIdx} = [collectedIntertrialWindowCenterBinsNormalized{a, posIdx}, centerBinValueNormalized];
                end
            end
        end
    end
    
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% =============================    Per-Sliding-Position d2 Analysis    =============================
fprintf('\n=== Per-Sliding-Position d2 Analysis (Reach vs Intertrial) ===\n');

% Initialize storage for per-window d2 values (for use in segment analysis)
% Structure: {area}{eventType}{slidingPosition} = [d2 values] and corresponding center times
perWindowD2 = struct();
perWindowD2.reach = cell(1, numAreas);
perWindowD2.intertrial = cell(1, numAreas);
perWindowD2.reachNormalized = cell(1, numAreas);
perWindowD2.intertrialNormalized = cell(1, numAreas);
perWindowD2.reachCenters = cell(1, numAreas);
perWindowD2.intertrialCenters = cell(1, numAreas);

for a = areasToTest
    perWindowD2.reach{a} = cell(1, numSlidingPositions);
    perWindowD2.intertrial{a} = cell(1, numSlidingPositions);
    perWindowD2.reachNormalized{a} = cell(1, numSlidingPositions);
    perWindowD2.intertrialNormalized{a} = cell(1, numSlidingPositions);
    perWindowD2.reachCenters{a} = cell(1, numSlidingPositions);
    perWindowD2.intertrialCenters{a} = cell(1, numSlidingPositions);
end

% Compute d2 metrics for each sliding position
for a = areasToTest
    fprintf('\nComputing d2 metrics per sliding position for area %s...\n', areas{a});
    
    for posIdx = 1:numSlidingPositions
        % Process reach windows - compute d2 for each window individually
        if ~isempty(collectedReachWindows{a, posIdx})
            windowDataList = collectedReachWindowData{a, posIdx};
            windowCenters = collectedReachWindowCenters{a, posIdx};
            numWindows = numel(windowDataList);
            
            d2PerWindow = nan(1, numWindows);
            d2ShuffledPerWindow = nan(numWindows, nShuffles);
            
            % Compute d2 for each window
            for w = 1:numWindows
                wDataMat = windowDataList{w};  % [time bins x neurons]
                wPopActivity = mean(wDataMat, 2);  % Population activity for this window
                
                % Compute d2 for this window
                if ~isempty(wPopActivity)
                    try
                        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
                        d2PerWindow(w) = getFixedPointDistance2(pOrder, critType, varphi);
                    catch
                        d2PerWindow(w) = nan;
                    end
                end
                
                % Compute shuffled d2 for normalization if requested
                if normalizeD2
                    for s = 1:nShuffles
                        numNeurons = size(wDataMat, 2);
                        numTimeBins = size(wDataMat, 1);
                        
                        % Circularly shift each neuron's activity independently
                        permutedDataMat = zeros(size(wDataMat));
                        for n = 1:numNeurons
                            shiftAmount = randi(numTimeBins);
                            permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                        end
                        
                        % Compute population activity from shuffled data
                        permutedPopActivity = mean(permutedDataMat, 2);
                        
                        % Compute d2 on permuted data
                        if ~isempty(permutedPopActivity)
                            try
                                [varphiPerm, ~] = myYuleWalker3(permutedPopActivity, pOrder);
                                d2ShuffledPerWindow(w, s) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                            catch
                                d2ShuffledPerWindow(w, s) = nan;
                            end
                        end
                    end
                end
            end
            
            % Average d2 values across windows at this sliding position
            d2Metrics.reach{a}(posIdx) = nanmean(d2PerWindow);
            
            % Store per-window d2 values and centers for segment analysis
            perWindowD2.reach{a}{posIdx} = d2PerWindow;
            perWindowD2.reachCenters{a}{posIdx} = windowCenters;
            
            % Normalize if requested
            if normalizeD2
                % Normalize each window individually, then average
                meanShuffledPerWindow = nanmean(d2ShuffledPerWindow, 2);  % Mean shuffled d2 per window
                d2NormalizedPerWindow = nan(1, numWindows);
                for w = 1:numWindows
                    if ~isnan(d2PerWindow(w)) && ~isnan(meanShuffledPerWindow(w)) && meanShuffledPerWindow(w) > 0
                        d2NormalizedPerWindow(w) = d2PerWindow(w) / meanShuffledPerWindow(w);
                    end
                end
                
                % Average normalized d2 values across windows
                d2Metrics.reachNormalized{a}(posIdx) = nanmean(d2NormalizedPerWindow);
                
                % Store normalized per-window values for segment analysis
                perWindowD2.reachNormalized{a}{posIdx} = d2NormalizedPerWindow;
            else
                d2Metrics.reachNormalized{a}(posIdx) = nan;
            end
        end
        
        % Process intertrial windows - compute d2 for each window individually
        if ~isempty(collectedIntertrialWindows{a, posIdx})
            windowDataList = collectedIntertrialWindowData{a, posIdx};
            windowCenters = collectedIntertrialWindowCenters{a, posIdx};
            numWindows = numel(windowDataList);
            
            d2PerWindow = nan(1, numWindows);
            d2ShuffledPerWindow = nan(numWindows, nShuffles);
            
            % Compute d2 for each window
            for w = 1:numWindows
                wDataMat = windowDataList{w};  % [time bins x neurons]
                wPopActivity = mean(wDataMat, 2);  % Population activity for this window
                
                % Compute d2 for this window
                if ~isempty(wPopActivity)
                    try
                        [varphi, ~] = myYuleWalker3(wPopActivity, pOrder);
                        d2PerWindow(w) = getFixedPointDistance2(pOrder, critType, varphi);
                    catch
                        d2PerWindow(w) = nan;
                    end
                end
                
                % Compute shuffled d2 for normalization if requested
                if normalizeD2
                    for s = 1:nShuffles
                        numNeurons = size(wDataMat, 2);
                        numTimeBins = size(wDataMat, 1);
                        
                        % Circularly shift each neuron's activity independently
                        permutedDataMat = zeros(size(wDataMat));
                        for n = 1:numNeurons
                            shiftAmount = randi(numTimeBins);
                            permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                        end
                        
                        % Compute population activity from shuffled data
                        permutedPopActivity = mean(permutedDataMat, 2);
                        
                        % Compute d2 on permuted data
                        if ~isempty(permutedPopActivity)
                            try
                                [varphiPerm, ~] = myYuleWalker3(permutedPopActivity, pOrder);
                                d2ShuffledPerWindow(w, s) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                            catch
                                d2ShuffledPerWindow(w, s) = nan;
                            end
                        end
                    end
                end
            end
            
            % Average d2 values across windows at this sliding position
            d2Metrics.intertrial{a}(posIdx) = nanmean(d2PerWindow);
            
            % Store per-window d2 values and centers for segment analysis
            perWindowD2.intertrial{a}{posIdx} = d2PerWindow;
            perWindowD2.intertrialCenters{a}{posIdx} = windowCenters;
            
            % Normalize if requested
            if normalizeD2
                % Normalize each window individually, then average
                meanShuffledPerWindow = nanmean(d2ShuffledPerWindow, 2);  % Mean shuffled d2 per window
                d2NormalizedPerWindow = nan(1, numWindows);
                for w = 1:numWindows
                    if ~isnan(d2PerWindow(w)) && ~isnan(meanShuffledPerWindow(w)) && meanShuffledPerWindow(w) > 0
                        d2NormalizedPerWindow(w) = d2PerWindow(w) / meanShuffledPerWindow(w);
                    end
                end
                
                % Average normalized d2 values across windows
                d2Metrics.intertrialNormalized{a}(posIdx) = nanmean(d2NormalizedPerWindow);
                
                % Store normalized per-window values for segment analysis
                perWindowD2.intertrialNormalized{a}{posIdx} = d2NormalizedPerWindow;
            else
                d2Metrics.intertrialNormalized{a}(posIdx) = nan;
            end
        end
    end
end

% =============================   Engagement Segment-level d2 Analysis    =============================
fprintf('\n=== Segment-level d2 Analysis (Reach vs Intertrial) ===\n');

segmentMetrics = struct();
segmentMetrics.reach = cell(numAreas, nSegments);
segmentMetrics.intertrial = cell(numAreas, nSegments);
segmentMetrics.reachNormalized = cell(numAreas, nSegments);
segmentMetrics.intertrialNormalized = cell(numAreas, nSegments);

for a = areasToTest
    fprintf('\nProcessing segments for area %s...\n', areas{a});
    for s = 1:nSegments
        if isempty(segmentWindowsList) || s > length(segmentWindowsList) || isempty(segmentWindowsList{s}) || any(isnan(segmentWindowsList{s}))
            % Initialize to nan if segment is invalid
            segmentMetrics.reach{a, s} = nan;
            segmentMetrics.intertrial{a, s} = nan;
            segmentMetrics.reachNormalized{a, s} = nan;
            segmentMetrics.intertrialNormalized{a, s} = nan;
            continue;
        end
        segWin = segmentWindowsList{s};
        tStart = segWin(1);
        tEnd   = segWin(2);

        % Find the sliding position index that corresponds to the alignment point (0)
        alignPosIdx = find(abs(slidingPositions) < stepSize/2, 1);  % Find position closest to 0
        if isempty(alignPosIdx)
            % If no exact match, use the first position (shouldn't happen if 0 is in range)
            alignPosIdx = 1;
        end
        
        % Collect normalized d2 values from windows centered at alignment point (sliding position = 0)
        % Only include windows whose event times (reach or intertrial midpoint) fall in this segment
        reachD2NormalizedInSegment = [];
        intertrialD2NormalizedInSegment = [];
        
        % Process reach windows - use normalized d2 values from windows at alignment point
        if ~isempty(perWindowD2.reachNormalized{a}{alignPosIdx}) && ~isempty(perWindowD2.reachCenters{a}{alignPosIdx})
            d2Vals = perWindowD2.reachNormalized{a}{alignPosIdx};
            windowCenters = perWindowD2.reachCenters{a}{alignPosIdx};
            
            % Ensure windowCenters and d2Vals are row vectors for consistent indexing
            if isscalar(windowCenters)
                windowCenters = windowCenters(:)';
            end
            windowCenters = windowCenters(:)';  % Force row vector
            if isscalar(d2Vals)
                d2Vals = d2Vals(:)';
            end
            d2Vals = d2Vals(:)';  % Force row vector
            
            % Window centers at alignment point are the reach times
            % Find reaches whose times fall in this segment
            inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
            if any(inSegment)
                reachD2NormalizedInSegment = [reachD2NormalizedInSegment, d2Vals(inSegment)];
            end
        end
        
        % Process intertrial windows - use normalized d2 values from windows at alignment point
        if ~isempty(perWindowD2.intertrialNormalized{a}{alignPosIdx}) && ~isempty(perWindowD2.intertrialCenters{a}{alignPosIdx})
            d2Vals = perWindowD2.intertrialNormalized{a}{alignPosIdx};
            windowCenters = perWindowD2.intertrialCenters{a}{alignPosIdx};
            
            % Ensure windowCenters and d2Vals are row vectors for consistent indexing
            if isscalar(windowCenters)
                windowCenters = windowCenters(:)';
            end
            windowCenters = windowCenters(:)';  % Force row vector
            if isscalar(d2Vals)
                d2Vals = d2Vals(:)';
            end
            d2Vals = d2Vals(:)';  % Force row vector
            
            % Window centers at alignment point are the intertrial midpoint times
            % Find intertrial midpoints whose times fall in this segment
            inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
            if any(inSegment)
                intertrialD2NormalizedInSegment = [intertrialD2NormalizedInSegment, d2Vals(inSegment)];
            end
        end

        % Average normalized d2 values for reach vs intertrial windows in this segment
        if ~isempty(reachD2NormalizedInSegment)
            segmentMetrics.reachNormalized{a, s} = nanmean(reachD2NormalizedInSegment);
        else
            segmentMetrics.reachNormalized{a, s} = nan;
        end

        if ~isempty(intertrialD2NormalizedInSegment)
            segmentMetrics.intertrialNormalized{a, s} = nanmean(intertrialD2NormalizedInSegment);
        else
            segmentMetrics.intertrialNormalized{a, s} = nan;
        end
        
        % For compatibility, also store raw d2 values (from alignment point windows)
        if ~isempty(perWindowD2.reach{a}{alignPosIdx}) && ~isempty(perWindowD2.reachCenters{a}{alignPosIdx})
            d2ValsRaw = perWindowD2.reach{a}{alignPosIdx};
            windowCenters = perWindowD2.reachCenters{a}{alignPosIdx};
            if isscalar(windowCenters)
                windowCenters = windowCenters(:)';
            end
            windowCenters = windowCenters(:)';
            if isscalar(d2ValsRaw)
                d2ValsRaw = d2ValsRaw(:)';
            end
            d2ValsRaw = d2ValsRaw(:)';
            inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
            if any(inSegment)
                segmentMetrics.reach{a, s} = nanmean(d2ValsRaw(inSegment));
            else
                segmentMetrics.reach{a, s} = nan;
            end
        else
            segmentMetrics.reach{a, s} = nan;
        end
        
        if ~isempty(perWindowD2.intertrial{a}{alignPosIdx}) && ~isempty(perWindowD2.intertrialCenters{a}{alignPosIdx})
            d2ValsRaw = perWindowD2.intertrial{a}{alignPosIdx};
            windowCenters = perWindowD2.intertrialCenters{a}{alignPosIdx};
            if isscalar(windowCenters)
                windowCenters = windowCenters(:)';
            end
            windowCenters = windowCenters(:)';
            if isscalar(d2ValsRaw)
                d2ValsRaw = d2ValsRaw(:)';
            end
            d2ValsRaw = d2ValsRaw(:)';
            inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
            if any(inSegment)
                segmentMetrics.intertrial{a, s} = nanmean(d2ValsRaw(inSegment));
            else
                segmentMetrics.intertrial{a, s} = nan;
            end
        else
            segmentMetrics.intertrial{a, s} = nan;
        end
    end
end

%% =============================    Save Results    =============================
results = struct();
results.areas = areas;
results.reachStart = reachStart;
results.intertrialMidpoints = intertrialMidpoints;
results.slidingWindowSize = slidingWindowSize;
results.windowBuffer = windowBuffer;
results.beforeAlign = beforeAlign;
results.afterAlign = afterAlign;
results.stepSize = stepSize;
results.slidingPositions = slidingPositions;
results.d2Metrics = d2Metrics;
results.segmentMetrics = segmentMetrics;
results.segmentNames = segmentNames;
if exist('sessionName', 'var')
    results.sessionName = sessionName;
end
if exist('idMatIdx', 'var')
    results.idMatIdx = idMatIdx;
end
if exist('segmentWindowsEng', 'var')
    results.segmentWindows = segmentWindowsEng;
end
results.binSize = binSize;  % Area-specific bin sizes
results.params.pcaFlag = pcaFlag;
results.params.pcaFirstFlag = pcaFirstFlag;
results.params.nDim = nDim;
results.params.pOrder = pOrder;
results.params.critType = critType;
results.params.minSpikesPerBin = minSpikesPerBin;
results.params.minBinsPerWindow = minBinsPerWindow;
results.params.nShuffles = nShuffles;
results.params.normalizeD2 = normalizeD2;

resultsPath = fullfile(saveDir, sprintf('criticality_reach_intertrial_d2_win%.1f_step%.1f.mat', slidingWindowSize, stepSize));
save(resultsPath, 'results');
fprintf('\nSaved results to: %s\n', resultsPath);

%% =============================    Plotting    =============================
% To plot from saved results without running analysis:
%   1. Set loadResultsForPlotting = true
%   2. Optionally specify resultsFileForPlotting (or leave empty for auto-detect)
%   3. Set makePlots = true
%   4. Run only the plotting section (or entire script)
%
% To plot from workspace variables (after running analysis):
%   1. Set loadResultsForPlotting = false
%   2. Set makePlots = true
%   3. Run plotting section (variables should be in workspace)

if ~makePlots
    fprintf('\n=== Skipping plots (makePlots = false) ===\n');
    return;
end

fprintf('\n=== Creating Summary Plots ===\n');

% Load saved results if requested
if loadResultsForPlotting
    fprintf('Loading saved results for plotting...\n');
    
    % Determine results file path
    if isempty(resultsFileForPlotting)
        % Auto-detect: look for most recent results file in saveDir
        if ~exist('saveDir', 'var') || isempty(saveDir)
            error('saveDir not defined. Please specify resultsFileForPlotting or run data loading section first.');
        end
        resultsFiles = dir(fullfile(saveDir, 'criticality_reach_intertrial_d2_win*.mat'));
        if isempty(resultsFiles)
            error('No results files found in %s. Run analysis first or specify resultsFileForPlotting.', saveDir);
        end
        % Sort by date (most recent first)
        [~, idx] = sort([resultsFiles.datenum], 'descend');
        resultsFileForPlotting = fullfile(saveDir, resultsFiles(idx(1)).name);
    end
    
    if ~exist(resultsFileForPlotting, 'file')
        error('Results file not found: %s', resultsFileForPlotting);
    end
    
    fprintf('Loading from: %s\n', resultsFileForPlotting);
    loadedResults = load(resultsFileForPlotting);
    results = loadedResults.results;
    
    % Extract variables from results structure
    areas = results.areas;
    reachStart = results.reachStart;
    intertrialMidpoints = results.intertrialMidpoints;
    slidingWindowSize = results.slidingWindowSize;
    windowBuffer = results.windowBuffer;
    beforeAlign = results.beforeAlign;
    afterAlign = results.afterAlign;
    stepSize = results.stepSize;
    slidingPositions = results.slidingPositions;
    d2Metrics = results.d2Metrics;
    segmentMetrics = results.segmentMetrics;
    segmentNames = results.segmentNames;
    binSize = results.binSize;
    if isfield(results, 'sessionName')
        sessionName = results.sessionName;
    end
    if isfield(results, 'idMatIdx')
        idMatIdx = results.idMatIdx;
    end
    
    % Determine saveDir from results file path if not already defined
    if ~exist('saveDir', 'var') || isempty(saveDir)
        [saveDir, ~, ~] = fileparts(resultsFileForPlotting);
    end
    
    % Determine areasToTest from loaded data
    areasToTest = 1:length(areas);
    
    fprintf('Loaded results: %d areas, %d sliding positions\n', length(areas), length(slidingPositions));
else
    fprintf('Using variables from workspace...\n');
    % Variables should already be in workspace from analysis above
    if ~exist('d2Metrics', 'var')
        error('d2Metrics not found in workspace. Run analysis section first or set loadResultsForPlotting = true.');
    end
    if ~exist('segmentMetrics', 'var')
        error('segmentMetrics not found in workspace. Run analysis section first or set loadResultsForPlotting = true.');
    end
    % Ensure saveDir exists
    if ~exist('saveDir', 'var') || isempty(saveDir)
        error('saveDir not defined. Please run data loading section first.');
    end
    % Ensure areasToTest is defined
    if ~exist('areasToTest', 'var') || isempty(areasToTest)
        if exist('areas', 'var')
            areasToTest = 1:length(areas);
        else
            error('areasToTest and areas not defined. Please run analysis section first or set loadResultsForPlotting = true.');
        end
    end
end

% Detect monitors and size figure to full screen (prefer second monitor if present)
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    targetPos = monitorTwo;
else
    targetPos = monitorOne;
end

% First, collect all y-values to determine shared y-axis limits
allYVals = [];
for a = areasToTest
    if normalizeD2
        reachVals = d2Metrics.reachNormalized{a};
        intertrialVals = d2Metrics.intertrialNormalized{a};
    else
        reachVals = d2Metrics.reach{a};
        intertrialVals = d2Metrics.intertrial{a};
    end
    allYVals = [allYVals, reachVals(~isnan(reachVals)), intertrialVals(~isnan(intertrialVals))];
    
    % Note: Segment values are now center bin means (different scale), so not included in yLimits
end

% Calculate shared y-axis limits with some padding
if ~isempty(allYVals)
    yMin = min(allYVals);
    yMax = max(allYVals);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 1;  % Avoid zero range
    end
    yLimits = [yMin - 0.05*yRange, yMax + 0.05*yRange];
else
    yLimits = [0, 1];  % Default if no data
end

% Determine layout: 2 rows if segments exist, 1 row otherwise
numAreasToPlot = length(areasToTest);
if nSegments > 0
    numRows = 2;
    numCols = numAreasToPlot;
else
    numRows = 1;
    numCols = numAreasToPlot;
end

% Create single figure with tight_subplot
figure(1001); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', targetPos);

% Use tight_subplot if available, otherwise use subplot
useTightSubplot = exist('tight_subplot', 'file');
if useTightSubplot
    ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.1 0.1], [0.08 0.04]);
else
    % Fallback to regular subplot
    ha = zeros(numRows * numCols, 1);
    for i = 1:numRows * numCols
        ha(i) = subplot(numRows, numCols, i);
    end
end

% Create sliding position plots for each area (line plots across sliding positions)
plotIdx = 0;
for a = areasToTest
    plotIdx = plotIdx + 1;
    axes(ha(plotIdx));
    hold on;
    
    % Extract reach and intertrial d2 values across sliding positions
    % Get number of neurons for this area
    if exist('idMatIdx', 'var') && ~isempty(idMatIdx) && a <= length(idMatIdx) && ~isempty(idMatIdx{a})
        numNeurons = length(idMatIdx{a});
        neuronStr = sprintf(' (n=%d)', numNeurons);
    else
        neuronStr = '';
    end
    
    if normalizeD2
        reachVals = d2Metrics.reachNormalized{a};
        intertrialVals = d2Metrics.intertrialNormalized{a};
        yLabelStr = 'd2 (normalized)';
        titleStr = sprintf('%s%s - Sliding Positions', areas{a}, neuronStr);
    else
        reachVals = d2Metrics.reach{a};
        intertrialVals = d2Metrics.intertrial{a};
        yLabelStr = 'd2';
        titleStr = sprintf('%s%s - Sliding Positions', areas{a}, neuronStr);
    end
    
    % Plot as lines
    plot(slidingPositions, reachVals, '-o', 'Color', [0 0 1], 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Reach');
    plot(slidingPositions, intertrialVals, '-s', 'Color', [1 0 0], 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Intertrial');
    
    xlabel('Sliding Position (s)', 'FontSize', 10);
    if plotIdx == 1 || (nSegments == 0 && plotIdx <= numCols)
        ylabel(yLabelStr, 'FontSize', 10);
    end
    title(titleStr, 'FontSize', 11);
    grid on;
    if plotIdx == 1
        legend('Location', 'best', 'FontSize', 9);
    end
    set(gca, 'XTickLabelMode', 'auto');
    set(gca, 'YTickLabelMode', 'auto');
    ylim(yLimits);
    
    % Add vertical line at 0 (alignment point)
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
    
    % Add horizontal line at y = 1
    yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
end

% Create segment-wise summary plots (bar plots for segments) if they exist
if nSegments > 0
    % Calculate y-axis limits for segment plots (center bin means, normalized if applicable)
    segmentYVals = [];
    for a = areasToTest
        nSeg = numel(segmentNames);
        for s = 1:nSeg
            if normalizeD2
                % Use normalized values for y-axis limits
                if ~isempty(segmentMetrics.reachNormalized{a, s}) && ~isnan(segmentMetrics.reachNormalized{a, s})
                    segmentYVals = [segmentYVals, segmentMetrics.reachNormalized{a, s}];
                end
                if ~isempty(segmentMetrics.intertrialNormalized{a, s}) && ~isnan(segmentMetrics.intertrialNormalized{a, s})
                    segmentYVals = [segmentYVals, segmentMetrics.intertrialNormalized{a, s}];
                end
            else
                % Use raw values for y-axis limits
                if ~isempty(segmentMetrics.reach{a, s}) && ~isnan(segmentMetrics.reach{a, s})
                    segmentYVals = [segmentYVals, segmentMetrics.reach{a, s}];
                end
                if ~isempty(segmentMetrics.intertrial{a, s}) && ~isnan(segmentMetrics.intertrial{a, s})
                    segmentYVals = [segmentYVals, segmentMetrics.intertrial{a, s}];
                end
            end
        end
    end
    
    % Calculate segment y-axis limits
    if ~isempty(segmentYVals)
        segYMin = min(segmentYVals);
        segYMax = max(segmentYVals);
        segYRange = segYMax - segYMin;
        if segYRange == 0
            segYRange = 1;  % Avoid zero range
        end
        segmentYLimits = [segYMin - 0.05*segYRange, segYMax + 0.05*segYRange];
    else
        segmentYLimits = [0, 1];  % Default if no data
    end
    
    plotIdx = numAreasToPlot;  % Start from second row
    for a = areasToTest
        plotIdx = plotIdx + 1;
        axes(ha(plotIdx));
        hold on;
        
        % Extract reach and intertrial center bin means across segments (normalized if applicable)
        nSeg = numel(segmentNames);
        reachVals = nan(1, nSeg);
        intertrialVals = nan(1, nSeg);
        for s = 1:nSeg
            if normalizeD2
                % Use normalized values
                if ~isempty(segmentMetrics.reachNormalized{a, s})
                    reachVals(s) = segmentMetrics.reachNormalized{a, s};
                else
                    reachVals(s) = nan;
                end
                if ~isempty(segmentMetrics.intertrialNormalized{a, s})
                    intertrialVals(s) = segmentMetrics.intertrialNormalized{a, s};
                else
                    intertrialVals(s) = nan;
                end
            else
                % Use raw values
                if ~isempty(segmentMetrics.reach{a, s})
                    reachVals(s) = segmentMetrics.reach{a, s};
                else
                    reachVals(s) = nan;
                end
                if ~isempty(segmentMetrics.intertrial{a, s})
                    intertrialVals(s) = segmentMetrics.intertrial{a, s};
                else
                    intertrialVals(s) = nan;
                end
            end
        end
        
        X = 1:nSeg;
        barWidth = 0.35;
        bar(X - barWidth/2, reachVals, barWidth, 'FaceColor', [0 0 1], 'DisplayName', 'Reach');
        bar(X + barWidth/2, intertrialVals, barWidth, 'FaceColor', [1 0 0], 'DisplayName', 'Intertrial');
        
        set(gca, 'XTick', 1:nSeg, 'XTickLabel', segmentNames);
        xtickangle(45);  % Rotate labels for readability
        xlabel('Segment', 'FontSize', 10);
        if plotIdx == numAreasToPlot + 1
            if normalizeD2
                ylabel('d2 (normalized)', 'FontSize', 10);
            else
                ylabel('d2', 'FontSize', 10);
            end
        end
        % Get number of neurons for this area
        if exist('idMatIdx', 'var') && ~isempty(idMatIdx) && a <= length(idMatIdx) && ~isempty(idMatIdx{a})
            numNeurons = length(idMatIdx{a});
            neuronStr = sprintf(' (n=%d)', numNeurons);
        else
            neuronStr = '';
        end
        title(sprintf('%s%s - Segments', areas{a}, neuronStr), 'FontSize', 11);
        grid on;
        if plotIdx == numAreasToPlot + 1
            legend('Location', 'best', 'FontSize', 9);
        end
        set(gca, 'YTickLabelMode', 'auto');
        ylim(segmentYLimits);
        
        % Add horizontal line at y = 1 for normalized plots
        if normalizeD2
            yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
        end
    end
end

% Add overall title
if exist('sessionName', 'var') && ~isempty(sessionName)
    sessionNameShort = sessionName(1:min(10, length(sessionName)));  % First 10 characters
    titlePrefix = [sessionNameShort, ' - '];
else
    titlePrefix = '';
end
if normalizeD2
    sgtitle(sprintf('%sd2 (normalized): Reach vs Intertrial (Window: %.1fs, Buffer: %.1fs)', ...
        titlePrefix, slidingWindowSize, windowBuffer), 'FontSize', 14, 'interpreter', 'none');
else
    sgtitle(sprintf('%sd2: Reach vs Intertrial (Window: %.1fs, Buffer: %.1fs)', ...
        titlePrefix, slidingWindowSize, windowBuffer), 'FontSize', 14, 'interpreter', 'none');
end

% Save figure
saveFile = fullfile(saveDir, sprintf('criticality_reach_intertrial_d2_all_areas_win%.1f_step%.1f.png', slidingWindowSize, stepSize));
exportgraphics(gcf, saveFile, 'Resolution', 300);
fprintf('Saved combined plot to: %s\n', saveFile);

fprintf('\n=== Analysis Complete ===\n');
