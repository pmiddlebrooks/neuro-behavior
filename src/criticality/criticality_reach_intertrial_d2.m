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

if ~exist('opts', 'var')
    opts = neuro_behavior_options;
    opts.frameSize = .001;
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0;
    opts.minFiringRate = .1;
    opts.maxFiringRate = 70;
end

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
    segmentNames = {'B1Eng', 'B1Not', 'B2Eng', 'B2Not'};
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

%% =============================    Configuration    =============================
% Sliding window parameters
beforeAlign = -2;  % Start sliding from this many seconds before alignment point
afterAlign  =  2;  % End sliding at this many seconds after alignment point
slidingWindowSize = 2;  % Window size for d2 analysis (user-defined, same for all areas)
stepSize    = .25; % Step size for sliding window (seconds)
windowBuffer = .5; % Minimum distance from window edge to event/midpoint (seconds)
minWindowSize = 1;  % Minimum window size required (seconds)

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

%% =============================    Find Optimal Bin Sizes Per Area    =============================
fprintf('\n=== Finding Optimal Bin Sizes Per Area ===\n');

% Add path to criticality functions
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils'));

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

%% =============================    Find Valid Reaches and Intertrial Midpoints    =============================
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
maxWindowFromSliding = abs(beforeAlign) + abs(afterAlign);
maxAllowedWindow = min(maxAllowedWindow, maxWindowFromSliding);

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
    % Also store the window data matrices for permutation analysis
    if a == areasToTest(1)
        % Initialize on first area
        collectedReachWindows = cell(numAreas, numSlidingPositions);
        collectedIntertrialWindows = cell(numAreas, numSlidingPositions);
        collectedReachWindowData = cell(numAreas, numSlidingPositions);  % Store [time x neurons] matrices
        collectedIntertrialWindowData = cell(numAreas, numSlidingPositions);  % Store [time x neurons] matrices
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
                
                % Store this window's population activity (as a row vector)
                % Store as: {area, position}(reachIndex, :)
                if isempty(collectedReachWindows{a, posIdx})
                    collectedReachWindows{a, posIdx} = wPopActivity(:)';
                    collectedReachWindowData{a, posIdx} = {wDataMat};
                else
                    collectedReachWindows{a, posIdx} = [collectedReachWindows{a, posIdx}; wPopActivity(:)'];
                    collectedReachWindowData{a, posIdx}{end+1} = wDataMat;
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
                
                % Store this window's population activity (as a row vector)
                % Store as: {area, position}(intertrialIndex, :)
                if isempty(collectedIntertrialWindows{a, posIdx})
                    collectedIntertrialWindows{a, posIdx} = wPopActivity(:)';
                    collectedIntertrialWindowData{a, posIdx} = {wDataMat};
                else
                    collectedIntertrialWindows{a, posIdx} = [collectedIntertrialWindows{a, posIdx}; wPopActivity(:)'];
                    collectedIntertrialWindowData{a, posIdx}{end+1} = wDataMat;
                end
            end
        end
    end
    
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

%% =============================    Per-Sliding-Position d2 Analysis    =============================
fprintf('\n=== Per-Sliding-Position d2 Analysis (Reach vs Intertrial) ===\n');

% Compute d2 metrics for each sliding position
for a = areasToTest
    fprintf('\nComputing d2 metrics per sliding position for area %s...\n', areas{a});
    
    for posIdx = 1:numSlidingPositions
        % Process reach windows
        if ~isempty(collectedReachWindows{a, posIdx})
            % Concatenate all reach windows at this sliding position
            concatReach = [];
            winData = collectedReachWindows{a, posIdx};
            if iscell(winData)
                for w = 1:numel(winData)
                    concatReach = [concatReach; winData{w}(:)];
                end
            else
                % winData is a matrix where each row is a window
                for w = 1:size(winData, 1)
                    concatReach = [concatReach; winData(w, :)'];
                end
            end
            
            % Perform d2 analysis on concatenated reach windows
            if ~isempty(concatReach)
                try
                    [varphi, ~] = myYuleWalker3(concatReach, pOrder);
                    d2Metrics.reach{a}(posIdx) = getFixedPointDistance2(pOrder, critType, varphi);
                catch
                    d2Metrics.reach{a}(posIdx) = nan;
                end
            end
            
            % Perform circular permutations and normalize if requested
            if normalizeD2 && ~isempty(collectedReachWindowData{a, posIdx})
                d2Shuffled = nan(1, nShuffles);
                windowDataList = collectedReachWindowData{a, posIdx};
                
                for s = 1:nShuffles
                    % For each shuffle, concatenate permuted windows
                    concatPermuted = [];
                    for w = 1:numel(windowDataList)
                        wDataMat = windowDataList{w};  % [time bins x neurons]
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
                        concatPermuted = [concatPermuted; permutedPopActivity];
                    end
                    
                    % Compute d2 on concatenated permuted data
                    if ~isempty(concatPermuted)
                        try
                            [varphiPerm, ~] = myYuleWalker3(concatPermuted, pOrder);
                            d2Shuffled(s) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                        catch
                            d2Shuffled(s) = nan;
                        end
                    end
                end
                
                % Normalize d2 by mean of shuffled d2 values
                meanShuffledD2 = nanmean(d2Shuffled);
                if ~isnan(d2Metrics.reach{a}(posIdx)) && ~isnan(meanShuffledD2) && meanShuffledD2 > 0
                    d2Metrics.reachNormalized{a}(posIdx) = d2Metrics.reach{a}(posIdx) / meanShuffledD2;
                else
                    d2Metrics.reachNormalized{a}(posIdx) = nan;
                end
            else
                d2Metrics.reachNormalized{a}(posIdx) = nan;
            end
        end
        
        % Process intertrial windows
        if ~isempty(collectedIntertrialWindows{a, posIdx})
            % Concatenate all intertrial windows at this sliding position
            concatIntertrial = [];
            winData = collectedIntertrialWindows{a, posIdx};
            if iscell(winData)
                for w = 1:numel(winData)
                    concatIntertrial = [concatIntertrial; winData{w}(:)];
                end
            else
                % winData is a matrix where each row is a window
                for w = 1:size(winData, 1)
                    concatIntertrial = [concatIntertrial; winData(w, :)'];
                end
            end
            
            % Perform d2 analysis on concatenated intertrial windows
            if ~isempty(concatIntertrial)
                try
                    [varphi, ~] = myYuleWalker3(concatIntertrial, pOrder);
                    d2Metrics.intertrial{a}(posIdx) = getFixedPointDistance2(pOrder, critType, varphi);
                catch
                    d2Metrics.intertrial{a}(posIdx) = nan;
                end
            end
            
            % Perform circular permutations and normalize if requested
            if normalizeD2 && ~isempty(collectedIntertrialWindowData{a, posIdx})
                d2Shuffled = nan(1, nShuffles);
                windowDataList = collectedIntertrialWindowData{a, posIdx};
                
                for s = 1:nShuffles
                    % For each shuffle, concatenate permuted windows
                    concatPermuted = [];
                    for w = 1:numel(windowDataList)
                        wDataMat = windowDataList{w};  % [time bins x neurons]
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
                        concatPermuted = [concatPermuted; permutedPopActivity];
                    end
                    
                    % Compute d2 on concatenated permuted data
                    if ~isempty(concatPermuted)
                        try
                            [varphiPerm, ~] = myYuleWalker3(concatPermuted, pOrder);
                            d2Shuffled(s) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                        catch
                            d2Shuffled(s) = nan;
                        end
                    end
                end
                
                % Normalize d2 by mean of shuffled d2 values
                meanShuffledD2 = nanmean(d2Shuffled);
                if ~isnan(d2Metrics.intertrial{a}(posIdx)) && ~isnan(meanShuffledD2) && meanShuffledD2 > 0
                    d2Metrics.intertrialNormalized{a}(posIdx) = d2Metrics.intertrial{a}(posIdx) / meanShuffledD2;
                else
                    d2Metrics.intertrialNormalized{a}(posIdx) = nan;
                end
            else
                d2Metrics.intertrialNormalized{a}(posIdx) = nan;
            end
        end
    end
end

%% =============================    Segment-level d2 Analysis    =============================
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
            continue;
        end
        segWin = segmentWindowsList{s};
        tStart = segWin(1);
        tEnd   = segWin(2);

        % Concatenate reach windows whose centers fall in this segment
        concatReach = [];
        reachWindowDataList = {};  % Store window data matrices for permutation
        for posIdx = 1:numSlidingPositions
            if isempty(collectedReachWindows{a, posIdx})
                continue;
            end
            % For reach windows, we need to calculate center times
            % Window center = reachTime + slidingPositions(posIdx)
            winData = collectedReachWindows{a, posIdx};
            numWindows = size(winData, 1);
            
            % Track row index in collected windows (windows are collected in order of reaches)
            rowIdx = 0;
            for r = 1:totalReaches
                reachTime = reachStart(r);
                windowCenter = reachTime + slidingPositions(posIdx);
                
                % Check if this window center falls in the segment
                if windowCenter >= tStart && windowCenter <= tEnd
                    % Check if this window was actually collected
                    % We need to re-check buffer constraints to see if it was collected
                    origReachIdx = validReachIndices(r);
                    prevMidpoint = [];
                    nextMidpoint = [];
                    if origReachIdx > 1
                        prevOrigMidpointIdx = origReachIdx - 1;
                        if prevOrigMidpointIdx <= length(intertrialMidpointsOriginal)
                            prevMidpoint = intertrialMidpointsOriginal(prevOrigMidpointIdx);
                        end
                    end
                    if origReachIdx <= length(intertrialMidpointsOriginal)
                        nextOrigMidpointIdx = origReachIdx;
                        if nextOrigMidpointIdx <= length(intertrialMidpointsOriginal)
                            nextMidpoint = intertrialMidpointsOriginal(nextOrigMidpointIdx);
                        end
                    end
                    
                    winStart = windowCenter - slidingWindowSize / 2;
                    winEnd = windowCenter + slidingWindowSize / 2;
                    constraintViolated = false;
                    if ~isempty(prevMidpoint) && winStart < prevMidpoint + windowBuffer
                        constraintViolated = true;
                    end
                    if ~isempty(nextMidpoint) && winEnd > nextMidpoint - windowBuffer
                        constraintViolated = true;
                    end
                    
                    if ~constraintViolated
                        % This window was collected, increment row index and collect it
                        rowIdx = rowIdx + 1;
                        if rowIdx <= numWindows
                            if iscell(winData)
                                if rowIdx <= numel(winData) && ~isempty(winData{rowIdx})
                                    concatReach = [concatReach; winData{rowIdx}(:)];
                                    if ~isempty(collectedReachWindowData{a, posIdx}) && rowIdx <= numel(collectedReachWindowData{a, posIdx})
                                        reachWindowDataList{end+1} = collectedReachWindowData{a, posIdx}{rowIdx};
                                    end
                                end
                            else
                                if rowIdx <= size(winData, 1)
                                    concatReach = [concatReach; winData(rowIdx, :)'];
                                    if ~isempty(collectedReachWindowData{a, posIdx}) && rowIdx <= numel(collectedReachWindowData{a, posIdx})
                                        reachWindowDataList{end+1} = collectedReachWindowData{a, posIdx}{rowIdx};
                                    end
                                end
                            end
                        end
                    end
                else
                    % Window center not in segment, but might have been collected
                    % Check if it was collected and skip it
                    origReachIdx = validReachIndices(r);
                    prevMidpoint = [];
                    nextMidpoint = [];
                    if origReachIdx > 1
                        prevOrigMidpointIdx = origReachIdx - 1;
                        if prevOrigMidpointIdx <= length(intertrialMidpointsOriginal)
                            prevMidpoint = intertrialMidpointsOriginal(prevOrigMidpointIdx);
                        end
                    end
                    if origReachIdx <= length(intertrialMidpointsOriginal)
                        nextOrigMidpointIdx = origReachIdx;
                        if nextOrigMidpointIdx <= length(intertrialMidpointsOriginal)
                            nextMidpoint = intertrialMidpointsOriginal(nextOrigMidpointIdx);
                        end
                    end
                    
                    winStart = windowCenter - slidingWindowSize / 2;
                    winEnd = windowCenter + slidingWindowSize / 2;
                    constraintViolated = false;
                    if ~isempty(prevMidpoint) && winStart < prevMidpoint + windowBuffer
                        constraintViolated = true;
                    end
                    if ~isempty(nextMidpoint) && winEnd > nextMidpoint - windowBuffer
                        constraintViolated = true;
                    end
                    
                    if ~constraintViolated
                        % This window was collected but not in segment, skip it
                        rowIdx = rowIdx + 1;
                    end
                end
            end
        end

        % Concatenate intertrial windows whose centers fall in this segment
        concatInter = [];
        intertrialWindowDataList = {};  % Store window data matrices for permutation
        for posIdx = 1:numSlidingPositions
            if isempty(collectedIntertrialWindows{a, posIdx})
                continue;
            end
            % For intertrial windows, we need to calculate center times
            winData = collectedIntertrialWindows{a, posIdx};
            numWindows = size(winData, 1);
            
            % Track row index in collected windows (windows are collected in order)
            rowIdx = 0;
            for idx = 1:length(intertrialMidpoints)
                midpointTime = intertrialMidpoints(idx);
                windowCenter = midpointTime + slidingPositions(posIdx);
                
                % Check if this window center falls in the segment
                if windowCenter >= tStart && windowCenter <= tEnd
                    % Find the original intertrial midpoint index
                    origIntertrialIdx = validIntertrialIndicesFiltered(idx);
                    
                    % Find previous and next reaches
                    prevReach = [];
                    nextReach = [];
                    if origIntertrialIdx >= 1 && origIntertrialIdx <= length(reachStartOriginal)
                        prevReach = reachStartOriginal(origIntertrialIdx);
                    end
                    if origIntertrialIdx + 1 <= length(reachStartOriginal)
                        nextReach = reachStartOriginal(origIntertrialIdx + 1);
                    end
                    
                    winStart = windowCenter - slidingWindowSize / 2;
                    winEnd = windowCenter + slidingWindowSize / 2;
                    constraintViolated = false;
                    if ~isempty(prevReach) && winStart < prevReach + windowBuffer
                        constraintViolated = true;
                    end
                    if ~isempty(nextReach) && winEnd > nextReach - windowBuffer
                        constraintViolated = true;
                    end
                    
                    if ~constraintViolated
                        % This window was collected, increment row index and collect it
                        rowIdx = rowIdx + 1;
                        if rowIdx <= numWindows
                            if iscell(winData)
                                if rowIdx <= numel(winData) && ~isempty(winData{rowIdx})
                                    concatInter = [concatInter; winData{rowIdx}(:)];
                                    if ~isempty(collectedIntertrialWindowData{a, posIdx}) && rowIdx <= numel(collectedIntertrialWindowData{a, posIdx})
                                        intertrialWindowDataList{end+1} = collectedIntertrialWindowData{a, posIdx}{rowIdx};
                                    end
                                end
                            else
                                if rowIdx <= size(winData, 1)
                                    concatInter = [concatInter; winData(rowIdx, :)'];
                                    if ~isempty(collectedIntertrialWindowData{a, posIdx}) && rowIdx <= numel(collectedIntertrialWindowData{a, posIdx})
                                        intertrialWindowDataList{end+1} = collectedIntertrialWindowData{a, posIdx}{rowIdx};
                                    end
                                end
                            end
                        end
                    end
                else
                    % Window center not in segment, but might have been collected
                    % Check if it was collected and skip it
                    origIntertrialIdx = validIntertrialIndicesFiltered(idx);
                    prevReach = [];
                    nextReach = [];
                    if origIntertrialIdx >= 1 && origIntertrialIdx <= length(reachStartOriginal)
                        prevReach = reachStartOriginal(origIntertrialIdx);
                    end
                    if origIntertrialIdx + 1 <= length(reachStartOriginal)
                        nextReach = reachStartOriginal(origIntertrialIdx + 1);
                    end
                    
                    winStart = windowCenter - slidingWindowSize / 2;
                    winEnd = windowCenter + slidingWindowSize / 2;
                    constraintViolated = false;
                    if ~isempty(prevReach) && winStart < prevReach + windowBuffer
                        constraintViolated = true;
                    end
                    if ~isempty(nextReach) && winEnd > nextReach - windowBuffer
                        constraintViolated = true;
                    end
                    
                    if ~constraintViolated
                        % This window was collected but not in segment, skip it
                        rowIdx = rowIdx + 1;
                    end
                end
            end
        end

        % d2 analysis: reach
        if ~isempty(concatReach)
            try
                [varphi, ~] = myYuleWalker3(concatReach, pOrder);
                segmentMetrics.reach{a, s} = getFixedPointDistance2(pOrder, critType, varphi);
            catch
                segmentMetrics.reach{a, s} = nan;
            end
        else
            segmentMetrics.reach{a, s} = nan;
        end

        % d2 analysis: intertrial
        if ~isempty(concatInter)
            try
                [varphi, ~] = myYuleWalker3(concatInter, pOrder);
                segmentMetrics.intertrial{a, s} = getFixedPointDistance2(pOrder, critType, varphi);
            catch
                segmentMetrics.intertrial{a, s} = nan;
            end
        else
            segmentMetrics.intertrial{a, s} = nan;
        end
        
        % Normalize d2 values using circular permutations if requested
        if normalizeD2
            % Normalize reach d2
            if ~isempty(reachWindowDataList) && ~isnan(segmentMetrics.reach{a, s})
                d2Shuffled = nan(1, nShuffles);
                for shuffle = 1:nShuffles
                    concatPermuted = [];
                    for w = 1:numel(reachWindowDataList)
                        wDataMat = reachWindowDataList{w};  % [time bins x neurons]
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
                        concatPermuted = [concatPermuted; permutedPopActivity];
                    end
                    
                    % Compute d2 on concatenated permuted data
                    if ~isempty(concatPermuted)
                        try
                            [varphiPerm, ~] = myYuleWalker3(concatPermuted, pOrder);
                            d2Shuffled(shuffle) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                        catch
                            d2Shuffled(shuffle) = nan;
                        end
                    end
                end
                
                meanShuffledD2 = nanmean(d2Shuffled);
                if ~isnan(segmentMetrics.reach{a, s}) && ~isnan(meanShuffledD2) && meanShuffledD2 > 0
                    segmentMetrics.reachNormalized{a, s} = segmentMetrics.reach{a, s} / meanShuffledD2;
                else
                    segmentMetrics.reachNormalized{a, s} = nan;
                end
            else
                segmentMetrics.reachNormalized{a, s} = nan;
            end
            
            % Normalize intertrial d2
            if ~isempty(intertrialWindowDataList) && ~isnan(segmentMetrics.intertrial{a, s})
                d2Shuffled = nan(1, nShuffles);
                for shuffle = 1:nShuffles
                    concatPermuted = [];
                    for w = 1:numel(intertrialWindowDataList)
                        wDataMat = intertrialWindowDataList{w};  % [time bins x neurons]
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
                        concatPermuted = [concatPermuted; permutedPopActivity];
                    end
                    
                    % Compute d2 on concatenated permuted data
                    if ~isempty(concatPermuted)
                        try
                            [varphiPerm, ~] = myYuleWalker3(concatPermuted, pOrder);
                            d2Shuffled(shuffle) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                        catch
                            d2Shuffled(shuffle) = nan;
                        end
                    end
                end
                
                meanShuffledD2 = nanmean(d2Shuffled);
                if ~isnan(segmentMetrics.intertrial{a, s}) && ~isnan(meanShuffledD2) && meanShuffledD2 > 0
                    segmentMetrics.intertrialNormalized{a, s} = segmentMetrics.intertrial{a, s} / meanShuffledD2;
                else
                    segmentMetrics.intertrialNormalized{a, s} = nan;
                end
            else
                segmentMetrics.intertrialNormalized{a, s} = nan;
            end
        else
            segmentMetrics.reachNormalized{a, s} = nan;
            segmentMetrics.intertrialNormalized{a, s} = nan;
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

% Create sliding position plots for each area (line plots across sliding positions)
for a = areasToTest
    figure(1000 + a); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    
    hold on;
    
    % Extract reach and intertrial d2 values across sliding positions
    if normalizeD2
        reachVals = d2Metrics.reachNormalized{a};
        intertrialVals = d2Metrics.intertrialNormalized{a};
        yLabelStr = 'd2 (normalized)';
        titleStr = sprintf('%s - d2 (normalized): Reach vs Intertrial across Sliding Positions', areas{a});
    else
        reachVals = d2Metrics.reach{a};
        intertrialVals = d2Metrics.intertrial{a};
        yLabelStr = 'd2';
        titleStr = sprintf('%s - d2: Reach vs Intertrial across Sliding Positions', areas{a});
    end
    
    % Plot as lines
    plot(slidingPositions, reachVals, '-o', 'Color', [0 0 1], 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Reach');
    plot(slidingPositions, intertrialVals, '-s', 'Color', [1 0 0], 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Intertrial');
    
    xlabel('Sliding Position (s)', 'FontSize', 12);
    ylabel(yLabelStr, 'FontSize', 12);
    title(titleStr, 'FontSize', 14);
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    set(gca, 'XTickLabelMode', 'auto');
    set(gca, 'YTickLabelMode', 'auto');
    
    % Add vertical line at 0 (alignment point)
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
    
    sgtitle(sprintf('%s - d2: Reach vs Intertrial (Window: %.1fs, Buffer: %.1fs, Bin: %.3fs)', ...
        areas{a}, slidingWindowSize, windowBuffer, binSize(a)), 'FontSize', 14);
    
    % Save figure
    saveFile = fullfile(saveDir, sprintf('criticality_reach_intertrial_d2_%s_sliding_win%.1f_step%.1f.png', areas{a}, slidingWindowSize, stepSize));
    exportgraphics(gcf, saveFile, 'Resolution', 300);
    fprintf('Saved sliding position plot for %s to: %s\n', areas{a}, saveFile);
end

% Also create segment-wise summary plot (bar plots for segments)
if nSegments > 0
    for a = areasToTest
        figure(2000 + a); clf;
        set(gcf, 'Units', 'pixels');
        set(gcf, 'Position', targetPos);
        
        hold on;
        
        % Extract reach and intertrial d2 values across segments
        nSeg = numel(segmentNames);
        reachVals = nan(1, nSeg);
        intertrialVals = nan(1, nSeg);
        for s = 1:nSeg
            if normalizeD2
                reachVals(s) = segmentMetrics.reachNormalized{a, s};
                intertrialVals(s) = segmentMetrics.intertrialNormalized{a, s};
            else
                reachVals(s) = segmentMetrics.reach{a, s};
                intertrialVals(s) = segmentMetrics.intertrial{a, s};
            end
        end
        
        X = 1:nSeg;
        barWidth = 0.35;
        bar(X - barWidth/2, reachVals, barWidth, 'FaceColor', [0 0 1], 'DisplayName', 'Reach');
        bar(X + barWidth/2, intertrialVals, barWidth, 'FaceColor', [1 0 0], 'DisplayName', 'Intertrial');
        
        set(gca, 'XTick', 1:nSeg, 'XTickLabel', segmentNames);
        xlabel('Segment', 'FontSize', 12);
        if normalizeD2
            ylabel('d2 (normalized)', 'FontSize', 12);
            title(sprintf('%s - d2 (normalized): Reach vs Intertrial (segments)', areas{a}), 'FontSize', 14);
        else
            ylabel('d2', 'FontSize', 12);
            title(sprintf('%s - d2: Reach vs Intertrial (segments)', areas{a}), 'FontSize', 14);
        end
        grid on;
        legend('Location', 'best', 'FontSize', 10);
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
        
        sgtitle(sprintf('%s - Segment-wise d2: Reach vs Intertrial (Window: %.1fs, Buffer: %.1fs, Bin: %.3fs)', ...
            areas{a}, slidingWindowSize, windowBuffer, binSize(a)), 'FontSize', 14);
        
        % Save figure
        saveFile = fullfile(saveDir, sprintf('criticality_reach_intertrial_d2_%s_segments_win%.1f_step%.1f.png', areas{a}, slidingWindowSize, stepSize));
        exportgraphics(gcf, saveFile, 'Resolution', 300);
        fprintf('Saved segment summary plot for %s to: %s\n', areas{a}, saveFile);
    end
end

fprintf('\n=== Analysis Complete ===\n');
