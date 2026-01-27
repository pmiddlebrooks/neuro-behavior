%%
% LZC Reach vs Intertrial Analysis
% Compares Lempel-Ziv complexity measures between reaches and intertrial intervals
% Performs sliding window LZC analysis around reach starts and intertrial midpoints
%
% Variables:
%   reachStart - Reach start times in seconds
%   intertrialMidpoints - Midpoint times between consecutive reaches
%   slidingWindowSize - Window size for LZC analysis (can be area-specific if useOptimalWindowSize is true)
%   windowBuffer - Minimum distance from window edge to event/midpoint
%   beforeAlign - Start of sliding window range (seconds before alignment point)
%   afterAlign - End of sliding window range (seconds after alignment point)
%   stepSize - Step size for sliding window (seconds)
%   nShuffles - Number of shuffles for LZC normalization
%   useBernoulliControl - Set to true to compute Bernoulli normalized metric (default: false)
%
% This script uses load_sliding_window_data() to load data (as in choose_task_and_session.m)
% and finds optimal bin sizes and window sizes per area (as in lzc_sliding_analysis.m)
%
% Normalization: LZC values are normalized by shuffling the binary sequence and computing
% LZC on shuffled sequences. The real LZC is divided by the mean of shuffled LZC values.
% Optionally, LZC can also be normalized by rate-matched Bernoulli control sequences.

%% Configure

% Want to parallelize the area-wise analysis?
runParallel = 1;

% Also analzye combined M23 nad M56?
includeM2356 = 1;
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

% Add combined M2356 area if requested and M23 and M56 exist
if includeM2356
    idxM23 = find(strcmp(areas, 'M23'));
    idxM56 = find(strcmp(areas, 'M56'));
    if ~isempty(idxM23) && ~isempty(idxM56) && ~any(strcmp(areas, 'M2356'))
        % Create combined M2356 area
        areas{end+1} = 'M2356';
        dataStruct.areas = areas;  % Update dataStruct.areas
        idMatIdx{end+1} = [idMatIdx{idxM23}(:); idMatIdx{idxM56}(:)];
        % Update dataStruct.idMatIdx if it exists
        if isfield(dataStruct, 'idMatIdx')
            dataStruct.idMatIdx{end+1} = idMatIdx{end};
        end
        if isfield(dataStruct, 'idLabel')
            dataStruct.idLabel{end+1} = [dataStruct.idLabel{idxM23}(:); dataStruct.idLabel{idxM56}(:)];
        end
        numAreas = length(areas);
        fprintf('\n=== Added combined M2356 area ===\n');
        fprintf('M2356: %d neurons (M23: %d, M56: %d)\n', ...
            length(idMatIdx{end}), ...
            length(idMatIdx{idxM23}), ...
            length(idMatIdx{idxM56}));
    elseif isempty(idxM23) || isempty(idxM56)
        fprintf('\n=== Warning: includeM2356 is true but M23 or M56 not found. Skipping M2356 creation. ===\n');
    end
end

% Get areasToTest
if isfield(dataStruct, 'areasToTest')
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

% If M2356 was created, ensure it's included in areasToTest
if includeM2356 && any(strcmp(areas, 'M2356'))
    m2356Idx = find(strcmp(areas, 'M2356'));
    if ~ismember(m2356Idx, areasToTest)
        areasToTest = [areasToTest, m2356Idx];
        fprintf('Added M2356 (index %d) to areasToTest\n', m2356Idx);
    end
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
slidingWindowSize = 5;  % Window size for LZC analysis (can be area-specific if useOptimalWindowSize is true)
stepSize    = 1/3; % Step size for sliding window (seconds)
windowBuffer = .5; % Minimum distance from window edge to event/midpoint (seconds)
minWindowSize = slidingWindowSize;  % Minimum window size required (seconds)

% LZC analysis parameters
nShuffles = 4;      % Number of shuffles for LZC normalization
useBernoulliControl = false;  % Set to true to compute Bernoulli normalized metric
useOptimalBinSize = true;  % Set to true to automatically calculate optimal bin size per area
useOptimalWindowSize = true;  % Set to true to automatically calculate optimal window size per area
minSpikesPerBin = 0.08;  % Minimum spikes per bin for optimal bin size calculation
minDataPoints = 2e4;  % Minimum data points per window for optimal window size calculation
minSlidingWindowSize = 6;  % Minimum window size (seconds)
maxSlidingWindowSize = 7;  % Maximum window size (seconds)
minBinSize = 0.01;  % Minimum bin size (seconds)
nMinNeurons = 10;  % Minimum number of neurons required per area
includeM2356 = true;  % Set to true to include combined M23+M56 area

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


% Calculate time range from spike data
fprintf('\n--- Using spike times for on-demand binning ---\n');
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end

% =============================    Find Optimal Bin and Window Sizes Per Area    =============================
fprintf('\n=== Finding Optimal Bin and Window Sizes Per Area ===\n');

% Filter out areas with insufficient neurons from areasToTest
% Note: M2356 is checked separately - it can be included even if M23 or M56 individually don't have enough neurons
fprintf('\n=== Filtering Areas by Neuron Count ===\n');
validAreasToTest = [];
for a = areasToTest
    % Skip M2356 for now - we'll check it separately after ensuring M23/M56 data is available
    if includeM2356 && any(strcmp(areas, 'M2356')) && a == find(strcmp(areas, 'M2356'))
        continue;  % Will check M2356 separately
    end
    
    if a <= length(idMatIdx) && ~isempty(idMatIdx{a})
        nNeurons = length(idMatIdx{a});
        if nNeurons >= nMinNeurons
            validAreasToTest = [validAreasToTest, a];
            fprintf('Area %s: %d neurons (included)\n', areas{a}, nNeurons);
        else
            fprintf('Area %s: %d neurons (excluded, < %d neurons)\n', areas{a}, nNeurons, nMinNeurons);
        end
    else
        fprintf('Area %s: no neurons found (excluded)\n', areas{a});
    end
end

% Check M2356 separately - include it if combined neurons >= nMinNeurons
if includeM2356 && any(strcmp(areas, 'M2356'))
    m2356Idx = find(strcmp(areas, 'M2356'));
    if m2356Idx <= length(idMatIdx) && ~isempty(idMatIdx{m2356Idx})
        nNeuronsM2356 = length(idMatIdx{m2356Idx});
        if nNeuronsM2356 >= nMinNeurons
            validAreasToTest = [validAreasToTest, m2356Idx];
            fprintf('Area %s: %d neurons (included, combined M23+M56)\n', areas{m2356Idx}, nNeuronsM2356);
        else
            fprintf('Area %s: %d neurons (excluded, < %d neurons even when combined)\n', areas{m2356Idx}, nNeuronsM2356, nMinNeurons);
        end
    end
end

if isempty(validAreasToTest)
    error('No areas have sufficient neurons (>= %d). Check nMinNeurons threshold or data.', nMinNeurons);
end

areasToTest = validAreasToTest;
fprintf('\nWill process %d area(s): %s\n', length(areasToTest), strjoin(areas(areasToTest), ', '));

% Find optimal bin size per area using spike times
binSize = zeros(1, numAreas);
for a = areasToTest
    neuronIDs = dataStruct.idLabel{a};
    
    % Calculate firing rate from spike times
    thisFiringRate = calculate_firing_rate_from_spikes(...
        dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange);
    
    if useOptimalBinSize
        [binSize(a), ~] = find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, 1);
        binSize(a) = max(binSize(a), minBinSize);
    else
        % Use a default bin size if not using optimal
        binSize(a) = 0.01;  % 10ms default
    end
    
    fprintf('Area %s: bin size = %.3f s, firing rate = %.2f spikes/s\n', ...
        areas{a}, binSize(a), thisFiringRate);
end

% Find optimal window size per area if requested
if useOptimalWindowSize
    fprintf('\n=== Finding Optimal Window Sizes Per Area ===\n');
    slidingWindowSize = zeros(1, numAreas);
    
    for a = areasToTest
        neuronIDs = dataStruct.idLabel{a};
        nNeurons = length(neuronIDs);
        
        % Calculate minimum window size needed: totalDataPoints = nNeurons × (windowSize / binSize)
        % We need: nNeurons × (windowSize / binSize) >= minDataPoints
        % So: windowSize >= (minDataPoints × binSize) / nNeurons
        minRequiredWindowSize = (minDataPoints * binSize(a)) / nNeurons;
        
        % Constrain to minSlidingWindowSize-maxSlidingWindowSize range and use minimum that satisfies requirement
        slidingWindowSize(a) = ceil(max(minSlidingWindowSize, min(maxSlidingWindowSize, minRequiredWindowSize)));
        
        % Calculate actual data points with this window size
        actualDataPoints = nNeurons * (slidingWindowSize(a) / binSize(a));
        
        fprintf('Area %s: %d neurons, binSize=%.3fs -> window=%.2fs (%.0f data points)\n', ...
            areas{a}, nNeurons, binSize(a), slidingWindowSize(a), actualDataPoints);
    end
else
    % Use scalar slidingWindowSize for all areas
    if isscalar(slidingWindowSize)
        slidingWindowSize = repmat(slidingWindowSize, 1, numAreas);
    end
end

% =============================    Find Valid Reaches and Intertrial Midpoints    =============================
fprintf('\n=== Finding Valid Reaches and Intertrial Midpoints ===\n');

% Find maximum possible window duration that satisfies buffer constraints
% For reach-centered windows: edges must not overlap with windowBuffer sec after 
%   previous intertrial midpoint or windowBuffer sec before next intertrial midpoint
% For intertrial-centered windows: edges must not overlap with windowBuffer sec after 
%   previous reach or windowBuffer sec before next reach

% Use minimum window size across areas for validation
% Exclude NaN values (areas that were skipped)
validWindowSizes = slidingWindowSize(areasToTest);
validWindowSizes = validWindowSizes(~isnan(validWindowSizes));
if isempty(validWindowSizes)
    error('No valid window sizes found. All areas were filtered out.');
end
minWindowSizeAcrossAreas = min(validWindowSizes);

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
    
    % Keep this reach if its maximum window is >= minWindowSizeAcrossAreas
    if maxWindowForThisReach >= minWindowSizeAcrossAreas
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
    
    % Keep this intertrial midpoint if its maximum window is >= minWindowSizeAcrossAreas
    if maxWindowForThisIntertrial >= minWindowSizeAcrossAreas
        validIntertrialIndices = [validIntertrialIndices, i];
    end
end

% Check if user-defined slidingWindowSize is valid
if isempty(validReachIndices) && isempty(validIntertrialIndices)
    error('No valid reaches or intertrial midpoints found with window size >= %.1f seconds. Try reducing windowBuffer or minWindowSize.', minWindowSizeAcrossAreas);
end

% For each area, check if its window size is valid
for a = areasToTest
    areaWindowSize = slidingWindowSize(a);
    
    % Skip areas with NaN window size (shouldn't happen after filtering, but check anyway)
    if isnan(areaWindowSize)
        continue;
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
    
    % Check if area-specific slidingWindowSize is valid
    if areaWindowSize > maxAllowedWindow
        warning('Area %s: slidingWindowSize (%.2f s) exceeds maximum allowed (%.2f s). Using maximum allowed.', ...
            areas{a}, areaWindowSize, maxAllowedWindow);
        slidingWindowSize(a) = maxAllowedWindow;
    end
    
    % Ensure slidingWindowSize is at least minWindowSizeAcrossAreas
    if slidingWindowSize(a) < minWindowSizeAcrossAreas
        slidingWindowSize(a) = minWindowSizeAcrossAreas;
    end
    
    % Ensure slidingWindowSize is positive and reasonable
    if slidingWindowSize(a) <= 0 || isnan(slidingWindowSize(a)) || isinf(slidingWindowSize(a))
        error('Area %s: Invalid slidingWindowSize. Try reducing windowBuffer or increasing inter-reach intervals.', areas{a});
    end
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

fprintf('Using area-specific sliding window sizes (min: %.2f s, max: %.2f s)\n', ...
    min(slidingWindowSize(areasToTest)), max(slidingWindowSize(areasToTest)));
fprintf('  Minimum window size: %.2f seconds\n', minWindowSizeAcrossAreas);
fprintf('  Valid reaches: %d/%d (%.1f%%)\n', length(validReachIndices), length(maxWindowPerReach), 100*length(validReachIndices)/length(maxWindowPerReach));
fprintf('  Valid intertrial midpoints: %d/%d (%.1f%%)\n', length(intertrialMidpoints), length(maxWindowPerIntertrial), 100*length(intertrialMidpoints)/length(maxWindowPerIntertrial));
fprintf('  Window buffer: %.2f seconds\n', windowBuffer);

%% =============================    Analysis    =============================
fprintf('\n=== Processing Areas ===\n');

% Check if parpool is already running, start one if not
if runParallel
    currentPool = gcp('nocreate');
if isempty(currentPool)
    NumWorkers = min(3, length(dataStruct.areas));
    parpool('local', NumWorkers);
    fprintf('Started parallel pool with %d workers\n', NumWorkers);
else
    fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
end
end

% Calculate sliding window positions
slidingPositions = beforeAlign:stepSize:afterAlign;
numSlidingPositions = length(slidingPositions);

% Initialize storage for LZC metrics (per sliding position, concatenated windows)
% Structure: {area}[slidingPosition]
lzcMetrics = struct();
lzcMetrics.reach = cell(1, numAreas);
lzcMetrics.intertrial = cell(1, numAreas);
lzcMetrics.reachNormalized = cell(1, numAreas);
lzcMetrics.intertrialNormalized = cell(1, numAreas);
if useBernoulliControl
    lzcMetrics.reachNormalizedBernoulli = cell(1, numAreas);
    lzcMetrics.intertrialNormalizedBernoulli = cell(1, numAreas);
end

for a = areasToTest
    lzcMetrics.reach{a} = nan(1, numSlidingPositions);
    lzcMetrics.intertrial{a} = nan(1, numSlidingPositions);
    lzcMetrics.reachNormalized{a} = nan(1, numSlidingPositions);
    lzcMetrics.intertrialNormalized{a} = nan(1, numSlidingPositions);
    if useBernoulliControl
        lzcMetrics.reachNormalizedBernoulli{a} = nan(1, numSlidingPositions);
        lzcMetrics.intertrialNormalizedBernoulli{a} = nan(1, numSlidingPositions);
    end
end

% Process each area
for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    
    aID = dataStruct.idMatIdx{a};
    neuronIDs = dataStruct.idLabel{a};
    areaWindowSize = slidingWindowSize(a);
    
    % Bin spikes on-demand at area-specific bin size for the full time range
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    numTimePoints = size(aDataMat, 1);
    
    % Initialize storage for collected windows and their center times
    % For each sliding position, collect all valid windows
    % Store as cell array indexed by area and position: collectedReachWindows{area, position}
    % Also store the window data matrices and window center times
    if a == areasToTest(1)
        % Initialize on first area
        collectedReachWindowData = cell(numAreas, numSlidingPositions);  % Store [time x neurons] matrices
        collectedIntertrialWindowData = cell(numAreas, numSlidingPositions);  % Store [time x neurons] matrices
        collectedReachWindowCenters = cell(numAreas, numSlidingPositions);  % Store window center times
        collectedIntertrialWindowCenters = cell(numAreas, numSlidingPositions);  % Store window center times
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
            winStart = windowCenter - areaWindowSize / 2;
            winEnd = windowCenter + areaWindowSize / 2;
            
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
                windowCenter, areaWindowSize, binSize(a), numTimePoints);
            
            % Check if window is valid (within bounds)
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                % Extract window data [time bins x neurons] for LZC analysis
                wDataMat = aDataMat(startIdx:endIdx, :);
                
                % Store this window's data matrix and center time
                if isempty(collectedReachWindowData{a, posIdx})
                    collectedReachWindowData{a, posIdx} = {wDataMat};
                    collectedReachWindowCenters{a, posIdx} = windowCenter;
                else
                    collectedReachWindowData{a, posIdx}{end+1} = wDataMat;
                    collectedReachWindowCenters{a, posIdx} = [collectedReachWindowCenters{a, posIdx}, windowCenter];
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
            winStart = windowCenter - areaWindowSize / 2;
            winEnd = windowCenter + areaWindowSize / 2;
            
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
                windowCenter, areaWindowSize, binSize(a), numTimePoints);
            
            % Check if window is valid (within bounds)
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                % Extract window data [time bins x neurons] for LZC analysis
                wDataMat = aDataMat(startIdx:endIdx, :);
                
                % Store this window's data matrix and center time
                if isempty(collectedIntertrialWindowData{a, posIdx})
                    collectedIntertrialWindowData{a, posIdx} = {wDataMat};
                    collectedIntertrialWindowCenters{a, posIdx} = windowCenter;
                else
                    collectedIntertrialWindowData{a, posIdx}{end+1} = wDataMat;
                    collectedIntertrialWindowCenters{a, posIdx} = [collectedIntertrialWindowCenters{a, posIdx}, windowCenter];
                end
            end
        end
    end
    
end

% =============================    Per-Sliding-Position LZC Analysis    =============================
fprintf('\n=== Per-Sliding-Position LZC Analysis (Reach vs Intertrial) ===\n');

% Initialize storage for per-window LZC values (for use in segment analysis)
% Structure: {area}{eventType}{slidingPosition} = [LZC values] and corresponding center times
perWindowLZC = struct();
perWindowLZC.reach = cell(1, numAreas);
perWindowLZC.intertrial = cell(1, numAreas);
perWindowLZC.reachNormalized = cell(1, numAreas);
perWindowLZC.intertrialNormalized = cell(1, numAreas);
if useBernoulliControl
    perWindowLZC.reachNormalizedBernoulli = cell(1, numAreas);
    perWindowLZC.intertrialNormalizedBernoulli = cell(1, numAreas);
end
perWindowLZC.reachCenters = cell(1, numAreas);
perWindowLZC.intertrialCenters = cell(1, numAreas);

for a = areasToTest
    perWindowLZC.reach{a} = cell(1, numSlidingPositions);
    perWindowLZC.intertrial{a} = cell(1, numSlidingPositions);
    perWindowLZC.reachNormalized{a} = cell(1, numSlidingPositions);
    perWindowLZC.intertrialNormalized{a} = cell(1, numSlidingPositions);
    if useBernoulliControl
        perWindowLZC.reachNormalizedBernoulli{a} = cell(1, numSlidingPositions);
        perWindowLZC.intertrialNormalizedBernoulli{a} = cell(1, numSlidingPositions);
    end
    perWindowLZC.reachCenters{a} = cell(1, numSlidingPositions);
    perWindowLZC.intertrialCenters{a} = cell(1, numSlidingPositions);
end

% Pre-allocate temporary cell arrays indexed by loop variable idx for parfor compatibility
% These will be assigned to the correct area indices after the parfor loop
numAreasToProcess = length(areasToTest);
tempLzcMetricsReach = cell(1, numAreasToProcess);
tempLzcMetricsIntertrial = cell(1, numAreasToProcess);
tempLzcMetricsReachNormalized = cell(1, numAreasToProcess);
tempLzcMetricsIntertrialNormalized = cell(1, numAreasToProcess);
if useBernoulliControl
    tempLzcMetricsReachNormalizedBernoulli = cell(1, numAreasToProcess);
    tempLzcMetricsIntertrialNormalizedBernoulli = cell(1, numAreasToProcess);
end
tempPerWindowLZCReach = cell(1, numAreasToProcess);
tempPerWindowLZCIntertrial = cell(1, numAreasToProcess);
tempPerWindowLZCReachNormalized = cell(1, numAreasToProcess);
tempPerWindowLZCIntertrialNormalized = cell(1, numAreasToProcess);
if useBernoulliControl
    tempPerWindowLZCReachNormalizedBernoulli = cell(1, numAreasToProcess);
    tempPerWindowLZCIntertrialNormalizedBernoulli = cell(1, numAreasToProcess);
end
tempPerWindowLZCReachCenters = cell(1, numAreasToProcess);
tempPerWindowLZCIntertrialCenters = cell(1, numAreasToProcess);

% Compute LZC metrics for each sliding position
parfor idx = 1:numAreasToProcess
    a = areasToTest(idx);  % Get actual area index
    tic
    fprintf('\nComputing LZC metrics per sliding position for area %s...\n', areas{a});
    
    % Initialize temporary arrays for this area
    tempLzcMetricsReach{idx} = nan(1, numSlidingPositions);
    tempLzcMetricsIntertrial{idx} = nan(1, numSlidingPositions);
    tempLzcMetricsReachNormalized{idx} = nan(1, numSlidingPositions);
    tempLzcMetricsIntertrialNormalized{idx} = nan(1, numSlidingPositions);
    if useBernoulliControl
        tempLzcMetricsReachNormalizedBernoulli{idx} = nan(1, numSlidingPositions);
        tempLzcMetricsIntertrialNormalizedBernoulli{idx} = nan(1, numSlidingPositions);
    end
    tempPerWindowLZCReach{idx} = cell(1, numSlidingPositions);
    tempPerWindowLZCIntertrial{idx} = cell(1, numSlidingPositions);
    tempPerWindowLZCReachNormalized{idx} = cell(1, numSlidingPositions);
    tempPerWindowLZCIntertrialNormalized{idx} = cell(1, numSlidingPositions);
    if useBernoulliControl
        tempPerWindowLZCReachNormalizedBernoulli{idx} = cell(1, numSlidingPositions);
        tempPerWindowLZCIntertrialNormalizedBernoulli{idx} = cell(1, numSlidingPositions);
    end
    tempPerWindowLZCReachCenters{idx} = cell(1, numSlidingPositions);
    tempPerWindowLZCIntertrialCenters{idx} = cell(1, numSlidingPositions);
    
    for posIdx = 1:numSlidingPositions
        % Process reach windows - compute LZC for each window individually
        if ~isempty(collectedReachWindowData{a, posIdx})
            windowDataList = collectedReachWindowData{a, posIdx};
            windowCenters = collectedReachWindowCenters{a, posIdx};
            numWindows = numel(windowDataList);
            
            lzcPerWindow = nan(1, numWindows);
            lzcNormalizedPerWindow = nan(1, numWindows);
            lzcNormalizedBernoulliPerWindow = nan(1, numWindows);  % Always initialize, even if not used
            
            % Compute LZC for each window
            for w = 1:numWindows
                wDataMat = windowDataList{w};  % [time bins x neurons]
                
                % Concatenate across neurons over time
                nNeurons = size(wDataMat, 2);
                nSamples = size(wDataMat, 1);
                concatenatedSeq = reshape(wDataMat', nSamples * nNeurons, 1);
                
                % Binarize: any value > 0 becomes 1, 0 stays 0
                binarySeq = double(concatenatedSeq > 0);
                
                % Calculate LZ complexity with controls
                [lzcPerWindow(w), lzcNormalizedPerWindow(w), lzcNormalizedBernoulliPerWindow(w)] = ...
                    compute_lz_complexity_with_controls(binarySeq, nShuffles, useBernoulliControl);
            end
            
            % Average LZC values across windows at this sliding position
            tempLzcMetricsReach{idx}(posIdx) = nanmean(lzcPerWindow);
            tempLzcMetricsReachNormalized{idx}(posIdx) = nanmean(lzcNormalizedPerWindow);
            if useBernoulliControl
                tempLzcMetricsReachNormalizedBernoulli{idx}(posIdx) = nanmean(lzcNormalizedBernoulliPerWindow);
            end
            
            % Store per-window LZC values and centers for segment analysis
            tempPerWindowLZCReach{idx}{posIdx} = lzcPerWindow;
            tempPerWindowLZCReachNormalized{idx}{posIdx} = lzcNormalizedPerWindow;
            if useBernoulliControl
                tempPerWindowLZCReachNormalizedBernoulli{idx}{posIdx} = lzcNormalizedBernoulliPerWindow;
            end
            tempPerWindowLZCReachCenters{idx}{posIdx} = windowCenters;
        end
        
        % Process intertrial windows - compute LZC for each window individually
        if ~isempty(collectedIntertrialWindowData{a, posIdx})
            windowDataList = collectedIntertrialWindowData{a, posIdx};
            windowCenters = collectedIntertrialWindowCenters{a, posIdx};
            numWindows = numel(windowDataList);
            
            lzcPerWindow = nan(1, numWindows);
            lzcNormalizedPerWindow = nan(1, numWindows);
            lzcNormalizedBernoulliPerWindow = nan(1, numWindows);  % Always initialize, even if not used
            
            % Compute LZC for each window
            for w = 1:numWindows
                wDataMat = windowDataList{w};  % [time bins x neurons]
                
                % Concatenate across neurons over time
                nNeurons = size(wDataMat, 2);
                nSamples = size(wDataMat, 1);
                concatenatedSeq = reshape(wDataMat', nSamples * nNeurons, 1);
                
                % Binarize: any value > 0 becomes 1, 0 stays 0
                binarySeq = double(concatenatedSeq > 0);
                
                % Calculate LZ complexity with controls
                [lzcPerWindow(w), lzcNormalizedPerWindow(w), lzcNormalizedBernoulliPerWindow(w)] = ...
                    compute_lz_complexity_with_controls(binarySeq, nShuffles, useBernoulliControl);
            end
            
            % Average LZC values across windows at this sliding position
            tempLzcMetricsIntertrial{idx}(posIdx) = nanmean(lzcPerWindow);
            tempLzcMetricsIntertrialNormalized{idx}(posIdx) = nanmean(lzcNormalizedPerWindow);
            if useBernoulliControl
                tempLzcMetricsIntertrialNormalizedBernoulli{idx}(posIdx) = nanmean(lzcNormalizedBernoulliPerWindow);
            end
            
            % Store per-window LZC values and centers for segment analysis
            tempPerWindowLZCIntertrial{idx}{posIdx} = lzcPerWindow;
            tempPerWindowLZCIntertrialNormalized{idx}{posIdx} = lzcNormalizedPerWindow;
            if useBernoulliControl
                tempPerWindowLZCIntertrialNormalizedBernoulli{idx}{posIdx} = lzcNormalizedBernoulliPerWindow;
            end
            tempPerWindowLZCIntertrialCenters{idx}{posIdx} = windowCenters;
        end
    end
    fprintf('\nArea %s done in %.1f min\n', areas{a}, toc/60);
end

% Map temporary arrays back to area-indexed arrays
for idx = 1:numAreasToProcess
    a = areasToTest(idx);
    lzcMetrics.reach{a} = tempLzcMetricsReach{idx};
    lzcMetrics.intertrial{a} = tempLzcMetricsIntertrial{idx};
    lzcMetrics.reachNormalized{a} = tempLzcMetricsReachNormalized{idx};
    lzcMetrics.intertrialNormalized{a} = tempLzcMetricsIntertrialNormalized{idx};
    if useBernoulliControl
        lzcMetrics.reachNormalizedBernoulli{a} = tempLzcMetricsReachNormalizedBernoulli{idx};
        lzcMetrics.intertrialNormalizedBernoulli{a} = tempLzcMetricsIntertrialNormalizedBernoulli{idx};
    end
    perWindowLZC.reach{a} = tempPerWindowLZCReach{idx};
    perWindowLZC.intertrial{a} = tempPerWindowLZCIntertrial{idx};
    perWindowLZC.reachNormalized{a} = tempPerWindowLZCReachNormalized{idx};
    perWindowLZC.intertrialNormalized{a} = tempPerWindowLZCIntertrialNormalized{idx};
    if useBernoulliControl
        perWindowLZC.reachNormalizedBernoulli{a} = tempPerWindowLZCReachNormalizedBernoulli{idx};
        perWindowLZC.intertrialNormalizedBernoulli{a} = tempPerWindowLZCIntertrialNormalizedBernoulli{idx};
    end
    perWindowLZC.reachCenters{a} = tempPerWindowLZCReachCenters{idx};
    perWindowLZC.intertrialCenters{a} = tempPerWindowLZCIntertrialCenters{idx};
end

% =============================   Engagement Segment-level LZC Analysis    =============================
fprintf('\n=== Segment-level LZC Analysis (Reach vs Intertrial) ===\n');

lzcSegmentMetrics = struct();
lzcSegmentMetrics.reach = cell(numAreas, nSegments);
lzcSegmentMetrics.intertrial = cell(numAreas, nSegments);
lzcSegmentMetrics.reachNormalized = cell(numAreas, nSegments);
lzcSegmentMetrics.intertrialNormalized = cell(numAreas, nSegments);
if useBernoulliControl
    lzcSegmentMetrics.reachNormalizedBernoulli = cell(numAreas, nSegments);
    lzcSegmentMetrics.intertrialNormalizedBernoulli = cell(numAreas, nSegments);
end

for a = areasToTest
    fprintf('\nProcessing segments for area %s...\n', areas{a});
    for s = 1:nSegments
        if isempty(segmentWindowsList) || s > length(segmentWindowsList) || isempty(segmentWindowsList{s}) || any(isnan(segmentWindowsList{s}))
            % Initialize to nan if segment is invalid
            lzcSegmentMetrics.reach{a, s} = nan;
            lzcSegmentMetrics.intertrial{a, s} = nan;
            lzcSegmentMetrics.reachNormalized{a, s} = nan;
            lzcSegmentMetrics.intertrialNormalized{a, s} = nan;
            if useBernoulliControl
                lzcSegmentMetrics.reachNormalizedBernoulli{a, s} = nan;
                lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s} = nan;
            end
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
        
        % Collect normalized LZC values from windows centered at alignment point (sliding position = 0)
        % Only include windows whose event times (reach or intertrial midpoint) fall in this segment
        reachLZCNormalizedInSegment = [];
        intertrialLZCNormalizedInSegment = [];
        if useBernoulliControl
            reachLZCNormalizedBernoulliInSegment = [];
            intertrialLZCNormalizedBernoulliInSegment = [];
        end
        
        % Process reach windows - use normalized LZC values from windows at alignment point
        if ~isempty(perWindowLZC.reachNormalized{a}{alignPosIdx}) && ~isempty(perWindowLZC.reachCenters{a}{alignPosIdx})
            lzcVals = perWindowLZC.reachNormalized{a}{alignPosIdx};
            windowCenters = perWindowLZC.reachCenters{a}{alignPosIdx};
            
            % Ensure windowCenters and lzcVals are row vectors for consistent indexing
            if isscalar(windowCenters)
                windowCenters = windowCenters(:)';
            end
            windowCenters = windowCenters(:)';  % Force row vector
            if isscalar(lzcVals)
                lzcVals = lzcVals(:)';
            end
            lzcVals = lzcVals(:)';  % Force row vector
            
            % Window centers at alignment point are the reach times
            % Find reaches whose times fall in this segment
            inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
            if any(inSegment)
                reachLZCNormalizedInSegment = [reachLZCNormalizedInSegment, lzcVals(inSegment)];
            end
        end
        
        % Process intertrial windows - use normalized LZC values from windows at alignment point
        if ~isempty(perWindowLZC.intertrialNormalized{a}{alignPosIdx}) && ~isempty(perWindowLZC.intertrialCenters{a}{alignPosIdx})
            lzcVals = perWindowLZC.intertrialNormalized{a}{alignPosIdx};
            windowCenters = perWindowLZC.intertrialCenters{a}{alignPosIdx};
            
            % Ensure windowCenters and lzcVals are row vectors for consistent indexing
            if isscalar(windowCenters)
                windowCenters = windowCenters(:)';
            end
            windowCenters = windowCenters(:)';  % Force row vector
            if isscalar(lzcVals)
                lzcVals = lzcVals(:)';
            end
            lzcVals = lzcVals(:)';  % Force row vector
            
            % Window centers at alignment point are the intertrial midpoint times
            % Find intertrial midpoints whose times fall in this segment
            inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
            if any(inSegment)
                intertrialLZCNormalizedInSegment = [intertrialLZCNormalizedInSegment, lzcVals(inSegment)];
            end
        end
        
        % Average normalized LZC values for reach vs intertrial windows in this segment
        if ~isempty(reachLZCNormalizedInSegment)
            lzcSegmentMetrics.reachNormalized{a, s} = nanmean(reachLZCNormalizedInSegment);
        else
            lzcSegmentMetrics.reachNormalized{a, s} = nan;
        end

        if ~isempty(intertrialLZCNormalizedInSegment)
            lzcSegmentMetrics.intertrialNormalized{a, s} = nanmean(intertrialLZCNormalizedInSegment);
        else
            lzcSegmentMetrics.intertrialNormalized{a, s} = nan;
        end
        
        % For compatibility, also store raw LZC values (from alignment point windows)
        if ~isempty(perWindowLZC.reach{a}{alignPosIdx}) && ~isempty(perWindowLZC.reachCenters{a}{alignPosIdx})
            lzcValsRaw = perWindowLZC.reach{a}{alignPosIdx};
            windowCenters = perWindowLZC.reachCenters{a}{alignPosIdx};
            if isscalar(windowCenters)
                windowCenters = windowCenters(:)';
            end
            windowCenters = windowCenters(:)';
            if isscalar(lzcValsRaw)
                lzcValsRaw = lzcValsRaw(:)';
            end
            lzcValsRaw = lzcValsRaw(:)';
            inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
            if any(inSegment)
                lzcSegmentMetrics.reach{a, s} = nanmean(lzcValsRaw(inSegment));
            else
                lzcSegmentMetrics.reach{a, s} = nan;
            end
        else
            lzcSegmentMetrics.reach{a, s} = nan;
        end
        
        if ~isempty(perWindowLZC.intertrial{a}{alignPosIdx}) && ~isempty(perWindowLZC.intertrialCenters{a}{alignPosIdx})
            lzcValsRaw = perWindowLZC.intertrial{a}{alignPosIdx};
            windowCenters = perWindowLZC.intertrialCenters{a}{alignPosIdx};
            if isscalar(windowCenters)
                windowCenters = windowCenters(:)';
            end
            windowCenters = windowCenters(:)';
            if isscalar(lzcValsRaw)
                lzcValsRaw = lzcValsRaw(:)';
            end
            lzcValsRaw = lzcValsRaw(:)';
            inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
            if any(inSegment)
                lzcSegmentMetrics.intertrial{a, s} = nanmean(lzcValsRaw(inSegment));
            else
                lzcSegmentMetrics.intertrial{a, s} = nan;
            end
        else
            lzcSegmentMetrics.intertrial{a, s} = nan;
        end
        
        % Process Bernoulli normalized values if enabled
        if useBernoulliControl
            if ~isempty(perWindowLZC.reachNormalizedBernoulli{a}{alignPosIdx}) && ~isempty(perWindowLZC.reachCenters{a}{alignPosIdx})
                lzcValsBernoulli = perWindowLZC.reachNormalizedBernoulli{a}{alignPosIdx};
                windowCenters = perWindowLZC.reachCenters{a}{alignPosIdx};
                if isscalar(windowCenters)
                    windowCenters = windowCenters(:)';
                end
                windowCenters = windowCenters(:)';
                if isscalar(lzcValsBernoulli)
                    lzcValsBernoulli = lzcValsBernoulli(:)';
                end
                lzcValsBernoulli = lzcValsBernoulli(:)';
                inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
                if any(inSegment)
                    lzcSegmentMetrics.reachNormalizedBernoulli{a, s} = nanmean(lzcValsBernoulli(inSegment));
                else
                    lzcSegmentMetrics.reachNormalizedBernoulli{a, s} = nan;
                end
            else
                lzcSegmentMetrics.reachNormalizedBernoulli{a, s} = nan;
            end
            
            if ~isempty(perWindowLZC.intertrialNormalizedBernoulli{a}{alignPosIdx}) && ~isempty(perWindowLZC.intertrialCenters{a}{alignPosIdx})
                lzcValsBernoulli = perWindowLZC.intertrialNormalizedBernoulli{a}{alignPosIdx};
                windowCenters = perWindowLZC.intertrialCenters{a}{alignPosIdx};
                if isscalar(windowCenters)
                    windowCenters = windowCenters(:)';
                end
                windowCenters = windowCenters(:)';
                if isscalar(lzcValsBernoulli)
                    lzcValsBernoulli = lzcValsBernoulli(:)';
                end
                lzcValsBernoulli = lzcValsBernoulli(:)';
                inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
                if any(inSegment)
                    lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s} = nanmean(lzcValsBernoulli(inSegment));
                else
                    lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s} = nan;
                end
            else
                lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s} = nan;
            end
        end
    end
end

%% =============================    Save Results    =============================
results = struct();
results.areas = areas;
results.reachStart = reachStart;
results.intertrialMidpoints = intertrialMidpoints;
results.slidingWindowSize = slidingWindowSize;  % Area-specific window sizes
results.windowBuffer = windowBuffer;
results.beforeAlign = beforeAlign;
results.afterAlign = afterAlign;
results.stepSize = stepSize;
results.slidingPositions = slidingPositions;
results.lzcMetrics = lzcMetrics;
results.lzcSegmentMetrics = lzcSegmentMetrics;
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
results.params.nShuffles = nShuffles;
results.params.useBernoulliControl = useBernoulliControl;
results.params.useOptimalBinSize = useOptimalBinSize;
results.params.useOptimalWindowSize = useOptimalWindowSize;
results.params.minSpikesPerBin = minSpikesPerBin;
results.params.minDataPoints = minDataPoints;
results.params.minSlidingWindowSize = minSlidingWindowSize;
results.params.maxSlidingWindowSize = maxSlidingWindowSize;
results.params.minBinSize = minBinSize;
results.params.nMinNeurons = nMinNeurons;
results.params.includeM2356 = includeM2356;

% Use mean window size for filename
meanWindowSize = mean(slidingWindowSize(areasToTest));
resultsPath = fullfile(saveDir, sprintf('lzc_reach_intertrial_win%.1f_step%.1f.mat', meanWindowSize, stepSize));
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
        resultsFiles = dir(fullfile(saveDir, 'lzc_reach_intertrial_win*.mat'));
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
    lzcMetrics = results.lzcMetrics;
    lzcSegmentMetrics = results.lzcSegmentMetrics;
    segmentNames = results.segmentNames;
    binSize = results.binSize;
    useBernoulliControl = results.params.useBernoulliControl;
    includeM2356 = results.params.includeM2356;
    if isfield(results, 'sessionName')
        sessionName = results.sessionName;
    end
    if isfield(results, 'idMatIdx')
        idMatIdx = results.idMatIdx;
    end
    
    % Reconstruct M2356 in idMatIdx if it exists in results but not in idMatIdx
    % (This can happen if results were saved with M2356 but idMatIdx wasn't updated)
    if includeM2356 && any(strcmp(areas, 'M2356'))
        m2356Idx = find(strcmp(areas, 'M2356'));
        if ~isempty(m2356Idx) && (isempty(idMatIdx) || length(idMatIdx) < m2356Idx || isempty(idMatIdx{m2356Idx}))
            % Try to reconstruct M2356 from M23 and M56
            idxM23 = find(strcmp(areas, 'M23'));
            idxM56 = find(strcmp(areas, 'M56'));
            if ~isempty(idxM23) && ~isempty(idxM56) && ~isempty(idMatIdx)
                if idxM23 <= length(idMatIdx) && idxM56 <= length(idMatIdx) && ...
                        ~isempty(idMatIdx{idxM23}) && ~isempty(idMatIdx{idxM56})
                    % Ensure idMatIdx is large enough
                    if length(idMatIdx) < m2356Idx
                        idMatIdx{m2356Idx} = [];
                    end
                    idMatIdx{m2356Idx} = [idMatIdx{idxM23}(:); idMatIdx{idxM56}(:)];
                    fprintf('Reconstructed M2356 in idMatIdx for plotting (n=%d neurons)\n', length(idMatIdx{m2356Idx}));
                end
            end
        end
    end
    
    % Determine nSegments from segmentNames
    if ~isempty(segmentNames)
        nSegments = numel(segmentNames);
    else
        nSegments = 0;
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
    if ~exist('lzcMetrics', 'var')
        error('lzcMetrics not found in workspace. Run analysis section first or set loadResultsForPlotting = true.');
    end
    if ~exist('lzcSegmentMetrics', 'var')
        error('lzcSegmentMetrics not found in workspace. Run analysis section first or set loadResultsForPlotting = true.');
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

% Collect all y-values to determine shared y-axis limits
allYVals = [];
for a = areasToTest
    if useBernoulliControl
        reachVals = lzcMetrics.reachNormalizedBernoulli{a};
        intertrialVals = lzcMetrics.intertrialNormalizedBernoulli{a};
    else
        reachVals = lzcMetrics.reachNormalized{a};
        intertrialVals = lzcMetrics.intertrialNormalized{a};
    end
    allYVals = [allYVals, reachVals(~isnan(reachVals)), intertrialVals(~isnan(intertrialVals))];
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
    
    % Extract reach and intertrial LZC values across sliding positions
    % Get number of neurons for this area
    if exist('idMatIdx', 'var') && ~isempty(idMatIdx) && a <= length(idMatIdx) && ~isempty(idMatIdx{a})
        numNeurons = length(idMatIdx{a});
        neuronStr = sprintf(' (n=%d)', numNeurons);
    else
        neuronStr = '';
    end
    
    if useBernoulliControl
        reachVals = lzcMetrics.reachNormalizedBernoulli{a};
        intertrialVals = lzcMetrics.intertrialNormalizedBernoulli{a};
        yLabelStr = 'LZC (Bernoulli normalized)';
        titleStr = sprintf('%s%s - Sliding Positions', areas{a}, neuronStr);
    else
        reachVals = lzcMetrics.reachNormalized{a};
        intertrialVals = lzcMetrics.intertrialNormalized{a};
        yLabelStr = 'LZC (normalized)';
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
    % Calculate y-axis limits for segment plots
    segmentYVals = [];
    for a = areasToTest
        nSeg = numel(segmentNames);
        for s = 1:nSeg
            if useBernoulliControl
                if ~isempty(lzcSegmentMetrics.reachNormalizedBernoulli{a, s}) && ~isnan(lzcSegmentMetrics.reachNormalizedBernoulli{a, s})
                    segmentYVals = [segmentYVals, lzcSegmentMetrics.reachNormalizedBernoulli{a, s}];
                end
                if ~isempty(lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s}) && ~isnan(lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s})
                    segmentYVals = [segmentYVals, lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s}];
                end
            else
                if ~isempty(lzcSegmentMetrics.reachNormalized{a, s}) && ~isnan(lzcSegmentMetrics.reachNormalized{a, s})
                    segmentYVals = [segmentYVals, lzcSegmentMetrics.reachNormalized{a, s}];
                end
                if ~isempty(lzcSegmentMetrics.intertrialNormalized{a, s}) && ~isnan(lzcSegmentMetrics.intertrialNormalized{a, s})
                    segmentYVals = [segmentYVals, lzcSegmentMetrics.intertrialNormalized{a, s}];
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
        
        % Extract reach and intertrial LZC values across segments
        nSeg = numel(segmentNames);
        reachVals = nan(1, nSeg);
        intertrialVals = nan(1, nSeg);
        for s = 1:nSeg
            if useBernoulliControl
                if ~isempty(lzcSegmentMetrics.reachNormalizedBernoulli{a, s})
                    reachVals(s) = lzcSegmentMetrics.reachNormalizedBernoulli{a, s};
                else
                    reachVals(s) = nan;
                end
                if ~isempty(lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s})
                    intertrialVals(s) = lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s};
                else
                    intertrialVals(s) = nan;
                end
            else
                if ~isempty(lzcSegmentMetrics.reachNormalized{a, s})
                    reachVals(s) = lzcSegmentMetrics.reachNormalized{a, s};
                else
                    reachVals(s) = nan;
                end
                if ~isempty(lzcSegmentMetrics.intertrialNormalized{a, s})
                    intertrialVals(s) = lzcSegmentMetrics.intertrialNormalized{a, s};
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
            if useBernoulliControl
                ylabel('LZC (Bernoulli normalized)', 'FontSize', 10);
            else
                ylabel('LZC (normalized)', 'FontSize', 10);
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
        
        % Add horizontal line at y = 1
        yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
    end
end

% Add overall title
if exist('sessionName', 'var') && ~isempty(sessionName)
    sessionNameShort = sessionName(1:min(10, length(sessionName)));
    titlePrefix = [sessionNameShort, ' - '];
else
    titlePrefix = '';
end
meanWindowSize = mean(slidingWindowSize(areasToTest));
if useBernoulliControl
    sgtitle(sprintf('%sLZC (Bernoulli normalized): Reach vs Intertrial (Window: %.1fs, Buffer: %.1fs)', ...
        titlePrefix, meanWindowSize, windowBuffer), 'FontSize', 14, 'interpreter', 'none');
else
    sgtitle(sprintf('%sLZC (normalized): Reach vs Intertrial (Window: %.1fs, Buffer: %.1fs)', ...
        titlePrefix, meanWindowSize, windowBuffer), 'FontSize', 14, 'interpreter', 'none');
end

% Save figure
saveFile = fullfile(saveDir, sprintf('lzc_reach_intertrial_all_areas_win%.1f_step%.1f.png', meanWindowSize, stepSize));
exportgraphics(gcf, saveFile, 'Resolution', 300);
fprintf('Saved combined plot to: %s\n', saveFile);

fprintf('\n=== Analysis Complete ===\n');

%% =============================    Helper Functions    =============================

function [lzComplexity, lzNormalized, lzNormalizedBernoulli] = compute_lz_complexity_with_controls(binarySeq, nShuffles, useBernoulliControl)
% COMPUTE_LZ_COMPLEXITY_WITH_CONTROLS Compute LZ complexity with shuffle and optional Bernoulli controls
%
% Variables:
%   binarySeq - Binary sequence
%   nShuffles - Number of shuffles for normalization
%   useBernoulliControl - Whether to compute Bernoulli normalized metric (default: false)
    
    if nargin < 3
        useBernoulliControl = false;
    end
    
    try
        % Calculate original LZ complexity
        lzComplexity = limpel_ziv_complexity(binarySeq, 'method', 'binary');
        
        % Normalize by shuffled version
        shuffledLZ = nan(1, nShuffles);
        for s = 1:nShuffles
            shuffledSeq = binarySeq(randperm(length(binarySeq)));
            shuffledLZ(s) = limpel_ziv_complexity(shuffledSeq, 'method', 'binary');
        end
        
        meanShuffledLZ = nanmean(shuffledLZ);
        if meanShuffledLZ > 0
            lzNormalized = lzComplexity / meanShuffledLZ;
        else
            lzNormalized = nan;
        end
        
        % Normalize by rate-matched Bernoulli control (optional)
        if useBernoulliControl
            firingRate = mean(binarySeq);
            bernoulliLZ = nan(1, nShuffles);
            for s = 1:nShuffles
                bernoulliSeq = double(rand(length(binarySeq), 1) < firingRate);
                bernoulliLZ(s) = limpel_ziv_complexity(bernoulliSeq, 'method', 'binary');
            end
            
            meanBernoulliLZ = nanmean(bernoulliLZ);
            if meanBernoulliLZ > 0
                lzNormalizedBernoulli = lzComplexity / meanBernoulliLZ;
            else
                lzNormalizedBernoulli = nan;
            end
        else
            lzNormalizedBernoulli = nan;  % Not computed
        end
    catch ME
        fprintf('    Warning: Error computing LZ complexity: %s\n', ME.message);
        lzComplexity = nan;
        lzNormalized = nan;
        lzNormalizedBernoulli = nan;
    end
end
