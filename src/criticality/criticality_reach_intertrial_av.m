%%
% Criticality Reach vs Intertrial Avalanche Analysis
% Compares avalanche analyses between reaches and intertrial intervals
% Performs sliding window avalanche analysis around reach starts and intertrial midpoints
%
% Variables:
%   reachStart - Reach start times in seconds
%   intertrialMidpoints - Midpoint times between consecutive reaches
%   avalancheWindow - Optimal window duration for avalanche analysis
%   windowBuffer - Minimum distance from window edge to event/midpoint
%   beforeAlign - Start of sliding window range (seconds before alignment point)
%   afterAlign - End of sliding window range (seconds after alignment point)
%   stepSize - Step size for sliding window (seconds)


%% =============================    Data Loading    =============================
paths = get_paths;

fprintf('\n=== Loading Reach Data ===\n');
% Reach data file
reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
% reachDataFile = fullfile(paths.reachDataPath, 'makeSpikes.mat');

[~, dataBaseName, ~] = fileparts(reachDataFile);
saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

dataR = load(reachDataFile);

opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = round(min(dataR.R(end,1) + 8000, max(dataR.CSV(:,1)*1000)) / 1000);
opts.minFiringRate = .1;
opts.maxFiringRate = 70;

[dataMat, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
areas = {'M23', 'M56', 'DS', 'VS'};
idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));
idList = {idM23, idM56, idDS, idVS};


%% =============================    Configuration    =============================
% Extract reach start times (in seconds)
reachStart = dataR.R(:,1) / 1000; % Convert from ms to seconds
totalReaches = length(reachStart);

fprintf('Loaded %d reaches\n', totalReaches);

% Calculate intertrial midpoints (halfway between consecutive reaches)
intertrialMidpoints = nan(1, totalReaches - 1);
for i = 1:totalReaches - 1
    intertrialMidpoints(i) = (reachStart(i) + reachStart(i+1)) / 2;
end
fprintf('Calculated %d intertrial midpoints\n', length(intertrialMidpoints));

% Sliding window parameters
beforeAlign = -2;  % Start sliding from this many seconds before alignment point
afterAlign = 2;    % End sliding at this many seconds after alignment point
binSize = .05;
stepSize = .25;       % Step size for sliding window (seconds)
windowBuffer = .5;   % Minimum distance from window edge to event/midpoint (seconds)
% Minimum avalanche window size (seconds)
minAvalancheWindow = 8;

% Areas to analyze
areasToTest = 1:4;

% PCA options
pcaFlag = 0;           % Set to 1 to use PCA
pcaFirstFlag = 1;      % Use first nDim dimensions if 1, last nDim if 0
nDim = 4;              % Number of PCA dimensions to use

% Threshold options
thresholdFlag = 1;     % Set to 1 to use threshold method
thresholdPct = 0.75;   % Threshold as percentage of median

% Plotting options
loadResultsForPlotting = false;  % Set to true to load saved results for plotting
                                  % Set to false to use variables from workspace
resultsFileForPlotting = '';     % Path to results file (empty = auto-detect from saveDir)
makePlots = true;                 % Set to true to generate plots

%% =============================    Find Optimal Avalanche Window    =============================
fprintf('\n=== Finding Optimal Avalanche Window ===\n');


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
        % Constraint: reachTime - avalancheWindow/2 >= prevMidpoint + windowBuffer
        % Rearranging: avalancheWindow <= 2 * (reachTime - prevMidpoint - windowBuffer)
        maxWindowFromPrev = 2 * (reachTime - prevMidpoint - windowBuffer);
        if maxWindowFromPrev < maxWindowForThisReach
            maxWindowForThisReach = maxWindowFromPrev;
        end
    end
    
    if r <= length(intertrialMidpoints)
        nextMidpoint = intertrialMidpoints(r);
        % Constraint: reachTime + avalancheWindow/2 <= nextMidpoint - windowBuffer
        % Rearranging: avalancheWindow <= 2 * (nextMidpoint - reachTime - windowBuffer)
        maxWindowFromNext = 2 * (nextMidpoint - reachTime - windowBuffer);
        if maxWindowFromNext < maxWindowForThisReach
            maxWindowForThisReach = maxWindowFromNext;
        end
    end
    
    maxWindowPerReach(r) = maxWindowForThisReach;
    
    % Keep this reach if its maximum window is >= minAvalancheWindow
    if maxWindowForThisReach >= minAvalancheWindow
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
    
    % Constraint: midpointTime - avalancheWindow/2 >= prevReach + windowBuffer
    % Rearranging: avalancheWindow <= 2 * (midpointTime - prevReach - windowBuffer)
    maxWindowFromPrev = 2 * (midpointTime - prevReach - windowBuffer);
    if maxWindowFromPrev < maxWindowForThisIntertrial
        maxWindowForThisIntertrial = maxWindowFromPrev;
    end
    
    % Constraint: midpointTime + avalancheWindow/2 <= nextReach - windowBuffer
    % Rearranging: avalancheWindow <= 2 * (nextReach - midpointTime - windowBuffer)
    maxWindowFromNext = 2 * (nextReach - midpointTime - windowBuffer);
    if maxWindowFromNext < maxWindowForThisIntertrial
        maxWindowForThisIntertrial = maxWindowFromNext;
    end
    
    maxWindowPerIntertrial(i) = maxWindowForThisIntertrial;
    
    % Keep this intertrial midpoint if its maximum window is >= minAvalancheWindow
    if maxWindowForThisIntertrial >= minAvalancheWindow
        validIntertrialIndices = [validIntertrialIndices, i];
    end
end

% Calculate avalancheWindow from valid events only
if isempty(validReachIndices) && isempty(validIntertrialIndices)
    error('No valid reaches or intertrial midpoints found with window size >= %.1f seconds. Try reducing windowBuffer or minAvalancheWindow.', minAvalancheWindow);
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
avalancheWindow = min(maxReachWindow, maxIntertrialWindow);

% Also need to consider the sliding range
maxWindowFromSliding = abs(beforeAlign) + abs(afterAlign);
avalancheWindow = min(avalancheWindow, maxWindowFromSliding);

% Ensure avalancheWindow is at least minAvalancheWindow
if avalancheWindow < minAvalancheWindow
    avalancheWindow = minAvalancheWindow;
end

% Ensure avalancheWindow is positive and reasonable
if avalancheWindow <= 0 || isnan(avalancheWindow) || isinf(avalancheWindow)
    error('Cannot find valid avalanche window. Try reducing windowBuffer or increasing inter-reach intervals.');
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

fprintf('Optimal avalanche window: %.2f seconds\n', avalancheWindow);
fprintf('  Minimum avalanche window: %.2f seconds\n', minAvalancheWindow);
fprintf('  Valid reaches: %d/%d (%.1f%%)\n', length(validReachIndices), length(maxWindowPerReach), 100*length(validReachIndices)/length(maxWindowPerReach));
fprintf('  Valid intertrial midpoints: %d/%d (%.1f%%)\n', length(intertrialMidpoints), length(maxWindowPerIntertrial), 100*length(intertrialMidpoints)/length(maxWindowPerIntertrial));
fprintf('  Window buffer: %.2f seconds\n', windowBuffer);

%% =============================    Analysis    =============================
fprintf('\n=== Processing Areas ===\n');

% Adjust areasToTest based on which areas have data
areasToTest = areasToTest(~cellfun(@isempty, idList));

% Step 1-2: Apply PCA to original data if requested
fprintf('\n--- Step 1-2: PCA on original data if requested ---\n');
reconstructedDataMat = cell(1, length(areas));
for a = areasToTest
    aID = idList{a}; 
    thisDataMat = dataMat(:, aID);
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


% Calculate sliding window positions
slidingPositions = beforeAlign:stepSize:afterAlign;
numSlidingPositions = length(slidingPositions);

% Initialize storage for avalanche metrics
% Structure: {area}{eventType}{metric}[slidingPosition]
% eventType: 'reach' or 'intertrial'
% metric: 'kappa', 'dcc', 'tau', 'alpha', 'paramSD', 'decades'
% Note: Metrics are calculated on collective data across all events for each sliding position
avalancheMetrics = struct();
avalancheMetrics.reach = struct();
avalancheMetrics.intertrial = struct();

% Initialize as cell arrays indexed by area
avalancheMetrics.reach.kappa = cell(1, length(areas));
avalancheMetrics.reach.dcc = cell(1, length(areas));
avalancheMetrics.reach.tau = cell(1, length(areas));
avalancheMetrics.reach.alpha = cell(1, length(areas));
avalancheMetrics.reach.paramSD = cell(1, length(areas));
avalancheMetrics.reach.decades = cell(1, length(areas));

avalancheMetrics.intertrial.kappa = cell(1, length(areas));
avalancheMetrics.intertrial.dcc = cell(1, length(areas));
avalancheMetrics.intertrial.tau = cell(1, length(areas));
avalancheMetrics.intertrial.alpha = cell(1, length(areas));
avalancheMetrics.intertrial.paramSD = cell(1, length(areas));
avalancheMetrics.intertrial.decades = cell(1, length(areas));

for a = areasToTest
    avalancheMetrics.reach.kappa{a} = nan(1, numSlidingPositions);
    avalancheMetrics.reach.dcc{a} = nan(1, numSlidingPositions);
    avalancheMetrics.reach.tau{a} = nan(1, numSlidingPositions);
    avalancheMetrics.reach.alpha{a} = nan(1, numSlidingPositions);
    avalancheMetrics.reach.paramSD{a} = nan(1, numSlidingPositions);
    avalancheMetrics.reach.decades{a} = nan(1, numSlidingPositions);
    
    avalancheMetrics.intertrial.kappa{a} = nan(1, numSlidingPositions);
    avalancheMetrics.intertrial.dcc{a} = nan(1, numSlidingPositions);
    avalancheMetrics.intertrial.tau{a} = nan(1, numSlidingPositions);
    avalancheMetrics.intertrial.alpha{a} = nan(1, numSlidingPositions);
    avalancheMetrics.intertrial.paramSD{a} = nan(1, numSlidingPositions);
    avalancheMetrics.intertrial.decades{a} = nan(1, numSlidingPositions);
end

% Process each area
for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    tic;
    
    aID = idList{a};
    
    % Bin the data for avalanche analysis
    aDataMat = neural_matrix_ms_to_frames(dataMat(:, aID), binSize);
    numTimePoints = size(aDataMat, 1);
    timePoints = (0:numTimePoints-1) * binSize; % Time axis in seconds
    
    % Apply PCA to binned data if requested
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
        forDim = find(cumsum(explained) > 30, 1); 
        forDim = max(3, min(6, forDim));
        nDim = 1:forDim; 
        aDataMat = score(:,nDim) * coeff(:,nDim)' + mu;
    end
    
% Sum the spikes across neurons
    aDataMat = round(sum(aDataMat, 2));
    
    % Convert window duration to samples
    avalancheWindowSamples = round(avalancheWindow / binSize);
    
    % Initialize storage for collected windows
    % For each sliding position, collect all valid windows
    collectedReachWindows = cell(1, numSlidingPositions);
    collectedIntertrialWindows = cell(1, numSlidingPositions);
    
    % Step 1: Collect all reach-aligned windows
    fprintf('  Collecting reach-aligned windows...\n');
    for posIdx = 1:numSlidingPositions
        offset = slidingPositions(posIdx);
        collectedReachWindows{posIdx} = [];
        
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
            winStart = windowCenter - avalancheWindow / 2;
            winEnd = windowCenter + avalancheWindow / 2;
            
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
            
            % Find indices in timePoints
            [~, startIdx] = min(abs(timePoints - winStart));
            [~, endIdx] = min(abs(timePoints - winEnd));
            
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wPopActivity = aDataMat(startIdx:endIdx);
                % Store this window (as a row vector)
                collectedReachWindows{posIdx} = [collectedReachWindows{posIdx}; wPopActivity(:)'];
            end
        end
    end
    
    % Step 2: Collect all intertrial-aligned windows
    fprintf('  Collecting intertrial-aligned windows...\n');
    for posIdx = 1:numSlidingPositions
        offset = slidingPositions(posIdx);
        collectedIntertrialWindows{posIdx} = [];
        
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
            winStart = windowCenter - avalancheWindow / 2;
            winEnd = windowCenter + avalancheWindow / 2;
            
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
            
            % Find indices in timePoints
            [~, startIdx] = min(abs(timePoints - winStart));
            [~, endIdx] = min(abs(timePoints - winEnd));
            
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wPopActivity = aDataMat(startIdx:endIdx);
                % Store this window (as a row vector)
                collectedIntertrialWindows{posIdx} = [collectedIntertrialWindows{posIdx}; wPopActivity(:)'];
            end
        end
    end
    
    % Initialize storage for individual windows (after thresholding and trimming)
    if ~exist('storedReachWindows', 'var')
        storedReachWindows = cell(length(areas), numSlidingPositions);
        storedIntertrialWindows = cell(length(areas), numSlidingPositions);
    end
    
    % Step 3: Perform avalanche analysis on collected windows
    fprintf('  Performing avalanche analysis on collected windows...\n');
    
    % Process reach-aligned windows
    for posIdx = 1:numSlidingPositions
        if ~isempty(collectedReachWindows{posIdx})
            % Calculate median across all reach windows for thresholding
            allReachData = collectedReachWindows{posIdx};
            if thresholdFlag
                medianReachData = median(allReachData(:));
                allReachData(allReachData < thresholdPct * medianReachData) = 0;
            end
            
            % Prepare storage for individual trimmed windows
            reachWindowsThisPos = {};
            concatenatedReachData = [];
            
            % For each window, trim so it begins and ends with a zero bin
            for w = 1:size(allReachData, 1)
                thisWindow = allReachData(w, :);
                zeroIdx = find(thisWindow == 0);
                if numel(zeroIdx) >= 2
                    firstZero = zeroIdx(1);
                    lastZero = zeroIdx(end);
                    if lastZero > firstZero
                        trimmedWindow = thisWindow(firstZero:lastZero);
                        % Store individual window
                        reachWindowsThisPos{end+1,1} = trimmedWindow(:); %#ok<AGROW>
                        % Concatenate for avalanche analysis
                        concatenatedReachData = [concatenatedReachData; trimmedWindow(:)]; %#ok<AGROW>
                    end
                end
            end
            
            % Store all individual windows for later analysis
            storedReachWindows{a, posIdx} = reachWindowsThisPos;
            
            % Perform avalanche analysis on concatenated data (if any)
            if ~isempty(concatenatedReachData)
                [kappa, dcc, tau, alpha, paramSD, decades] = perform_avalanche_analysis(concatenatedReachData, binSize);
                
                avalancheMetrics.reach.kappa{a}(posIdx) = kappa;
                avalancheMetrics.reach.dcc{a}(posIdx) = dcc;
                avalancheMetrics.reach.tau{a}(posIdx) = tau;
                avalancheMetrics.reach.alpha{a}(posIdx) = alpha;
                avalancheMetrics.reach.paramSD{a}(posIdx) = paramSD;
                avalancheMetrics.reach.decades{a}(posIdx) = decades;
            end
        end
    end
    
    % Process intertrial-aligned windows
    for posIdx = 1:numSlidingPositions
        if ~isempty(collectedIntertrialWindows{posIdx})
            % Calculate median across all intertrial windows for thresholding
            allIntertrialData = collectedIntertrialWindows{posIdx};
            if thresholdFlag
                medianIntertrialData = median(allIntertrialData(:));
                allIntertrialData(allIntertrialData < thresholdPct * medianIntertrialData) = 0;
            end
            
            % Prepare storage for individual trimmed windows
            intertrialWindowsThisPos = {};
            concatenatedIntertrialData = [];
            
            % For each window, trim so it begins and ends with a zero bin
            for w = 1:size(allIntertrialData, 1)
                thisWindow = allIntertrialData(w, :);
                zeroIdx = find(thisWindow == 0);
                if numel(zeroIdx) >= 2
                    firstZero = zeroIdx(1);
                    lastZero = zeroIdx(end);
                    if lastZero > firstZero
                        trimmedWindow = thisWindow(firstZero:lastZero);
                        % Store individual window
                        intertrialWindowsThisPos{end+1,1} = trimmedWindow(:); %#ok<AGROW>
                        % Concatenate for avalanche analysis
                        concatenatedIntertrialData = [concatenatedIntertrialData; trimmedWindow(:)]; %#ok<AGROW>
                    end
                end
            end
            
            % Store all individual windows for later analysis
            storedIntertrialWindows{a, posIdx} = intertrialWindowsThisPos;
            
            % Perform avalanche analysis on concatenated data (if any)
            if ~isempty(concatenatedIntertrialData)
                [kappa, dcc, tau, alpha, paramSD, decades] = perform_avalanche_analysis(concatenatedIntertrialData, binSize);
                
                avalancheMetrics.intertrial.kappa{a}(posIdx) = kappa;
                avalancheMetrics.intertrial.dcc{a}(posIdx) = dcc;
                avalancheMetrics.intertrial.tau{a}(posIdx) = tau;
                avalancheMetrics.intertrial.alpha{a}(posIdx) = alpha;
                avalancheMetrics.intertrial.paramSD{a}(posIdx) = paramSD;
                avalancheMetrics.intertrial.decades{a}(posIdx) = decades;
            end
        end
    end
    
    fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end

% =============================    Save Results    =============================
results = struct();
results.areas = areas;
results.reachStart = reachStart;
results.intertrialMidpoints = intertrialMidpoints;
results.avalancheWindow = avalancheWindow;
results.windowBuffer = windowBuffer;
results.beforeAlign = beforeAlign;
results.afterAlign = afterAlign;
results.stepSize = stepSize;
results.slidingPositions = slidingPositions;
results.avalancheMetrics = avalancheMetrics;
if exist('storedReachWindows', 'var')
    results.windows.reach = storedReachWindows;
    results.windows.intertrial = storedIntertrialWindows;
end
results.binSize = binSize;
results.params.pcaFlag = pcaFlag;
results.params.pcaFirstFlag = pcaFirstFlag;
results.params.nDim = nDim;
results.params.thresholdFlag = thresholdFlag;
results.params.thresholdPct = thresholdPct;

resultsPath = fullfile(saveDir, sprintf('criticality_reach_intertrial_av_win%.1f_step%.1f.mat', avalancheWindow, stepSize));
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
        resultsFiles = dir(fullfile(saveDir, 'criticality_reach_intertrial_av_win*.mat'));
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
    avalancheWindow = results.avalancheWindow;
    windowBuffer = results.windowBuffer;
    beforeAlign = results.beforeAlign;
    afterAlign = results.afterAlign;
    stepSize = results.stepSize;
    slidingPositions = results.slidingPositions;
    avalancheMetrics = results.avalancheMetrics;
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
    if ~exist('avalancheMetrics', 'var') || ~exist('slidingPositions', 'var')
        error('Required variables not found in workspace. Set loadResultsForPlotting = true to load from saved results.');
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

% Define metrics to plot
metricNames = {'kappa', 'dcc', 'tau', 'alpha', 'paramSD', 'decades'};
numMetrics = length(metricNames);

% Detect monitors and size figure to full screen (prefer second monitor if present)
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    targetPos = monitorTwo;
else
    targetPos = monitorOne;
end

% Create summary plot for each area
for a = areasToTest
    figure(1000 + a); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', targetPos);
    
    % Create subplots: 1 row x numMetrics columns
    ha = tight_subplot(1, numMetrics, [0.08 0.04], [0.15 0.1], [0.08 0.04]);
    
    for m = 1:numMetrics
        metricName = metricNames{m};
        axes(ha(m));
        hold on;
        
        % Extract reach and intertrial data for this metric
        reachData = avalancheMetrics.reach.(metricName){a};
        intertrialData = avalancheMetrics.intertrial.(metricName){a};
        
        % Plot reach data
        if ~isempty(reachData) && any(~isnan(reachData))
            plot(slidingPositions, reachData, 'b-', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 6, 'DisplayName', 'Reach');
        end
        
        % Plot intertrial data
        if ~isempty(intertrialData) && any(~isnan(intertrialData))
            plot(slidingPositions, intertrialData, 'r-', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 6, 'DisplayName', 'Intertrial');
        end
        
        % Formatting
        xlabel('Time relative to alignment (s)', 'FontSize', 10);
        ylabel(metricName, 'FontSize', 10);
        title(sprintf('%s - %s', areas{a}, metricName), 'FontSize', 12);
        grid on;
        legend('Location', 'best', 'FontSize', 8);
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
    end
    
    sgtitle(sprintf('%s - Avalanche Metrics: Reach vs Intertrial (Window: %.1fs, Buffer: %.1fs)', areas{a}, avalancheWindow, windowBuffer), 'FontSize', 14);
    
    % Save figure
    saveFile = fullfile(saveDir, sprintf('criticality_reach_intertrial_av_%s_summary_win%.1f_step%.1f.png', areas{a}, avalancheWindow, stepSize));
    exportgraphics(gcf, saveFile, 'Resolution', 300);
    fprintf('Saved summary plot for %s to: %s\n', areas{a}, saveFile);
end

fprintf('\n=== Analysis Complete ===\n');

%% =============================    Helper Functions    =============================

function [kappa, dcc, tau, alpha, paramSD, decades] = perform_avalanche_analysis(wPopActivity, binSize)
% Perform avalanche analysis on concatenated population activity
% Note: Thresholding is already applied before calling this function
% Inputs:
%   wPopActivity - concatenated population activity (column vector, already thresholded)
%   binSize - bin size in seconds
%   medianData - median activity (not used, kept for compatibility)
%   thresholdPct - threshold percentage (not used, kept for compatibility)
% Returns: kappa, dcc, tau, alpha, paramSD, decades

kappa = nan;
dcc = nan;
tau = nan;
alpha = nan;
paramSD = nan;
decades = nan;

% Data is already thresholded, so use it directly
% Check if there are enough zero bins for avalanche detection
zeroBins = find(wPopActivity == 0);
if length(zeroBins) > 1 && any(diff(zeroBins)>1)
    try
        % Create avalanche data
        % asdfMat = rastertoasdf2(wPopActivity', binSize*1000, 'CBModel', 'Spikes', 'DS');
        % Av = avprops(asdfMat, 'ratio', 'fingerprint');
        % 
        % % Calculate avalanche parameters
        % [tau, pSz, tauC, pSzC, alpha, pDr, paramSD, decades] = avalanche_log(Av, 0);
        
            % Woody's new code:
[sizes, durs] = getAvalanches(wPopActivity', .5, 1);
gof = .8;
plotAv = 0;
[tau, plrS, minavS, maxavS, ~, ~, ~] = plfit2023(sizes, gof, plotAv, 0);
[alpha, plrD, minavD, maxavD, ~, ~, ~] = plfit2023(durs, gof, plotAv, 0);
[paramSD, sigmaNuZInvStd, logCoeff] = size_given_duration(sizes, durs, 'durmin', minavD, 'durmax', maxavD)

        % dcc (distance to criticality from avalanche analysis)
        dcc = distance_to_criticality(tau, alpha, paramSD);
        
        % kappa (avalanche shape parameter)
        kappa = compute_kappa(sizes);
    catch
        % Leave as NaN if calculation fails
    end
end
end

function [tau, pSz, tauC, pSzC, alpha, pDr, paramSD, decades] = avalanche_log(Av, plotFlag)
% Calculate avalanche parameters from avalanche structure
% (Copied from criticality_compare_av.m)

if plotFlag == 1
    plotFlag = 'plot';
else
    plotFlag = 'nothing';
end

% size distribution (SZ)
[tau, xminSZ, xmaxSZ, sigmaSZ, pSz, pCritSZ, ksDR, DataSZ] =...
    avpropvals(Av.size, 'size', plotFlag);
tau = cell2mat(tau);
pSz = cell2mat(pSz);

decades = log10(cell2mat(xmaxSZ)/cell2mat(xminSZ));

% size distribution (SZ) with cutoffs
tauC = nan;
pSzC = nan;
UniqSizes = unique(Av.size);
Occurances = hist(Av.size,UniqSizes);
AllowedSizes = UniqSizes(Occurances >= 4);
AllowedSizes(AllowedSizes < 4) = [];
if length(AllowedSizes) > 1
    LimSize = Av.size(ismember(Av.size,AllowedSizes));
    [tauC, xminSZ, xmaxSZ, sigmaSZ, pSzC, pCritSZ, DataSZ] =...
        avpropvals(LimSize, 'size', plotFlag);
    tauC = cell2mat(tauC);
    pSzC = cell2mat(pSzC);
end

% duration distribution (DR)
if length(unique(Av.duration)) > 1
    [alpha, xminDR, xmaxDR, sigmaDR, pDr, pCritDR, ksDR, DataDR] =...
        avpropvals(Av.duration, 'duration', plotFlag);
    alpha = cell2mat(alpha);
    pDr = cell2mat(pDr);
    % size given duration distribution (SD)
    [paramSD, waste, waste, sigmaSD] = avpropvals({Av.size, Av.duration},...
        'sizgivdur', 'durmin', xminDR{1}, 'durmax', xmaxDR{1}, plotFlag);
else
    alpha = nan;
    paramSD = nan;
end
end
