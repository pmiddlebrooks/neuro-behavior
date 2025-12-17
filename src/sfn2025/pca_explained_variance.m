%% PCA_EXPLAINED_VARIANCE - Assess explained variance of neural activity using PCA
%
% This script performs PCA analysis on reach and naturalistic neural data
% to assess how many components are needed to explain variance in each brain area.
%
% Two analyses:
% 1. Whole session: All data without event alignment
% 2. Event-aligned: Windows around events
%    - Reach: All reach onsets
%    - Naturalistic: Onsets of bhvID(s) specified in natBhvID (can be vector to collapse multiple behaviors)
%
% Outputs:
% - Cumulative explained variance plots (1x4 figure, one per area)
% - Number of components needed to explain at least 50% variance

% clear; close all;

%% Parameters
binSize = 0.05; % seconds
eventWindow = [-.5, .5]; % seconds [before, after] relative to event onset for event-aligned analysis
natBhvID = [6:8]; % Behavior ID(s) for naturalistic event-aligned analysis (can be vector to collapse multiple behaviors)
natBhvID = [15]; % Behavior ID(s) for naturalistic event-aligned analysis (can be vector to collapse multiple behaviors)

minBhvDur = 0.03; % Minimum behavior duration (seconds)
minSeparation = 0.2; % Minimum time between same behaviors (seconds)

areasToTest = 2:3;
%% Brain areas: 1=M23, 2=M56, 3=DS, 4=VS
areas = {'M23', 'M56', 'DS', 'VS'};
areaNumeric = [1, 2, 3, 4];

% Setup paths and options
paths = get_paths;
opts = neuro_behavior_options;
opts.removeSome = true;
opts.firingRateCheckTime = 5 * 60;
opts.minFiringRate = 0.2;
opts.maxFiringRate = 70;

% ==================== REACH DATA ====================
fprintf('\n=== Loading Reach Data ===\n');

% Load reach data
sessionName = 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat';
reachDataFile = fullfile(paths.reachDataPath, sessionName);
dataR = load(reachDataFile);

% Set collection window
opts.collectStart = 0;
opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
opts.dataPath = reachDataFile;

% Get spike data per area
spikeDataReach = spike_times_per_area_reach(opts);

% Extract reach onset times (convert from ms to seconds)
reachOnsets = dataR.R(:,1) / 1000; % Convert milliseconds to seconds
reachOnsets = reachOnsets(reachOnsets >= opts.collectStart & reachOnsets <= opts.collectEnd);

fprintf('Loaded %d reaches\n', length(reachOnsets));

%% ==================== REACH / INTERTRIAL WINDOW CONFIG (MODELED ON CRITICALITY SCRIPT) ====================
% We will define windows around reach starts and intertrial midpoints and
% later restrict the reach PCA to these windows (rather than whole session).

% Extract reach start times in seconds (alias for clarity with criticality script)
reachStart = reachOnsets;
totalReaches = length(reachStart);

% Calculate intertrial midpoints (halfway between consecutive reaches)
intertrialMidpoints = nan(1, totalReaches - 1);
for i = 1:totalReaches - 1
    intertrialMidpoints(i) = (reachStart(i) + reachStart(i+1)) / 2;
end

fprintf('Calculated %d intertrial midpoints\n', length(intertrialMidpoints));

% Window / buffer configuration (mirrors criticality_reach_intertrial_av.m)
beforeAlign = -2;   % seconds (only used to cap maximum window length)
afterAlign  =  2;   % seconds
stepSize    =  0.25; %#ok<NASGU> % kept for reference; not used directly here
windowBuffer = 0.5; % Minimum distance from window edge to neighboring event
minAvalancheWindow = 8; % Minimum window duration (seconds)

%% Find optimal common window duration that satisfies buffer constraints
validReachIndices = [];
maxWindowPerReach = nan(1, totalReaches);

for r = 1:totalReaches
    reachTime = reachStart(r);
    
    maxWindowForThisReach = inf;
    
    % Previous intertrial midpoint (between previous and current reach)
    if r > 1
        prevMidpoint = intertrialMidpoints(r-1);
        maxWindowFromPrev = 2 * (reachTime - prevMidpoint - windowBuffer);
        if maxWindowFromPrev < maxWindowForThisReach
            maxWindowForThisReach = maxWindowFromPrev;
        end
    end
    
    % Next intertrial midpoint (between current and next reach)
    if r <= length(intertrialMidpoints)
        nextMidpoint = intertrialMidpoints(r);
        maxWindowFromNext = 2 * (nextMidpoint - reachTime - windowBuffer);
        if maxWindowFromNext < maxWindowForThisReach
            maxWindowForThisReach = maxWindowFromNext;
        end
    end
    
    maxWindowPerReach(r) = maxWindowForThisReach;
    
    if maxWindowForThisReach >= minAvalancheWindow
        validReachIndices = [validReachIndices, r]; %#ok<AGROW>
    end
end

validIntertrialIndices = [];
maxWindowPerIntertrial = nan(1, length(intertrialMidpoints));

for i = 1:length(intertrialMidpoints)
    midpointTime = intertrialMidpoints(i);
    
    % Intertrial midpoint i is between reach i and reach i+1
    prevReach = reachStart(i);
    nextReach = reachStart(i+1);
    
    maxWindowForThisIntertrial = inf;
    
    maxWindowFromPrev = 2 * (midpointTime - prevReach - windowBuffer);
    if maxWindowFromPrev < maxWindowForThisIntertrial
        maxWindowForThisIntertrial = maxWindowFromPrev;
    end
    
    maxWindowFromNext = 2 * (nextReach - midpointTime - windowBuffer);
    if maxWindowFromNext < maxWindowForThisIntertrial
        maxWindowForThisIntertrial = maxWindowFromNext;
    end
    
    maxWindowPerIntertrial(i) = maxWindowForThisIntertrial;
    
    if maxWindowForThisIntertrial >= minAvalancheWindow
        validIntertrialIndices = [validIntertrialIndices, i]; %#ok<AGROW>
    end
end

if isempty(validReachIndices) && isempty(validIntertrialIndices)
    error('No valid reaches or intertrial midpoints with window >= %.1f s. Adjust windowBuffer or minAvalancheWindow.', minAvalancheWindow);
end

maxReachWindow = inf;
if ~isempty(validReachIndices)
    maxReachWindow = min(maxWindowPerReach(validReachIndices));
end

maxIntertrialWindow = inf;
if ~isempty(validIntertrialIndices)
    maxIntertrialWindow = min(maxWindowPerIntertrial(validIntertrialIndices));
end

% Use the smaller of the two and also respect the sliding-range cap
avalancheWindow = min(maxReachWindow, maxIntertrialWindow);
maxWindowFromSliding = abs(beforeAlign) + abs(afterAlign);
avalancheWindow = min(avalancheWindow, maxWindowFromSliding);

if avalancheWindow < minAvalancheWindow
    avalancheWindow = minAvalancheWindow;
end

if avalancheWindow <= 0 || isnan(avalancheWindow) || isinf(avalancheWindow)
    error('Cannot find valid window duration. Adjust windowBuffer or inter-reach intervals.');
end

% Keep original arrays for reference
reachStartOriginal = reachStart;
intertrialMidpointsOriginal = intertrialMidpoints; %#ok<NASGU>

% Filter reachStart to only include indices that can support this window
reachStart = reachStart(validReachIndices);
totalReaches = length(reachStart);

% Recompute intertrial midpoints consistent with filtered reaches and
% ensure that only midpoints that passed the window test are retained.
intertrialMidpointsFiltered = nan(1, totalReaches - 1);
intertrialMidpointValid = false(1, totalReaches - 1);

for i = 1:totalReaches - 1
    midpointTime = (reachStart(i) + reachStart(i+1)) / 2;
    
    origReachIdx1 = validReachIndices(i);
    origReachIdx2 = validReachIndices(i+1);
    
    if origReachIdx2 == origReachIdx1 + 1
        if ismember(origReachIdx1, validIntertrialIndices)
            intertrialMidpointsFiltered(i) = midpointTime;
            intertrialMidpointValid(i) = true;
        end
    end
end

intertrialMidpoints = intertrialMidpointsFiltered(intertrialMidpointValid);

fprintf('Windowed PCA config: window = %.2f s, buffer = %.2f s\n', avalancheWindow, windowBuffer);
fprintf('  Valid reach windows: %d / %d\n', length(reachStart), length(reachStartOriginal));
fprintf('  Valid intertrial windows: %d / %d\n', length(intertrialMidpoints), length(maxWindowPerIntertrial));

% ==================== NATURALISTIC DATA ====================
fprintf('\n=== Loading Naturalistic Data ===\n');

% Naturalistic data parameters
animal = 'ag25290';
sessionBhv = '112321_1';
sessionNrn = '112321';
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectEnd = 45 * 60; % seconds

% Load behavior data
bhvDataPath = strcat(paths.bhvDataPath, 'animal_', animal, '/');
bhvFileName = ['behavior_labels_', animal, '_', sessionBhv, '.csv'];
opts.dataPath = bhvDataPath;
opts.fileName = bhvFileName;
dataBhv = load_data(opts, 'behavior');

% Load spike data
freeDataPath = strcat(paths.freeDataPath, 'animal_', animal, '/', sessionNrn, '/recording1/');
opts.dataPath = freeDataPath;
spikeDataNat = spike_times_per_area(opts);

%% Extract behavior onset times for natBhvID with additional filtering
% Filter criteria:
%   1. Duration >= 0.03 seconds
%   2. Valid behavior (from behavior_selection)
%   3. Separated from any behavior in natBhvID by at least 0.5 seconds
%
% If natBhvID is a vector, all specified behaviors are collapsed together

% Ensure natBhvID is a row vector
if iscolumn(natBhvID)
    natBhvID = natBhvID';
end

% Initial mask: any behavior ID in natBhvID, valid, and long enough duration
bhvMask = ismember(dataBhv.ID, natBhvID) & (dataBhv.Valid == 1) & (dataBhv.Dur >= minBhvDur);

% Get all candidate behavior indices
candidateIndices = find(bhvMask);
candidateStartTimes = dataBhv.StartTime(candidateIndices);
candidateDurations = dataBhv.Dur(candidateIndices);
candidateBhvIDs = dataBhv.ID(candidateIndices);

% Filter by minimum separation from any behavior in natBhvID
% Sort candidate indices by start time to process chronologically
[~, sortOrder] = sort(candidateStartTimes);
candidateIndices = candidateIndices(sortOrder);
candidateStartTimes = candidateStartTimes(sortOrder);
candidateBhvIDs = candidateBhvIDs(sortOrder);

validIndices = [];
for i = 1:length(candidateIndices)
    idx = candidateIndices(i);
    startTime = dataBhv.StartTime(idx);
    
    % Check if there's a previous valid behavior (any in natBhvID) within minSeparation
    hasRecentSame = false;
    if ~isempty(validIndices)
        % Get the most recent valid behavior start time
        lastValidStart = dataBhv.StartTime(validIndices(end));
        timeSinceLast = startTime - lastValidStart;
        
        % If the last valid behavior was in natBhvID and too recent, skip this one
        if timeSinceLast < minSeparation
            hasRecentSame = true;
        end
    end
    
    % Keep this behavior if no recent behavior found
    if ~hasRecentSame
        validIndices = [validIndices; idx];
    end
end

% Extract onset times for valid behaviors
natEventOnsets = dataBhv.StartTime(validIndices);
if isscalar(natBhvID)
    fprintf('Found %d onsets for behavior ID %d (after filtering: Dur>=%.2fs, Sep>=%.2fs)\n', ...
        length(natEventOnsets), natBhvID, minBhvDur, minSeparation);
else
    fprintf('Found %d onsets for behavior IDs [%s] (collapsed, after filtering: Dur>=%.2fs, Sep>=%.2fs)\n', ...
        length(natEventOnsets), num2str(natBhvID), minBhvDur, minSeparation);
end

%% ==================== ANALYSIS 1: WHOLE SESSION ====================
fprintf('\n=== Analysis 1: Whole Session PCA ===\n');

% Initialize storage for whole session results
cumVarReachWhole = cell(1, length(areas));
cumVarIntertrialWhole = cell(1, length(areas));
cumVarNatWhole = cell(1, length(areas));
numComp50ReachWhole = zeros(1, length(areas));
numComp50IntertrialWhole = zeros(1, length(areas));
numComp50NatWhole = zeros(1, length(areas));

for a = areasToTest
    fprintf('\nProcessing area %s (whole session)...\n', areas{a});
    
    %% Reach data - whole session
    areaMask = spikeDataReach(:,3) == areaNumeric(a);
    neuronIdsReach = unique(spikeDataReach(areaMask, 2));
    
    %% Naturalistic data - whole session (get neuron count first for comparison)
    areaMask = spikeDataNat(:,3) == areaNumeric(a);
    neuronIdsNat = unique(spikeDataNat(areaMask, 2));
    
    % Determine minimum neuron count and downsample if needed
    numReach = length(neuronIdsReach);
    numNat = length(neuronIdsNat);
    minNeurons = min(numReach, numNat);
    
    if minNeurons == 0
        fprintf('  No neurons found in area %s\n', areas{a});
        cumVarReachWhole{a} = [];
        cumVarNatWhole{a} = [];
        numComp50ReachWhole(a) = NaN;
        numComp50NatWhole(a) = NaN;
        continue;
    end
    
    % Downsample to match minimum
    if numReach > minNeurons
        rng(42); % Set seed for reproducibility
        neuronIdsReach = neuronIdsReach(randperm(numReach, minNeurons));
        fprintf('  Reach: downsampled from %d to %d neurons\n', numReach, minNeurons);
    else
        fprintf('  Reach: %d neurons\n', numReach);
    end
    
    if numNat > minNeurons
        rng(43); % Different seed for naturalistic
        neuronIdsNat = neuronIdsNat(randperm(numNat, minNeurons));
        fprintf('  Naturalistic: downsampled from %d to %d neurons\n', numNat, minNeurons);
    else
        fprintf('  Naturalistic: %d neurons\n', numNat);
    end
    
    %% Process reach data
    if ~isempty(neuronIdsReach)
        neuronIds = neuronIdsReach;
        
        % Bin all spike data for this area
        timeEdges = opts.collectStart:binSize:opts.collectEnd;
        numBins = length(timeEdges) - 1;
        dataMatReachFull = zeros(numBins, length(neuronIds));
        
        for n = 1:length(neuronIds)
            neuronId = neuronIds(n);
            neuronMask = (spikeDataReach(:,2) == neuronId) & (spikeDataReach(:,3) == areaNumeric(a));
            spikeTimes = spikeDataReach(neuronMask, 1);
            spikeTimes = spikeTimes(spikeTimes >= opts.collectStart & spikeTimes <= opts.collectEnd);
            
            % Bin spikes
            spikeCounts = histcounts(spikeTimes, timeEdges);
            dataMatReachFull(:, n) = spikeCounts';
        end
        
        % Remove neurons with no spikes
        validNeurons = sum(dataMatReachFull, 1) > 0;
        dataMatReachFull = dataMatReachFull(:, validNeurons);
        
        if size(dataMatReachFull, 2) > 0
            % Build time-bin centers for window selection
            binCenters = timeEdges(1:end-1) + diff(timeEdges)/2;
            
            % Select bins that fall within any reach window
            reachBinMask = false(size(binCenters));
            for r = 1:length(reachStart)
                winStart = reachStart(r) - avalancheWindow/2;
                winEnd   = reachStart(r) + avalancheWindow/2;
                reachBinMask = reachBinMask | (binCenters >= winStart & binCenters <= winEnd);
            end
            
            % Select bins that fall within any intertrial window
            intertrialBinMask = false(size(binCenters));
            for iWin = 1:length(intertrialMidpoints)
                winStart = intertrialMidpoints(iWin) - avalancheWindow/2;
                winEnd   = intertrialMidpoints(iWin) + avalancheWindow/2;
                intertrialBinMask = intertrialBinMask | (binCenters >= winStart & binCenters <= winEnd);
            end
            
            dataMatReach = dataMatReachFull(reachBinMask, :);
            dataMatIntertrial = dataMatReachFull(intertrialBinMask, :);
            
            % Run PCA for reach windows
            if ~isempty(dataMatReach) && size(dataMatReach, 2) > 0
                [coeffReach, scoreReach, ~, ~, explainedReach] = pca(dataMatReach); %#ok<ASGLU>
                cumVarReachWhole{a} = cumsum(explainedReach);
                numComp50ReachWhole(a) = find(cumVarReachWhole{a} >= 50, 1);
                if isempty(numComp50ReachWhole(a))
                    numComp50ReachWhole(a) = length(cumVarReachWhole{a});
                end
                fprintf('  Reach windows: %d components needed for 50%% variance\n', numComp50ReachWhole(a));
            else
                cumVarReachWhole{a} = [];
                numComp50ReachWhole(a) = NaN;
                fprintf('  Reach windows: no data for PCA\n');
            end
            
            % Run PCA for intertrial windows
            if ~isempty(dataMatIntertrial) && size(dataMatIntertrial, 2) > 0
                [coeffInter, scoreInter, ~, ~, explainedInter] = pca(dataMatIntertrial); %#ok<ASGLU>
                cumVarIntertrialWhole{a} = cumsum(explainedInter);
                numComp50IntertrialWhole(a) = find(cumVarIntertrialWhole{a} >= 50, 1);
                if isempty(numComp50IntertrialWhole(a))
                    numComp50IntertrialWhole(a) = length(cumVarIntertrialWhole{a});
                end
                fprintf('  Intertrial windows: %d components needed for 50%% variance\n', numComp50IntertrialWhole(a));
            else
                cumVarIntertrialWhole{a} = [];
                numComp50IntertrialWhole(a) = NaN;
                fprintf('  Intertrial windows: no data for PCA\n');
            end
        else
            cumVarReachWhole{a} = [];
            cumVarIntertrialWhole{a} = [];
            numComp50ReachWhole(a) = NaN;
            numComp50IntertrialWhole(a) = NaN;
        end
    end
    
    %% Process naturalistic data
    if ~isempty(neuronIdsNat)
        neuronIds = neuronIdsNat;
        
        % Bin all spike data for this area
        timeEdges = opts.collectStart:binSize:opts.collectEnd;
        numBins = length(timeEdges) - 1;
        dataMatNat = zeros(numBins, length(neuronIds));
        
        for n = 1:length(neuronIds)
            neuronId = neuronIds(n);
            neuronMask = (spikeDataNat(:,2) == neuronId) & (spikeDataNat(:,3) == areaNumeric(a));
            spikeTimes = spikeDataNat(neuronMask, 1);
            spikeTimes = spikeTimes(spikeTimes >= opts.collectStart & spikeTimes <= opts.collectEnd);
            
            % Bin spikes
            spikeCounts = histcounts(spikeTimes, timeEdges);
            dataMatNat(:, n) = spikeCounts';
        end
        
        % Remove neurons with no spikes
        validNeurons = sum(dataMatNat, 1) > 0;
        dataMatNat = dataMatNat(:, validNeurons);
        
        if size(dataMatNat, 2) > 0
            % Run PCA
            [coeff, score, ~, ~, explained] = pca(dataMatNat);
            
            % Calculate cumulative explained variance
            cumVarNatWhole{a} = cumsum(explained);
            
            % Find number of components for 50% variance
            numComp50NatWhole(a) = find(cumVarNatWhole{a} >= 50, 1);
            if isempty(numComp50NatWhole(a))
                numComp50NatWhole(a) = length(cumVarNatWhole{a});
            end
            
            fprintf('  Naturalistic (whole session): %d components needed for 50%% variance\n', numComp50NatWhole(a));
        else
            cumVarNatWhole{a} = [];
            numComp50NatWhole(a) = NaN;
        end
    else
        cumVarNatWhole{a} = [];
        numComp50NatWhole(a) = NaN;
    end
end

% ==================== ANALYSIS 2: EVENT-ALIGNED ====================
fprintf('\n=== Analysis 2: Event-Aligned PCA ===\n');

% Subsample events to match minimum between reach and naturalistic
numReachEvents = length(reachOnsets);
numNatEvents = length(natEventOnsets);
minEvents = min(numReachEvents, numNatEvents);

if minEvents == 0
    fprintf('No events available for event-aligned analysis\n');
    % Initialize empty results
    cumVarReachEvent = cell(1, length(areas));
    cumVarNatEvent = cell(1, length(areas));
    numComp50ReachEvent = zeros(1, length(areas));
    numComp50NatEvent = zeros(1, length(areas));
else
    % Downsample events to match minimum
    if numReachEvents > minEvents
        rng(46); % Set seed for reproducibility
        reachOnsetsSub = reachOnsets(randperm(numReachEvents, minEvents));
        fprintf('Reach events: downsampled from %d to %d events\n', numReachEvents, minEvents);
    else
        reachOnsetsSub = reachOnsets;
        fprintf('Reach events: %d events\n', numReachEvents);
    end
    
    if numNatEvents > minEvents
        rng(47); % Different seed for naturalistic
        natEventOnsetsSub = natEventOnsets(randperm(numNatEvents, minEvents));
        fprintf('Naturalistic events: downsampled from %d to %d events\n', numNatEvents, minEvents);
    else
        natEventOnsetsSub = natEventOnsets;
        fprintf('Naturalistic events: %d events\n', numNatEvents);
    end
    
    % Initialize storage for event-aligned results
    cumVarReachEvent = cell(1, length(areas));
    cumVarNatEvent = cell(1, length(areas));
    numComp50ReachEvent = zeros(1, length(areas));
    numComp50NatEvent = zeros(1, length(areas));
    
    for a = areasToTest
    fprintf('\nProcessing area %s (event-aligned)...\n', areas{a});
    
    %% Reach data - event-aligned
    areaMask = spikeDataReach(:,3) == areaNumeric(a);
    neuronIdsReach = unique(spikeDataReach(areaMask, 2));
    
    %% Naturalistic data - event-aligned (get neuron count first for comparison)
    areaMask = spikeDataNat(:,3) == areaNumeric(a);
    neuronIdsNat = unique(spikeDataNat(areaMask, 2));
    
    % Determine minimum neuron count and downsample if needed
    numReach = length(neuronIdsReach);
    numNat = length(neuronIdsNat);
    minNeurons = min(numReach, numNat);
    
    if minNeurons == 0
        fprintf('  No neurons found in area %s\n', areas{a});
        cumVarReachEvent{a} = [];
        cumVarNatEvent{a} = [];
        numComp50ReachEvent(a) = NaN;
        numComp50NatEvent(a) = NaN;
        continue;
    end
    
    % Downsample to match minimum
    if numReach > minNeurons
        rng(44); % Set seed for reproducibility (different from whole session)
        neuronIdsReach = neuronIdsReach(randperm(numReach, minNeurons));
        fprintf('  Reach: downsampled from %d to %d neurons\n', numReach, minNeurons);
    else
        fprintf('  Reach: %d neurons\n', numReach);
    end
    
    if numNat > minNeurons
        rng(45); % Different seed for naturalistic
        neuronIdsNat = neuronIdsNat(randperm(numNat, minNeurons));
        fprintf('  Naturalistic: downsampled from %d to %d neurons\n', numNat, minNeurons);
    else
        fprintf('  Naturalistic: %d neurons\n', numNat);
    end
    
    %% Process reach data
    if ~isempty(neuronIdsReach)
        neuronIds = neuronIdsReach;
        
        % Extract windows around each reach onset
        windowDuration = eventWindow(2) - eventWindow(1);
        numBinsPerWindow = round(windowDuration / binSize);
        timeEdgesWindow = eventWindow(1):binSize:eventWindow(2);
        numBinsWindow = length(timeEdgesWindow) - 1;
        
        % Initialize data matrix: [timeBins x neurons x events]
        dataMatReachEvent = nan(numBinsWindow, length(neuronIds), minEvents);
        
        for r = 1:minEvents
            reachTime = reachOnsetsSub(r);
            windowStart = reachTime + eventWindow(1);
            windowEnd = reachTime + eventWindow(2);
            
            for n = 1:length(neuronIds)
                neuronId = neuronIds(n);
                neuronMask = (spikeDataReach(:,2) == neuronId) & (spikeDataReach(:,3) == areaNumeric(a));
                spikeTimes = spikeDataReach(neuronMask, 1);
                
                % Extract spikes in window
                spikeWindow = spikeTimes(spikeTimes >= windowStart & spikeTimes <= windowEnd);
                
                % Convert to relative times (eventWindow(1) to eventWindow(2))
                spikeWindowRel = spikeWindow - reachTime;
                
                % Bin spikes
                spikeCounts = histcounts(spikeWindowRel, timeEdgesWindow);
                if length(spikeCounts) == numBinsWindow
                    dataMatReachEvent(:, n, r) = spikeCounts';
                end
            end
        end
        
        % Reshape to [timeBins*events x neurons]
        dataMatReachEvent = reshape(dataMatReachEvent, numBinsWindow * minEvents, length(neuronIds));
        
        % Remove neurons with no spikes
        validNeurons = sum(dataMatReachEvent, 1, 'omitnan') > 0;
        dataMatReachEvent = dataMatReachEvent(:, validNeurons);
        
        % Remove rows with all NaN
        validRows = ~all(isnan(dataMatReachEvent), 2);
        dataMatReachEvent = dataMatReachEvent(validRows, :);
        
        if size(dataMatReachEvent, 2) > 0 && size(dataMatReachEvent, 1) > 0
            % Run PCA
            [coeff, score, ~, ~, explained] = pca(dataMatReachEvent, 'Rows', 'complete');
            
            % Calculate cumulative explained variance
            cumVarReachEvent{a} = cumsum(explained);
            
            % Find number of components for 50% variance
            numComp50ReachEvent(a) = find(cumVarReachEvent{a} >= 50, 1);
            if isempty(numComp50ReachEvent(a))
                numComp50ReachEvent(a) = length(cumVarReachEvent{a});
            end
            
            fprintf('  Reach: %d components needed for 50%% variance\n', numComp50ReachEvent(a));
        else
            cumVarReachEvent{a} = [];
            numComp50ReachEvent(a) = NaN;
        end
    end
    
    %% Process naturalistic data
    if ~isempty(neuronIdsNat)
        neuronIds = neuronIdsNat;
        
        % Extract windows around each behavior onset
        windowDuration = eventWindow(2) - eventWindow(1);
        numBinsPerWindow = round(windowDuration / binSize);
        timeEdgesWindow = eventWindow(1):binSize:eventWindow(2);
        numBinsWindow = length(timeEdgesWindow) - 1;
        
        % Initialize data matrix: [timeBins x neurons x events]
        dataMatNatEvent = nan(numBinsWindow, length(neuronIds), minEvents);
        
        for e = 1:minEvents
            eventTime = natEventOnsetsSub(e);
            windowStart = eventTime + eventWindow(1);
            windowEnd = eventTime + eventWindow(2);
            
            for n = 1:length(neuronIds)
                neuronId = neuronIds(n);
                neuronMask = (spikeDataNat(:,2) == neuronId) & (spikeDataNat(:,3) == areaNumeric(a));
                spikeTimes = spikeDataNat(neuronMask, 1);
                
                % Extract spikes in window
                spikeWindow = spikeTimes(spikeTimes >= windowStart & spikeTimes <= windowEnd);
                
                % Convert to relative times (eventWindow(1) to eventWindow(2))
                spikeWindowRel = spikeWindow - eventTime;
                
                % Bin spikes
                spikeCounts = histcounts(spikeWindowRel, timeEdgesWindow);
                if length(spikeCounts) == numBinsWindow
                    dataMatNatEvent(:, n, e) = spikeCounts';
                end
            end
        end
        
        % Reshape to [timeBins*events x neurons]
        dataMatNatEvent = reshape(dataMatNatEvent, numBinsWindow * minEvents, length(neuronIds));
        
        % Remove neurons with no spikes
        validNeurons = sum(dataMatNatEvent, 1, 'omitnan') > 0;
        dataMatNatEvent = dataMatNatEvent(:, validNeurons);
        
        % Remove rows with all NaN
        validRows = ~all(isnan(dataMatNatEvent), 2);
        dataMatNatEvent = dataMatNatEvent(validRows, :);
        
        if size(dataMatNatEvent, 2) > 0 && size(dataMatNatEvent, 1) > 0
            % Run PCA
            [coeff, score, ~, ~, explained] = pca(dataMatNatEvent, 'Rows', 'complete');
            
            % Calculate cumulative explained variance
            cumVarNatEvent{a} = cumsum(explained);
            
            % Find number of components for 50% variance
            numComp50NatEvent(a) = find(cumVarNatEvent{a} >= 50, 1);
            if isempty(numComp50NatEvent(a))
                numComp50NatEvent(a) = length(cumVarNatEvent{a});
            end
            
            fprintf('  Naturalistic: %d components needed for 50%% variance\n', numComp50NatEvent(a));
        else
            cumVarNatEvent{a} = [];
            numComp50NatEvent(a) = NaN;
        end
    end
    end % End of else block for event-aligned analysis
end % End of if minEvents == 0

%% ==================== PLOTTING ====================
fprintf('\n=== Plotting Results ===\n');

% Create figure for whole session
figure(1); clf;
set(gcf, 'Units', 'pixels');
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
    targetMonitor = monitorPositions(size(monitorPositions, 1), :);
else
    targetMonitor = monitorPositions(1, :);
end
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)/2]);

for a = 1:length(areas)
    subplot(1, 4, a);
    hold on;
    
    % Plot reach-window data
    if ~isempty(cumVarReachWhole{a})
        plot(1:length(cumVarReachWhole{a}), cumVarReachWhole{a}, 'b-', 'LineWidth', 4, 'DisplayName', sprintf('Reach windows (n=%d)', numComp50ReachWhole(a)));
    end
    
    % Plot intertrial-window data
    if ~isempty(cumVarIntertrialWhole{a})
        plot(1:length(cumVarIntertrialWhole{a}), cumVarIntertrialWhole{a}, 'g-', 'LineWidth', 4, 'DisplayName', sprintf('Intertrial windows (n=%d)', numComp50IntertrialWhole(a)));
    end
    
    % Plot naturalistic data
    if ~isempty(cumVarNatWhole{a})
        plot(1:length(cumVarNatWhole{a}), cumVarNatWhole{a}, 'r-', 'LineWidth', 4, 'DisplayName', sprintf('Naturalistic whole (n=%d)', numComp50NatWhole(a)));
    end
    
    % Add 50% line
    % yline(50, 'k--', 'LineWidth', 2, 'DisplayName', '50%');
    
    xlabel('PCA Component');
    ylabel('Explained Variance (%)');
    title(sprintf('%s - Reach vs Intertrial vs Naturalistic', areas{a}));
    % legend('Location', 'best'); % legend suppressed for clarity across subplots
    grid on;
    ylim([0, 100]);
end

sgtitle('PCA Explained Variance - Reach Windows vs Intertrial Windows vs Naturalistic', 'FontSize', 14, 'FontWeight', 'bold');

% Save whole session figure
saveDir = fullfile(paths.dropPath, 'sfn2025', 'explained_variance');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end
saveFileWhole = fullfile(saveDir, 'pca_explained_variance_whole_session.eps');
exportgraphics(figure(1), saveFileWhole, 'ContentType', 'vector');
fprintf('Saved whole session figure to: %s\n', saveFileWhole);

% Create figure for event-aligned
figure(2); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)/2]);

for a = 1:length(areas)
    subplot(1, 4, a);
    hold on;
    
    % Plot reach data
    if ~isempty(cumVarReachEvent{a})
        plot(1:length(cumVarReachEvent{a}), cumVarReachEvent{a}, 'b-', 'LineWidth', 4, 'DisplayName', sprintf('Reach (n=%d)', numComp50ReachEvent(a)));
    end
    
    % Plot naturalistic data
    if ~isempty(cumVarNatEvent{a})
        plot(1:length(cumVarNatEvent{a}), cumVarNatEvent{a}, 'r-', 'LineWidth', 4, 'DisplayName', sprintf('Naturalistic (n=%d)', numComp50NatEvent(a)));
    end
    
    % Add 50% line
    % yline(50, 'k--', 'LineWidth', 1, 'DisplayName', '50%');
    
    xlabel('PCA Component');
    ylabel('Explained Variance (%)');
    title(sprintf('%s - Event-Aligned', areas{a}));
    % legend('Location', 'best');
    grid on;
    ylim([0, 100]);
end

if isscalar(natBhvID)
    sgtitle(sprintf('PCA Explained Variance - Event-Aligned (Reach: all onsets, Naturalistic: bhvID=%d)', natBhvID), 'FontSize', 14, 'FontWeight', 'bold');
    bhvIDStr = sprintf('%d', natBhvID);
else
    sgtitle(sprintf('PCA Explained Variance - Event-Aligned (Reach: all onsets, Naturalistic: bhvID=[%s])', num2str(natBhvID)), 'FontSize', 14, 'FontWeight', 'bold');
    bhvIDStr = sprintf('%s', strrep(num2str(natBhvID), ' ', '_'));
end

% Save event-aligned figure
saveFileEvent = fullfile(saveDir, sprintf('pca_explained_variance_event_aligned_bhvID_%s.eps', bhvIDStr));
exportgraphics(figure(2), saveFileEvent, 'ContentType', 'vector');
fprintf('Saved event-aligned figure to: %s\n', saveFileEvent);

%% ==================== SUMMARY REPORT ====================
fprintf('\n=== SUMMARY: Components Needed for 50%% Variance ===\n');
fprintf('\nWhole Session:\n');
fprintf('  Reach windows:\n');
for a = 1:length(areas)
    if ~isnan(numComp50ReachWhole(a))
        fprintf('    %s: %d components\n', areas{a}, numComp50ReachWhole(a));
    else
        fprintf('    %s: No data\n', areas{a});
    end
end
fprintf('  Intertrial windows:\n');
for a = 1:length(areas)
    if ~isnan(numComp50IntertrialWhole(a))
        fprintf('    %s: %d components\n', areas{a}, numComp50IntertrialWhole(a));
    else
        fprintf('    %s: No data\n', areas{a});
    end
end
fprintf('  Naturalistic:\n');
for a = 1:length(areas)
    if ~isnan(numComp50NatWhole(a))
        fprintf('    %s: %d components\n', areas{a}, numComp50NatWhole(a));
    else
        fprintf('    %s: No data\n', areas{a});
    end
end

fprintf('\nEvent-Aligned:\n');
fprintf('  Reach:\n');
for a = 1:length(areas)
    if ~isnan(numComp50ReachEvent(a))
        fprintf('    %s: %d components\n', areas{a}, numComp50ReachEvent(a));
    else
        fprintf('    %s: No data\n', areas{a});
    end
end
fprintf('  Naturalistic:\n');
for a = 1:length(areas)
    if ~isnan(numComp50NatEvent(a))
        fprintf('    %s: %d components\n', areas{a}, numComp50NatEvent(a));
    else
        fprintf('    %s: No data\n', areas{a});
    end
end

fprintf('\n=== Complete ===\n');

