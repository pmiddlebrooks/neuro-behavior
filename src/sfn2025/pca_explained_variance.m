%% PCA_EXPLAINED_VARIANCE - Assess explained variance of neural activity using PCA
%
% This script performs PCA analysis on reach and naturalistic neural data
% to assess how many components are needed to explain variance in each brain area.
%
% Two analyses:
% 1. Whole session: All data without event alignment
% 2. Event-aligned: 2 second windows around events
%    - Reach: All reach onsets
%    - Naturalistic: Onsets of bhvID = 10
%
% Outputs:
% - Cumulative explained variance plots (1x4 figure, one per area)
% - Number of components needed to explain at least 50% variance

clear; close all;

%% Parameters
binSize = 0.02; % seconds
eventWindow = 2; % seconds (window around events for event-aligned analysis)
natBhvID = 10; % Behavior ID for naturalistic event-aligned analysis

% Brain areas: 1=M23, 2=M56, 3=DS, 4=VS
areas = {'M23', 'M56', 'DS', 'VS'};
areaNumeric = [1, 2, 3, 4];

%% Setup paths and options
paths = get_paths;
opts = neuro_behavior_options;
opts.removeSome = true;
opts.firingRateCheckTime = 5 * 60;
opts.minFiringRate = 0.5;
opts.maxFiringRate = 70;

%% ==================== REACH DATA ====================
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

%% ==================== NATURALISTIC DATA ====================
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
nrnDataPath = strcat(paths.nrnDataPath, 'animal_', animal, '/', sessionNrn, '/recording1/');
opts.dataPath = nrnDataPath;
spikeDataNat = spike_times_per_area(opts);

% Extract behavior onset times for bhvID = 10
bhvMask = (dataBhv.ID == natBhvID) & (dataBhv.Valid == 1);
natEventOnsets = dataBhv.StartTime(bhvMask);
fprintf('Found %d onsets for behavior ID %d\n', length(natEventOnsets), natBhvID);

%% ==================== ANALYSIS 1: WHOLE SESSION ====================
fprintf('\n=== Analysis 1: Whole Session PCA ===\n');

% Initialize storage for whole session results
cumVarReachWhole = cell(1, length(areas));
cumVarNatWhole = cell(1, length(areas));
numComp50ReachWhole = zeros(1, length(areas));
numComp50NatWhole = zeros(1, length(areas));

for a = 1:length(areas)
    fprintf('\nProcessing area %s (whole session)...\n', areas{a});
    
    %% Reach data - whole session
    areaMask = spikeDataReach(:,3) == areaNumeric(a);
    neuronIds = unique(spikeDataReach(areaMask, 2));
    
    if isempty(neuronIds)
        fprintf('  No neurons found in area %s (reach)\n', areas{a});
        cumVarReachWhole{a} = [];
        numComp50ReachWhole(a) = NaN;
    else
        fprintf('  Reach: %d neurons\n', length(neuronIds));
        
        % Bin all spike data for this area
        timeEdges = opts.collectStart:binSize:opts.collectEnd;
        numBins = length(timeEdges) - 1;
        dataMatReach = zeros(numBins, length(neuronIds));
        
        for n = 1:length(neuronIds)
            neuronId = neuronIds(n);
            neuronMask = (spikeDataReach(:,2) == neuronId) & (spikeDataReach(:,3) == areaNumeric(a));
            spikeTimes = spikeDataReach(neuronMask, 1);
            spikeTimes = spikeTimes(spikeTimes >= opts.collectStart & spikeTimes <= opts.collectEnd);
            
            % Bin spikes
            spikeCounts = histcounts(spikeTimes, timeEdges);
            dataMatReach(:, n) = spikeCounts';
        end
        
        % Remove neurons with no spikes
        validNeurons = sum(dataMatReach, 1) > 0;
        dataMatReach = dataMatReach(:, validNeurons);
        
        if size(dataMatReach, 2) > 0
            % Run PCA
            [coeff, score, ~, ~, explained] = pca(dataMatReach);
            
            % Calculate cumulative explained variance
            cumVarReachWhole{a} = cumsum(explained);
            
            % Find number of components for 50% variance
            numComp50ReachWhole(a) = find(cumVarReachWhole{a} >= 50, 1);
            if isempty(numComp50ReachWhole(a))
                numComp50ReachWhole(a) = length(cumVarReachWhole{a});
            end
            
            fprintf('  Reach: %d components needed for 50%% variance\n', numComp50ReachWhole(a));
        else
            cumVarReachWhole{a} = [];
            numComp50ReachWhole(a) = NaN;
        end
    end
    
    %% Naturalistic data - whole session
    areaMask = spikeDataNat(:,3) == areaNumeric(a);
    neuronIds = unique(spikeDataNat(areaMask, 2));
    
    if isempty(neuronIds)
        fprintf('  No neurons found in area %s (naturalistic)\n', areas{a});
        cumVarNatWhole{a} = [];
        numComp50NatWhole(a) = NaN;
    else
        fprintf('  Naturalistic: %d neurons\n', length(neuronIds));
        
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
            
            fprintf('  Naturalistic: %d components needed for 50%% variance\n', numComp50NatWhole(a));
        else
            cumVarNatWhole{a} = [];
            numComp50NatWhole(a) = NaN;
        end
    end
end

%% ==================== ANALYSIS 2: EVENT-ALIGNED ====================
fprintf('\n=== Analysis 2: Event-Aligned PCA ===\n');

% Initialize storage for event-aligned results
cumVarReachEvent = cell(1, length(areas));
cumVarNatEvent = cell(1, length(areas));
numComp50ReachEvent = zeros(1, length(areas));
numComp50NatEvent = zeros(1, length(areas));

for a = 1:length(areas)
    fprintf('\nProcessing area %s (event-aligned)...\n', areas{a});
    
    %% Reach data - event-aligned
    areaMask = spikeDataReach(:,3) == areaNumeric(a);
    neuronIds = unique(spikeDataReach(areaMask, 2));
    
    if isempty(neuronIds) || isempty(reachOnsets)
        fprintf('  No neurons or reaches found in area %s (reach)\n', areas{a});
        cumVarReachEvent{a} = [];
        numComp50ReachEvent(a) = NaN;
    else
        fprintf('  Reach: %d neurons, %d events\n', length(neuronIds), length(reachOnsets));
        
        % Extract 2 second windows around each reach onset
        numBinsPerWindow = round(eventWindow / binSize);
        timeEdgesWindow = 0:binSize:eventWindow;
        numBinsWindow = length(timeEdgesWindow) - 1;
        
        % Initialize data matrix: [timeBins x neurons x events]
        dataMatReachEvent = nan(numBinsWindow, length(neuronIds), length(reachOnsets));
        
        for r = 1:length(reachOnsets)
            reachTime = reachOnsets(r);
            windowStart = reachTime;
            windowEnd = reachTime + eventWindow;
            
            for n = 1:length(neuronIds)
                neuronId = neuronIds(n);
                neuronMask = (spikeDataReach(:,2) == neuronId) & (spikeDataReach(:,3) == areaNumeric(a));
                spikeTimes = spikeDataReach(neuronMask, 1);
                
                % Extract spikes in window
                spikeWindow = spikeTimes(spikeTimes >= windowStart & spikeTimes <= windowEnd);
                
                % Convert to relative times (0 to eventWindow)
                spikeWindowRel = spikeWindow - windowStart;
                
                % Bin spikes
                spikeCounts = histcounts(spikeWindowRel, timeEdgesWindow);
                if length(spikeCounts) == numBinsWindow
                    dataMatReachEvent(:, n, r) = spikeCounts';
                end
            end
        end
        
        % Reshape to [timeBins*events x neurons]
        dataMatReachEvent = reshape(dataMatReachEvent, numBinsWindow * length(reachOnsets), length(neuronIds));
        
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
    
    %% Naturalistic data - event-aligned
    areaMask = spikeDataNat(:,3) == areaNumeric(a);
    neuronIds = unique(spikeDataNat(areaMask, 2));
    
    if isempty(neuronIds) || isempty(natEventOnsets)
        fprintf('  No neurons or events found in area %s (naturalistic)\n', areas{a});
        cumVarNatEvent{a} = [];
        numComp50NatEvent(a) = NaN;
    else
        fprintf('  Naturalistic: %d neurons, %d events\n', length(neuronIds), length(natEventOnsets));
        
        % Extract 2 second windows around each behavior onset
        numBinsPerWindow = round(eventWindow / binSize);
        timeEdgesWindow = 0:binSize:eventWindow;
        numBinsWindow = length(timeEdgesWindow) - 1;
        
        % Initialize data matrix: [timeBins x neurons x events]
        dataMatNatEvent = nan(numBinsWindow, length(neuronIds), length(natEventOnsets));
        
        for e = 1:length(natEventOnsets)
            eventTime = natEventOnsets(e);
            windowStart = eventTime;
            windowEnd = eventTime + eventWindow;
            
            for n = 1:length(neuronIds)
                neuronId = neuronIds(n);
                neuronMask = (spikeDataNat(:,2) == neuronId) & (spikeDataNat(:,3) == areaNumeric(a));
                spikeTimes = spikeDataNat(neuronMask, 1);
                
                % Extract spikes in window
                spikeWindow = spikeTimes(spikeTimes >= windowStart & spikeTimes <= windowEnd);
                
                % Convert to relative times (0 to eventWindow)
                spikeWindowRel = spikeWindow - windowStart;
                
                % Bin spikes
                spikeCounts = histcounts(spikeWindowRel, timeEdgesWindow);
                if length(spikeCounts) == numBinsWindow
                    dataMatNatEvent(:, n, e) = spikeCounts';
                end
            end
        end
        
        % Reshape to [timeBins*events x neurons]
        dataMatNatEvent = reshape(dataMatNatEvent, numBinsWindow * length(natEventOnsets), length(neuronIds));
        
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
end

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
    
    % Plot reach data
    if ~isempty(cumVarReachWhole{a})
        plot(1:length(cumVarReachWhole{a}), cumVarReachWhole{a}, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Reach (n=%d)', numComp50ReachWhole(a)));
    end
    
    % Plot naturalistic data
    if ~isempty(cumVarNatWhole{a})
        plot(1:length(cumVarNatWhole{a}), cumVarNatWhole{a}, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Naturalistic (n=%d)', numComp50NatWhole(a)));
    end
    
    % Add 50% line
    yline(50, 'k--', 'LineWidth', 1, 'DisplayName', '50%');
    
    xlabel('PCA Component');
    ylabel('Cumulative Explained Variance (%)');
    title(sprintf('%s - Whole Session', areas{a}));
    legend('Location', 'best');
    grid on;
    ylim([0, 100]);
end

sgtitle('PCA Explained Variance - Whole Session', 'FontSize', 14, 'FontWeight', 'bold');

% Create figure for event-aligned
figure(2); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)/2]);

for a = 1:length(areas)
    subplot(1, 4, a);
    hold on;
    
    % Plot reach data
    if ~isempty(cumVarReachEvent{a})
        plot(1:length(cumVarReachEvent{a}), cumVarReachEvent{a}, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Reach (n=%d)', numComp50ReachEvent(a)));
    end
    
    % Plot naturalistic data
    if ~isempty(cumVarNatEvent{a})
        plot(1:length(cumVarNatEvent{a}), cumVarNatEvent{a}, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Naturalistic (n=%d)', numComp50NatEvent(a)));
    end
    
    % Add 50% line
    yline(50, 'k--', 'LineWidth', 1, 'DisplayName', '50%');
    
    xlabel('PCA Component');
    ylabel('Cumulative Explained Variance (%)');
    title(sprintf('%s - Event-Aligned', areas{a}));
    legend('Location', 'best');
    grid on;
    ylim([0, 100]);
end

sgtitle(sprintf('PCA Explained Variance - Event-Aligned (Reach: all onsets, Naturalistic: bhvID=%d)', natBhvID), 'FontSize', 14, 'FontWeight', 'bold');

%% ==================== SUMMARY REPORT ====================
fprintf('\n=== SUMMARY: Components Needed for 50%% Variance ===\n');
fprintf('\nWhole Session:\n');
fprintf('  Reach:\n');
for a = 1:length(areas)
    if ~isnan(numComp50ReachWhole(a))
        fprintf('    %s: %d components\n', areas{a}, numComp50ReachWhole(a));
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

