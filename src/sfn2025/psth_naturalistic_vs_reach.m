%% PSTH_NATURALISTIC_VS_REACH - Compare PSTHs for reach and naturalistic data
%
% Produces PSTHs of spikes around:
%   - Reach onset (reach data)
%   - Behavior onsets (naturalistic data)
%
% Plots are stacked by brain area (M23, M56, DS, VS) with:
%   - Reach: single column (peri-reach)
%   - Naturalistic: one column per behavior
%
% Variables:
%   binSize - Bin size for PSTH (seconds)
%   psthWindow - Window around event onset (seconds, total window = 2*psthWindow)
%   baselineWindow - Baseline window for z-scoring [start, end] relative to event (seconds)
%   areas - Brain areas to analyze
%   spikeDataReach - Spike data for reach [time, neuronId, areaNumeric]
%   spikeDataNat - Spike data for naturalistic [time, neuronId, areaNumeric]
%   reachOnsets - Reach onset times (seconds)
%   behaviorOnsets - Cell array of behavior onset times per behavior
%   psthReach - PSTH data for reach [timeBins x areas]
%   psthNat - PSTH data for naturalistic [timeBins x areas x behaviors]

clear; close all;

%% Parameters
binSize = 0.02; % seconds
psthWindow = 2; % seconds (window extends from -psthWindow to +psthWindow around event)
baselineWindow = [-6, -2]; % seconds relative to event onset
sortWindow = [-0.2, 0.2]; % seconds relative to event onset for sorting neurons

% Ensure PSTH window includes baseline window
if baselineWindow(1) < -psthWindow
    psthWindow = abs(baselineWindow(1)); % Extend window to include baseline
    fprintf('Extended PSTH window to %.1f seconds to include baseline\n', psthWindow);
end

% Brain areas: 1=M23, 2=M56, 3=DS, 4=VS
areas = {'M23', 'M56', 'DS', 'VS'};
areaNumeric = [1, 2, 3, 4];

%% Setup paths and options
paths = get_paths;
opts = neuro_behavior_options;
opts.removeSome = true;
opts.firingRateCheckTime = 5 * 60;
opts.minFiringRate = 0.5;
opts.maxFiringRate = 40;

% Add path to figure_tools if needed
if exist('E:/Projects/figure_tools', 'dir')
    addpath('E:/Projects/figure_tools');
elseif exist('Z:/middlebrooks/Projects/figure_tools', 'dir')
    addpath('Z:/middlebrooks/Projects/figure_tools');
elseif exist('/Users/paulmiddlebrooks/Projects/figure_tools', 'dir')
    addpath('/Users/paulmiddlebrooks/Projects/figure_tools');
end

%% ==================== REACH DATA ====================
fprintf('\n=== Loading Reach Data ===\n');

% Load reach data (example session)
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

% Get behavior codes and names
codes = unique(dataBhv.ID);
behaviors = {};
for iBhv = 1:length(codes)
    firstIdx = find(dataBhv.ID == codes(iBhv), 1);
    behaviors = [behaviors, dataBhv.Name{firstIdx}];
end

% Filter behaviors: keep only valid behaviors with at least 20 occurrences
validBehaviors = [];
validCodes = [];
for i = 1:length(codes)
    if codes(i) ~= -1 && sum(dataBhv.ID == codes(i) & dataBhv.Valid) >= 20
        validBehaviors = [validBehaviors, behaviors(i)];
        validCodes = [validCodes, codes(i)];
    end
end

fprintf('Found %d valid behaviors\n', length(validCodes));

% Load spike data
nrnDataPath = strcat(paths.nrnDataPath, 'animal_', animal, '/', sessionNrn, '/recording1/');
opts.dataPath = nrnDataPath;
spikeDataNat = spike_times_per_area(opts);

% Extract behavior onset times for each valid behavior
behaviorOnsets = cell(1, length(validCodes));
for b = 1:length(validCodes)
    bhvMask = (dataBhv.ID == validCodes(b)) & (dataBhv.Valid == 1);
    behaviorOnsets{b} = dataBhv.StartTime(bhvMask);
    fprintf('  Behavior %d (%s): %d onsets\n', validCodes(b), validBehaviors{b}, length(behaviorOnsets{b}));
end

%% ==================== COMPUTE PSTHs ====================

% Time axis for PSTH (centered on event at time 0)
timeAxis = (-psthWindow:binSize:psthWindow)';
numTimeBins = length(timeAxis);

% Bin edges for histcounts (need one more edge than bins)
binEdges = [timeAxis; timeAxis(end) + binSize];

% Baseline indices
baselineIdx = timeAxis >= baselineWindow(1) & timeAxis <= baselineWindow(2);

% Sort window indices
sortIdx = timeAxis >= sortWindow(1) & timeAxis <= sortWindow(2);

fprintf('\n=== Computing Reach PSTHs ===\n');

% Initialize PSTH storage for reach
psthReach = cell(1, length(areas));

for a = 1:length(areas)
    fprintf('Processing area %s...\n', areas{a});
    
    % Get neurons in this area
    areaMask = spikeDataReach(:,3) == areaNumeric(a);
    neuronIds = unique(spikeDataReach(areaMask, 2));
    
    if isempty(neuronIds)
        fprintf('  No neurons found in area %s\n', areas{a});
        psthReach{a} = nan(numTimeBins, 1);
        continue;
    end
    
    fprintf('  Found %d neurons\n', length(neuronIds));
    
    % Initialize storage: [timeBins x neurons x reaches]
    psthPerNeuron = nan(numTimeBins, length(neuronIds), length(reachOnsets));
    
    % Process each reach
    for r = 1:length(reachOnsets)
        reachTime = reachOnsets(r);
        
        % Process each neuron
        for n = 1:length(neuronIds)
            neuronId = neuronIds(n);
            
            % Get spikes for this neuron
            neuronMask = (spikeDataReach(:,2) == neuronId) & (spikeDataReach(:,3) == areaNumeric(a));
            spikeTimes = spikeDataReach(neuronMask, 1);
            
            % Extract spikes in PSTH window relative to reach onset
            windowStart = reachTime - psthWindow;
            windowEnd = reachTime + psthWindow;
            spikeWindow = spikeTimes(spikeTimes >= windowStart & spikeTimes <= windowEnd);
            
            % Convert to relative times (centered on reach onset)
            spikeWindowRel = spikeWindow - reachTime;
            
            % Bin spikes
            spikeCounts = histcounts(spikeWindowRel, binEdges)';
            
            % Z-score using baseline
            baselineCounts = spikeCounts(baselineIdx);
            baselineMean = mean(baselineCounts);
            baselineStd = std(baselineCounts);
            
            if baselineStd > 0
                psthPerNeuron(:, n, r) = (spikeCounts - baselineMean) / baselineStd;
            else
                psthPerNeuron(:, n, r) = zeros(size(spikeCounts));
            end
        end
    end
    
    % Average across reaches to get per-neuron PSTH [timeBins x neurons]
    psthPerNeuronAvg = nanmean(psthPerNeuron, 3);
    
    % Calculate sort metric: mean z-scored value in sort window for each neuron
    sortMetric = nanmean(psthPerNeuronAvg(sortIdx, :), 1);
    
    % Sort neurons from highest to lowest sort metric
    [~, sortOrder] = sort(sortMetric, 'descend');
    
    % Reorder neurons according to sort order
    psthPerNeuronSorted = psthPerNeuronAvg(:, sortOrder);
    
    % Store sorted per-neuron PSTH [timeBins x neurons]
    psthReach{a} = psthPerNeuronSorted;
    
    fprintf('  Sorted %d neurons by mean z-score in sort window\n', length(neuronIds));
end

fprintf('\n=== Computing Naturalistic PSTHs ===\n');

% Initialize PSTH storage for naturalistic
psthNat = cell(1, length(areas));

for a = 1:length(areas)
    fprintf('Processing area %s...\n', areas{a});
    
    % Get neurons in this area
    areaMask = spikeDataNat(:,3) == areaNumeric(a);
    neuronIds = unique(spikeDataNat(areaMask, 2));
    
    if isempty(neuronIds)
        fprintf('  No neurons found in area %s\n', areas{a});
        psthNat{a} = nan(numTimeBins, length(validCodes));
        continue;
    end
    
    fprintf('  Found %d neurons\n', length(neuronIds));
    
    % Initialize storage: [timeBins x neurons x behaviors]
    psthPerBehavior = cell(1, length(validCodes));
    
    % Process each behavior
    for b = 1:length(validCodes)
        behaviorOnsets_b = behaviorOnsets{b};
        
        if isempty(behaviorOnsets_b)
            psthPerBehavior{b} = nan(numTimeBins, length(neuronIds));
            continue;
        end
        
        % Initialize storage: [timeBins x neurons x bouts]
        psthPerNeuron = nan(numTimeBins, length(neuronIds), length(behaviorOnsets_b));
        
        % Process each behavior onset
        for bout = 1:length(behaviorOnsets_b)
            behaviorTime = behaviorOnsets_b(bout);
            
            % Process each neuron
            for n = 1:length(neuronIds)
                neuronId = neuronIds(n);
                
                % Get spikes for this neuron
                neuronMask = (spikeDataNat(:,2) == neuronId) & (spikeDataNat(:,3) == areaNumeric(a));
                spikeTimes = spikeDataNat(neuronMask, 1);
                
                % Extract spikes in PSTH window relative to behavior onset
                windowStart = behaviorTime - psthWindow;
                windowEnd = behaviorTime + psthWindow;
                spikeWindow = spikeTimes(spikeTimes >= windowStart & spikeTimes <= windowEnd);
                
                % Convert to relative times (centered on behavior onset)
                spikeWindowRel = spikeWindow - behaviorTime;
                
                % Bin spikes
                spikeCounts = histcounts(spikeWindowRel, binEdges)';
                
                % Z-score using baseline
                baselineCounts = spikeCounts(baselineIdx);
                baselineMean = mean(baselineCounts);
                baselineStd = std(baselineCounts);
                
                if baselineStd > 0
                    psthPerNeuron(:, n, bout) = (spikeCounts - baselineMean) / baselineStd;
                else
                    psthPerNeuron(:, n, bout) = zeros(size(spikeCounts));
                end
            end
        end
        
        % Average across bouts to get per-neuron PSTH [timeBins x neurons]
        psthPerNeuronAvg = nanmean(psthPerNeuron, 3);
        
        % Calculate sort metric: mean z-scored value in sort window for each neuron
        sortMetric = nanmean(psthPerNeuronAvg(sortIdx, :), 1);
        
        % Sort neurons from highest to lowest sort metric
        [~, sortOrder] = sort(sortMetric, 'descend');
        
        % Reorder neurons according to sort order
        psthPerNeuronSorted = psthPerNeuronAvg(:, sortOrder);
        
        % Store sorted per-neuron PSTH [timeBins x neurons]
        psthPerBehavior{b} = psthPerNeuronSorted;
    end
    
    % Store as cell array [behaviors] of [timeBins x neurons] matrices
    psthNat{a} = psthPerBehavior;
    
    fprintf('  Sorted neurons for each behavior\n');
end

%% ==================== PLOTTING ====================

fprintf('\n=== Creating Plots ===\n');

% Find common color scale range
allValues = [];
for a = 1:length(areas)
    if ~isempty(psthReach{a}) && ~all(isnan(psthReach{a}(:)))
        allValues = [allValues; psthReach{a}(:)];
    end
    if ~isempty(psthNat{a})
        for b = 1:length(psthNat{a})
            if ~isempty(psthNat{a}{b}) && ~all(isnan(psthNat{a}{b}(:)))
                allValues = [allValues; psthNat{a}{b}(:)];
            end
        end
    end
end

if isempty(allValues)
    error('No valid PSTH data to plot');
end

colorRange = [min(allValues), max(allValues)];
% Make symmetric around zero for better visualization
maxAbs = max(abs(colorRange));
colorRange = [-maxAbs, maxAbs];

% Create colormap (function returns colormap and sets clim on current axes)
customColormap = bluewhitered_custom(colorRange);

% Create figure for reach data
figure('Position', [100, 100, 400, 800]);
for a = 1:length(areas)
    subplot(length(areas), 1, a);
    
    if ~isempty(psthReach{a}) && ~all(isnan(psthReach{a}(:)))
        numNeurons = size(psthReach{a}, 2);
        imagesc(timeAxis, 1:numNeurons, psthReach{a}');
        colormap(customColormap);
        caxis(colorRange);
        hold on;
        plot([0, 0], [0.5, numNeurons+0.5], 'k-', 'LineWidth', 1); % Vertical line at event onset
        hold off;
        if a == 1
            colorbar;
        end
        ylabel(areas{a});
        if a == length(areas)
            xlabel('Time from reach onset (s)');
        end
        title(sprintf('Reach PSTH (%d neurons)', numNeurons));
        set(gca, 'YTick', []);
    else
        text(0, 0.5, sprintf('No data for %s', areas{a}), 'HorizontalAlignment', 'center');
        ylabel(areas{a});
    end
end
sgtitle('Reach Data - Peri-Reach PSTH (sorted by mean z-score in [-0.2, 0.2]s)');

% Create figure for naturalistic data
figure('Position', [550, 100, 400*length(validCodes), 800]);
for a = 1:length(areas)
    for b = 1:length(validCodes)
        subplot(length(areas), length(validCodes), (a-1)*length(validCodes) + b);
        
        if ~isempty(psthNat{a}) && length(psthNat{a}) >= b && ~isempty(psthNat{a}{b}) && ~all(isnan(psthNat{a}{b}(:)))
            numNeurons = size(psthNat{a}{b}, 2);
            imagesc(timeAxis, 1:numNeurons, psthNat{a}{b}');
            colormap(customColormap);
            caxis(colorRange);
            hold on;
            plot([0, 0], [0.5, numNeurons+0.5], 'k-', 'LineWidth', 1); % Vertical line at event onset
            hold off;
            if a == 1 && b == length(validCodes)
                colorbar;
            end
            ylabel(areas{a});
            if a == length(areas)
                xlabel('Time from behavior onset (s)');
            end
            if a == 1
                title(sprintf('%s (%d neurons)', validBehaviors{b}, numNeurons), 'Interpreter', 'none');
            end
        else
            text(0, 0.5, sprintf('No data'), 'HorizontalAlignment', 'center');
            ylabel(areas{a});
            if a == 1
                title(validBehaviors{b}, 'Interpreter', 'none');
            end
        end
        set(gca, 'YTick', []);
    end
end
sgtitle('Naturalistic Data - Peri-Behavior PSTH (sorted by mean z-score in [-0.2, 0.2]s)');

fprintf('\n=== Complete ===\n');

