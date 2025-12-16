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
binSize = 0.05; % seconds
psthWindow = 3; % seconds (window extends from -psthWindow to +psthWindow around event) - used for plotting
baselineWindow = [-5, 5]; % seconds relative to event onset
sortWindow = [-0.25, 0.25]; % seconds relative to event onset for sorting neurons

natCodesToPlot = [2 3 8 9 10 11 12 13 14 15];

% Calculate analysis window (may extend beyond psthWindow to include baseline)
% This is used for data extraction, while psthWindow is used for plotting
analysisWindow = psthWindow; % Start with plotting window
if baselineWindow(1) < -psthWindow
    analysisWindow = abs(baselineWindow(1)); % Extend window to include baseline
    fprintf('Analysis window extended to %.1f seconds to include baseline (plotting window remains %.1f seconds)\n', analysisWindow, psthWindow);
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
opts.maxFiringRate = 70;

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
freeDataPath = strcat(paths.freeDataPath, 'animal_', animal, '/', sessionNrn, '/recording1/');
opts.dataPath = freeDataPath;
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
            
            % Extract spikes in analysis window relative to reach onset
            windowStart = reachTime - analysisWindow;
            windowEnd = reachTime + analysisWindow;
            spikeWindow = spikeTimes(spikeTimes >= windowStart & spikeTimes <= windowEnd);
            
            % Convert to relative times (centered on reach onset)
            spikeWindowRel = spikeWindow - reachTime;
            
            % Bin spikes (use extended bin edges if analysis window is larger than psthWindow)
            if analysisWindow > psthWindow
                % Create extended bin edges for analysis window
                extendedTimeAxis = (-analysisWindow:binSize:analysisWindow)';
                extendedBinEdges = [extendedTimeAxis; extendedTimeAxis(end) + binSize];
                spikeCountsExtended = histcounts(spikeWindowRel, extendedBinEdges)';
                
                % Calculate baseline from extended data (where baseline actually falls)
                extendedBaselineIdx = extendedTimeAxis >= baselineWindow(1) & extendedTimeAxis <= baselineWindow(2);
                baselineCounts = spikeCountsExtended(extendedBaselineIdx);
                baselineMean = mean(baselineCounts);
                baselineStd = std(baselineCounts);
                
                % Extract only the portion corresponding to psthWindow
                psthStartIdx = find(extendedTimeAxis >= -psthWindow, 1);
                psthEndIdx = find(extendedTimeAxis <= psthWindow, 1, 'last');
                spikeCounts = spikeCountsExtended(psthStartIdx:psthEndIdx);
            else
                % Use standard bin edges
                spikeCounts = histcounts(spikeWindowRel, binEdges)';
                
                % Z-score using baseline
                baselineCounts = spikeCounts(baselineIdx);
                baselineMean = mean(baselineCounts);
                baselineStd = std(baselineCounts);
            end
            
            % Z-score using baseline
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
                
                % Extract spikes in analysis window relative to behavior onset
                windowStart = behaviorTime - analysisWindow;
                windowEnd = behaviorTime + analysisWindow;
                spikeWindow = spikeTimes(spikeTimes >= windowStart & spikeTimes <= windowEnd);
                
                % Convert to relative times (centered on behavior onset)
                spikeWindowRel = spikeWindow - behaviorTime;
                
                % Bin spikes (use extended bin edges if analysis window is larger than psthWindow)
                if analysisWindow > psthWindow
                    % Create extended bin edges for analysis window
                    extendedTimeAxis = (-analysisWindow:binSize:analysisWindow)';
                    extendedBinEdges = [extendedTimeAxis; extendedTimeAxis(end) + binSize];
                    spikeCountsExtended = histcounts(spikeWindowRel, extendedBinEdges)';
                    
                    % Calculate baseline from extended data (where baseline actually falls)
                    extendedBaselineIdx = extendedTimeAxis >= baselineWindow(1) & extendedTimeAxis <= baselineWindow(2);
                    baselineCounts = spikeCountsExtended(extendedBaselineIdx);
                    baselineMean = mean(baselineCounts);
                    baselineStd = std(baselineCounts);
                    
                    % Extract only the portion corresponding to psthWindow
                    psthStartIdx = find(extendedTimeAxis >= -psthWindow, 1);
                    psthEndIdx = find(extendedTimeAxis <= psthWindow, 1, 'last');
                    spikeCounts = spikeCountsExtended(psthStartIdx:psthEndIdx);
                else
                    % Use standard bin edges
                    spikeCounts = histcounts(spikeWindowRel, binEdges)';
                    
                    % Z-score using baseline
                    baselineCounts = spikeCounts(baselineIdx);
                    baselineMean = mean(baselineCounts);
                    baselineStd = std(baselineCounts);
                end
                
                % Z-score using baseline
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

xlimPlot = [-1 1];

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

% Set colormap range to saturate at ±2
colorRange = [-2, 2];
colorRange = [-.8, .8];

% Create colormap (function returns colormap and sets clim on current axes)
customColormap = bluewhitered_custom(colorRange);

% Monitor setup - prefer second monitor if available
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    targetMonitor = monitorTwo;
else
    targetMonitor = monitorOne;
end

% Create figure for reach data - stack all areas in single plot
figure(33); clf;
ha_rea = tight_subplot(1, 1, [0.02 0.01], [0.05 0.08], [0.02 0.02]);
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)/4]);
axes;

% Combine all areas with separator rows between them
separatorWidth = 2; % Number of separator rows between areas
psthCombined = [];
areaBoundaries = nan(1, length(areas)); % Track where each area starts (for potential labeling)
currentRow = 0;

for a = 1:length(areas)
    if ~isempty(psthReach{a}) && ~all(isnan(psthReach{a}(:)))
        % Add separator row before this area (except first area)
        if a > 1
            separatorRow = zeros(separatorWidth, size(psthReach{a}, 1)); % [separatorWidth x timeBins] - zeros show as white
            psthCombined = [psthCombined; separatorRow]; % Concatenate vertically
            currentRow = currentRow + separatorWidth;
        end
        
        % Track area boundary
        areaBoundaries(a) = currentRow + 1;
        
        % Add this area's data
        psthCombined = [psthCombined; psthReach{a}'];
        currentRow = currentRow + size(psthReach{a}, 2);
    end
end

if ~isempty(psthCombined)
    numTotalNeurons = size(psthCombined, 1);
    imagesc(timeAxis, 1:numTotalNeurons, psthCombined);
    xlim(xlimPlot);
    colormap(customColormap);
    caxis(colorRange);
    hold on;
    
    % Add vertical line at event onset
    plot([0, 0], [0.5, numTotalNeurons+0.5], 'k-', 'LineWidth', 1);
    
    % Add horizontal black lines between areas
    for a = 2:length(areas)
        if ~isnan(areaBoundaries(a)) && areaBoundaries(a) > 1
            boundaryY = areaBoundaries(a) - separatorWidth/2 - 0.5;
            plot(xlimPlot, [boundaryY, boundaryY], 'k-', 'LineWidth', 6);
        end
    end
    
    hold off;
    title(sprintf('Reach PSTH (All Areas, %d neurons)', numTotalNeurons));
else
    text(0, 0.5, 'No data available', 'HorizontalAlignment', 'center');
end

set(gca, 'YTick', [], 'XTick', [], 'XTickLabel', [], 'YTickLabel', []);
sgtitle('Reach Data - Peri-Reach PSTH (sorted by mean z-score in [-0.4, 0.4]s)');

% Create figure for naturalistic data - stack all areas in single plot per behavior
% Filter behaviors to only plot those in natCodesToPlot
plotBehaviorIndices = [];
plotBehaviorCodes = [];
plotBehaviorNames = {};
for i = 1:length(validCodes)
    if ismember(validCodes(i), natCodesToPlot)
        plotBehaviorIndices = [plotBehaviorIndices, i]; % Original index in validCodes
        plotBehaviorCodes = [plotBehaviorCodes, validCodes(i)];
        plotBehaviorNames = [plotBehaviorNames, validBehaviors(i)];
    end
end

fprintf('Plotting %d of %d behaviors (codes: %s)\n', length(plotBehaviorCodes), length(validCodes), mat2str(plotBehaviorCodes));

figure(34); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), targetMonitor(3), targetMonitor(4)]);
ha_nat = tight_subplot(1, length(plotBehaviorCodes), [0.02 0.01], [0.05 0.08], [0.02 0.02]);

for plotIdx = 1:length(plotBehaviorCodes)
    axes(ha_nat(plotIdx));
    
    % Get original behavior index
    origB = plotBehaviorIndices(plotIdx);
    
    % Combine all areas with separator rows between them for this behavior
    separatorWidth = 2; % Number of separator rows between areas
    psthCombined = [];
    areaBoundaries = nan(1, length(areas)); % Track where each area starts
    currentRow = 0;
    
    for a = 1:length(areas)
        if ~isempty(psthNat{a}) && length(psthNat{a}) >= origB && ~isempty(psthNat{a}{origB}) && ~all(isnan(psthNat{a}{origB}(:)))
            % Add separator row before this area (except first area)
            if a > 1
                separatorRow = zeros(separatorWidth, size(psthNat{a}{origB}, 1)); % [separatorWidth x timeBins] - zeros show as white
                psthCombined = [psthCombined; separatorRow]; % Concatenate vertically
                currentRow = currentRow + separatorWidth;
            end
            
            % Track area boundary
            areaBoundaries(a) = currentRow + 1;
            
            % Add this area's data
            psthCombined = [psthCombined; psthNat{a}{origB}'];
            currentRow = currentRow + size(psthNat{a}{origB}, 2);
        end
    end
    
    if ~isempty(psthCombined)
        numTotalNeurons = size(psthCombined, 1);
        imagesc(timeAxis, 1:numTotalNeurons, psthCombined);
        xlim(xlimPlot);
        colormap(ha_nat(plotIdx), customColormap);
        caxis(ha_nat(plotIdx), colorRange);
        hold on;
        
        % Add vertical line at event onset
        plot([0, 0], [0.5, numTotalNeurons+0.5], 'k-', 'LineWidth', 1);
        
        % Add horizontal black lines between areas
        for a = 2:length(areas)
            if ~isnan(areaBoundaries(a)) && areaBoundaries(a) > 1
                boundaryY = areaBoundaries(a) - separatorWidth/2 - 0.5;
                plot(xlimPlot, [boundaryY, boundaryY], 'k-', 'LineWidth', 6);
            end
        end
        
        hold off;
        title(sprintf('%s (%d neurons)', plotBehaviorNames{plotIdx}, numTotalNeurons), 'Interpreter', 'none');
    else
        text(0, 0.5, sprintf('No data'), 'HorizontalAlignment', 'center');
        title(plotBehaviorNames{plotIdx}, 'Interpreter', 'none');
    end
    
    set(gca, 'YTick', [], 'XTick', [], 'XTickLabel', [], 'YTickLabel', []);
end
sgtitle('Naturalistic Data - Peri-Behavior PSTH (sorted by mean z-score in [-0.4, 0.4]s)');

%% Save figures as .eps files
figuresDir = fullfile('/Users/paulmiddlebrooks/Projects/neuro-behavior/src/sfn2025/figures');
if ~exist(figuresDir, 'dir')
    mkdir(figuresDir);
    fprintf('Created figures directory: %s\n', figuresDir);
end

% Save reach figure
figure(33);
saveFileReach = fullfile(figuresDir, 'psth_reach.eps');
print(gcf, '-depsc', '-painters', saveFileReach);
fprintf('Saved reach figure to: %s\n', saveFileReach);

% Save naturalistic figure
figure(34);
saveFileNat = fullfile(figuresDir, 'psth_naturalistic.eps');
print(gcf, '-depsc', '-painters', saveFileNat);
fprintf('Saved naturalistic figure to: %s\n', saveFileNat);

fprintf('\n=== Complete ===\n');


%%
%% Create standalone colorbar figure
figure(35); clf;
set(gcf, 'Units', 'pixels');
% Make it a narrow vertical figure for the colorbar
colorbarWidth = 100; % pixels
colorbarHeight = 400; % pixels
set(gcf, 'Position', [targetMonitor(1), targetMonitor(2), colorbarWidth, colorbarHeight]);

% Create a dummy axes with the colormap
axes('Position', [0.1 0.1 0.3 0.8]);
colormap(customColormap);
caxis(colorRange);

% Create the colorbar
c = colorbar('Location', 'east');
c.Label.String = 'Z-scored firing rate';
c.Label.FontSize = 12;
c.Ticks = linspace(colorRange(1), colorRange(2), 5); % 5 ticks from min to max
c.TickLabels = arrayfun(@(x) sprintf('%.1f', x), c.Ticks, 'UniformOutput', false);

% Remove the axes (we only want the colorbar)
axis off;

% Save colorbar figure
saveFileColorbar = fullfile(figuresDir, 'psth_colorbar.eps');
print(gcf, '-depsc', '-painters', saveFileColorbar);
fprintf('Saved colorbar figure to: %s\n', saveFileColorbar);