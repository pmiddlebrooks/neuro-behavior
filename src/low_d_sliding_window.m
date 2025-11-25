%%
% Low-Dimensional Sliding Window Analysis
% Compares how average low-d representations change across a session
% Projects whole session data into low-d space (PCA) and analyzes using sliding windows
%
% Analysis modes:
%   - 'trans': Average component values aligned around behavior onsets
%   - 'within': Mean component values during behavior bouts
%


% Write a new script, low_d_sliding_window.m
% The purpose of this analysis is to compare how average low-d representations change across a session.
% This script projects the whole session of naturalistic or reach spiking
% data (with binSize opts.frameSize) into low-d representations (start with
% PCA), and uses a slidingWindowSize and a stepSize to collect data across the session for analysis. 
% Allow user to specify which bhvID labels to collect/analyze.
% Allow user to specify which dataType to analyze:
% Naturalistic:
% - if user chooses trans
%   - Within each window, for each of the first nDim components of the
%   low-projection, collect the average component value algned in a
%   transWindow around the onset of each bhvID category (except bhvID ==
%   -1). Plot the mean transWindows for each bhvID, one peri-onset line per
%   slidingWindow. Each bhvID category gets its own column of axes in a
%   tight_subplot. Each row of the plot is one of the nDim components.   
% - if user chooses within
%   - Within each window (for each of the first nDim components), collect
%   the mean component value meanWithin of all collected bhvID indices for
%   each bhvID category, beginning from afterOnset after the onset of a
%   bout until beforeNext before the onset of whatever next bhvID occurs
%   next. Plot the meanWithin value of each comopnent as a function of the
%   center of the slidingWindow throughout the session. Each column is a
%   bhvID label, each row is one of the nDim components.   
% Reach:
% - Do the same, using define_reach_bhv_labels to get bhvID.
% 
% This script follows much of the same logic as @svm_decoding_compare.m  

paths = get_paths;

%% =============================    Configuration    =============================
% Data type selection
dataType = 'naturalistic';  % 'reach' or 'naturalistic'

% Analysis type
analysisType = 'trans';  % 'trans' or 'within'

% Low-d projection parameters
nDim = 6;  % Number of PCA dimensions to use

% Sliding window parameters
slidingWindowSize = 10*60;  % Window size in seconds
stepSize = 5*60;  % Step size in seconds

% Frame/bin size
frameSize = 0.05;  % seconds

% Behavior IDs to analyze (empty = all valid behaviors, exclude -1)
bhvIDsToAnalyze = [];  % Empty = analyze all valid behaviors
bhvIDsToAnalyze = [1,6,8,9,10,11,12,13,15];  % Empty = analyze all valid behaviors

% Analysis-specific parameters
if strcmp(analysisType, 'trans')
    transWindow = [-0.5, 0.5];  % Time window around behavior onset [before, after] in seconds
elseif strcmp(analysisType, 'within')
    afterOnset = 0.1;  % Start collecting after this time (seconds) from behavior onset
    beforeNext = 0.1;  % Stop collecting before this time (seconds) before next behavior
end

% Brain areas to analyze
areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 2:3;  % All areas

% Plotting options
savePlots = false;
saveDir = fullfile(paths.dropPath, 'low_d_sliding_window');

%% =============================    Data Loading    =============================
fprintf('\n=== Loading %s data ===\n', dataType);

    opts = neuro_behavior_options;
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0;
    opts.minFiringRate = 0.5;
    opts.maxFiringRate = 70;
    opts.frameSize = frameSize;

if strcmp(dataType, 'reach')
    % Load reach data
    sessionName = 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat';
    reachDataFile = fullfile(paths.reachDataPath, sessionName);
    
    dataR = load(reachDataFile);
    
    opts.collectEnd = round(min(dataR.R(end,1) + 8000, max(dataR.CSV(:,1)*1000)) / 1000);
    
    [dataMat, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
    
    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idList = {idM23, idM56, idDS, idVS};
    
    % Get behavior labels for reach task
    bhvOpts = struct();
    bhvOpts.frameSize = frameSize;
    bhvOpts.collectStart = opts.collectStart;
    bhvOpts.collectEnd = opts.collectEnd;
    bhvID = define_reach_bhv_labels(reachDataFile, bhvOpts);
    
    % Behavior labels for reach
    behaviors = {'pre-reach', 'reach', 'pre-reward', 'reward', 'post-reward', 'intertrial'};
    
    fprintf('Loaded reach data: %d neurons, %d time points\n', size(dataMat, 2), size(dataMat, 1));
    
elseif strcmp(dataType, 'naturalistic')
    % Load naturalistic data
    opts.minActTime = 0.16;
    opts.collectEnd = 45 * 60; % seconds
    
    % Get neural data
    getDataType = 'spikes';
    get_standard_data
    
    % Curate behavior labels
    [dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);
    
    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idList = {idM23, idM56, idDS, idVS};
    
    % Behavior labels for naturalistic (from get_standard_data)
    behaviors = {'investigate_1', 'investigate_2', 'investigate_3', ...
        'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
        'head_groom', 'contra_body_groom', 'ipsi_body_groom', 'contra_itch', ...
        'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};
    
    fprintf('Loaded naturalistic data: %d neurons, %d time points\n', size(dataMat, 2), size(dataMat, 1));
else
    error('Invalid dataType. Must be ''reach'' or ''naturalistic''');
end

% Determine which behavior IDs to analyze
if isempty(bhvIDsToAnalyze)
    % Get all valid behavior IDs (exclude -1)
    uniqueBhvIDs = unique(bhvID);
    bhvIDsToAnalyze = uniqueBhvIDs(uniqueBhvIDs >= 0);
end

fprintf('Analyzing behavior IDs: %s\n', mat2str(bhvIDsToAnalyze));

%% =============================    PCA Projection    =============================
fprintf('\n=== Computing PCA projection ===\n');

% Project whole session data into low-d space for each area
pcaResults = cell(1, length(areas));
for a = areasToTest
    fprintf('Computing PCA for area %s...\n', areas{a});
    idSelect = idList{a};
    
    if length(idSelect) < nDim
        fprintf('  Skipping %s: not enough neurons (%d < %d)\n', areas{a}, length(idSelect), nDim);
        continue;
    end
    
    % Compute PCA on z-scored data
    [coeff, score, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));
    
    % Store first nDim components
    pcaResults{a} = struct();
    pcaResults{a}.latents = score;
    pcaResults{a}.coeff = coeff;
    pcaResults{a}.explained = explained;
    
    fprintf('  Area %s: Explained variance: %.2f%%\n', areas{a}, sum(explained(1:nDim)));
end

%% =============================    Find Behavior Onsets    =============================
fprintf('\n=== Finding behavior onsets ===\n');

% Find onset times for each behavior ID
bhvOnsets = cell(1, length(bhvIDsToAnalyze));
for b = 1:length(bhvIDsToAnalyze)
    bhvIDVal = bhvIDsToAnalyze(b);
    
    if strcmp(dataType, 'naturalistic')
        % For naturalistic: use dataBhv.StartTime directly
        if exist('dataBhv', 'var')
            bhvMask = (dataBhv.ID == bhvIDVal) & (dataBhv.Valid == 1);
            bhvOnsets{b} = dataBhv.StartTime(bhvMask);
        else
            % Fallback: find transitions in bhvID
            isTargetBhv = (bhvID == bhvIDVal);
            transitions = find(diff([0; isTargetBhv]) == 1);
            bhvOnsets{b} = (transitions - 1) * frameSize;
        end
    else
        % For reach: find where bhvID changes to this value
        isTargetBhv = (bhvID == bhvIDVal);
        transitions = find(diff([0; isTargetBhv]) == 1);
        bhvOnsets{b} = (transitions - 1) * frameSize;
    end
    
    fprintf('  Behavior ID %d: %d onsets\n', bhvIDVal, length(bhvOnsets{b}));
end

%% =============================    Sliding Window Analysis    =============================
fprintf('\n=== Sliding window analysis (type: %s) ===\n', analysisType);

% Calculate number of windows
totalTime = size(dataMat, 1) * frameSize;
numWindows = floor((totalTime - slidingWindowSize) / stepSize) + 1;
windowCenters = (0:numWindows-1) * stepSize + slidingWindowSize / 2;

fprintf('Total time: %.1f s, Window size: %.1f s, Step size: %.1f s\n', totalTime, slidingWindowSize, stepSize);
fprintf('Number of windows: %d\n', numWindows);

% Initialize results storage
if strcmp(analysisType, 'trans')
    % For trans: store aligned traces for each window
    % Structure: results{area}{bhvID}{component}(window, timeBin)
    results = cell(1, length(areas));
    for a = areasToTest
        results{a} = cell(1, length(bhvIDsToAnalyze));
        for b = 1:length(bhvIDsToAnalyze)
            results{a}{b} = cell(1, nDim);
            for d = 1:nDim
                results{a}{b}{d} = cell(numWindows, 1);
            end
        end
    end
else
    % For within: store mean values for each window
    % Structure: results{area}{bhvID}{component}(window)
    results = cell(1, length(areas));
    for a = areasToTest
        results{a} = cell(1, length(bhvIDsToAnalyze));
        for b = 1:length(bhvIDsToAnalyze)
            results{a}{b} = cell(1, nDim);
            for d = 1:nDim
                results{a}{b}{d} = nan(numWindows, 1);
            end
        end
    end
end

% Process each sliding window
for w = 1:numWindows
    windowStart = (w - 1) * stepSize;
    windowEnd = windowStart + slidingWindowSize;
    windowCenter = windowStart + slidingWindowSize / 2;
    
    % Convert to bin indices
    binStart = floor(windowStart / frameSize) + 1;
    binEnd = min(floor(windowEnd / frameSize), size(dataMat, 1));
    
    if binEnd <= binStart
        continue;
    end
    
    windowBins = binStart:binEnd;
    windowBhvID = bhvID(windowBins);
    windowTime = (windowBins - 1) * frameSize;
    
    if mod(w, max(1, round(numWindows/10))) == 0
        fprintf('  Processing window %d/%d (center: %.1f s)\n', w, numWindows, windowCenter);
    end
    
    % Process each area
    for a = areasToTest
        if isempty(pcaResults{a})
            continue;
        end
        
        latents = pcaResults{a}.latents;
        windowLatents = latents(windowBins, :);  % [nBins x nDim]
        
        % Process each behavior ID
        for b = 1:length(bhvIDsToAnalyze)
            bhvIDVal = bhvIDsToAnalyze(b);
            
                if strcmp(analysisType, 'trans')
                % Find onsets of this behavior within the window
                windowOnsets = bhvOnsets{b}(bhvOnsets{b} >= windowStart & bhvOnsets{b} < windowEnd);
                
                if isempty(windowOnsets)
                    continue;
                end
                
                % For each component, collect and average aligned traces
                for d = 1:nDim
                    allAlignedTraces = [];  % Collect all [time, value] pairs from all onsets
                    
                    for o = 1:length(windowOnsets)
                        onsetTime = windowOnsets(o);
                        
                        % Find time bins within transWindow around this onset
                        transStart = onsetTime + transWindow(1);
                        transEnd = onsetTime + transWindow(2);
                        
                        % Check if full transWindow is within the sliding window
                        if transStart < windowStart || transEnd > windowEnd
                            continue;  % Skip this onset if full window is not available
                        end
                        
                        % Find bins within this window (relative to windowStart)
                        binMask = windowTime >= transStart & windowTime <= transEnd;
                        if ~any(binMask)
                            continue;
                        end
                        
                        % Get component values
                        componentVals = windowLatents(binMask, d);
                        
                        % Create time axis relative to onset
                        timeAxisRel = windowTime(binMask) - onsetTime;
                        
                        % Ensure both are column vectors
                        if isrow(componentVals)
                            componentVals = componentVals';
                        end
                        if isrow(timeAxisRel)
                            timeAxisRel = timeAxisRel';
                        end
                        
                        % Collect [time, value] pairs from this onset
                        if isempty(allAlignedTraces)
                            allAlignedTraces = [timeAxisRel, componentVals];
                        else
                            allAlignedTraces = [allAlignedTraces; timeAxisRel, componentVals];
                        end
                    end
                    
                    % Average aligned traces by binning time points
                    if ~isempty(allAlignedTraces)
                        % Create time bin edges based on transWindow and frameSize
                        timeBinEdges = transWindow(1):frameSize:transWindow(2);
                        % Add the upper edge
                        if timeBinEdges(end) < transWindow(2)
                            timeBinEdges = [timeBinEdges, transWindow(2)];
                        end
                        
                        % Calculate bin centers
                        timeBinCenters = (timeBinEdges(1:end-1) + timeBinEdges(2:end)) / 2;
                        numTimeBins = length(timeBinCenters);
                        
                        % Initialize averaged trace
                        averagedTrace = zeros(numTimeBins, 2);
                        averagedTrace(:, 1) = timeBinCenters';
                        
                        % Bin and average values
                        for t = 1:numTimeBins
                            % Find all values within this time bin
                            timeMask = allAlignedTraces(:, 1) >= timeBinEdges(t) & ...
                                      allAlignedTraces(:, 1) < timeBinEdges(t+1);
                            % Handle the last bin edge inclusively
                            if t == numTimeBins
                                timeMask = allAlignedTraces(:, 1) >= timeBinEdges(t) & ...
                                          allAlignedTraces(:, 1) <= timeBinEdges(t+1);
                            end
                            
                            if any(timeMask)
                                averagedTrace(t, 2) = nanmean(allAlignedTraces(timeMask, 2));
                            else
                                averagedTrace(t, 2) = NaN;
                            end
                        end
                        
                        % Store averaged trace for this window
                        results{a}{b}{d}{w} = averagedTrace;
                    end
                end
                
            else  % analysisType == 'within'
                % Find all time bins within behavior bouts
                % A bout starts afterOnset after an onset and ends beforeNext before the next behavior
                
                % Find behavior onsets within window
                windowOnsets = bhvOnsets{b}(bhvOnsets{b} >= windowStart & bhvOnsets{b} < windowEnd);
                
                if isempty(windowOnsets)
                    continue;
                end
                
                % For each component, collect mean values during behavior bouts
                for d = 1:nDim
                    componentVals = [];
                    
                    for o = 1:length(windowOnsets)
                        onsetTime = windowOnsets(o);
                        
                        % Find the next behavior onset (any behavior)
                        nextOnsetIdx = find(bhvOnsets{b} > onsetTime, 1);
                        if isempty(nextOnsetIdx)
                            % Use end of window as next onset
                            nextOnsetTime = windowEnd;
                        else
                            % Check all behavior IDs for next onset
                            nextOnsetTime = windowEnd;
                            for b2 = 1:length(bhvIDsToAnalyze)
                                nextOnsets = bhvOnsets{b2}(bhvOnsets{b2} > onsetTime);
                                if ~isempty(nextOnsets) && nextOnsets(1) < nextOnsetTime
                                    nextOnsetTime = nextOnsets(1);
                                end
                            end
                            nextOnsetTime = min(nextOnsetTime, windowEnd);
                        end
                        
                        % Define bout window: afterOnset after onset to beforeNext before next
                        boutStart = onsetTime + afterOnset;
                        boutEnd = nextOnsetTime - beforeNext;
                        
                        if boutEnd <= boutStart
                            continue;
                        end
                        
                        % Find bins within this bout
                        boutBins = windowBins(windowTime >= boutStart & windowTime < boutEnd);
                        if isempty(boutBins)
                            continue;
                        end
                        
                        % Get component values for these bins
                        boutBinIndices = boutBins - binStart + 1;  % Relative to windowBins
                        boutComponentVals = windowLatents(boutBinIndices, d);
                        
                        componentVals = [componentVals; boutComponentVals];
                    end
                    
                    % Store mean value for this window
                    if ~isempty(componentVals)
                        results{a}{b}{d}(w) = nanmean(componentVals);
                    end
                end
            end
        end
    end
end

%% =============================    Plotting    =============================
fprintf('\n=== Creating plots ===\n');

if ~exist(saveDir, 'dir') && savePlots
    mkdir(saveDir);
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

for a = areasToTest
    if isempty(pcaResults{a})
        continue;
    end
    
    fprintf('Plotting area %s...\n', areas{a});
    
    % Determine number of behavior IDs to plot
    nBhvToPlot = length(bhvIDsToAnalyze);
    
    % Create figure with subplots: rows = components, columns = behaviors
    fig = figure('Position', targetPos);
    
    if strcmp(analysisType, 'trans')
        % For trans: plot aligned traces, one line per window
        ha = tight_subplot(nDim, nBhvToPlot, [0.02 0.02], [0.08 0.05], [0.06 0.03]);
        
        for d = 1:nDim
            for b = 1:nBhvToPlot
                axIdx = (d-1)*nBhvToPlot + b;
                axes(ha(axIdx));
                hold on;
                
                % Collect all traces across windows
                allTraces = [];
                windowIndices = [];
                for w = 1:numWindows
                    if ~isempty(results{a}{b}{d}{w})
                        allTraces = [allTraces; results{a}{b}{d}{w}];
                        windowIndices = [windowIndices; w*ones(size(results{a}{b}{d}{w}, 1), 1)];
                    end
                end
                
                if ~isempty(allTraces)
                    % Create color gradient from saturated red to mid-gray
                    % Saturated red: [1, 0, 0]
                    % Mid-gray: [0.5, 0.5, 0.5]
                    redColor = [1, 0, 0];
                    grayColor = [0.5, 0.5, 0.5];
                    
                    % Get unique windows and normalize their positions based on window center times
                    uniqueWindows = unique(windowIndices);
                    if length(uniqueWindows) > 1
                        % Use window center times to determine position in session
                        uniqueWindowCenters = windowCenters(uniqueWindows);
                        windowPositions = (uniqueWindowCenters - min(uniqueWindowCenters)) / (max(uniqueWindowCenters) - min(uniqueWindowCenters));
                    else
                        windowPositions = 0.5;  % Single window, use middle color
                    end
                    
                    % Plot individual traces (one per window) with color gradient
                    for wIdx = 1:length(uniqueWindows)
                        w = uniqueWindows(wIdx);
                        traceMask = windowIndices == w;
                        traceData = allTraces(traceMask, :);
                        if ~isempty(traceData)
                            % Interpolate color based on window position in session
                            color = redColor * (1 - windowPositions(wIdx)) + grayColor * windowPositions(wIdx);
                            plot(traceData(:, 1), traceData(:, 2), '-', 'LineWidth', 1.5, 'Color', color);
                        end
                    end
                    
                end
                
                % Formatting
                xlabel('Time from onset (s)');
                ylabel(sprintf('PC%d', d));
                if d == 1
                    if strcmp(dataType, 'reach')
                        title(sprintf('%s', behaviors{bhvIDsToAnalyze(b)}));
                    else
                        title(sprintf('bhvID %d', bhvIDsToAnalyze(b)));
                    end
                end
                if b == 1
                    ylabel(sprintf('PC%d', d));
                end
                xline(0, 'k--', 'LineWidth', 1);
                grid on;
            end
        end
        
        sgtitle(sprintf('%s - %s (transWindow: [%.2f, %.2f] s)', areas{a}, analysisType, transWindow(1), transWindow(2)), ...
            'FontSize', 14, 'FontWeight', 'bold');
        
    else  % analysisType == 'within'
        % For within: plot mean values vs window center time
        ha = tight_subplot(nDim, nBhvToPlot, [0.05 0.02], [0.08 0.05], [0.06 0.03]);
        
        for d = 1:nDim
            for b = 1:nBhvToPlot
                axIdx = (d-1)*nBhvToPlot + b;
                axes(ha(axIdx));
                hold on;
                
                % Get mean values for this component and behavior
                meanVals = results{a}{b}{d};
                validWindows = ~isnan(meanVals);
                
                if any(validWindows)
                    plot(windowCenters(validWindows), meanVals(validWindows), '-', 'LineWidth', 2, 'Color', [0 0 1]);
                end
                
                % Formatting
                xlabel('Window center time (s)');
                ylabel(sprintf('PC%d mean', d));
                if d == 1
                    if strcmp(dataType, 'reach')
                        title(sprintf('%s', behaviors{bhvIDsToAnalyze(b)}));
                    else
                        title(sprintf('bhvID %d', bhvIDsToAnalyze(b)));
                    end
                end
                if b == 1
                    ylabel(sprintf('PC%d mean', d));
                end
                grid on;
            end
        end
        
        sgtitle(sprintf('%s - %s (afterOnset: %.2f s, beforeNext: %.2f s)', areas{a}, analysisType, afterOnset, beforeNext), ...
            'FontSize', 14, 'FontWeight', 'bold');
    end
    
    % Save figure
    if savePlots
        filename = sprintf('low_d_%s_%s_%s_win%.0f_step%.0f_nDim%d.eps', ...
            dataType, analysisType, areas{a}, slidingWindowSize, stepSize, nDim);
        filepath = fullfile(saveDir, filename);
        exportgraphics(fig, filepath, 'ContentType', 'vector');
        fprintf('  Saved: %s\n', filename);
    end
end

fprintf('\n=== Analysis complete ===\n');

