%%
% Peri-Behavior HMM State Analysis for Naturalistic Data
% Loads results from hmm_mazz_nat.m and analyzes HMM state sequences
% around behavior onset times for each brain area
%
% Variables:
%   results - loaded HMM analysis results
%   dataBhv - naturalistic behavioral data
%   areas - brain areas to analyze
%   bhvStartIDs - behavior IDs to align on
%   stateWindows - HMM state sequences in windows around each behavior onset
%   stateImages - imagesc plots of state sequences for each behavior

%% Load existing results if requested
paths = get_paths;
binSize = .01;
minDur = .04;
bhvStartIDs = [1, 5, 11, 15];

areasToTest = 1:4;

% Sliding metrics window (slides across the peri window for entropy/behavior/kin)
slidingWindowSize = 2.0;  % seconds
metricsStepSec = [];     % default: one HMM bin step (set per area)

% Define window parameters
periBhvWindow = 30; % peri-behavior window around each behavior onset (used for all analyses)

% Determine save directory and filename based on parameters
hmmdir = fullfile(paths.dropPath, 'metastability');
filename = sprintf('hmm_mazz_nat_bin%.3f_minDur%.3f_bhv%s.mat', binSize, minDur, mat2str(bhvStartIDs));
resultsPath = fullfile(hmmdir, filename);

% Extract first 10 characters of filename for titles and file names
filePrefix = 'Nat';

% Load HMM analysis results
fprintf('Loading HMM analysis results from: %s\n', resultsPath);
if ~exist(resultsPath, 'file')
    error('Results file not found: %s\nMake sure hmm_mazz_nat.m has been run for this dataset.', resultsPath);
end
results = load(resultsPath);
results = results.results;

% Extract areas and parameters
areas = results.areas;
binSizes = results.binSize;
numStates = results.numStates;
hmmResults = results.hmm_results;

% Prepare distinct colormaps per area for states and metastates
stateCmaps = cell(1, length(areas));
metaCmaps = cell(1, length(areas));
for a = 1:length(areas)
    try
        if numStates(a) > 0
            rgb = maxdistcolor(numStates(a), @sRGB_to_OKLab);
            stateCmaps{a} = [1 1 1; rgb]; % state 0 is white
        else
            stateCmaps{a} = lines(max(1, numStates(a)+1));
            stateCmaps{a}(1,:) = [1 1 1];
        end
    catch
        % Fallback if maxdistcolor not available
        ctmp = lines(max(1, numStates(a)+1));
        ctmp(1,:) = [1 1 1];
        stateCmaps{a} = ctmp;
    end
    % Metastate colormap (if available)
    if ~isempty(hmmResults{a}) && isfield(hmmResults{a}, 'metastate_results') && isfield(hmmResults{a}.metastate_results, 'num_metastates')
        nMeta = hmmResults{a}.metastate_results.num_metastates;
        try
            if nMeta > 0
                rgbm = maxdistcolor(nMeta, @sRGB_to_OKLab);
                metaCmaps{a} = [1 1 1; rgbm]; % metastate 0 is white
            else
                metaCmaps{a} = lines(1); metaCmaps{a}(1,:) = [1 1 1];
            end
        catch
            ctmp = lines(max(1, nMeta+1));
            ctmp(1,:) = [1 1 1];
            metaCmaps{a} = ctmp;
        end
    else
        metaCmaps{a} = [];
    end
end

% Load naturalistic behavioral data
fprintf('Loading naturalistic behavioral data...\n');
opts = neuro_behavior_options;
% Use first area's bin size for behavior data consistency
opts.frameSize = binSizes(find(binSizes>0,1,'first'));
getDataType = 'behavior';
get_standard_data
[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);
binSizeBhv = opts.frameSize;

% Find behavior onsets for each bhvStartID
fprintf('\n=== Finding Behavior Onsets ===\n');
bhvOnsetTimes = cell(1, length(bhvStartIDs));
bhvOnsetIndices = cell(1, length(bhvStartIDs));
for b = 1:length(bhvStartIDs)
    bhvID_target = bhvStartIDs(b);
    % Find transitions into this behavior (onset)
    isTargetBhv = (bhvID == bhvID_target);
    transitions = find(diff([0; isTargetBhv]) == 1); % transitions into this behavior
    bhvOnsetIndices{b} = transitions;
    bhvOnsetTimes{b} = (transitions - 1) * binSizeBhv; % Convert to seconds
    fprintf('Behavior ID %d: %d onsets found\n', bhvID_target, length(transitions));
end

% ==============================================     Peri-Behavior Analysis     ==============================================

% Initialize storage for peri-behavior state sequences (separate by behavior ID)
stateWindowsByBhv = cell(length(bhvStartIDs), length(areas));

% Initialize storage for peri-behavior metastate sequences (separate by behavior ID)
metastateWindowsByBhv = cell(length(bhvStartIDs), length(areas));

% Flags indicating availability of data to analyze/plot per area
hasHmmArea = false(1, length(areas));

fprintf('\n=== Peri-Behavior HMM State Analysis ===\n');

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});

    % Get HMM results for this area
    hmmRes = hmmResults{a};
    
    % Check if HMM analysis was successful for this area
    if isempty(hmmRes) || ~strcmp(hmmRes.metadata.analysis_status, 'SUCCESS')
        fprintf('Skipping area %s due to failed HMM analysis.\n', areas{a});
        continue
    end
    
    hasHmmArea(a) = true;
    
    % Get continuous state sequence and parameters
    continuousSequence = hmmRes.continuous_results.sequence;
    binSize = hmmRes.HmmParam.BinSize;
    totalTimeBins = length(continuousSequence);
    
    % Get continuous metastate sequence if available
    if isfield(hmmRes, 'metastate_results') && ~isempty(hmmRes.metastate_results.continuous_metastates)
        continuousMetastates = hmmRes.metastate_results.continuous_metastates;
        hasMetastates = true;
    else
        continuousMetastates = [];
        hasMetastates = false;
    end
    
    % Calculate window duration in bins
    windowDurationBins = ceil(periBhvWindow / binSize);
    halfWindow = floor(windowDurationBins / 2);
    
    % Extract around each behavior onset for each behavior ID
    for b = 1:length(bhvStartIDs)
        numOnsets = length(bhvOnsetIndices{b});
        
        % Initialize storage for this behavior ID
        stateWindowsByBhv{b, a} = nan(numOnsets, windowDurationBins + 1);
        
        if hasMetastates
            metastateWindowsByBhv{b, a} = nan(numOnsets, windowDurationBins + 1);
        end
        
        validOnsets = 0;
        for o = 1:numOnsets
            % Convert behavior onset time to HMM bin index
            bhvOnsetTime = bhvOnsetTimes{b}(o);
            bhvBin = round(bhvOnsetTime / binSize) + 1; % Convert to bin index (1-indexed)
            
            winStart = bhvBin - halfWindow;
            winEnd = bhvBin + halfWindow;
            
            if winStart >= 1 && winEnd <= totalTimeBins
                stateWindowsByBhv{b, a}(o, :) = continuousSequence(winStart:winEnd);
                if hasMetastates
                    metastateWindowsByBhv{b, a}(o, :) = continuousMetastates(winStart:winEnd);
                end
                validOnsets = validOnsets + 1;
            else
                stateWindowsByBhv{b, a}(o, :) = nan(1, windowDurationBins + 1);
                if hasMetastates
                    metastateWindowsByBhv{b, a}(o, :) = nan(1, windowDurationBins + 1);
                end
            end
        end
        
        fprintf('Area %s, Behavior ID %d: %d valid onsets\n', areas{a}, bhvStartIDs(b), validOnsets);
    end
end

% ==============================================     Sliding Window Metrics (State/Metastate/Behavior/Kin)     ==============================================

% Initialize storage for sliding-window metrics per area
stateEntropyWindows = cell(1, length(areas));        % (all onsets x frames)
metaEntropyWindows = cell(1, length(areas));         % (all onsets x frames)
behaviorSwitchesWindows = cell(1, length(areas));    % (all onsets x frames)
behaviorProportionWindows = cell(1, length(areas));  % (all onsets x frames)
behaviorEntropyWindows = cell(1, length(areas));     % (all onsets x frames)
kinPCAStdWindows = cell(1, length(areas));           % (all onsets x frames)

% Build a binary behavior timeline at reference binning for behavior metrics
refBinSize = binSizes(find(binSizes>0,1,'first')); if isempty(refBinSize), refBinSize = 0.01; end
recordingDurationSec = max(cellfun(@max, bhvOnsetTimes));
recordingDurationSec = max(recordingDurationSec, length(bhvID) * binSizeBhv);
bhvTimelineLen = ceil(recordingDurationSec / refBinSize);
bhvTimeline = zeros(bhvTimelineLen,1);
% Create timeline where each behavior ID is marked
for i = 1:length(bhvID)
    if bhvID(i) > 0
        idx = max(1, min(bhvTimelineLen, floor((i-1)*binSizeBhv/refBinSize)+1));
        bhvTimeline(idx) = bhvID(i);
    end
end

% Optional kinematics (load if available)
opts.frameSize = refBinSize;
getDataType = 'kinematics';
try
    get_standard_data
    kinDown = zscore(kinData(:,1)); % Use first kinematic component
    hasKin = true;
catch
    kinDown = [];
    hasKin = false;
    fprintf('Warning: Kinematic data not available, skipping kinPCAStd calculation.\n');
end

% Note: Entropy computation now matches hmm_correlations.m approach (inline code below)

for a = areasToTest
    if ~hasHmmArea(a), continue; end
    hmmRes = hmmResults{a};
    binSizeA = binSizes(a);
    winBinsA = ceil(periBhvWindow / binSizeA);
    halfWinA = floor(winBinsA/2);
    nFramesPeri = winBinsA + 1;

    % local step/size
    stepBins = max(1, round((isempty(metricsStepSec) || isnan(metricsStepSec) || metricsStepSec<=0)*binSizeA/binSizeA + (~(isempty(metricsStepSec) || isnan(metricsStepSec) || metricsStepSec<=0))*metricsStepSec/binSizeA));
    metWinBins = max(1, round(slidingWindowSize / binSizeA));

    seq = hmmRes.continuous_results.sequence;
    hasMeta = isfield(hmmRes,'metastate_results') && isfield(hmmRes.metastate_results,'continuous_metastates') && ~isempty(hmmRes.metastate_results.continuous_metastates);
    if hasMeta, metaSeq = hmmRes.metastate_results.continuous_metastates; else, metaSeq = []; end
    tA = (0:length(seq)-1) * binSizeA;
    mapIdx = @(tSec) max(1, min(length(bhvTimeline), round(tSec/refBinSize)+1));

    % Collect all behavior onsets across all behavior IDs
    allBhvOnsetTimes = [];
    for b = 1:length(bhvStartIDs)
        allBhvOnsetTimes = [allBhvOnsetTimes; bhvOnsetTimes{b}];
    end
    allBhvOnsetTimes = sort(allBhvOnsetTimes);
    
    stateEntropyWindows{a} = nan(length(allBhvOnsetTimes), nFramesPeri);
    metaEntropyWindows{a} = nan(length(allBhvOnsetTimes), nFramesPeri);
    behaviorSwitchesWindows{a} = nan(length(allBhvOnsetTimes), nFramesPeri);
    behaviorProportionWindows{a} = nan(length(allBhvOnsetTimes), nFramesPeri);
    behaviorEntropyWindows{a} = nan(length(allBhvOnsetTimes), nFramesPeri);
    kinPCAStdWindows{a} = nan(length(allBhvOnsetTimes), nFramesPeri);

    for r = 1:length(allBhvOnsetTimes)
        centerTime = allBhvOnsetTimes(r);
        [~, cIdx] = min(abs(tA - centerTime));
        periStart = cIdx - halfWinA; periEnd = cIdx + halfWinA;
        if periStart < 1 || periEnd > length(seq), continue; end

        for p = periStart:stepBins:periEnd
            swStart = max(periStart, p - floor(metWinBins/2));
            swEnd = min(periEnd, swStart + metWinBins - 1);
            swStart = max(periStart, swEnd - metWinBins + 1);
            if swEnd <= swStart, continue; end
            periPos = (p - periStart) + 1;

            % State entropy (same approach as hmm_correlations.m)
            segS = seq(swStart:swEnd); segS = segS(segS>0);
            if isempty(segS) || all(isnan(segS))
                stateEntropyWindows{a}(r, periPos) = NaN;
            elseif length(unique(segS)) <= 1
                stateEntropyWindows{a}(r, periPos) = 0;
            else
                % Count occurrences of each state
                stateCounts = histcounts(segS, 1:(numStates(a)+1));
                stateProportions = stateCounts / length(segS);
                stateProportions = stateProportions(stateProportions > 0); % Remove zero probabilities
                if ~isempty(stateProportions)
                    stateEntropyWindows{a}(r, periPos) = -sum(stateProportions .* log2(stateProportions));
                else
                    stateEntropyWindows{a}(r, periPos) = NaN;
                end
            end

            % Metastate entropy (same approach as hmm_correlations.m)
            if hasMeta
                segM = metaSeq(swStart:swEnd); segM = segM(segM>0);
                if isempty(segM) || all(isnan(segM))
                    metaEntropyWindows{a}(r, periPos) = NaN;
                elseif length(unique(segM)) <= 1
                    metaEntropyWindows{a}(r, periPos) = 0;
                else
                    % Get maximum metastate value
                    maxMetastate = hmmRes.metastate_results.num_metastates;
                    % Count occurrences of each metastate
                    metaCounts = histcounts(segM, 1:(maxMetastate+1));
                    metaProportions = metaCounts / length(segM);
                    metaProportions = metaProportions(metaProportions > 0); % Remove zero probabilities
                    if ~isempty(metaProportions)
                        metaEntropyWindows{a}(r, periPos) = -sum(metaProportions .* log2(metaProportions));
                    else
                        metaEntropyWindows{a}(r, periPos) = NaN;
                    end
                end
            end

            % Behavior metrics from behavior timeline
            tStart = (swStart-1)*binSizeA; tEnd = (swEnd-1)*binSizeA;
            bStart = mapIdx(tStart); bEnd = mapIdx(tEnd);
            if bEnd >= bStart
                segB = bhvTimeline(bStart:bEnd);
                if ~isempty(segB)
                    behaviorSwitchesWindows{a}(r, periPos) = sum(diff(segB)~=0);
                    behaviorProportionWindows{a}(r, periPos) = mean(segB > 0); % Proportion of time in any behavior
                    % Behavior entropy: distribution of behavior IDs
                    validBhv = segB(segB > 0);
                    if ~isempty(validBhv) && length(unique(validBhv)) > 1
                        bhvCounts = histcounts(validBhv, min(validBhv):max(validBhv)+1);
                        bhvProportions = bhvCounts / length(validBhv);
                        bhvProportions = bhvProportions(bhvProportions > 0);
                        behaviorEntropyWindows{a}(r, periPos) = -sum(bhvProportions .* log2(bhvProportions));
                    elseif ~isempty(validBhv)
                        behaviorEntropyWindows{a}(r, periPos) = 0; % Single behavior
                    else
                        behaviorEntropyWindows{a}(r, periPos) = NaN;
                    end
                end
            end

            % Kin PCA std proxy from kinematic data
            if hasKin && ~isempty(kinDown)
                kStart = bStart; kEnd = min(bEnd, length(kinDown));
                if kEnd >= kStart
                    kinPCAStdWindows{a}(r, periPos) = std(kinDown(kStart:kEnd), 'omitnan');
                end
            end
        end
    end
end

%% ==============================================     Sliding Window Metrics Plotting     ==============================================

% Create behavior ID indices
bhvIdxByID = cell(1, length(bhvStartIDs));
allBhvOnsetTimes = [];
for b = 1:length(bhvStartIDs)
    allBhvOnsetTimes = [allBhvOnsetTimes; bhvOnsetTimes{b}];
end
allBhvOnsetTimes = sort(allBhvOnsetTimes);
for b = 1:length(bhvStartIDs)
    bhvIdxByID{b} = ismember(allBhvOnsetTimes, bhvOnsetTimes{b});
end

% Number of rows = number of behavior IDs
numRows = length(bhvStartIDs);

% Metrics to plot
metricsToPlot = {'stateEntropy', 'behaviorSwitches', 'behaviorEntropy', 'kinPCAStd'};
metricNames = {'State Entropy', 'Behavior Switches', 'Behavior Entropy', 'Kin PCA Std'};
metricWindows = {stateEntropyWindows, behaviorSwitchesWindows, behaviorEntropyWindows, kinPCAStdWindows};

% Create figure for metrics
figure(402); clf;
monitorPositions = get(0, 'MonitorPositions');
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    set(gcf, 'Position', monitorTwo);
else
    set(gcf, 'Position', monitorPositions(1, :));
end

% Layout: rows = behavior IDs, columns = areas
numCols = length(areasToTest);
ha_metrics = tight_subplot(numRows, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

% Create time axes for each area (same as peri-behavior window)
timeAxisPerArea = cell(1, length(areas));
for a = areasToTest
    if hasHmmArea(a)
        binSizeA = binSizes(a);
        winBinsA = ceil(periBhvWindow / binSizeA);
        halfWinA = floor(winBinsA / 2);
        timeAxisPerArea{a} = (-halfWinA:halfWinA) * binSizeA;
    end
end

% Colors for metrics (different colors for each metric)
metricColors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560]}; % Blue, Red-orange, Yellow, Purple
metricLineStyles = {'-', '--', ':', '-.'};

plotIdx = 0;
for bhvRowIdx = 1:numRows
    bhvID_target = bhvStartIDs(bhvRowIdx);
    condIdx_use = bhvIdxByID{bhvRowIdx};
    condName = sprintf('Behavior ID %d', bhvID_target);
    
    % Plot each area
    for areaIdx = 1:length(areasToTest)
        a = areasToTest(areaIdx);
        plotIdx = plotIdx + 1;
        
        axes(ha_metrics(plotIdx));
        hold on;
        
        % Check if we have any data for this area
        hasData = false;
        for mIdx = 1:length(metricsToPlot)
            metricWindows_m = metricWindows{mIdx};
            if hasHmmArea(a) && ~isempty(metricWindows_m{a}) && any(~isnan(metricWindows_m{a}(:)))
                hasData = true;
                break;
            end
        end
        
        if ~hasData
            xlim([-periBhvWindow/2, periBhvWindow/2]);
            ylim([0 1]);
            title(sprintf('%s - %s\n(No Data)', areas{a}, condName), 'FontSize', 10);
            continue;
        end
        
        % Get time axis for this area
        if ~isempty(timeAxisPerArea{a})
            timeAxis = timeAxisPerArea{a};
        else
            % Fallback
            binSizeA = binSizes(a);
            winBinsA = ceil(periBhvWindow / binSizeA);
            halfWinA = floor(winBinsA / 2);
            timeAxis = (-halfWinA:halfWinA) * binSizeA;
        end
        
        timeAxis_plot = timeAxis; % Initialize with full time axis
        
        % Plot each metric on the same axes
        for mIdx = 1:length(metricsToPlot)
            metricWindows_m = metricWindows{mIdx};
            
            % Check if data exists
            if ~hasHmmArea(a) || isempty(metricWindows_m{a}) || all(isnan(metricWindows_m{a}(:)))
                continue;
            end
            
            % Extract data for this behavior ID
            metricData = metricWindows_m{a}(condIdx_use, :);
            validRows = ~all(isnan(metricData), 2);
            
            if ~any(validRows)
                continue;
            end
            
            metricDataValid = metricData(validRows, :);
            
            % Calculate mean and std
            meanMetric = nanmean(metricDataValid, 1);
            stdMetric = nanstd(metricDataValid, 0, 1);
            
            % Ensure timeAxis matches data length
            if length(timeAxis) ~= length(meanMetric)
                minLen = min(length(timeAxis), length(meanMetric));
                timeAxis_plot_metric = timeAxis(1:minLen);
                meanMetric = meanMetric(1:minLen);
                stdMetric = stdMetric(1:minLen);
            else
                timeAxis_plot_metric = timeAxis;
            end
            
            % Update timeAxis_plot to the shortest one (in case metrics have different lengths)
            if length(timeAxis_plot_metric) < length(timeAxis_plot)
                timeAxis_plot = timeAxis_plot_metric;
            end
            
            % Plot std ribbon
            fill([timeAxis_plot_metric, fliplr(timeAxis_plot_metric)], ...
                 [meanMetric + stdMetric, fliplr(meanMetric - stdMetric)], ...
                 metricColors{mIdx}, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', [metricNames{mIdx} ' ± std']);
            
            % Plot mean line
            plot(timeAxis_plot_metric, meanMetric, 'Color', metricColors{mIdx}, 'LineWidth', 2, ...
                 'LineStyle', metricLineStyles{mIdx}, 'DisplayName', metricNames{mIdx});
        end
        
        % Add vertical line at behavior onset
        plot([0 0], ylim, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        
        % Formatting
        xlabel('Time (s)', 'FontSize', 10);
        ylabel('Metric Value', 'FontSize', 10);
        title(sprintf('%s - %s', areas{a}, condName), 'FontSize', 10);
        grid on;
        
        % Set axis limits
        if ~isempty(timeAxis_plot)
            xlim([timeAxis_plot(1), timeAxis_plot(end)]);
        else
            xlim([-periBhvWindow/2, periBhvWindow/2]);
        end
        
        % Auto-scale y-axis with some padding
        ylim('auto');
        yLim = ylim;
        if ~isnan(yLim(1)) && ~isnan(yLim(2)) && yLim(2) > yLim(1)
            ylim([yLim(1) - 0.05*range(yLim), yLim(2) + 0.05*range(yLim)]);
        end
        
        % Set tick labels
        if ~isempty(timeAxis_plot)
            xTicks = ceil(timeAxis_plot(1)):floor(timeAxis_plot(end));
            if isempty(xTicks)
                xTicks = linspace(timeAxis_plot(1), timeAxis_plot(end), 5);
            end
            xticks(xTicks);
            xticklabels(string(xTicks));
        end
        
        % Add legend only on first subplot (top-left)
        if bhvRowIdx == 1 && areaIdx == 1
            legend('Location', 'best', 'FontSize', 8);
        end
    end
end

% Add overall title
sgtitle(sprintf('%s - Sliding Window Metrics (Window: %.1fs, Peri: %.1fs)', filePrefix, slidingWindowSize, periBhvWindow), 'FontSize', 14);

% Save figure
filename_metrics = fullfile(hmmdir, sprintf('%s_peri_bhv_metrics_win%.1f_peri%.1f.png', filePrefix, slidingWindowSize, periBhvWindow));
exportgraphics(gcf, filename_metrics, 'Resolution', 300);
fprintf('Saved peri-behavior metrics plot to: %s\n', filename_metrics);

%% ==============================================     Plotting Results     ==============================================

% Create peri-behavior plots for each area: behavior IDs x areas
figure(400); clf;
% Prefer plotting on second screen if available
monitorPositions = get(0, 'MonitorPositions');
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    set(gcf, 'Position', monitorTwo);
else
    set(gcf, 'Position', monitorPositions(1, :));
end

% Use tight_subplot for layout: behavior IDs (rows) x areas (columns)
numCols = length(areasToTest);
ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

% Create time axis for peri-behavior window (centered on behavior onset)
% Use first area's bin size for time axis (will be recalculated per area as needed)
halfWindow = floor(ceil(periBhvWindow / binSizes(1)) / 2);
timeAxisPeriBhv = (-halfWindow:halfWindow) * binSizes(1);

% Plot each behavior ID (row) and area (column)
plotIdx = 0;
for bhvRowIdx = 1:numRows
    for areaIdx = 1:length(areasToTest)
        a = areasToTest(areaIdx);
        plotIdx = plotIdx + 1;
        
        axes(ha(plotIdx));
        hold on;
        
        % Get state data for this behavior ID and area
        stateData = stateWindowsByBhv{bhvRowIdx, a};
        bhvID_target = bhvStartIDs(bhvRowIdx);
        condName = sprintf('Behavior ID %d', bhvID_target);
        
        % Check if we have data for this area and behavior ID
        if ~hasHmmArea(a) || isempty(stateData) || all(isnan(stateData(:)))
            % No data - show blank plot
            xlim([timeAxisPeriBhv(1), timeAxisPeriBhv(end)]);
            ylim([0.5, 1.5]);
            title(sprintf('%s - %s\n(No Data)', areas{a}, condName), 'FontSize', 10);
            continue;
        end
        
        % Remove rows that are all NaN and ensure onsets are in order
        validRows = ~all(isnan(stateData), 2);
        if ~any(validRows)
            xlim([timeAxisPeriBhv(1), timeAxisPeriBhv(end)]);
            ylim([0.5, 1.5]);
            title(sprintf('%s - %s\n(No Valid Onsets)', areas{a}, condName), 'FontSize', 10);
            continue;
        end
        
        % Keep onsets in original order (top to bottom)
        stateDataValid = stateData(validRows, :);
        
        % Create imagesc plot
        imagesc(timeAxisPeriBhv, 1:size(stateDataValid, 1), stateDataValid);
        
        % Set colormap to be consistent within area; state 0 is white
        if ~isempty(stateCmaps{a})
            colormap(ha(plotIdx), stateCmaps{a});
            caxis(ha(plotIdx), [0, max(1, numStates(a))]);
        end
        
        % Add colorbar only on top row, and keep fixed ticks/labels per area
        if bhvRowIdx == 1
            c = colorbar('peer', ha(plotIdx));
            c.Ticks = 0:max(1, numStates(a));
            c.TickLabels = string(0:max(1, numStates(a)));
            c.Label.String = 'HMM State';
            c.Label.FontSize = 8;
        end
        
        % Add vertical line at behavior onset
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
        
        % Formatting
        xlabel('Time relative to behavior onset (s)', 'FontSize', 8);
        ylabel('Onset Trial', 'FontSize', 8);
        title(sprintf('%s - %s\n(%d onsets)', areas{a}, condName, sum(validRows)), 'FontSize', 10);
        
        % Set axis limits
        xlim([timeAxisPeriBhv(1), timeAxisPeriBhv(end)]);
        ylim([0.5, size(stateDataValid, 1) + 0.5]);
        
        % Set tick labels
        xTicks = ceil(timeAxisPeriBhv(1)):floor(timeAxisPeriBhv(end));
        if isempty(xTicks)
            xTicks = linspace(timeAxisPeriBhv(1), timeAxisPeriBhv(end), 5);
        end
        xticks(xTicks);
        xticklabels(string(xTicks));
        
        % Set y-axis ticks to show trial numbers
        if size(stateDataValid, 1) <= 10
            yticks(1:size(stateDataValid, 1));
        else
            yticks(1:5:size(stateDataValid, 1));
        end
        
        grid on;
        set(gca, 'GridAlpha', 0.3);
    end
end

% Add overall title
sgtitle(sprintf('%s - Peri-Behavior HMM State Sequences (Window: %gs)', filePrefix, periBhvWindow), 'FontSize', 16);

% Save combined figure
filename = fullfile(hmmdir, sprintf('%s_peri_bhv_hmm_states_win%gs.png', filePrefix, periBhvWindow));
exportgraphics(gcf, filename, 'Resolution', 300);
fprintf('Saved peri-behavior HMM state plot to: %s\n', filename);

% ==============================================     Metastate Plotting     ==============================================

% Check if any area has metastate data
hasAnyMetastates = false;
for a = areasToTest
    if hasHmmArea(a) && ~isempty(results.hmm_results{a}) && isfield(results.hmm_results{a}, 'metastate_results') && ~isempty(results.hmm_results{a}.metastate_results.continuous_metastates)
        hasAnyMetastates = true;
        break;
    end
end

if hasAnyMetastates
    fprintf('\n=== Creating Peri-Behavior Metastate Plot ===\n');
    
    % Create peri-behavior metastate plots for each area: behavior IDs x areas
    figure(401); clf;
    % Prefer plotting on second screen if available
    monitorPositions = get(0, 'MonitorPositions');
    monitorTwo = monitorPositions(size(monitorPositions, 1), :);
    if size(monitorPositions, 1) >= 2
        set(gcf, 'Position', monitorTwo);
    else
        set(gcf, 'Position', monitorPositions(1, :));
    end

    % Use tight_subplot for layout: behavior IDs (rows) x areas (columns)
    numCols = length(areasToTest);
    ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

    % Plot each behavior ID (row) and area (column)
    plotIdx = 0;
    for bhvRowIdx = 1:numRows
        for areaIdx = 1:length(areasToTest)
            a = areasToTest(areaIdx);
            plotIdx = plotIdx + 1;
            
            axes(ha(plotIdx));
            hold on;
            
            % Get metastate data for this behavior ID and area
            metastateData = metastateWindowsByBhv{bhvRowIdx, a};
            bhvID_target = bhvStartIDs(bhvRowIdx);
            condName = sprintf('Behavior ID %d', bhvID_target);
            
            % Check if we have metastate data for this area and behavior ID
            if ~hasHmmArea(a) || isempty(metastateData) || all(isnan(metastateData(:)))
                % No data - show blank plot
                xlim([timeAxisPeriBhv(1), timeAxisPeriBhv(end)]);
                ylim([0.5, 1.5]);
                title(sprintf('%s - %s\n(No Metastate Data)', areas{a}, condName), 'FontSize', 10);
                continue;
            end
            
            % Remove rows that are all NaN and ensure onsets are in order
            validRows = ~all(isnan(metastateData), 2);
            if ~any(validRows)
                xlim([timeAxisPeriBhv(1), timeAxisPeriBhv(end)]);
                ylim([0.5, 1.5]);
                title(sprintf('%s - %s\n(No Valid Onsets)', areas{a}, condName), 'FontSize', 10);
                continue;
            end
            
            % Keep onsets in original order (top to bottom)
            metastateDataValid = metastateData(validRows, :);
            
            % Create imagesc plot
            imagesc(timeAxisPeriBhv, 1:size(metastateDataValid, 1), metastateDataValid);
            
            % Set colormap for metastates; metastate 0 is white
            if ~isempty(metaCmaps{a})
                colormap(ha(plotIdx), metaCmaps{a});
                nMeta = size(metaCmaps{a},1) - 1;
                caxis(ha(plotIdx), [0, max(1, nMeta)]);
            end
            
            % Add colorbar only on top row, and keep fixed ticks/labels per area
            if bhvRowIdx == 1
                c = colorbar('peer', ha(plotIdx));
                if exist('nMeta','var') && ~isempty(nMeta)
                    c.Ticks = 0:max(1, nMeta);
                    c.TickLabels = string(0:max(1, nMeta));
                end
                c.Label.String = 'Metastate';
                c.Label.FontSize = 8;
            end
            
            % Add vertical line at behavior onset
            plot([0 0], ylim, 'k--', 'LineWidth', 2);
            
            % Formatting
            xlabel('Time relative to behavior onset (s)', 'FontSize', 8);
            ylabel('Onset Trial', 'FontSize', 8);
            title(sprintf('%s - %s\n(%d onsets)', areas{a}, condName, sum(validRows)), 'FontSize', 10);
            
            % Set axis limits
            xlim([timeAxisPeriBhv(1), timeAxisPeriBhv(end)]);
            ylim([0.5, size(metastateDataValid, 1) + 0.5]);
            
            % Set tick labels
            xTicks = ceil(timeAxisPeriBhv(1)):floor(timeAxisPeriBhv(end));
            if isempty(xTicks)
                xTicks = linspace(timeAxisPeriBhv(1), timeAxisPeriBhv(end), 5);
            end
            xticks(xTicks);
            xticklabels(string(xTicks));
            
            % Set y-axis ticks to show trial numbers
            if size(metastateDataValid, 1) <= 10
                yticks(1:size(metastateDataValid, 1));
            else
                yticks(1:5:size(metastateDataValid, 1));
            end
            
            grid on;
            set(gca, 'GridAlpha', 0.3);
        end
    end

    % Add overall title
    sgtitle(sprintf('%s - Peri-Behavior Metastate Sequences (Window: %gs)', filePrefix, periBhvWindow), 'FontSize', 16);

    % Save combined metastate figure
    filename = fullfile(hmmdir, sprintf('%s_peri_bhv_metastates_win%gs.png', filePrefix, periBhvWindow));
    exportgraphics(gcf, filename, 'Resolution', 300);
    fprintf('Saved peri-behavior metastate plot to: %s\n', filename);
else
    fprintf('\nNo metastate data available for plotting\n');
end

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Peri-Behavior HMM State Analysis Summary ===\n');
for a = areasToTest
    if hasHmmArea(a)
        fprintf('\nArea %s:\n', areas{a});
        for b = 1:length(bhvStartIDs)
            fprintf('  Behavior ID %d: %d valid onsets\n', bhvStartIDs(b), sum(~all(isnan(stateWindowsByBhv{b, a}), 2)));
        end
        fprintf('  Number of HMM states: %d\n', numStates(a));
        fprintf('  Bin size: %.6f seconds\n', binSizes(a));
    else
        fprintf('\nArea %s: HMM analysis failed\n', areas{a});
    end
end

fprintf('\nPeri-behavior HMM state analysis complete!\n');
