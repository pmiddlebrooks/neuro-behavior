%%
% Criticality + HMM + Behavior Sliding Window Correlations Script
% Loads results from criticality_decoding_accuracy.m and HMM analysis, then performs
% correlation analyses with behavior metrics using sliding windows
% 
% BIN SIZE HANDLING:
% - HMM data: hmmBinSize (from HMM results)
% - Behavior data: bhvBinSize (matches HMM bin size for consistency)
% - Criticality data: criticalityBinSize (from criticality results)
% - Kinematics data: kinBinSize = 0.1 seconds (fixed)
%
% CORRELATION ANALYSES:
% 1. Criticality vs Behavior metrics
% 2. HMM vs Behavior metrics  
% 3. Criticality vs HMM metrics
%
% VISUALIZATIONS:
% - Separate scatter plot grids for each correlation type
% - Rows = one metric type, Columns = other metric type
% - Uses tight_subplot for clean layout

%%    Load behavior IDs
opts = neuro_behavior_options;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 45 * 60; % seconds

paths = get_paths;

% Monitor setup
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

%% ==============================================     Analysis Parameters     ==============================================

% Correlation analysis parameters
correlationMethod = 'pearson';  % 'pearson', 'spearman', or 'kendall'
minValidPoints = 10;           % Minimum number of valid points for correlation
significanceLevel = 0.05;      % Significance level for correlation tests

% HMM analysis parameters
natOrReach = 'Nat';            % 'Nat' or 'Reach'
brainArea = 'M56';             % 'M23', 'M56', 'DS', 'VS'

% Areas to analyze (will be updated based on loaded data)
areasToTest = 2:3;             % Default for criticality analysis

%% ==============================================     Load Criticality Decoding Results     ==============================================

fprintf('\n=== Loading Criticality Decoding Results ===\n');

% Load results from criticality_decoding_accuracy.m
try
    % Try to load the most recent results file
    resultFiles = dir(fullfile(paths.dropPath, 'criticality_decoding_results_*.mat'));
    if isempty(resultFiles)
        error('No criticality decoding results found');
    end

    % Get the most recent file
    [~, idx] = max([resultFiles.datenum]);
    latestFile = resultFiles(idx).name;

    load(fullfile(paths.dropPath, latestFile), 'results');
    fprintf('Loaded results from: %s\n', latestFile);

    % Extract key data
    areas = results.areas;
    d2Nat = results.naturalistic.d2;
    decodingAccuracyNat = results.naturalistic.decodingAccuracy;
    startSNat = results.naturalistic.startS;
    criticalityBinSize = results.naturalistic.criticalityBinSize;
    criticalityWindowSize = results.naturalistic.criticalityWindowSize;

    % Extract parameters
    dimReductionMethod = results.dimReductionMethod;
    transOrWithin = results.params.transOrWithin;
    svmBinSize = results.params.svmBinSize;
    stepSize = results.params.stepSize;

    fprintf('Criticality analysis parameters:\n');
    fprintf('  Dimensionality reduction: %s\n', dimReductionMethod);
    fprintf('  Behavior analysis: %s\n', transOrWithin);
    fprintf('  SVM bin size: %.3f s\n', svmBinSize);
    fprintf('  Criticality bin size: %.3f s\n', criticalityBinSize);
    fprintf('  Window size: %.1f s\n', criticalityWindowSize);
    fprintf('  Step size: %.1f s\n', stepSize);

catch e
    fprintf('Error loading criticality decoding results: %s\n', e.message);
    fprintf('Please run criticality_decoding_accuracy.m first to generate the required data.\n');
    return;
end

%% ==============================================     Load HMM Results     ==============================================

fprintf('\n=== Loading HMM Results ===\n');

try
    % Load HMM results for the specified data type and brain area
    [hmm_results] = hmm_load_saved_model(natOrReach, brainArea);
    fprintf('Loaded HMM model for %s data in %s area\n', natOrReach, brainArea);

    % Extract HMM parameters
    hmmBinSize = hmm_results.HmmParam.BinSize;
    numStates = hmm_results.best_model.num_states;

    % Extract continuous results
    hmmSequence = hmm_results.continuous_results.sequence;
    hmmProbabilities = hmm_results.continuous_results.pStates;
    totalHmmBins = length(hmmSequence);

    fprintf('HMM parameters:\n');
    fprintf('  Number of states: %d\n', numStates);
    fprintf('  Bin size: %.6f s\n', hmmBinSize);
    fprintf('  Total time bins: %d\n', totalHmmBins);
    fprintf('  Total time: %.1f s\n', totalHmmBins * hmmBinSize);

catch e
    fprintf('Error loading HMM results: %s\n', e.message);
    fprintf('Please ensure HMM analysis has been completed for %s data in %s area.\n', natOrReach, brainArea);
    return;
end

%% ==============================================     Load Behavior and Kinematics Data     ==============================================

fprintf('\n=== Loading Behavior and Kinematics Data ===\n');

% Load behavior data with HMM-matched frame size for consistency
opts.frameSize = hmmBinSize;
bhvBinSize = opts.frameSize;
getDataType = 'behavior';
get_standard_data
% [dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);

% Load kinematics (in 0.1 s binSize) and perform PCA analysis
kinBinSize = 0.1;
getDataType = 'kinematics';
get_standard_data

startFrame = 1 + opts.collectStart / kinBinSize;
endFrame = startFrame - 1 + (opts.collectFor / kinBinSize);
kinData = kinData(startFrame:endFrame,:);

[coeff, score, ~, ~, explained] = pca(zscore(kinData));
kinPCA1 = score(:, 1);  % First principal component
kinPCA2 = score(:, 2);  % Second principal component

% Display PCA variance explained
fprintf('PCA variance explained: PC1 = %.1f%%, PC2 = %.1f%%\n', explained(1), explained(2));

fprintf('Data bin sizes:\n');
fprintf('  HMM: %.6f s\n', hmmBinSize);
fprintf('  Behavior: %.6f s (matched to HMM)\n', bhvBinSize);
fprintf('  Criticality: %.3f s\n', criticalityBinSize);
fprintf('  Kinematics: %.1f s\n', kinBinSize);

%% ==============================================     Calculate Behavior Metrics     ==============================================

fprintf('\n=== Calculating Behavior Metrics ===\n');

% Behavior metrics are the same across brain areas (global behavioral metrics)
% Calculate once and use for all areas
fprintf('Calculating behavior switches and proportion (global metrics)...\n');

% Define which behavior labels to count for proportion (user can modify this list)
propBhvList = [0 1 2 13 14 15];  % Example: grooming behaviors
fprintf('Calculating proportion of behaviors: %s\n', mat2str(propBhvList));

% Get all valid d2 time points (use first area as reference)
validD2Idx = ~isnan(d2Nat{areasToTest(1)});
validD2Times = startSNat{areasToTest(1)}(validD2Idx);

% Initialize behavior metric arrays (same size as d2 arrays)
behaviorSwitches = nan(size(d2Nat{areasToTest(1)}));
behaviorProportion = nan(size(d2Nat{areasToTest(1)}));
behaviorDwellTimesMean = nan(size(d2Nat{areasToTest(1)}));
behaviorDwellTimesStd = nan(size(d2Nat{areasToTest(1)}));
behaviorOccupancyEntropy = nan(size(d2Nat{areasToTest(1)}));
kinPCA1Std = nan(size(d2Nat{areasToTest(1)}));
kinPCA2Std = nan(size(d2Nat{areasToTest(1)}));

% Process each criticality window
for i = 1:length(validD2Times)
    centerTime = validD2Times(i);

    % Calculate window boundaries (same as used for criticality)
    windowStartTime = centerTime - criticalityWindowSize/2;
    windowEndTime = centerTime + criticalityWindowSize/2;

    % Convert to frame indices for behavior data
    % bhvID is in HMM bin size, so convert times to frame indices
    startFrame = max(1, round(windowStartTime / bhvBinSize));
    endFrame = min(length(bhvID), round(windowEndTime / bhvBinSize));

    % Get behavior labels for this window
    windowBhvID = bhvID(startFrame:endFrame);
    
    % Filter out invalid behavior IDs (-1 values)
    validBhvID = windowBhvID(windowBhvID ~= -1);

    % Calculate behavior switches and dwell times
    % Note: For dwell time calculations, we only include behaviors that are completely
    % contained within the window (start and end within the window boundaries)
    if length(validBhvID) > 1
        % Find where behavior changes (switches)
        behaviorChanges = diff(validBhvID) ~= 0;
        numSwitches = sum(behaviorChanges);
        behaviorSwitches(i) = numSwitches;

        % Calculate behavior dwell times (only for complete behaviors within window)
        if numSwitches > 0
            % Find indices where behaviors change
            changeIndices = find(behaviorChanges);

            % Calculate dwell times for behaviors that start and end within the window
            % Exclude first behavior (may start before window) and last behavior (may continue after window)
            if length(changeIndices) >= 2
                % Get dwell times for behaviors that are completely within the window
                dwellTimes = diff(changeIndices) * bhvBinSize; % Convert to seconds
                behaviorDwellTimesMean(i) = mean(dwellTimes);
                behaviorDwellTimesStd(i) = std(dwellTimes);
            else
                % Only one behavior change - can't calculate complete dwell times
                behaviorDwellTimesMean(i) = nan;
                behaviorDwellTimesStd(i) = nan;
            end
        else
            % No behavior changes - single behavior throughout window
            % Check if this behavior starts and ends within the window
            if startFrame > 1 && endFrame < length(bhvID)
                % Check if behavior changes at window boundaries
                if bhvID(startFrame-1) ~= windowBhvID(1) && bhvID(endFrame+1) ~= windowBhvID(end)
                    % Behavior is complete within window
                    behaviorDwellTimesMean(i) = length(validBhvID) * bhvBinSize;
                    behaviorDwellTimesStd(i) = 0; % Single behavior, no variation
                else
                    % Behavior extends beyond window boundaries
                    behaviorDwellTimesMean(i) = nan;
                    behaviorDwellTimesStd(i) = nan;
                end
            else
                % Window is at data boundaries - can't determine if behavior is complete
                behaviorDwellTimesMean(i) = nan;
                behaviorDwellTimesStd(i) = nan;
            end
        end

        % Calculate behavior occupancy entropy (for all valid behaviors in window)
        if length(unique(validBhvID)) > 1
            % Count occurrences of each behavior
            behaviorCounts = histcounts(validBhvID, min(validBhvID):max(validBhvID));
            behaviorProportions = behaviorCounts / length(validBhvID);
            behaviorProportions = behaviorProportions(behaviorProportions > 0); % Remove zero probabilities
            if ~isempty(behaviorProportions)
                behaviorOccupancyEntropy(i) = -sum(behaviorProportions .* log2(behaviorProportions));
            end
        else
            % Single behavior throughout window
            behaviorOccupancyEntropy(i) = 0;
        end
    else
        % Single frame or no valid behaviors - can't calculate dwell times
        behaviorSwitches(i) = 0;
        behaviorDwellTimesMean(i) = nan;
        behaviorDwellTimesStd(i) = nan;
        behaviorOccupancyEntropy(i) = 0;
    end

    % Calculate proportion of specified behaviors (only for valid behaviors)
    if length(validBhvID) > 0
        % For proportion calculation, we'll use all valid behaviors in the window
        % but note that this may include incomplete behaviors at boundaries
        specifiedBehaviors = ismember(validBhvID, propBhvList);
        proportion = sum(specifiedBehaviors) / length(validBhvID);
        behaviorProportion(i) = proportion;
    else
        behaviorProportion(i) = 0;
    end

    % Calculate standard deviation of kinPCA1 and kinPCA2 within the window
    % These represent the variability of the first and second principal components
    % of kinematics data within each time window
    if length(validBhvID) > 0
        % Get kinPCA data for the same window
        % Kinematics are in 0.1 second bins, different from behavior/HMM binning
        kinStartFrame = max(1, round(windowStartTime / kinBinSize));
        kinEndFrame = min(length(kinPCA1), round(windowEndTime / kinBinSize));
        
        % Get kinPCA1 and kinPCA2 data for the same time window
        windowKinPCA1 = kinPCA1(kinStartFrame:kinEndFrame);
        windowKinPCA2 = kinPCA2(kinStartFrame:kinEndFrame);
        kinPCA1Std(i) = std(windowKinPCA1);
        kinPCA2Std(i) = std(windowKinPCA2);
    else
        kinPCA1Std(i) = nan;
        kinPCA2Std(i) = nan;
    end
end

fprintf('Behavior metrics calculated for %d time windows\n', length(validD2Times));

% Add behavior metrics to metrics structure (same for all areas)
metrics = struct();
metrics.behaviorSwitches = behaviorSwitches;
metrics.behaviorProportion = behaviorProportion;
metrics.behaviorDwellTimesMean = behaviorDwellTimesMean;
metrics.behaviorDwellTimesStd = behaviorDwellTimesStd;
metrics.behaviorOccupancyEntropy = behaviorOccupancyEntropy;
metrics.kinPCA1Std = kinPCA1Std;
metrics.kinPCA2Std = kinPCA2Std;
metrics.timePoints = startSNat{areasToTest(1)};

%% ==============================================     Calculate HMM Metrics     ==============================================

fprintf('\n=== Calculating HMM Metrics ===\n');

% Calculate time points for HMM sliding windows
windowSizeBins = round(criticalityWindowSize / hmmBinSize);
stepSizeBins = round(stepSize / hmmBinSize);
numWindows = floor((totalHmmBins - windowSizeBins) / stepSizeBins) + 1;

% Initialize HMM metric arrays
stateOccupancyEntropy = nan(numWindows, 1);
stateDwellTimesMean = nan(numWindows, 1);
stateDwellTimesStd = nan(numWindows, 1);
numUniqueStates = nan(numWindows, 1);
meanMaxProbability = nan(numWindows, 1);
stateProbabilityVariance = nan(numWindows, 1);
klDivergenceMean = nan(numWindows, 1);
hmmWindowCenterTimes = nan(numWindows, 1);

% Process each HMM sliding window
for i = 1:numWindows
    % Calculate window boundaries in HMM bins
    startBin = (i-1) * stepSizeBins + 1;
    endBin = startBin + windowSizeBins - 1;

    % Ensure we don't exceed data bounds
    if endBin > totalHmmBins
        break;
    end

    % Calculate center time of this window
    centerBin = (startBin + endBin) / 2;
    hmmWindowCenterTimes(i) = centerBin * hmmBinSize;

    % Get HMM data for this window
    windowSequence = hmmSequence(startBin:endBin);
    windowProbabilities = hmmProbabilities(startBin:endBin, :);

    % Remove invalid states (0 values)
    validStates = windowSequence(windowSequence > 0);
    validProbs = windowProbabilities(windowSequence > 0, :);

    if ~isempty(validStates)
        % 1. State occupancy entropy (only for complete states within window)
        % Check if states at window boundaries are complete
        if startBin > 1 && endBin < totalHmmBins
            % Check if state changes at window boundaries
            if hmmSequence(startBin-1) ~= windowSequence(1) && hmmSequence(endBin+1) ~= windowSequence(end)
                % States are complete within window - calculate entropy for all states
                stateCounts = histcounts(validStates, 1:(numStates+1));
                stateProportions = stateCounts / length(validStates);
                stateProportions = stateProportions(stateProportions > 0); % Remove zero probabilities
                if ~isempty(stateProportions)
                    stateOccupancyEntropy(i) = -sum(stateProportions .* log2(stateProportions));
                end
            else
                % States extend beyond window boundaries - find complete states within window
                % Find where states change within the window
                stateChanges = diff(windowSequence) ~= 0;
                if any(stateChanges)
                    changeIndices = find(stateChanges);
                    
                    % Get states that are completely within the window
                    % Start from the first state change, end at the last state change
                    if length(changeIndices) >= 2
                        startCompleteIdx = changeIndices(1);
                        endCompleteIdx = changeIndices(end);
                        
                        % Extract complete states (excluding first and last incomplete states)
                        completeStates = windowSequence(startCompleteIdx:endCompleteIdx);
                        
                        % Calculate entropy for complete states only
                        stateCounts = histcounts(completeStates, 1:(numStates+1));
                        stateProportions = stateCounts / length(completeStates);
                        stateProportions = stateProportions(stateProportions > 0); % Remove zero probabilities
                        if ~isempty(stateProportions)
                            stateOccupancyEntropy(i) = -sum(stateProportions .* log2(stateProportions));
                        end
                    else
                        % Only one state change - can't calculate entropy for complete states
                        stateOccupancyEntropy(i) = nan;
                    end
                else
                    % Single state throughout window - can't determine if complete
                    stateOccupancyEntropy(i) = nan;
                end
            end
        else
            % Window is at data boundaries - can't determine if states are complete
            stateOccupancyEntropy(i) = nan;
        end

        % 2. State dwell times mean and std (only for complete states within window)
        if length(validStates) > 1
            % Find state transitions
            stateChanges = diff(validStates) ~= 0;
            if any(stateChanges)
                % Calculate dwell times for states that start and end within the window
                changeIndices = find(stateChanges);

                % Exclude first state (may start before window) and last state (may continue after window)
                if length(changeIndices) >= 2
                    % Get dwell times for states that are completely within the window
                    dwellTimes = diff(changeIndices) * hmmBinSize; % Convert to seconds
                    stateDwellTimesMean(i) = mean(dwellTimes);
                    stateDwellTimesStd(i) = std(dwellTimes);
                else
                    % Only one state change - can't calculate complete dwell times
                    stateDwellTimesMean(i) = nan;
                    stateDwellTimesStd(i) = nan;
                end
            else
                % Single state throughout window - check if complete
                if startBin > 1 && endBin < totalHmmBins
                    if hmmSequence(startBin-1) ~= windowSequence(1) && hmmSequence(endBin+1) ~= windowSequence(end)
                        % State is complete within window
                        stateDwellTimesMean(i) = length(validStates) * hmmBinSize;
                        stateDwellTimesStd(i) = 0;
                    else
                        % State extends beyond window boundaries
                        stateDwellTimesMean(i) = nan;
                        stateDwellTimesStd(i) = nan;
                    end
                else
                    % Window is at data boundaries - can't determine if state is complete
                    stateDwellTimesMean(i) = nan;
                    stateDwellTimesStd(i) = nan;
                end
            end
        else
            % Single state - check if complete
            if startBin > 1 && endBin < totalHmmBins
                if hmmSequence(startBin-1) ~= windowSequence(1) && hmmSequence(endBin+1) ~= windowSequence(end)
                    % State is complete within window
                    stateDwellTimesMean(i) = hmmBinSize;
                    stateDwellTimesStd(i) = 0;
                else
                    % State extends beyond window boundaries
                    stateDwellTimesMean(i) = nan;
                    stateDwellTimesStd(i) = nan;
                end
            else
                % Window is at data boundaries - can't determine if state is complete
                stateDwellTimesMean(i) = nan;
                stateDwellTimesStd(i) = nan;
            end
        end

        % 3. Number of unique states (only for complete states within window)
        if startBin > 1 && endBin < totalHmmBins
            % Check if states at window boundaries are complete
            if hmmSequence(startBin-1) ~= windowSequence(1) && hmmSequence(endBin+1) ~= windowSequence(end)
                % States are complete within window
                numUniqueStates(i) = length(unique(validStates));
            else
                % States extend beyond window boundaries - find complete states within window
                % Find where states change within the window
                stateChanges = diff(windowSequence) ~= 0;
                if any(stateChanges)
                    changeIndices = find(stateChanges);
                    
                    % Get states that are completely within the window
                    % Start from the first state change, end at the last state change
                    if length(changeIndices) >= 2
                        startCompleteIdx = changeIndices(1);
                        endCompleteIdx = changeIndices(end);
                        
                        % Extract complete states (excluding first and last incomplete states)
                        completeStates = windowSequence(startCompleteIdx:endCompleteIdx);
                        
                        % Count unique states for complete states only
                        numUniqueStates(i) = length(unique(completeStates));
                    else
                        % Only one state change - can't calculate unique states for complete states
                        numUniqueStates(i) = nan;
                    end
                else
                    % Single state throughout window - can't determine if complete
                    numUniqueStates(i) = nan;
                end
            end
        else
            % Window is at data boundaries - can't determine if states are complete
            numUniqueStates(i) = nan;
        end
        
        % 4. Mean maximum probability per bin
        if ~isempty(validProbs)
            maxProbs = max(validProbs, [], 2);
            meanMaxProbability(i) = mean(maxProbs);
        end

        % 5. Variance of state probabilities
        if ~isempty(validProbs)
            stateProbabilityVariance(i) = var(validProbs(:));
        end

        % 6. KL divergence between consecutive probability distributions
        if size(validProbs, 1) > 1
            klDivs = zeros(size(validProbs, 1) - 1, 1);
            for j = 1:(size(validProbs, 1) - 1)
                p = validProbs(j, :);
                q = validProbs(j + 1, :);

                % Ensure probabilities sum to 1 and handle zeros
                p = p / sum(p);
                q = q / sum(q);

                % Add small epsilon to avoid log(0)
                epsilon = 1e-10;
                p = p + epsilon;
                q = q + epsilon;
                p = p / sum(p);
                q = q / sum(q);

                % Calculate KL divergence
                klDivs(j) = sum(p .* log2(p ./ q));
            end
            klDivergenceMean(i) = mean(klDivs);
        end
    end
end

% Trim arrays to actual number of windows processed
numWindows = i - 1;
stateOccupancyEntropy = stateOccupancyEntropy(1:numWindows);
stateDwellTimesMean = stateDwellTimesMean(1:numWindows);
stateDwellTimesStd = stateDwellTimesStd(1:numWindows);
numUniqueStates = numUniqueStates(1:numWindows);
meanMaxProbability = meanMaxProbability(1:numWindows);
stateProbabilityVariance = stateProbabilityVariance(1:numWindows);
klDivergenceMean = klDivergenceMean(1:numWindows);
hmmWindowCenterTimes = hmmWindowCenterTimes(1:numWindows);

fprintf('HMM metrics calculated for %d time windows\n', numWindows);

% Add HMM metrics to metrics structure
metrics.stateOccupancyEntropy = stateOccupancyEntropy;
metrics.stateDwellTimesMean = stateDwellTimesMean;
metrics.stateDwellTimesStd = stateDwellTimesStd;
metrics.numUniqueStates = numUniqueStates;
metrics.meanMaxProbability = meanMaxProbability;
metrics.stateProbabilityVariance = stateProbabilityVariance;
metrics.klDivergenceMean = klDivergenceMean;
metrics.hmmWindowCenterTimes = hmmWindowCenterTimes;

%% ==============================================     Calculate Valid Time Points     ==============================================

fprintf('\n=== Calculating Valid Time Points ===\n');

% Get valid time points for each correlation type
validTimePoints = cell(1, length(areas));

for a = areasToTest
    % Criticality vs Behavior: need d2, decoding accuracy, and behavior metrics
    validD2Idx = ~isnan(d2Nat{a});
    validAccIdx = ~isnan(decodingAccuracyNat{a});
    validBhvSwitchIdx = ~isnan(behaviorSwitches);
    validBhvPropIdx = ~isnan(behaviorProportion);
    validBhvDwellMeanIdx = ~isnan(behaviorDwellTimesMean);
    validBhvDwellStdIdx = ~isnan(behaviorDwellTimesStd);
    validBhvEntropyIdx = ~isnan(behaviorOccupancyEntropy);
    validKin1Idx = ~isnan(kinPCA1Std);
    validKin2Idx = ~isnan(kinPCA2Std);
    
    % HMM vs Behavior: need HMM metrics and behavior metrics
    validHmmEntropyIdx = ~isnan(stateOccupancyEntropy);
    validHmmDwellMeanIdx = ~isnan(stateDwellTimesMean);
    validHmmDwellStdIdx = ~isnan(stateDwellTimesStd);
    validHmmUniqueIdx = ~isnan(numUniqueStates);
    validHmmMaxProbIdx = ~isnan(meanMaxProbability);
    validHmmProbVarIdx = ~isnan(stateProbabilityVariance);
    validHmmKLIdx = ~isnan(klDivergenceMean);
    
    % Criticality vs HMM: need criticality metrics and HMM metrics
    % For this comparison, we need to interpolate HMM metrics to criticality time points
    % or vice versa. For now, we'll use the overlapping time windows.
    
    % Store valid indices for each correlation type
    validTimePoints{a}.criticalityVsBehavior = validD2Idx & validAccIdx & validBhvSwitchIdx & ...
        validBhvPropIdx & validBhvDwellMeanIdx & validBhvDwellStdIdx & validBhvEntropyIdx & validKin1Idx & validKin2Idx;
    
    validTimePoints{a}.hmmVsBehavior = validHmmEntropyIdx & validHmmDwellMeanIdx & validHmmDwellStdIdx & ...
        validHmmUniqueIdx & validHmmMaxProbIdx & validHmmProbVarIdx & validHmmKLIdx & ...
        validBhvSwitchIdx & validBhvPropIdx & validBhvDwellMeanIdx & validBhvDwellStdIdx & ...
        validBhvEntropyIdx & validKin1Idx & validKin2Idx;
    
    % For criticality vs HMM, we need to find overlapping time windows
    % This is more complex and will be handled in the correlation analysis
    
    fprintf('Area %s:\n', areas{a});
    fprintf('  Criticality vs Behavior: %d valid time points\n', sum(validTimePoints{a}.criticalityVsBehavior));
    fprintf('  HMM vs Behavior: %d valid time points\n', sum(validTimePoints{a}.hmmVsBehavior));
end

%% ==============================================     Correlation Analysis 1: Criticality vs Behavior     ==============================================

fprintf('\n=== Correlation Analysis 1: Criticality vs Behavior ===\n');

% Initialize correlation results structure
corrResultsCriticalityVsBehavior = [];
corrIdx = 1;

% Define metric names for correlation matrix
criticalityMetricNames = {'d2', 'decodingAccuracy'};
behaviorMetricNames = {'behaviorSwitches', 'behaviorProportion', 'behaviorDwellTimesMean', ...
    'behaviorDwellTimesStd', 'behaviorOccupancyEntropy', 'kinPCA1Std', 'kinPCA2Std'};
% Note: kinPCA1Std and kinPCA2Std represent the standard deviation of the first and second
% principal components of kinematics data within each sliding window

% Initialize area results structure
areaResultsCriticalityVsBehavior = struct();

% Calculate correlation matrix for each area
for a = areasToTest
    fprintf('\n--- Area: %s ---\n', areas{a});

    % Get valid data points
    validIdx = validTimePoints{a}.criticalityVsBehavior;

    if sum(validIdx) >= minValidPoints
        % Extract all metrics for this area
        d2Data = d2Nat{a}(validIdx)';
        accData = decodingAccuracyNat{a}(validIdx)';
        switchData = behaviorSwitches(validIdx)';
        propData = behaviorProportion(validIdx)';
        bhvDwellMeanData = behaviorDwellTimesMean(validIdx)';
        bhvDwellStdData = behaviorDwellTimesStd(validIdx)';
        bhvEntropyData = behaviorOccupancyEntropy(validIdx)';
        kin1Data = kinPCA1Std(validIdx)';
        kin2Data = kinPCA2Std(validIdx)';

        % Create data matrix for correlation analysis
        dataMatrix = [d2Data, accData, switchData, propData, bhvDwellMeanData, ...
            bhvDwellStdData, bhvEntropyData, kin1Data, kin2Data];

        % Calculate correlation matrix
        [R, P] = corrcoef(dataMatrix, 'rows', 'complete');

        % Display correlation matrix
        fprintf('Correlation Matrix:\n');
        fprintf('%-25s', 'Metric');
        allMetricNames = [criticalityMetricNames, behaviorMetricNames];
        for i = 1:length(allMetricNames)
            fprintf('%-20s', allMetricNames{i});
        end
        fprintf('\n');

        for i = 1:length(allMetricNames)
            fprintf('%-25s', allMetricNames{i});
            for j = 1:length(allMetricNames)
                if i == j
                    fprintf('%-20s', '1.000');
                else
                    fprintf('%-20.3f', R(i,j));
                end
            end
            fprintf('\n');
        end

        % Store significant correlations in results structure
        for i = 1:length(criticalityMetricNames)
            for j = 1:length(behaviorMetricNames)
                metric1Idx = i;
                metric2Idx = j + length(criticalityMetricNames);
                
                if P(metric1Idx, metric2Idx) < significanceLevel
                    corrResultsCriticalityVsBehavior(corrIdx).area = areas{a};
                    corrResultsCriticalityVsBehavior(corrIdx).metric1 = criticalityMetricNames{i};
                    corrResultsCriticalityVsBehavior(corrIdx).metric2 = behaviorMetricNames{j};
                    corrResultsCriticalityVsBehavior(corrIdx).correlation = R(metric1Idx, metric2Idx);
                    corrResultsCriticalityVsBehavior(corrIdx).p_value = P(metric1Idx, metric2Idx);
                    corrResultsCriticalityVsBehavior(corrIdx).n_valid_points = sum(validIdx);
                    corrResultsCriticalityVsBehavior(corrIdx).significant = true;
                    corrIdx = corrIdx + 1;

                    fprintf('\nSignificant correlation: %s vs %s: r = %.3f, p = %.3f\n', ...
                        criticalityMetricNames{i}, behaviorMetricNames{j}, R(metric1Idx, metric2Idx), P(metric1Idx, metric2Idx));
                end
            end
        end

        fprintf('Total valid points: %d\n', sum(validIdx));

    else
        fprintf('Insufficient data points (%d < %d)\n', sum(validIdx), minValidPoints);
        % Initialize empty matrices for this area
        R = nan(length(allMetricNames), length(allMetricNames));
        P = nan(length(allMetricNames), length(allMetricNames));
    end

    % Store matrices for this area
    areaResultsCriticalityVsBehavior(a).correlationMatrix = R;
    areaResultsCriticalityVsBehavior(a).pValueMatrix = P;
    areaResultsCriticalityVsBehavior(a).metrics = allMetricNames;
end

%% ==============================================     Correlation Analysis 2: HMM vs Behavior     ==============================================

fprintf('\n=== Correlation Analysis 2: HMM vs Behavior ===\n');

% Initialize correlation results structure
corrResultsHmmVsBehavior = [];
corrIdx = 1;

% Define metric names for correlation matrix
hmmMetricNames = {'stateOccupancyEntropy', 'stateDwellTimesMean', 'stateDwellTimesStd', ...
    'numUniqueStates', 'meanMaxProbability', 'stateProbabilityVariance', 'klDivergenceMean'};

% Initialize area results structure
areaResultsHmmVsBehavior = struct();

% Calculate correlation matrix for each area
for a = areasToTest
    fprintf('\n--- Area: %s ---\n', areas{a});

    % Get valid data points
    validIdx = validTimePoints{a}.hmmVsBehavior;

    if sum(validIdx) >= minValidPoints
        % Extract all metrics for this area
        entropyData = stateOccupancyEntropy(validIdx)';
        dwellMeanData = stateDwellTimesMean(validIdx)';
        dwellStdData = stateDwellTimesStd(validIdx)';
        uniqueData = numUniqueStates(validIdx)';
        maxProbData = meanMaxProbability(validIdx)';
        probVarData = stateProbabilityVariance(validIdx)';
        klData = klDivergenceMean(validIdx)';
        switchData = behaviorSwitches(validIdx)';
        propData = behaviorProportion(validIdx)';
        bhvDwellMeanData = behaviorDwellTimesMean(validIdx)';
        bhvDwellStdData = behaviorDwellTimesStd(validIdx)';
        bhvEntropyData = behaviorOccupancyEntropy(validIdx)';
        kin1Data = kinPCA1Std(validIdx)';
        kin2Data = kinPCA2Std(validIdx)';

        % Create data matrix for correlation analysis
        dataMatrix = [entropyData, dwellMeanData, dwellStdData, uniqueData, ...
            maxProbData, probVarData, klData, switchData, propData, bhvDwellMeanData, ...
            bhvDwellStdData, bhvEntropyData, kin1Data, kin2Data];

        % Calculate correlation matrix
        [R, P] = corrcoef(dataMatrix, 'rows', 'complete');

        % Display correlation matrix
        fprintf('Correlation Matrix:\n');
        fprintf('%-25s', 'Metric');
        allMetricNames = [hmmMetricNames, behaviorMetricNames];
        for i = 1:length(allMetricNames)
            fprintf('%-20s', allMetricNames{i});
        end
        fprintf('\n');

        for i = 1:length(allMetricNames)
            fprintf('%-25s', allMetricNames{i});
            for j = 1:length(allMetricNames)
                if i == j
                    fprintf('%-20s', '1.000');
                else
                    fprintf('%-20.3f', R(i,j));
                end
            end
            fprintf('\n');
        end

        % Store significant correlations in results structure
        for i = 1:length(hmmMetricNames)
            for j = 1:length(behaviorMetricNames)
                metric1Idx = i;
                metric2Idx = j + length(hmmMetricNames);
                
                if P(metric1Idx, metric2Idx) < significanceLevel
                    corrResultsHmmVsBehavior(corrIdx).area = areas{a};
                    corrResultsHmmVsBehavior(corrIdx).metric1 = hmmMetricNames{i};
                    corrResultsHmmVsBehavior(corrIdx).metric2 = behaviorMetricNames{j};
                    corrResultsHmmVsBehavior(corrIdx).correlation = R(metric1Idx, metric2Idx);
                    corrResultsHmmVsBehavior(corrIdx).p_value = P(metric1Idx, metric2Idx);
                    corrResultsHmmVsBehavior(corrIdx).n_valid_points = sum(validIdx);
                    corrResultsHmmVsBehavior(corrIdx).significant = true;
                    corrIdx = corrIdx + 1;

                    fprintf('\nSignificant correlation: %s vs %s: r = %.3f, p = %.3f\n', ...
                        hmmMetricNames{i}, behaviorMetricNames{j}, R(metric1Idx, metric2Idx), P(metric1Idx, metric2Idx));
                end
            end
        end

        fprintf('Total valid points: %d\n', sum(validIdx));

    else
        fprintf('Insufficient data points (%d < %d)\n', sum(validIdx), minValidPoints);
        % Initialize empty matrices for this area
        R = nan(length(allMetricNames), length(allMetricNames));
        P = nan(length(allMetricNames), length(allMetricNames));
    end

    % Store matrices for this area
    areaResultsHmmVsBehavior(a).correlationMatrix = R;
    areaResultsHmmVsBehavior(a).pValueMatrix = P;
    areaResultsHmmVsBehavior(a).metrics = allMetricNames;
end

%% ==============================================     Correlation Analysis 3: Criticality vs HMM     ==============================================

fprintf('\n=== Correlation Analysis 3: Criticality vs HMM ===\n');

% Initialize correlation results structure
corrResultsCriticalityVsHmm = [];
corrIdx = 1;

% Initialize area results structure
areaResultsCriticalityVsHmm = struct();

% Calculate correlation matrix for each area
for a = areasToTest
    fprintf('\n--- Area: %s ---\n', areas{a});

    % For criticality vs HMM, we need to interpolate HMM metrics to criticality time points
    % or find overlapping time windows. We'll use interpolation for now.
    
    % Get valid criticality time points
    validCriticalityIdx = ~isnan(d2Nat{a}) & ~isnan(decodingAccuracyNat{a});
    validCriticalityTimes = startSNat{a}(validCriticalityIdx);
    
    % Interpolate HMM metrics to criticality time points
    % Note: This assumes HMM metrics are available at hmmWindowCenterTimes
    if ~isempty(validCriticalityTimes) && ~isempty(hmmWindowCenterTimes)
        % Interpolate HMM metrics to criticality time points
        interpolatedHmmEntropy = interp1(hmmWindowCenterTimes, stateOccupancyEntropy, validCriticalityTimes, 'linear', 'extrap');
        interpolatedHmmDwellMean = interp1(hmmWindowCenterTimes, stateDwellTimesMean, validCriticalityTimes, 'linear', 'extrap');
        interpolatedHmmDwellStd = interp1(hmmWindowCenterTimes, stateDwellTimesStd, validCriticalityTimes, 'linear', 'extrap');
        interpolatedHmmUnique = interp1(hmmWindowCenterTimes, numUniqueStates, validCriticalityTimes, 'linear', 'extrap');
        interpolatedHmmMaxProb = interp1(hmmWindowCenterTimes, meanMaxProbability, validCriticalityTimes, 'linear', 'extrap');
        interpolatedHmmProbVar = interp1(hmmWindowCenterTimes, stateProbabilityVariance, validCriticalityTimes, 'linear', 'extrap');
        interpolatedHmmKL = interp1(hmmWindowCenterTimes, klDivergenceMean, validCriticalityTimes, 'linear', 'extrap');
        
        % Remove any NaN values from interpolation
        validInterpIdx = ~isnan(interpolatedHmmEntropy) & ~isnan(interpolatedHmmDwellMean) & ...
            ~isnan(interpolatedHmmDwellStd) & ~isnan(interpolatedHmmUnique) & ...
            ~isnan(interpolatedHmmMaxProb) & ~isnan(interpolatedHmmProbVar) & ~isnan(interpolatedHmmKL);
        
        if sum(validInterpIdx) >= minValidPoints
            % Extract all metrics for this area
            d2Data = d2Nat{a}(validCriticalityIdx);
            accData = decodingAccuracyNat{a}(validCriticalityIdx);
            
            % Get interpolated HMM metrics
            hmmEntropyData = interpolatedHmmEntropy(validInterpIdx)';
            hmmDwellMeanData = interpolatedHmmDwellMean(validInterpIdx)';
            hmmDwellStdData = interpolatedHmmDwellStd(validInterpIdx)';
            hmmUniqueData = interpolatedHmmUnique(validInterpIdx)';
            hmmMaxProbData = interpolatedHmmMaxProb(validInterpIdx)';
            hmmProbVarData = interpolatedHmmProbVar(validInterpIdx)';
            hmmKLData = interpolatedHmmKL(validInterpIdx)';

            % Create data matrix for correlation analysis
            dataMatrix = [d2Data(validInterpIdx)', accData(validInterpIdx)', hmmEntropyData, hmmDwellMeanData, ...
                hmmDwellStdData, hmmUniqueData, hmmMaxProbData, hmmProbVarData, hmmKLData];

            % Calculate correlation matrix
            [R, P] = corrcoef(dataMatrix, 'rows', 'complete');

            % Display correlation matrix
            fprintf('Correlation Matrix:\n');
            fprintf('%-25s', 'Metric');
            allMetricNames = [criticalityMetricNames, hmmMetricNames];
            for i = 1:length(allMetricNames)
                fprintf('%-20s', allMetricNames{i});
            end
            fprintf('\n');

            for i = 1:length(allMetricNames)
                fprintf('%-25s', allMetricNames{i});
                for j = 1:length(allMetricNames)
                    if i == j
                        fprintf('%-20s', '1.000');
                    else
                        fprintf('%-20.3f', R(i,j));
                    end
                end
                fprintf('\n');
            end

            % Store significant correlations in results structure
            for i = 1:length(criticalityMetricNames)
                for j = 1:length(hmmMetricNames)
                    metric1Idx = i;
                    metric2Idx = j + length(criticalityMetricNames);
                    
                    if P(metric1Idx, metric2Idx) < significanceLevel
                        corrResultsCriticalityVsHmm(corrIdx).area = areas{a};
                        corrResultsCriticalityVsHmm(corrIdx).metric1 = criticalityMetricNames{i};
                        corrResultsCriticalityVsHmm(corrIdx).metric2 = hmmMetricNames{j};
                        corrResultsCriticalityVsHmm(corrIdx).correlation = R(metric1Idx, metric2Idx);
                        corrResultsCriticalityVsHmm(corrIdx).p_value = P(metric1Idx, metric2Idx);
                        corrResultsCriticalityVsHmm(corrIdx).n_valid_points = sum(validInterpIdx);
                        corrResultsCriticalityVsHmm(corrIdx).significant = true;
                        corrIdx = corrIdx + 1;

                        fprintf('\nSignificant correlation: %s vs %s: r = %.3f, p = %.3f\n', ...
                            criticalityMetricNames{i}, hmmMetricNames{j}, R(metric1Idx, metric2Idx), P(metric1Idx, metric2Idx));
                    end
                end
            end

            fprintf('Total valid points: %d\n', sum(validInterpIdx));

        else
            fprintf('Insufficient interpolated data points (%d < %d)\n', sum(validInterpIdx), minValidPoints);
            % Initialize empty matrices for this area
            R = nan(length(allMetricNames), length(allMetricNames));
            P = nan(length(allMetricNames), length(allMetricNames));
        end
    else
        fprintf('No valid criticality or HMM data for interpolation\n');
        % Initialize empty matrices for this area
        R = nan(length([criticalityMetricNames, hmmMetricNames]), length([criticalityMetricNames, hmmMetricNames]));
        P = nan(length([criticalityMetricNames, hmmMetricNames]), length([criticalityMetricNames, hmmMetricNames]));
    end

    % Store matrices for this area
    areaResultsCriticalityVsHmm(a).correlationMatrix = R;
    areaResultsCriticalityVsHmm(a).pValueMatrix = P;
    areaResultsCriticalityVsHmm(a).metrics = [criticalityMetricNames, hmmMetricNames];
end

%% ==============================================     Visualization     ==============================================

fprintf('\n=== Creating Visualizations ===\n');

% Create scatter plot grids for each correlation type using tight_subplot
% Each plot will have rows = one metric type, columns = other metric type

%% 1. Criticality vs Behavior Scatter Plots
figure(400); clf;
set(gcf, 'Position', monitorOne);
sgtitle('Criticality vs Behavior Metric Correlations', 'FontSize', 16);

% Calculate subplot layout: rows = criticality metrics, columns = behavior metrics
nCriticalityMetrics = length(criticalityMetricNames);
nBehaviorMetrics = length(behaviorMetricNames);

% Use tight_subplot for clean layout
ha = tight_subplot(nCriticalityMetrics, nBehaviorMetrics, [0.1 0.1], [0.1 0.1], [0.1 0.1]);

% Plot scatter for each criticality vs behavior metric pair
for i = 1:nCriticalityMetrics
    for j = 1:nBehaviorMetrics
        subplotIdx = (i-1) * nBehaviorMetrics + j;
        axes(ha(subplotIdx));
        
        % Get data for this area (use first area for now)
        a = areasToTest(1);
        validIdx = validTimePoints{a}.criticalityVsBehavior;
        
        if sum(validIdx) >= minValidPoints
            % Get data for the criticality metric (y-axis)
            switch criticalityMetricNames{i}
                case 'd2'
                    metric1Data = d2Nat{a}(validIdx);
                case 'decodingAccuracy'
                    metric1Data = decodingAccuracyNat{a}(validIdx);
            end
            
            % Get data for the behavior metric (x-axis)
            switch behaviorMetricNames{j}
                case 'behaviorSwitches'
                    metric2Data = behaviorSwitches(validIdx);
                case 'behaviorProportion'
                    metric2Data = behaviorProportion(validIdx);
                case 'behaviorDwellTimesMean'
                    metric2Data = behaviorDwellTimesMean(validIdx);
                case 'behaviorDwellTimesStd'
                    metric2Data = behaviorDwellTimesStd(validIdx);
                case 'behaviorOccupancyEntropy'
                    metric2Data = behaviorOccupancyEntropy(validIdx);
                case 'kinPCA1Std'
                    metric2Data = kinPCA1Std(validIdx);
                case 'kinPCA2Std'
                    metric2Data = kinPCA2Std(validIdx);
            end
            
            scatter(metric2Data, metric1Data, 20, 'filled', 'MarkerFaceAlpha', .6);
            
            % Only label leftmost and bottom plots to avoid clutter
            if j == 1
                ylabel(criticalityMetricNames{i}, 'Interpreter', 'none');
            else
                set(gca, 'YTickLabel', []);
            end
            if i == nCriticalityMetrics
                xlabel(behaviorMetricNames{j}, 'Interpreter', 'none');
            else
                set(gca, 'XTickLabel', []);
            end
            
            % Find correlation value for this specific pair
            metric1Idx = i;
            metric2Idx = j + nCriticalityMetrics;
            if ~isempty(areaResultsCriticalityVsBehavior) && isfield(areaResultsCriticalityVsBehavior, 'correlationMatrix')
                R = areaResultsCriticalityVsBehavior(a).correlationMatrix;
                if ~isempty(R) && ~all(isnan(R(:)))
                    corrValue = R(metric1Idx, metric2Idx);
                    title(sprintf('r=%.3f', corrValue), 'FontSize', 9);
                else
                    title('', 'FontSize', 9);
                end
            else
                title('', 'FontSize', 9);
            end
            grid on;
        else
            text(0.5, 0.5, 'Insufficient Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
            set(gca, 'XTick', [], 'YTick', []);
        end
    end
end

%% 2. HMM vs Behavior Scatter Plots
figure(401); clf;
set(gcf, 'Position', monitorTwo);
sgtitle('HMM vs Behavior Metric Correlations', 'FontSize', 16);

% Calculate subplot layout: rows = HMM metrics, columns = behavior metrics
nHmmMetrics = length(hmmMetricNames);

% Use tight_subplot for clean layout
ha = tight_subplot(nHmmMetrics, nBehaviorMetrics, [0.1 0.1], [0.1 0.1], [0.1 0.1]);

% Plot scatter for each HMM vs behavior metric pair
for i = 1:nHmmMetrics
    for j = 1:nBehaviorMetrics
        subplotIdx = (i-1) * nBehaviorMetrics + j;
        axes(ha(subplotIdx));
        
        % Get data for this area (use first area for now)
        a = areasToTest(1);
        validIdx = validTimePoints{a}.hmmVsBehavior;
        
        if sum(validIdx) >= minValidPoints
            % Get data for the HMM metric (y-axis)
            switch hmmMetricNames{i}
                case 'stateOccupancyEntropy'
                    metric1Data = stateOccupancyEntropy(validIdx);
                case 'stateDwellTimesMean'
                    metric1Data = stateDwellTimesMean(validIdx);
                case 'stateDwellTimesStd'
                    metric1Data = stateDwellTimesStd(validIdx);
                case 'numUniqueStates'
                    metric1Data = numUniqueStates(validIdx);
                case 'meanMaxProbability'
                    metric1Data = meanMaxProbability(validIdx);
                case 'stateProbabilityVariance'
                    metric1Data = stateProbabilityVariance(validIdx);
                case 'klDivergenceMean'
                    metric1Data = klDivergenceMean(validIdx);
            end
            
            % Get data for the behavior metric (x-axis)
            switch behaviorMetricNames{j}
                case 'behaviorSwitches'
                    metric2Data = behaviorSwitches(validIdx);
                case 'behaviorProportion'
                    metric2Data = behaviorProportion(validIdx);
                case 'behaviorDwellTimesMean'
                    metric2Data = behaviorDwellTimesMean(validIdx);
                case 'behaviorDwellTimesStd'
                    metric2Data = behaviorDwellTimesStd(validIdx);
                case 'behaviorOccupancyEntropy'
                    metric2Data = behaviorOccupancyEntropy(validIdx);
                case 'kinPCA1Std'
                    metric2Data = kinPCA1Std(validIdx);
                case 'kinPCA2Std'
                    metric2Data = kinPCA2Std(validIdx);
            end
            
            scatter(metric2Data, metric1Data, 20, 'filled', 'MarkerFaceAlpha', .6);
            
            % Only label leftmost and bottom plots to avoid clutter
            if j == 1
                ylabel(hmmMetricNames{i}, 'Interpreter', 'none');
            else
                set(gca, 'YTickLabel', []);
            end
            if i == nHmmMetrics
                xlabel(behaviorMetricNames{j}, 'Interpreter', 'none');
            else
                set(gca, 'XTickLabel', []);
            end
            
            % Find correlation value for this specific pair
            metric1Idx = i;
            metric2Idx = j + nHmmMetrics;
            if ~isempty(areaResultsHmmVsBehavior) && isfield(areaResultsHmmVsBehavior, 'correlationMatrix')
                R = areaResultsHmmVsBehavior(a).correlationMatrix;
                if ~isempty(R) && ~all(isnan(R(:)))
                    corrValue = R(metric1Idx, metric2Idx);
                    title(sprintf('r=%.3f', corrValue), 'FontSize', 9);
                else
                    title('', 'FontSize', 9);
                end
            else
                title('', 'FontSize', 9);
            end
            grid on;
        else
            text(0.5, 0.5, 'Insufficient Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
            set(gca, 'XTick', [], 'YTick', []);
        end
    end
end

%% 3. Criticality vs HMM Scatter Plots
figure(402); clf;
set(gcf, 'Position', monitorTwo);
sgtitle('Criticality vs HMM Metric Correlations', 'FontSize', 16);

% Use tight_subplot for clean layout
ha = tight_subplot(nCriticalityMetrics, nHmmMetrics, [0.1 0.1], [0.1 0.1], [0.1 0.1]);

% Plot scatter for each criticality vs HMM metric pair
for i = 1:nCriticalityMetrics
    for j = 1:nHmmMetrics
        subplotIdx = (i-1) * nHmmMetrics + j;
        axes(ha(subplotIdx));
        
        % Get data for this area (use first area for now)
        a = areasToTest(1);
        validIdx = validTimePoints{a}.criticalityVsBehavior; % Use criticality valid points
        
        if sum(validIdx) >= minValidPoints
            % Get data for the criticality metric (y-axis)
            switch criticalityMetricNames{i}
                case 'd2'
                    metric1Data = d2Nat{a}(validIdx);
                case 'decodingAccuracy'
                    metric1Data = decodingAccuracyNat{a}(validIdx);
            end
            
            % For HMM metrics, we need to interpolate to criticality time points
            validCriticalityTimes = startSNat{a}(validIdx);
            
            % Interpolate HMM metric to criticality time points
            switch hmmMetricNames{j}
                case 'stateOccupancyEntropy'
                    metric2Data = interp1(hmmWindowCenterTimes, stateOccupancyEntropy, validCriticalityTimes, 'linear', 'extrap');
                case 'stateDwellTimesMean'
                    metric2Data = interp1(hmmWindowCenterTimes, stateDwellTimesMean, validCriticalityTimes, 'linear', 'extrap');
                case 'stateDwellTimesStd'
                    metric2Data = interp1(hmmWindowCenterTimes, stateDwellTimesStd, validCriticalityTimes, 'linear', 'extrap');
                case 'numUniqueStates'
                    metric2Data = interp1(hmmWindowCenterTimes, numUniqueStates, validCriticalityTimes, 'linear', 'extrap');
                case 'meanMaxProbability'
                    metric2Data = interp1(hmmWindowCenterTimes, meanMaxProbability, validCriticalityTimes, 'linear', 'extrap');
                case 'stateProbabilityVariance'
                    metric2Data = interp1(hmmWindowCenterTimes, stateProbabilityVariance, validCriticalityTimes, 'linear', 'extrap');
                case 'klDivergenceMean'
                    metric2Data = interp1(hmmWindowCenterTimes, klDivergenceMean, validCriticalityTimes, 'linear', 'extrap');
            end
            
            % Remove any NaN values from interpolation
            validInterpIdx = ~isnan(metric2Data);
            if sum(validInterpIdx) >= minValidPoints
                metric1Data = metric1Data(validInterpIdx);
                metric2Data = metric2Data(validInterpIdx);
                
                scatter(metric2Data, metric1Data, 20, 'filled', 'MarkerFaceAlpha', .6);
                
                % Only label leftmost and bottom plots to avoid clutter
                if j == 1
                    ylabel(criticalityMetricNames{i}, 'Interpreter', 'none');
                else
                    set(gca, 'YTickLabel', []);
                end
                if i == nCriticalityMetrics
                    xlabel(hmmMetricNames{j}, 'Interpreter', 'none');
                else
                    set(gca, 'XTickLabel', []);
                end
                
                % Find correlation value for this specific pair
                metric1Idx = i;
                metric2Idx = j + nCriticalityMetrics;
                if ~isempty(areaResultsCriticalityVsHmm) && isfield(areaResultsCriticalityVsHmm, 'correlationMatrix')
                    R = areaResultsCriticalityVsHmm(a).correlationMatrix;
                    if ~isempty(R) && ~all(isnan(R(:)))
                        corrValue = R(metric1Idx, metric2Idx);
                        title(sprintf('r=%.3f', corrValue), 'FontSize', 9);
                    else
                        title('', 'FontSize', 9);
                    end
                else
                    title('', 'FontSize', 9);
                end
                grid on;
            else
                text(0.5, 0.5, 'Insufficient Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
                set(gca, 'XTick', [], 'YTick', []);
            end
        else
            text(0.5, 0.5, 'Insufficient Data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
            set(gca, 'XTick', [], 'YTick', []);
        end
    end
end

%% ==============================================     Save Results     ==============================================

% Save correlation results
correlationResults = struct();

% Store all three correlation analyses
correlationResults.criticalityVsBehavior = struct();
correlationResults.criticalityVsBehavior.corrResults = corrResultsCriticalityVsBehavior;
correlationResults.criticalityVsBehavior.areaResults = areaResultsCriticalityVsBehavior;
correlationResults.criticalityVsBehavior.metricNames = [criticalityMetricNames, behaviorMetricNames];

correlationResults.hmmVsBehavior = struct();
correlationResults.hmmVsBehavior.corrResults = corrResultsHmmVsBehavior;
correlationResults.hmmVsBehavior.areaResults = areaResultsHmmVsBehavior;
correlationResults.hmmVsBehavior.metricNames = [hmmMetricNames, behaviorMetricNames];

correlationResults.criticalityVsHmm = struct();
correlationResults.criticalityVsHmm.corrResults = corrResultsCriticalityVsHmm;
correlationResults.criticalityVsHmm.areaResults = areaResultsCriticalityVsHmm;
correlationResults.criticalityVsHmm.metricNames = [criticalityMetricNames, hmmMetricNames];

% Store analysis parameters
correlationResults.parameters.correlationMethod = correlationMethod;
correlationResults.parameters.minValidPoints = minValidPoints;
correlationResults.parameters.significanceLevel = significanceLevel;
correlationResults.parameters.areasToTest = areasToTest;
correlationResults.parameters.propBhvList = propBhvList;

% Store data collection timing parameters
correlationResults.parameters.collectStart = opts.collectStart;
correlationResults.parameters.collectFor = opts.collectFor;

% Store bin sizes for each data type
correlationResults.parameters.hmmBinSize = hmmBinSize;
correlationResults.parameters.criticalityBinSize = criticalityBinSize;
correlationResults.parameters.bhvBinSize = bhvBinSize;
correlationResults.parameters.kinBinSize = kinBinSize;

% Store window and step sizes
correlationResults.parameters.criticalityWindowSize = criticalityWindowSize;
correlationResults.parameters.stepSize = stepSize;

% Store HMM parameters
correlationResults.parameters.natOrReach = natOrReach;
correlationResults.parameters.brainArea = brainArea;
correlationResults.parameters.numStates = numStates;

% Store metric categories
correlationResults.parameters.criticalityMetrics = criticalityMetricNames;
correlationResults.parameters.behaviorMetrics = behaviorMetricNames;
correlationResults.parameters.hmmMetrics = hmmMetricNames;

% Store all metrics in the main structure
correlationResults.metrics = metrics;

% Save to file
save(fullfile(paths.dropPath, sprintf('criticality_hmm_behavior_correlations_%s_%s_%s.mat', ...
    correlationMethod, natOrReach, brainArea)), 'correlationResults');

fprintf('\nAnalysis complete! Results saved to criticality_hmm_behavior_correlations_%s_%s_%s.mat\n', ...
    correlationMethod, natOrReach, brainArea);

%% ==============================================     Summary     ==============================================

fprintf('\n=== Analysis Summary ===\n');
fprintf('Successfully completed comprehensive correlation analysis:\n');
fprintf('1. Criticality vs Behavior metrics: %d significant correlations\n', length(corrResultsCriticalityVsBehavior));
fprintf('2. HMM vs Behavior metrics: %d significant correlations\n', length(corrResultsHmmVsBehavior));
fprintf('3. Criticality vs HMM metrics: %d significant correlations\n', length(corrResultsCriticalityVsHmm));

fprintf('\nData collection parameters:\n');
fprintf('  Collection start: %.1f s (%.1f min)\n', opts.collectStart, opts.collectStart/60);
fprintf('  Collection duration: %.1f s (%.1f min)\n', opts.collectFor, opts.collectFor/60);
fprintf('  Collection end: %.1f s (%.1f min)\n', opts.collectStart + opts.collectFor, (opts.collectStart + opts.collectFor)/60);

fprintf('\nData bin sizes used:\n');
fprintf('  HMM data: %.6f s\n', hmmBinSize);
fprintf('  Behavior data: %.6f s (matched to HMM)\n', bhvBinSize);
fprintf('  Criticality data: %.3f s\n', criticalityBinSize);
fprintf('  Kinematics data: %.1f s\n', kinBinSize);

fprintf('\nBehavior metrics calculated:\n');
fprintf('  - Behavior switches\n');
fprintf('  - Behavior proportion\n');
fprintf('  - Behavior dwell times (mean and std)\n');
fprintf('  - Behavior occupancy entropy\n');
fprintf('  - Kinematics PCA1 standard deviation\n');
fprintf('  - Kinematics PCA2 standard deviation\n');

fprintf('\nHMM metrics calculated:\n');
fprintf('  - State occupancy entropy\n');
fprintf('  - State dwell times (mean and std)\n');
fprintf('  - Number of unique states\n');
fprintf('  - Mean maximum probability\n');
fprintf('  - State probability variance\n');
fprintf('  - KL divergence between consecutive probability distributions\n');

fprintf('\nVisualizations created:\n');
fprintf('  - Figure 400: Criticality vs Behavior scatter plots\n');
fprintf('  - Figure 401: HMM vs Behavior scatter plots\n');
fprintf('  - Figure 402: Criticality vs HMM scatter plots\n');
fprintf('  - All plots use tight_subplot with rows = one metric type, columns = other metric type\n');

fprintf('\nScript completed successfully!\n');
