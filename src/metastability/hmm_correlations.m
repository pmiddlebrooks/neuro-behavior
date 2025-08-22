%%
% HMM Correlations Script
% Loads HMM results and performs correlation analyses with behavior and kinematic variables
% Computes sliding window metrics for HMM states and probabilities
%
% PARAMETER LOADING:
% This script automatically loads optimal step and window sizes from criticality_compare_results.mat
% to ensure consistency with the criticality analysis. The parameters are:
% - stepSize: from results.params.stepSize
% - windowSize: from results.naturalistic.unifiedWindowSize or results.reach.unifiedWindowSize
% depending on the data type being analyzed.

%%    Load behavior IDs
paths = get_paths;

% Monitor setup
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

%% Load behavior data
opts = neuro_behavior_options;

%%   Load kinematics (in .1 s binSize right now)
kinBinSize = .1;
getDataType = 'kinematics';
get_standard_data

startFrame = 1 + opts.collectStart / kinBinSize;
endFrame = startFrame - 1 + (opts.collectFor / kinBinSize);
kinData = kinData(startFrame:endFrame,:);

[coeff, score, ~, ~, explained] = pca(zscore(kinData));
kinPCA = score(:, 1);

%% ==============================================     Analysis Parameters     ==============================================

% Correlation analysis parameters
correlationMethod = 'pearson';  % 'pearson', 'spearman', or 'kendall'
minValidPoints = 10;           % Minimum number of valid points for correlation
significanceLevel = 0.05;      % Significance level for correlation tests

% HMM analysis parameters
natOrReach = 'Nat';            % 'Nat' or 'Reach'
brainArea = 'M56';             % 'M23', 'M56', 'DS', 'VS'

% Areas to analyze (corresponds to brain areas with HMM results)
areasToTest = 1;               % Will be updated based on loaded HMM data

%% ==============================================     Load Criticality Results for Optimal Parameters     ==============================================

fprintf('\n=== Loading Criticality Results for Optimal Parameters ===\n');

try
    % Load results from criticality_compare.m to get optimal step and window sizes
    criticalityFile = fullfile(paths.dropPath, 'criticality_compare_results.mat');
    if ~exist(criticalityFile, 'file')
        error('Criticality results file not found: %s\nPlease run criticality_compare.m first to generate the required data.', criticalityFile);
    end

    load(criticalityFile, 'results');
    fprintf('Loaded criticality results from: %s\n', criticalityFile);

    % Extract optimal parameters based on data type
    if strcmp(natOrReach, 'Nat')
        stepSize = results.params.stepSize;
        windowSize = results.naturalistic.unifiedWindowSize;
        fprintf('Using naturalistic data parameters:\n');
        fprintf('  Step size: %.1f s\n', stepSize);
        fprintf('  Window size: %.1f s\n', windowSize);
    else
        stepSize = results.params.stepSize;
        windowSize = results.reach.unifiedWindowSize;
        fprintf('Using reach data parameters:\n');
        fprintf('  Step size: %.1f s\n', stepSize);
        fprintf('  Window size: %.1f s\n', windowSize);
    end

catch e
    fprintf('Error loading criticality results: %s\n', e.message);
    fprintf('Using default parameters: stepSize = 2.0s, windowSize = 5.0s\n');
    stepSize = 2.0;
    windowSize = 5.0;
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

    % Update areas to test based on loaded data
    areas = {brainArea};
    areasToTest = 1;

    % Update behavior data frame size to match HMM bin size for consistency
    opts.frameSize = hmmBinSize;
    bhvBinSize = opts.frameSize;

    % Reload behavior data with updated frame size to match HMM binning
    fprintf('Loading behavior data with HMM-matched frame size: %.6f s\n', opts.frameSize);
    opts.frameSize = hmm_results.HmmParam.BinSize;
    getDataType = 'behavior';
    get_standard_data


catch e
    fprintf('Error loading HMM results: %s\n', e.message);
    fprintf('Please ensure HMM analysis has been completed for %s data in %s area.\n', natOrReach, brainArea);
    return;
end

%% ==============================================     Prepare Metric Vectors     ==============================================

fprintf('\n=== Preparing Metric Vectors ===\n');

% Initialize metric structure
metrics = struct();

% Store existing metrics
metrics.timePoints = [];
metrics.kinPCAStd = [];

% Note: validTimePoints will be calculated after all metrics are computed

%% ==============================================     Calculate Behavior Metrics     ==============================================

fprintf('\n=== Calculating Behavior Metrics ===\n');

% Behavior metrics are the same across brain areas (global behavioral metrics)
% Calculate once and use for all areas
fprintf('Calculating behavior switches and proportion (global metrics)...\n');

% Define which behavior labels to count for proportion (user can modify this list)
propBhvList = [0 1 2 13 14 15];  % Example: grooming behaviors
fprintf('Calculating proportion of behaviors: %s\n', mat2str(propBhvList));

% Note: Different bin sizes for different data types:
% - HMM data: hmmBinSize (from HMM results)
% - Behavior data: bhvBinSize (matches HMM bin size for consistency)
% - Kinematics data: kinBinSize = 0.1 seconds (fixed)
fprintf('Data bin sizes: HMM=%.6fs, Behavior=%.6fs, Kinematics=%.1fs\n', ...
    hmmBinSize, bhvBinSize, kinBinSize);

% Calculate time points for sliding windows
windowSizeBins = round(windowSize / hmmBinSize);
stepSizeBins = round(stepSize / hmmBinSize);
numWindows = floor((totalHmmBins - windowSizeBins) / stepSizeBins) + 1;

% Initialize behavior metric arrays
behaviorSwitches = nan(numWindows, 1);
behaviorProportion = nan(numWindows, 1);
behaviorDwellTimesMean = nan(numWindows, 1);
behaviorDwellTimesStd = nan(numWindows, 1);
behaviorOccupancyEntropy = nan(numWindows, 1);
kinPCAStd = nan(numWindows, 1);
windowCenterTimes = nan(numWindows, 1);

% Process each sliding window
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
    windowCenterTimes(i) = centerBin * hmmBinSize;

    % Convert HMM bin indices to behavior/kinematics frame indices
    % Behavior and kinematics are in 0.1 second bins
    startTime = (startBin - 1) * hmmBinSize;
    endTime = endBin * hmmBinSize;

    startFrame = max(1, round(startTime / bhvBinSize));
    endFrame = min(length(bhvID), round(endTime / bhvBinSize));

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
                    behaviorDwellTimesMean(i) = length(windowBhvID) * bhvBinSize;
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
        % Single frame - can't calculate dwell times
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

    % Calculate standard deviation of kinPCA within the window
    if length(validBhvID) > 0
        % Convert HMM window boundaries to kinematics frame indices
        % Kinematics are in 0.1 second bins, different from behavior/HMM binning
        startTime = (startBin - 1) * hmmBinSize;
        endTime = endBin * hmmBinSize;
        
        kinStartFrame = max(1, round(startTime / kinBinSize));
        kinEndFrame = min(length(kinPCA), round(endTime / kinBinSize));
        
        % Get kinPCA data for the same time window
        windowKinPCA = kinPCA(kinStartFrame:kinEndFrame);
        kinPCAStd(i) = std(windowKinPCA);
    else
        kinPCAStd(i) = nan;
    end
end

% Trim arrays to actual number of windows processed
numWindows = i - 1;
behaviorSwitches = behaviorSwitches(1:numWindows);
behaviorProportion = behaviorProportion(1:numWindows);
behaviorDwellTimesMean = behaviorDwellTimesMean(1:numWindows);
behaviorDwellTimesStd = behaviorDwellTimesStd(1:numWindows);
behaviorOccupancyEntropy = behaviorOccupancyEntropy(1:numWindows);
kinPCAStd = kinPCAStd(1:numWindows);
windowCenterTimes = windowCenterTimes(1:numWindows);

fprintf('Behavior metrics calculated for %d time windows\n', numWindows);

% Add behavior metrics to metrics structure (same for all areas)
metrics.behaviorSwitches = behaviorSwitches;
metrics.behaviorProportion = behaviorProportion;
metrics.behaviorDwellTimesMean = behaviorDwellTimesMean;
metrics.behaviorDwellTimesStd = behaviorDwellTimesStd;
metrics.behaviorOccupancyEntropy = behaviorOccupancyEntropy;
metrics.kinPCAStd = kinPCAStd;
metrics.timePoints = windowCenterTimes;

%% ==============================================     Calculate HMM Metrics     ==============================================

fprintf('\n=== Calculating HMM Metrics ===\n');

% Initialize HMM metric arrays
stateOccupancyEntropy = nan(numWindows, 1);
stateDwellTimesMean = nan(numWindows, 1);
stateDwellTimesStd = nan(numWindows, 1);
numUniqueStates = nan(numWindows, 1);
meanMaxProbability = nan(numWindows, 1);
stateProbabilityVariance = nan(numWindows, 1);
klDivergenceMean = nan(numWindows, 1);

% Process each sliding window
for i = 1:numWindows
    % Calculate window boundaries in HMM bins
    startBin = (i-1) * stepSizeBins + 1;
    endBin = startBin + windowSizeBins - 1;

    % Ensure we don't exceed data bounds
    if endBin > totalHmmBins
        break;
    end

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

fprintf('HMM metrics calculated for %d time windows\n', numWindows);

% Add HMM metrics to metrics structure
metrics.stateOccupancyEntropy = stateOccupancyEntropy;
metrics.stateDwellTimesMean = stateDwellTimesMean;
metrics.stateDwellTimesStd = stateDwellTimesStd;
metrics.numUniqueStates = numUniqueStates;
metrics.meanMaxProbability = meanMaxProbability;
metrics.stateProbabilityVariance = stateProbabilityVariance;
metrics.klDivergenceMean = klDivergenceMean;

%% ==============================================     Calculate Valid Time Points     ==============================================

fprintf('\n=== Calculating Valid Time Points ===\n');

% Get valid time points (where we have all metrics)
validTimePoints = cell(1, length(areas));
for a = areasToTest
    validBhvSwitchIdx = ~isnan(behaviorSwitches);
    validBhvPropIdx = ~isnan(behaviorProportion);
    validBhvDwellMeanIdx = ~isnan(behaviorDwellTimesMean);
    validBhvDwellStdIdx = ~isnan(behaviorDwellTimesStd);
    validBhvEntropyIdx = ~isnan(behaviorOccupancyEntropy);
    validKinIdx = ~isnan(kinPCAStd);
    validEntropyIdx = ~isnan(stateOccupancyEntropy);
    validDwellMeanIdx = ~isnan(stateDwellTimesMean);
    validDwellStdIdx = ~isnan(stateDwellTimesStd);
    validUniqueIdx = ~isnan(numUniqueStates);
    validMaxProbIdx = ~isnan(meanMaxProbability);
    validProbVarIdx = ~isnan(stateProbabilityVariance);
    validKLIdx = ~isnan(klDivergenceMean);

    validTimePoints{a} = validBhvSwitchIdx & validBhvPropIdx & validBhvDwellMeanIdx & validBhvDwellStdIdx & validBhvEntropyIdx & validKinIdx & ...
        validEntropyIdx & validDwellMeanIdx & validDwellStdIdx & ...
        validUniqueIdx & validMaxProbIdx & validProbVarIdx & validKLIdx;

    fprintf('Area %s: %d valid time points\n', areas{a}, sum(validTimePoints{a}));
end

%% ==============================================     Correlation Analysis     ==============================================

fprintf('\n=== Correlation Analysis ===\n');

% Initialize correlation results structure
corrResults = [];
corrIdx = 1;

% Define metric names for correlation matrix
metricNames = {'stateOccupancyEntropy', 'stateDwellTimesMean', 'stateDwellTimesStd', ...
    'numUniqueStates', 'meanMaxProbability', 'stateProbabilityVariance', ...
    'klDivergenceMean', 'behaviorSwitches', 'behaviorProportion', 'behaviorDwellTimesMean', ...
    'behaviorDwellTimesStd', 'behaviorOccupancyEntropy', 'kinPCAStd'};

% Initialize area results structure
areaResults = struct();

% Calculate correlation matrix for each area
for a = areasToTest
    fprintf('\n--- Area: %s ---\n', areas{a});

    % Get valid data points
    validIdx = validTimePoints{a};

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
        kinData = kinPCAStd(validIdx)';

        % Create data matrix for correlation analysis
        % Ensure dataMatrix is (observations x variables)
        dataMatrix = [entropyData(:), dwellMeanData(:), dwellStdData(:), uniqueData(:), ...
            maxProbData(:), probVarData(:), klData(:), switchData(:), propData(:), bhvDwellMeanData(:), ...
            bhvDwellStdData(:), bhvEntropyData(:), kinData(:)];

        % Calculate correlation matrix
        [R, P] = corrcoef(dataMatrix, 'rows', 'complete');

        % Display correlation matrix
        fprintf('Correlation Matrix:\n');
        fprintf('%-25s', 'Metric');
        for i = 1:length(metricNames)
            fprintf('%-20s', metricNames{i});
        end
        fprintf('\n');

        for i = 1:length(metricNames)
            fprintf('%-25s', metricNames{i});
            for j = 1:length(metricNames)
                if i == j
                    fprintf('%-20s', '1.000');
                else
                    fprintf('%-20.3f', R(i,j));
                end
            end
            fprintf('\n');
        end

        % Display p-values
        fprintf('\nP-values:\n');
        fprintf('%-25s', 'Metric');
        for i = 1:length(metricNames)
            fprintf('%-20s', metricNames{i});
        end
        fprintf('\n');

        for i = 1:length(metricNames)
            fprintf('%-25s', metricNames{i});
            for j = 1:length(metricNames)
                if i == j
                    fprintf('%-20s', '1.000');
                else
                    fprintf('%-20.3f', P(i,j));
                end
            end
            fprintf('\n');
        end

        % Store significant correlations in results structure (only behavior vs HMM metrics)
        % This reduces the number of correlations from 78 (13x13) to 35 (5x7) and focuses on
        % the most relevant comparisons between behavioral and neural dynamics
        behaviorMetrics = {'behaviorSwitches', 'behaviorProportion', 'behaviorDwellTimesMean', 'behaviorDwellTimesStd', 'behaviorOccupancyEntropy'};
        hmmMetrics = {'stateOccupancyEntropy', 'stateDwellTimesMean', 'stateDwellTimesStd', 'numUniqueStates', 'meanMaxProbability', 'stateProbabilityVariance', 'klDivergenceMean'};

        for i = 1:length(metricNames)
            for j = 1:length(metricNames)
                % Only compare behavior metrics to HMM metrics
                isBehaviorMetric = ismember(metricNames{i}, behaviorMetrics);
                isHmmMetric = ismember(metricNames{j}, hmmMetrics);

                if (isBehaviorMetric && isHmmMetric) || (isHmmMetric && isBehaviorMetric)
                    if P(i,j) < significanceLevel
                        corrResults(corrIdx).area = areas{a};
                        corrResults(corrIdx).metric1 = metricNames{i};
                        corrResults(corrIdx).metric2 = metricNames{j};
                        corrResults(corrIdx).correlation = R(i,j);
                        corrResults(corrIdx).p_value = P(i,j);
                        corrResults(corrIdx).n_valid_points = sum(validIdx);
                        corrResults(corrIdx).significant = true;
                        corrIdx = corrIdx + 1;

                        fprintf('\nSignificant correlation: %s vs %s: r = %.3f, p = %.3f\n', ...
                            metricNames{i}, metricNames{j}, R(i,j), P(i,j));
                    end
                end
            end
        end

        fprintf('Total valid points: %d\n', sum(validIdx));

    else
        fprintf('Insufficient data points (%d < %d)\n', sum(validIdx), minValidPoints);
        % Initialize empty matrices for this area
        R = nan(length(metricNames), length(metricNames));
        P = nan(length(metricNames), length(metricNames));
    end

    % Store matrices for this area
    areaResults(a).correlationMatrix = R;
    areaResults(a).pValueMatrix = P;
    areaResults(a).metrics = metricNames;
end

%% ==============================================     Visualization     ==============================================

fprintf('\n=== Creating Visualizations ===\n');

% Plot correlation summary
figure(400); clf;
set(gcf, 'Position', monitorOne);

% Extract correlation values and significance (only if corrResults has entries)
if isfield(corrResults, 'correlation') && ~isempty(corrResults)
    corrVals = [corrResults.correlation];
    pVals = [corrResults.p_value];
    significant = [corrResults.significant];
else
    corrVals = [];
    pVals = [];
    significant = [];
end

% Create labels for each correlation
corrLabels = {};
for i = 1:length(corrResults)
    corrLabels{i} = sprintf('%s: %s vs %s', corrResults(i).area, corrResults(i).metric1, corrResults(i).metric2);
end

% Create bar plot
if ~isempty(corrVals)
    bar(corrVals);
    set(gca, 'XTickLabel', corrLabels, 'XTickLabelRotation', 45);
    ylabel('Correlation Coefficient');
    title(sprintf('HMM Correlation Analysis (%s)', upper(correlationMethod)));
    grid on;

    % Add significance markers
    hold on;
    for i = 1:length(significant)
        if significant(i)
            plot(i, corrVals(i) + 0.05, '*', 'Color', 'black', 'MarkerSize', 60);
        end
    end

    % Add correlation values as text
    for i = 1:length(corrVals)
        if ~isnan(corrVals(i))
            text(i, corrVals(i) + 0.02, sprintf('%.3f', corrVals(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10);
        end
    end
end

% Create correlation matrix heatmap plots
figure(401); clf;
set(gcf, 'Position', monitorTwo);

% Use tight_subplot for 2x1 layout (correlation matrices and p-value matrices)
ha = tight_subplot(2, length(areasToTest), [0.1 0.1], [0.1 0.1], [0.1 0.1]);

% Find global min/max correlation for consistent colorbar
allCorrs = [];
for a = areasToTest
    allCorrs = [allCorrs; areaResults(a).correlationMatrix(:)];
end
corrLims = [min(allCorrs) max(allCorrs)];

% Plot correlation matrices (top row)
for i = 1:length(areasToTest)
    a = areasToTest(i);
    axes(ha(i));

    % Get correlation and p-value matrices
    R = areaResults(a).correlationMatrix;
    P = areaResults(a).pValueMatrix;
    metrics = areaResults(a).metrics;

    % Plot correlation heatmap
    imagesc(R);
    colorbar;
    caxis(corrLims);
    customColorMap = bluewhitered_custom([-.8 .8]);
    colormap(customColorMap);

    % Add significance markers
    [row, col] = find(P < 0.05);
    hold on;
    for k = 1:length(row)
        text(col(k), row(k), '*', 'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontSize', 14);
    end

    % Labels and title
    set(gca, 'XTick', 1:length(metrics), 'YTick', 1:length(metrics));
    set(gca, 'XTickLabel', metrics, 'YTickLabel', metrics);
    xtickangle(45);
    title(sprintf('%s - Correlations', areas{a}));
end

% Plot p-value matrices (bottom row)
for i = 1:length(areasToTest)
    a = areasToTest(i);
    axes(ha(i + length(areasToTest)));

    % Get p-value matrix and create binary significance map
    P = areaResults(a).pValueMatrix;
    sigMap = double(P >= 0.05);  % White for significant (p < 0.05), black otherwise

    % Plot p-value heatmap
    imagesc(sigMap);
    colormap(gca, [0 0 0; 1 1 1]);  % Black and white colormap

    % Labels
    set(gca, 'XTick', 1:length(metrics), 'YTick', 1:length(metrics));
    set(gca, 'XTickLabel', metrics, 'YTickLabel', metrics);
    xtickangle(45);
    title(sprintf('%s - P-values', areas{a}));
end

    figure(402); clf;
% Plot scatter plots for each area
for a = areasToTest
    % Create new figure for each area
    set(gcf, 'Position', monitorTwo);

    % Get metrics for this area
    metrics = areaResults(a).metrics;
    R = areaResults(a).correlationMatrix;

    % Define behavior and HMM metrics
    behaviorMetrics = {'behaviorSwitches', 'behaviorProportion', 'behaviorDwellTimesMean', 'behaviorDwellTimesStd', 'behaviorOccupancyEntropy'};
    hmmMetrics = {'stateOccupancyEntropy', 'stateDwellTimesMean', 'stateDwellTimesStd', 'numUniqueStates', 'meanMaxProbability', 'stateProbabilityVariance', 'klDivergenceMean'};

    nBehaviorMetrics = length(behaviorMetrics);
    nHmmMetrics = length(hmmMetrics);

    % Plot grid: rows = HMM metrics, columns = behavior metrics
    for hmmIdx = 1:nHmmMetrics
        for behIdx = 1:nBehaviorMetrics
            subplot(nHmmMetrics, nBehaviorMetrics, (hmmIdx-1)*nBehaviorMetrics + behIdx);
            validIdx = validTimePoints{a};

            % Get data for the behavior metric (x-axis)
            switch behaviorMetrics{behIdx}
                case 'behaviorSwitches'
                    metric1Data = behaviorSwitches(validIdx);
                case 'behaviorProportion'
                    metric1Data = behaviorProportion(validIdx);
                case 'behaviorDwellTimesMean'
                    metric1Data = behaviorDwellTimesMean(validIdx);
                case 'behaviorDwellTimesStd'
                    metric1Data = behaviorDwellTimesStd(validIdx);
                case 'behaviorOccupancyEntropy'
                    metric1Data = behaviorOccupancyEntropy(validIdx);
            end

            % Get data for the HMM metric (y-axis)
            switch hmmMetrics{hmmIdx}
                case 'stateOccupancyEntropy'
                    metric2Data = stateOccupancyEntropy(validIdx);
                case 'stateDwellTimesMean'
                    metric2Data = stateDwellTimesMean(validIdx);
                case 'stateDwellTimesStd'
                    metric2Data = stateDwellTimesStd(validIdx);
                case 'numUniqueStates'
                    metric2Data = numUniqueStates(validIdx);
                case 'meanMaxProbability'
                    metric2Data = meanMaxProbability(validIdx);
                case 'stateProbabilityVariance'
                    metric2Data = stateProbabilityVariance(validIdx);
                case 'klDivergenceMean'
                    metric2Data = klDivergenceMean(validIdx);
            end

            scatter(metric1Data, metric2Data, 20, 'filled', 'MarkerFaceAlpha', .6);
            % Only label leftmost and bottom plots to avoid clutter
            if behIdx == 1
                ylabel(hmmMetrics{hmmIdx}, 'Interpreter', 'none');
            else
                set(gca, 'YTickLabel', []);
            end
            if hmmIdx == nHmmMetrics
                xlabel(behaviorMetrics{behIdx}, 'Interpreter', 'none');
            else
                set(gca, 'XTickLabel', []);
            end

            % Find correlation value for this specific pair
            metric1Idx = find(strcmp(metricNames, behaviorMetrics{behIdx}));
            metric2Idx = find(strcmp(metricNames, hmmMetrics{hmmIdx}));
            if ~isempty(metric1Idx) && ~isempty(metric2Idx)
                corrValue = R(metric1Idx, metric2Idx);
                title(sprintf('r=%.3f', corrValue), 'FontSize', 9);
            else
                title('', 'FontSize', 9);
            end
            grid on;
        end
    end

    sgtitle(sprintf('Behavior vs HMM Metric Correlations - %s', areas{a}));
end

%% ==============================================     Save Results     ==============================================

% Save correlation results
correlationResults = struct();
correlationResults.metrics = metricNames;
correlationResults.corrResults = corrResults;
correlationResults.areaResults = areaResults;
correlationResults.parameters.correlationMethod = correlationMethod;
correlationResults.parameters.minValidPoints = minValidPoints;
correlationResults.parameters.significanceLevel = significanceLevel;
correlationResults.parameters.areasToTest = areasToTest;
correlationResults.parameters.propBhvList = propBhvList;
correlationResults.parameters.hmmBinSize = hmmBinSize;
correlationResults.parameters.windowSize = windowSize;
correlationResults.parameters.stepSize = stepSize;
correlationResults.parameters.natOrReach = natOrReach;
correlationResults.parameters.brainArea = brainArea;
correlationResults.parameters.bhvBinSize = bhvBinSize;
correlationResults.parameters.kinBinSize = kinBinSize;
correlationResults.parameters.behaviorMetrics = {'behaviorSwitches', 'behaviorProportion', 'behaviorDwellTimesMean', 'behaviorDwellTimesStd', 'behaviorOccupancyEntropy'};
correlationResults.parameters.hmmMetrics = {'stateOccupancyEntropy', 'stateDwellTimesMean', 'stateDwellTimesStd', 'numUniqueStates', 'meanMaxProbability', 'stateProbabilityVariance', 'klDivergenceMean'};
correlationResults.parameters.kinematicsMetrics = {'kinPCAStd'};

% Save to file
save(fullfile(paths.dropPath, sprintf('hmm_correlations_%s_%s_%s.mat', correlationMethod, natOrReach, brainArea)), 'correlationResults');

fprintf('\nAnalysis complete! Results saved to hmm_correlations_%s_%s_%s.mat\n', correlationMethod, natOrReach, brainArea);

%% ==============================================     Framework for Additional Metrics     ==============================================

fprintf('\n=== Framework for Additional Metrics ===\n');
fprintf('To add additional metrics:\n');
fprintf('1. Load or calculate the new metric vector\n');
fprintf('2. Add it to the metrics structure: metrics.newMetric = newMetricVector;\n');
fprintf('3. Update validTimePoints to include the new metric\n');
fprintf('4. Add correlation analysis for the new metric\n');
fprintf('5. Update visualizations to include the new metric\n');

% Example structure for adding new metrics:
%
% % Load additional metric data
% additionalMetric = loadAdditionalMetric();
%
% % Add to metrics structure
% metrics.additionalMetric = additionalMetric;
%
% % Update valid time points
% for a = areasToTest
%     validAdditionalIdx = ~isnan(additionalMetric);
%     validTimePoints{a} = validTimePoints{a} & validAdditionalIdx;
% end
%
% % Add correlation analysis
% for a = areasToTest
%     validIdx = validTimePoints{a};
%     entropyData = stateOccupancyEntropy(validIdx);
%     additionalData = additionalMetric(validIdx);
%
%     [rho, pval] = corr(entropyData, additionalData, 'type', correlationMethod);
%     % Store results...
% end
