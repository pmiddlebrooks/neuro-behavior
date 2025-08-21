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
    hmmBinSize = hmm_results.HmmParam.binSize;
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

% Calculate time points for sliding windows
windowSizeBins = round(windowSize / hmmBinSize);
stepSizeBins = round(stepSize / hmmBinSize);
numWindows = floor((totalHmmBins - windowSizeBins) / stepSizeBins) + 1;

% Initialize behavior metric arrays
behaviorSwitches = nan(numWindows, 1);
behaviorProportion = nan(numWindows, 1);
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
    
    % Calculate behavior switches
    if length(windowBhvID) > 1
        % Find where behavior changes (switches)
        behaviorChanges = diff(windowBhvID) ~= 0;
        numSwitches = sum(behaviorChanges);
        behaviorSwitches(i) = numSwitches;
    else
        behaviorSwitches(i) = 0;
    end
    
    % Calculate proportion of specified behaviors
    if length(windowBhvID) > 0
        % Count how many frames have behaviors in the specified list
        specifiedBehaviors = ismember(windowBhvID, propBhvList);
        proportion = sum(specifiedBehaviors) / length(windowBhvID);
        behaviorProportion(i) = proportion;
    else
        behaviorProportion(i) = 0;
    end
    
    % Calculate standard deviation of kinPCA within the window
    if length(windowBhvID) > 0
        % Get kinPCA data for the same window
        windowKinPCA = kinPCA(startFrame:endFrame);
        kinPCAStd(i) = std(windowKinPCA);
    else
        kinPCAStd(i) = nan;
    end
end

% Trim arrays to actual number of windows processed
numWindows = i - 1;
behaviorSwitches = behaviorSwitches(1:numWindows);
behaviorProportion = behaviorProportion(1:numWindows);
kinPCAStd = kinPCAStd(1:numWindows);
windowCenterTimes = windowCenterTimes(1:numWindows);

fprintf('Behavior metrics calculated for %d time windows\n', numWindows);

% Add behavior metrics to metrics structure (same for all areas)
metrics.behaviorSwitches = behaviorSwitches;
metrics.behaviorProportion = behaviorProportion;
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
        % 1. State occupancy entropy
        stateCounts = histcounts(validStates, 1:(numStates+1));
        stateProportions = stateCounts / length(validStates);
        stateProportions = stateProportions(stateProportions > 0); % Remove zero probabilities
        if ~isempty(stateProportions)
            stateOccupancyEntropy(i) = -sum(stateProportions .* log2(stateProportions));
        end
        
        % 2. State dwell times mean and std
        if length(validStates) > 1
            % Find state transitions
            stateChanges = diff(validStates) ~= 0;
            if any(stateChanges)
                % Calculate dwell times
                changeIndices = find(stateChanges);
                dwellTimes = [changeIndices(1); diff(changeIndices); length(validStates) - changeIndices(end)];
                dwellTimes = dwellTimes * hmmBinSize; % Convert to seconds
                
                stateDwellTimesMean(i) = mean(dwellTimes);
                stateDwellTimesStd(i) = std(dwellTimes);
            else
                % Single state throughout window
                stateDwellTimesMean(i) = length(validStates) * hmmBinSize;
                stateDwellTimesStd(i) = 0;
            end
        else
            % Single state
            stateDwellTimesMean(i) = hmmBinSize;
            stateDwellTimesStd(i) = 0;
        end
        
        % 3. Number of unique states
        numUniqueStates(i) = length(unique(validStates));
        
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
    validKinIdx = ~isnan(kinPCAStd);
    validEntropyIdx = ~isnan(stateOccupancyEntropy);
    validDwellMeanIdx = ~isnan(stateDwellTimesMean);
    validDwellStdIdx = ~isnan(stateDwellTimesStd);
    validUniqueIdx = ~isnan(numUniqueStates);
    validMaxProbIdx = ~isnan(meanMaxProbability);
    validProbVarIdx = ~isnan(stateProbabilityVariance);
    validKLIdx = ~isnan(klDivergenceMean);
    
    validTimePoints{a} = validBhvSwitchIdx & validBhvPropIdx & validKinIdx & ...
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
               'klDivergenceMean', 'behaviorSwitches', 'behaviorProportion', 'kinPCAStd'};

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
        kinData = kinPCAStd(validIdx)';
        
        % Create data matrix for correlation analysis
        dataMatrix = [entropyData, dwellMeanData, dwellStdData, uniqueData, ...
                     maxProbData, probVarData, klData, switchData, propData, kinData];
        
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
        
        % Store significant correlations in results structure
        for i = 1:length(metricNames)
            for j = (i+1):length(metricNames)  % Only upper triangle to avoid duplicates
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

% Plot scatter plots for each area
for a = areasToTest
    % Create new figure for each area
    figure(400 + a); clf;
    set(gcf, 'Position', monitorTwo);
    
    % Get metrics for this area
    metrics = areaResults(a).metrics;
    R = areaResults(a).correlationMatrix;
    
    % Calculate number of subplots needed
    nMetrics = length(metrics);
    nComparisons = nMetrics * (nMetrics-1) / 2;
    nRows = ceil(sqrt(nComparisons));
    nCols = ceil(nComparisons/nRows);
    
    % Plot scatter for each metric pair
    subplotIdx = 1;
    for i = 1:length(metrics)
        for j = (i+1):length(metrics)
            subplot(nRows, nCols, subplotIdx);
            validIdx = validTimePoints{a};
            
            % Get data for the two metrics being compared
            switch metrics{i}
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
                case 'behaviorSwitches'
                    metric1Data = behaviorSwitches(validIdx);
                case 'behaviorProportion'
                    metric1Data = behaviorProportion(validIdx);
                case 'kinPCAStd'
                    metric1Data = kinPCAStd(validIdx);
            end
            
            switch metrics{j}
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
                case 'behaviorSwitches'
                    metric2Data = behaviorSwitches(validIdx);
                case 'behaviorProportion'
                    metric2Data = behaviorProportion(validIdx);
                case 'kinPCAStd'
                    metric2Data = kinPCAStd(validIdx);
            end
            
            scatter(metric1Data, metric2Data, 20, 'filled', 'MarkerFaceAlpha', .6);
            xlabel(metrics{i});
            ylabel(metrics{j});
            title(sprintf('%s vs %s\nr=%.3f', metrics{i}, metrics{j}, R(i,j)));
            grid on;
            subplotIdx = subplotIdx + 1;
        end
    end
    
    sgtitle(sprintf('HMM Correlation Scatter Plots - %s', areas{a}));
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
