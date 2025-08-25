%%
% Criticality Correlations Script
% Loads results from criticality_decoding_accuracy.m and performs correlation analyses
% with multiple metrics including d2, decoding accuracy, and additional metrics to be added

%%    Load behavior IDs
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 30 * 60; % seconds
opts.minFiringRate = .05;

paths = get_paths;

% Monitor setup
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

% Load behavior data
getDataType = 'behavior';
opts.frameSize = .1;
bhvBinSize = opts.frameSize;
get_standard_data
[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);


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

% Areas to analyze
areasToTest = 2:3;

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

    fprintf('Analysis parameters:\n');
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

%% ==============================================     Prepare Metric Vectors     ==============================================

fprintf('\n=== Preparing Metric Vectors ===\n');

% Initialize metric structure
metrics = struct();

% Store existing metrics
metrics.d2 = d2Nat;
metrics.decodingAccuracy = decodingAccuracyNat;
metrics.timePoints = startSNat;
metrics.kinPCAStd = kinPCAStd;

% Note: validTimePoints will be calculated after all metrics are computed

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
kinPCAStd = nan(size(d2Nat{areasToTest(1)}));

% Process each criticality window
for i = 1:length(validD2Times)
    centerTime = validD2Times(i);

    % Calculate window boundaries (same as used for criticality)
    windowStartTime = centerTime - criticalityWindowSize/2;
    windowEndTime = centerTime + criticalityWindowSize/2;

    % Convert to frame indices for behavior data
    % bhvID is in 0.1 second bins, so convert times to frame indices
    startFrame = max(1, round(windowStartTime / bhvBinSize));
    endFrame = min(length(bhvID), round(windowEndTime / bhvBinSize));

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

fprintf('Behavior metrics calculated for %d time windows\n', length(validD2Times));

% Add behavior metrics to metrics structure (same for all areas)
metrics.behaviorSwitches = behaviorSwitches;
metrics.behaviorProportion = behaviorProportion;
metrics.kinPCAStd = kinPCAStd;

%% ==============================================     Calculate Valid Time Points     ==============================================

fprintf('\n=== Calculating Valid Time Points ===\n');

% Get valid time points (where we have d2, decoding accuracy, behavior switches, behavior proportion, and kinPCAStd)
validTimePoints = cell(1, length(areas));
for a = areasToTest
    validD2Idx = ~isnan(d2Nat{a});
    validAccIdx = ~isnan(decodingAccuracyNat{a});
    validSwitchIdx = ~isnan(behaviorSwitches);
    validPropIdx = ~isnan(behaviorProportion);
    validKinIdx = ~isnan(kinPCAStd);
    validTimePoints{a} = validD2Idx & validAccIdx & validSwitchIdx & validPropIdx & validKinIdx;

    fprintf('Area %s: %d valid time points\n', areas{a}, sum(validTimePoints{a}));
end

%% ==============================================     Correlation Analysis     ==============================================

fprintf('\n=== Correlation Analysis ===\n');

% Initialize correlation results structure
corrResults = [];
corrIdx = 1;

% Define metric names for correlation matrix (same for all areas)
metricNames = {'d2', 'decodingAccuracy', 'behaviorSwitches', 'behaviorProportion', 'kinPCAStd'};

% Initialize area results structure
areaResults = struct();

% Calculate correlation matrix for each area
for a = areasToTest
    fprintf('\n--- Area: %s ---\n', areas{a});

    % Get valid data points
    validIdx = validTimePoints{a};

    if sum(validIdx) >= minValidPoints
        % Extract all metrics for this area
        d2Data = d2Nat{a}(validIdx)';
        accData = decodingAccuracyNat{a}(validIdx)';
        switchData = behaviorSwitches(validIdx)';
        propData = behaviorProportion(validIdx)';
        kinData = kinPCAStd(validIdx)';

        % Create data matrix for correlation analysis
        dataMatrix = [d2Data, accData, switchData, propData, kinData];

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
        R = nan(5, 5);
        P = nan(5, 5);
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
bar(corrVals);
set(gca, 'XTickLabel', corrLabels, 'XTickLabelRotation', 45);
ylabel('Correlation Coefficient');
title(sprintf('Correlation Analysis (%s)', upper(correlationMethod)));
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
% Create correlation matrix heatmap plots
figure(401); clf;
set(gcf, 'Position', monitorTwo);

% Use tight_subplot for 2x2 layout (correlation matrices and p-value matrices)
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
    % colormap('parula');

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

exportgraphics(gcf, fullfile(paths.dropPath, sprintf('criticality/correlation_summary_%s.png', correlationMethod)), 'Resolution', 300);

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
                case 'd2'
                    metric1Data = d2Nat{a}(validIdx);
                case 'decodingAccuracy'
                    metric1Data = decodingAccuracyNat{a}(validIdx);
                case 'behaviorSwitches'
                    metric1Data = behaviorSwitches(validIdx);
                case 'behaviorProportion'
                    metric1Data = behaviorProportion(validIdx);
                case 'kinPCAStd'
                    metric1Data = kinPCAStd(validIdx);
            end

            switch metrics{j}
                case 'd2'
                    metric2Data = d2Nat{a}(validIdx);
                case 'decodingAccuracy'
                    metric2Data = decodingAccuracyNat{a}(validIdx);
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

    sgtitle(sprintf('Correlation Scatter Plots - %s', areas{a}));
end

exportgraphics(gcf, fullfile(paths.dropPath, sprintf('criticality/correlation_scatter_%s.png', correlationMethod)), 'Resolution', 300);

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

% Save to file
save(fullfile(paths.dropPath, sprintf('criticality_correlations_%s_%s.mat', correlationMethod, transOrWithin)), 'correlationResults');

fprintf('\nAnalysis complete! Results saved to criticality_correlations_%s_%s.mat\n', correlationMethod, transOrWithin);

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
%     validAdditionalIdx = ~isnan(additionalMetric{a});
%     validTimePoints{a} = validTimePoints{a} & validAdditionalIdx;
% end
%
% % Add correlation analysis
% for a = areasToTest
%     validIdx = validTimePoints{a};
%     d2Data = d2Nat{a}(validIdx);
%     additionalData = additionalMetric{a}(validIdx);
%
%     [rho, pval] = corr(d2Data, additionalData, 'type', correlationMethod);
%     % Store results...
% end