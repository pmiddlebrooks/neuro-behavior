%%
% Criticality Mutual Information Analysis
% Calculates mutual information between brain areas for different criticality measures
% Assumes criticality measures are pre-calculated and provided as input
% Tests hypothesis that MI is maximal near criticality and minimal far from criticality

%%
paths = get_paths;

% Monitor setup
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

%% ==============================================     Input Data     ==============================================

% Define brain areas
areas = {'M23', 'M56', 'DS', 'VS'};

% Define criticality measures
measures = {'mrBr', 'd2', 'dcc', 'kappa'};
measureNames = {'MR Branching Ratio', 'Distance to Criticality (d2)', 'Distance to Criticality (dcc)', 'Kappa'};

% Load pre-calculated criticality measures
% You can modify this section to load your own data
% Example structure: criticalityData.area.measure = time_series_vector

% For demonstration, we'll create sample data structure
% Replace this with your actual data loading
fprintf('Loading criticality measures data...\n');

% Example: Load from saved results
% load('criticality_compare_results.mat');
% criticalityData = results;

% For now, create sample data structure (replace with your actual data)
criticalityData = struct();

% Sample data for naturalistic
criticalityData.naturalistic = struct();
for a = 1:length(areas)
    for m = 1:length(measures)
        % Create sample time series (replace with your actual data)
        criticalityData.naturalistic.(areas{a}).(measures{m}) = randn(100, 1);
    end
end

% Sample data for reach
criticalityData.reach = struct();
for a = 1:length(areas)
    for m = 1:length(measures)
        for m = 1:length(measures)
            % Create sample time series (replace with your actual data)
            criticalityData.reach.(areas{a}).(measures{m}) = randn(100, 1);
        end
    end
end

%% ==============================================     Analysis Parameters     ==============================================

% Mutual information parameters
nBins = 20;            % Number of bins for histogram-based MI calculation
nPermutations = 100;   % Number of permutations for significance testing

% Criticality thresholds for analysis
criticalityThresholds = struct();
criticalityThresholds.mrBr = [0.8, 1.2];    % Near criticality range for MR branching ratio
criticalityThresholds.d2 = [0.1, 0.3];      % Near criticality range for d2
criticalityThresholds.dcc = [0.1, 0.3];     % Near criticality range for dcc
criticalityThresholds.kappa = [1.0, 1.5];   % Near criticality range for kappa

%% ==============================================     Mutual Information Calculation     ==============================================

fprintf('\n=== Mutual Information Analysis ===\n');

% Function to calculate mutual information between two vectors
function mi = calculate_mutual_information(x, y, nBins)
    % Remove NaN values
    validIdx = ~isnan(x) & ~isnan(y);
    x = x(validIdx);
    y = y(validIdx);
    
    if length(x) < 10  % Need sufficient data points
        mi = nan;
        return;
    end
    
    % Create histograms
    [~, edgesX] = histcounts(x, nBins);
    [~, edgesY] = histcounts(y, nBins);
    
    % Calculate joint and marginal distributions
    [jointHist, ~, ~] = histcounts2(x, y, edgesX, edgesY);
    
    % Normalize to get probabilities
    jointProb = jointHist / sum(jointHist(:));
    marginalX = sum(jointProb, 2);
    marginalY = sum(jointProb, 1);
    
    % Calculate mutual information
    mi = 0;
    for i = 1:size(jointProb, 1)
        for j = 1:size(jointProb, 2)
            if jointProb(i, j) > 0 && marginalX(i) > 0 && marginalY(j) > 0
                mi = mi + jointProb(i, j) * log2(jointProb(i, j) / (marginalX(i) * marginalY(j)));
            end
        end
    end
end

% Calculate mutual information for all area pairs and measures
areaPairs = nchoosek(1:length(areas), 2);
numPairs = size(areaPairs, 1);

% Initialize MI results for both datasets
miResultsNat = struct();
miResultsRea = struct();

% Process both datasets
datasets = {'naturalistic', 'reach'};

for d = 1:length(datasets)
    dataset = datasets{d};
    fprintf('\nProcessing %s data...\n', dataset);
    
    % Initialize MI matrices
    miResults = struct();
    miResults.miMatrix = nan(length(measures), numPairs);
    miResults.areaPairs = areaPairs;
    miResults.areaPairNames = cell(numPairs, 1);
    
    % Calculate MI for each measure and area pair
    for m = 1:length(measures)
        measure = measures{m};
        fprintf('Calculating MI for %s...\n', measureNames{m});
        
        for p = 1:numPairs
            area1 = areaPairs(p, 1);
            area2 = areaPairs(p, 2);
            
            % Get data for this pair
            data1 = criticalityData.(dataset).(areas{area1}).(measure);
            data2 = criticalityData.(dataset).(areas{area2}).(measure);
            
            % Calculate mutual information
            mi = calculate_mutual_information(data1, data2, nBins);
            miResults.miMatrix(m, p) = mi;
            
            % Store area pair name
            if m == 1  % Only need to do this once per pair
                miResults.areaPairNames{p} = sprintf('%s-%s', areas{area1}, areas{area2});
            end
        end
    end
    
    % Store results
    if d == 1
        miResultsNat = miResults;
    else
        miResultsRea = miResults;
    end
end

%% ==============================================     Statistical Significance Testing     ==============================================

fprintf('\n=== Statistical Significance Testing ===\n');

% Function to calculate permutation-based p-values
function pValue = permutation_test(data1, data2, nPermutations, nBins)
    % Remove NaN values
    validIdx = ~isnan(data1) & ~isnan(data2);
    data1 = data1(validIdx);
    data2 = data2(validIdx);
    
    if length(data1) < 10
        pValue = nan;
        return;
    end
    
    % Calculate observed MI
    observedMI = calculate_mutual_information(data1, data2, nBins);
    
    % Permutation test
    nullMI = zeros(nPermutations, 1);
    for perm = 1:nPermutations
        % Shuffle one of the vectors
        shuffledData2 = data2(randperm(length(data2)));
        nullMI(perm) = calculate_mutual_information(data1, shuffledData2, nBins);
    end
    
    % Calculate p-value
    pValue = sum(nullMI >= observedMI) / nPermutations;
end

% Calculate p-values for all MI values
for d = 1:length(datasets)
    dataset = datasets{d};
    fprintf('\nCalculating p-values for %s data...\n', dataset);
    
    % Initialize p-value matrix
    miResults = struct();
    miResults.pValues = nan(length(measures), numPairs);
    
    % Calculate p-values for each measure and area pair
    for m = 1:length(measures)
        measure = measures{m};
        fprintf('Calculating p-values for %s...\n', measureNames{m});
        
        for p = 1:numPairs
            area1 = areaPairs(p, 1);
            area2 = areaPairs(p, 2);
            
            % Get data for this pair
            data1 = criticalityData.(dataset).(areas{area1}).(measure);
            data2 = criticalityData.(dataset).(areas{area2}).(measure);
            
            % Calculate p-value
            pValue = permutation_test(data1, data2, nPermutations, nBins);
            miResults.pValues(m, p) = pValue;
        end
    end
    
    % Store results
    if d == 1
        miResultsNat.pValues = miResults.pValues;
    else
        miResultsRea.pValues = miResults.pValues;
    end
end

%% ==============================================     Criticality-Based MI Analysis     ==============================================

fprintf('\n=== Criticality-Based MI Analysis ===\n');

% Function to classify criticality state
function state = classify_criticality_state(value, measure, thresholds)
    if strcmp(measure, 'mrBr')
        if value >= thresholds.mrBr(1) && value <= thresholds.mrBr(2)
            state = 'near_critical';
        else
            state = 'far_critical';
        end
    elseif strcmp(measure, 'd2')
        if value >= thresholds.d2(1) && value <= thresholds.d2(2)
            state = 'near_critical';
        else
            state = 'far_critical';
        end
    elseif strcmp(measure, 'dcc')
        if value >= thresholds.dcc(1) && value <= thresholds.dcc(2)
            state = 'near_critical';
        else
            state = 'far_critical';
        end
    elseif strcmp(measure, 'kappa')
        if value >= thresholds.kappa(1) && value <= thresholds.kappa(2)
            state = 'near_critical';
        else
            state = 'far_critical';
        end
    end
end

% Analyze MI as a function of criticality state
criticalityAnalysis = struct();

for d = 1:length(datasets)
    dataset = datasets{d};
    fprintf('\nAnalyzing criticality-based MI for %s data...\n', dataset);
    
    criticalityAnalysis.(dataset) = struct();
    
    for m = 1:length(measures)
        measure = measures{m};
        fprintf('Analyzing %s...\n', measureNames{m});
        
        % Initialize arrays for near and far criticality
        nearCriticalMI = [];
        farCriticalMI = [];
        
        % Analyze each area pair
        for p = 1:numPairs
            area1 = areaPairs(p, 1);
            area2 = areaPairs(p, 2);
            
            % Get data for this pair
            data1 = criticalityData.(dataset).(areas{area1}).(measure);
            data2 = criticalityData.(dataset).(areas{area2}).(measure);
            
            % Remove NaN values
            validIdx = ~isnan(data1) & ~isnan(data2);
            data1 = data1(validIdx);
            data2 = data2(validIdx);
            
            if length(data1) < 10
                continue;
            end
            
            % Calculate MI for each time point
            for t = 1:length(data1)
                % Create time windows around current time point
                windowSize = min(20, length(data1));
                startIdx = max(1, t - windowSize/2);
                endIdx = min(length(data1), t + windowSize/2);
                
                windowData1 = data1(startIdx:endIdx);
                windowData2 = data2(startIdx:endIdx);
                
                % Calculate MI for this window
                windowMI = calculate_mutual_information(windowData1, windowData2, nBins);
                
                if ~isnan(windowMI)
                    % Classify criticality state based on current value
                    currentValue = data1(t);
                    state = classify_criticality_state(currentValue, measure, criticalityThresholds);
                    
                    if strcmp(state, 'near_critical')
                        nearCriticalMI = [nearCriticalMI, windowMI];
                    else
                        farCriticalMI = [farCriticalMI, windowMI];
                    end
                end
            end
        end
        
        % Store results
        criticalityAnalysis.(dataset).(measure).nearCriticalMI = nearCriticalMI;
        criticalityAnalysis.(dataset).(measure).farCriticalMI = farCriticalMI;
        criticalityAnalysis.(dataset).(measure).meanNearCritical = nanmean(nearCriticalMI);
        criticalityAnalysis.(dataset).(measure).meanFarCritical = nanmean(farCriticalMI);
        criticalityAnalysis.(dataset).(measure).stdNearCritical = nanstd(nearCriticalMI);
        criticalityAnalysis.(dataset).(measure).stdFarCritical = nanstd(farCriticalMI);
    end
end

%% ==============================================     Visualization     ==============================================

% Create visualization plots
fprintf('\n=== Creating Visualizations ===\n');

% 1. MI heatmaps for each measure
for m = 1:length(measures)
    figure(300 + m); clf;
    set(gcf, 'Position', monitorTwo);
    
    % Create MI matrix for visualization
    miMatrixNat = nan(length(areas), length(areas));
    miMatrixRea = nan(length(areas), length(areas));
    
    % Fill in the MI values
    for p = 1:numPairs
        area1 = areaPairs(p, 1);
        area2 = areaPairs(p, 2);
        
        miMatrixNat(area1, area2) = miResultsNat.miMatrix(m, p);
        miMatrixNat(area2, area1) = miMatrixNat(area1, area2);  % Symmetric
        
        miMatrixRea(area1, area2) = miResultsRea.miMatrix(m, p);
        miMatrixRea(area2, area1) = miMatrixRea(area1, area2);  % Symmetric
    end
    
    % Plot heatmaps
    subplot(1, 2, 1);
    imagesc(miMatrixNat);
    colorbar;
    title(sprintf('%s - Naturalistic', measureNames{m}));
    xticks(1:length(areas)); yticks(1:length(areas));
    xticklabels(areas); yticklabels(areas);
    axis square;
    
    subplot(1, 2, 2);
    imagesc(miMatrixRea);
    colorbar;
    title(sprintf('%s - Reach', measureNames{m}));
    xticks(1:length(areas)); yticks(1:length(areas));
    xticklabels(areas); yticklabels(areas);
    axis square;
    
    sgtitle(sprintf('Mutual Information - %s', measureNames{m}));
end

% 2. Comparison of MI across measures
figure(400); clf;
set(gcf, 'Position', monitorTwo);

% Create bar plot comparing MI across measures
ha = tight_subplot(2, 1, [0.1 0.05], [0.15 0.1], [0.1 0.05]);

% Naturalistic data
axes(ha(1));
barData = miResultsNat.miMatrix';
bar(barData);
xlabel('Area Pairs');
ylabel('Mutual Information');
title('Naturalistic Data');
legend(measureNames, 'Location', 'best');
xticks(1:numPairs);
xticklabels(miResultsNat.areaPairNames);
grid on;

% Reach data
axes(ha(2));
barData = miResultsRea.miMatrix';
bar(barData);
xlabel('Area Pairs');
ylabel('Mutual Information');
title('Reach Data');
legend(measureNames, 'Location', 'best');
xticks(1:numPairs);
xticklabels(miResultsRea.areaPairNames);
grid on;

sgtitle('Mutual Information Comparison Across Measures');

% 3. Criticality-based MI analysis
figure(500); clf;
set(gcf, 'Position', monitorTwo);

% Create subplots for each measure
ha = tight_subplot(2, 2, [0.1 0.05], [0.15 0.1], [0.1 0.05]);

for m = 1:length(measures)
    measure = measures{m};
    axes(ha(m));
    
    % Prepare data for plotting
    natNear = criticalityAnalysis.naturalistic.(measure).nearCriticalMI;
    natFar = criticalityAnalysis.naturalistic.(measure).farCriticalMI;
    reaNear = criticalityAnalysis.reach.(measure).nearCriticalMI;
    reaFar = criticalityAnalysis.reach.(measure).farCriticalMI;
    
    % Create box plot
    data = [natNear, natFar, reaNear, reaFar];
    groups = [ones(1,length(natNear)), 2*ones(1,length(natFar)), ...
              3*ones(1,length(reaNear)), 4*ones(1,length(reaFar))];
    
    boxplot(data, groups, 'Labels', {'Nat Near', 'Nat Far', 'Rea Near', 'Rea Far'});
    ylabel('Mutual Information');
    title(sprintf('%s', measureNames{m}));
    grid on;
end

sgtitle('MI Comparison: Near vs Far from Criticality');

% 4. Statistical comparison
figure(600); clf;
set(gcf, 'Position', monitorTwo);

% Create summary plot
measures_to_plot = 1:length(measures);
nearCriticalMeans = zeros(length(measures), 2);
farCriticalMeans = zeros(length(measures), 2);

for m = 1:length(measures)
    measure = measures{m};
    nearCriticalMeans(m, 1) = criticalityAnalysis.naturalistic.(measure).meanNearCritical;
    nearCriticalMeans(m, 2) = criticalityAnalysis.reach.(measure).meanNearCritical;
    farCriticalMeans(m, 1) = criticalityAnalysis.naturalistic.(measure).meanFarCritical;
    farCriticalMeans(m, 2) = criticalityAnalysis.reach.(measure).meanFarCritical;
end

% Plot comparison
subplot(1, 2, 1);
bar([nearCriticalMeans(:,1), farCriticalMeans(:,1)]);
xlabel('Criticality Measures');
ylabel('Mean Mutual Information');
title('Naturalistic Data');
legend({'Near Critical', 'Far from Critical'}, 'Location', 'best');
xticks(1:length(measures));
xticklabels(measureNames);
grid on;

subplot(1, 2, 2);
bar([nearCriticalMeans(:,2), farCriticalMeans(:,2)]);
xlabel('Criticality Measures');
ylabel('Mean Mutual Information');
title('Reach Data');
legend({'Near Critical', 'Far from Critical'}, 'Location', 'best');
xticks(1:length(measures));
xticklabels(measureNames);
grid on;

sgtitle('MI Comparison: Near vs Far from Criticality (Summary)');

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Summary Statistics ===\n');

% Print summary statistics for MI
for d = 1:length(datasets)
    dataset = datasets{d};
    if d == 1
        miResults = miResultsNat;
    else
        miResults = miResultsRea;
    end
    
    fprintf('\n%s Data:\n', dataset);
    fprintf('Mean MI across all measures and area pairs: %.3f\n', nanmean(miResults.miMatrix(:)));
    fprintf('Std MI across all measures and area pairs: %.3f\n', nanstd(miResults.miMatrix(:)));
    
    for m = 1:length(measures)
        fprintf('%s - Mean MI: %.3f, Max MI: %.3f\n', measureNames{m}, ...
            nanmean(miResults.miMatrix(m, :)), nanmax(miResults.miMatrix(m, :)));
    end
    
    % Significant MI values
    significantMI = miResults.pValues < 0.05;  % p < 0.05
    fprintf('Number of significant MI values (p < 0.05): %d/%d\n', sum(significantMI(:)), numel(significantMI));
end

% Print criticality-based analysis results
fprintf('\n=== Criticality-Based Analysis Results ===\n');

for d = 1:length(datasets)
    dataset = datasets{d};
    fprintf('\n%s Data:\n', dataset);
    
    for m = 1:length(measures)
        measure = measures{m};
        analysis = criticalityAnalysis.(dataset).(measure);
        
        fprintf('%s:\n', measureNames{m});
        fprintf('  Near criticality - Mean MI: %.3f ± %.3f (n=%d)\n', ...
            analysis.meanNearCritical, analysis.stdNearCritical, length(analysis.nearCriticalMI));
        fprintf('  Far from criticality - Mean MI: %.3f ± %.3f (n=%d)\n', ...
            analysis.meanFarCritical, analysis.stdFarCritical, length(analysis.farCriticalMI));
        
        % Calculate effect size (Cohen's d)
        if length(analysis.nearCriticalMI) > 0 && length(analysis.farCriticalMI) > 0
            pooledStd = sqrt(((length(analysis.nearCriticalMI)-1)*analysis.stdNearCritical^2 + ...
                (length(analysis.farCriticalMI)-1)*analysis.stdFarCritical^2) / ...
                (length(analysis.nearCriticalMI) + length(analysis.farCriticalMI) - 2));
            cohensD = (analysis.meanNearCritical - analysis.meanFarCritical) / pooledStd;
            fprintf('  Effect size (Cohen''s d): %.3f\n', cohensD);
        end
    end
end

%% ==============================================     Save Results     ==============================================

% Save all results
results = struct();
results.areas = areas;
results.measures = measures;
results.measureNames = measureNames;
results.areaPairs = areaPairs;
results.areaPairNames = miResultsNat.areaPairNames;

% MI results
results.miResults.naturalistic = miResultsNat;
results.miResults.reach = miResultsRea;

% Criticality analysis results
results.criticalityAnalysis = criticalityAnalysis;

% Analysis parameters
results.params.nBins = nBins;
results.params.nPermutations = nPermutations;
results.params.criticalityThresholds = criticalityThresholds;

% Save to file
save(fullfile(paths.dropPath, 'criticality_mutual_information_results.mat'), 'results');

fprintf('\nAnalysis complete! Results saved to criticality_mutual_information_results.mat\n'); 