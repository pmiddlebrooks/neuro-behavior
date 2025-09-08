%%
% Criticality Decoding Accuracy Script
% Combines criticality measurements (d2) with decoding accuracy analysis
% Uses sliding window analysis on naturalistic data
% Tests correlation between d2 and decoding accuracy for multiple methods
% 
% This script loads SVM decoding results from svm_decoding_compare.m based on
% user-specified parameters (frame size, nDim, transOrWithin) and calculates
% correlations between criticality (d2) and decoding accuracy for all methods:
% - PCA
% - UMAP  
% - PSID with kinematics (psidKin)
% - PSID with behavior labels (psidBhv)
% - ICG
%
% User Parameters (modify at top of script):
% - frameSize: Frame size in seconds (must match svm_decoding_compare.m)
% - nDim: Number of dimensions (must match svm_decoding_compare.m)  
% - transOrWithin: Analysis type ('all', 'trans', 'transPost', 'within')
% - areasToTest: Brain areas to analyze

%% ==============================================     Load Existing Criticality Data     ==============================================

fprintf('\n=== Loading Existing Criticality Data ===\n');

% Load existing criticality results
try
    load(fullfile(paths.dropPath, 'criticality_compare_results.mat'), 'results');
    fprintf('Loaded existing criticality results\n');
    
    % Extract d2 data and parameters
    d2Nat = results.naturalistic.d2;
    startSNat = results.naturalistic.startS;
    criticalityBinSize = results.naturalistic.unifiedBinSize;
    criticalityWindowSize = results.naturalistic.unifiedWindowSize;
    stepSize = results.params.stepSize;
    
    fprintf('Criticality parameters: bin size = %.3f s, window size = %.1f s, step size = %.1f s\n', ...
        criticalityBinSize, criticalityWindowSize, stepSize);
    
catch
    fprintf('Error: Could not load criticality_compare_results.mat\n');
    fprintf('Please run criticality_compare.m first to generate the required data.\n');
    return;
end

collectStart = results.naturalistic.collectStart;
collectFor = results.naturalistic.collectFor;


%% ==============================================     USER PARAMETERS     ==============================================

% User-specified parameters for SVM decoding results
frameSize = 0.15;  % Frame size in seconds (must match svm_decoding_compare.m)
nDim = 6;          % Number of dimensions (must match svm_decoding_compare.m)
transOrWithin = 'all';  % Analysis type: 'all', 'trans', 'transPost', 'within'

% Define brain areas
areas = {'M23', 'M56', 'DS', 'VS'};

% Choose which areas to test
areasToTest = 2:4;

fprintf('\n=== USER PARAMETERS ===\n');
fprintf('Frame size: %.3f seconds\n', frameSize);
fprintf('Number of dimensions: %d\n', nDim);
fprintf('Analysis type: %s\n', transOrWithin);
fprintf('Areas to test: %s\n', strjoin(areas(areasToTest), ', '));
fprintf('Expected methods: PCA, UMAP, PSIDKin, PSIDBhv, PSIDKin_remaining, PSIDBhv_remaining, ICG\n');



%% ==============================================     Load Existing Decoding Results     ==============================================

fprintf('\n=== Loading Existing Multi-Area Decoding Results ===\n');

% Load existing multi-area SVM decoding comparison results
svmResultsPath = fullfile(paths.dropPath, 'decoding');
svmFiles = dir(fullfile(svmResultsPath, 'svm_decoding_compare_multi_area_*.mat'));

if isempty(svmFiles)
    fprintf('Error: No multi-area SVM decoding comparison files found in %s\n', svmResultsPath);
    fprintf('Please run svm_decoding_compare.m first to generate the required data.\n');
    return;
end

% Find the file that matches our parameters
matchingFile = '';
for i = 1:length(svmFiles)
    filename = svmFiles(i).name;
    % Extract parameters from filename: svm_decoding_compare_multi_area_{transOrWithin}_nDim{nDim}_bin{frameSize}_nShuffles{nShuffles}.mat
    pattern = sprintf('svm_decoding_compare_multi_area_%s_nDim%d_bin%.2f_', transOrWithin, nDim, frameSize);
    if contains(filename, pattern)
        matchingFile = filename;
        break;
    end
end

if isempty(matchingFile)
    fprintf('Error: No matching SVM decoding file found for parameters:\n');
    fprintf('  transOrWithin: %s\n', transOrWithin);
    fprintf('  nDim: %d\n', nDim);
    fprintf('  frameSize: %.2f\n', frameSize);
    fprintf('\nAvailable files:\n');
    for i = 1:length(svmFiles)
        fprintf('  %s\n', svmFiles(i).name);
    end
    fprintf('\nPlease run svm_decoding_compare.m with the correct parameters first.\n');
    return;
end

% Load the matching file
svmResultsFile = fullfile(svmResultsPath, matchingFile);
load(svmResultsFile, 'allResults');
fprintf('Loaded multi-area SVM decoding results from: %s\n', matchingFile);

% Extract results for the areas we want to test
svmLatents = cell(1, length(areas));
svmMethods = cell(1, length(areas));
svmBinSize = cell(1, length(areas));
svmSvmInd = cell(1, length(areas));
svmSvmID = cell(1, length(areas));
svmAccuracy = cell(1, length(areas));
svmModels = cell(1, length(areas));
allPredictions = cell(1, length(areas));
allPredictionIndices = cell(1, length(areas));

% Extract data for each area to test
for a = areasToTest
    areaName = areas{a};
    fprintf('\nExtracting results for area %s...\n', areaName);
    
    % Check if this area exists in the loaded results
    if ~isempty(allResults.latents{a})
        svmLatents{a} = allResults.latents{a};
        svmMethods{a} = allResults.methods{a};
        svmBinSize{a} = allResults.parameters.frameSize;
        svmSvmInd{a} = allResults.svmInd{a};
        svmSvmID{a} = allResults.svmID{a};
        svmAccuracy{a} = allResults.accuracy{a};
        svmModels{a} = allResults.svmModels{a};
        allPredictions{a} = allResults.allPredictions{a};
        allPredictionIndices{a} = allResults.allPredictionIndices{a};
        
        fprintf('Area %s - SVM parameters: bin size = %.3f s, methods = %s\n', ...
            areaName, svmBinSize{a}, strjoin(svmMethods{a}, ', '));
        fprintf('Area %s - SVM accuracies: %s\n', ...
            areaName, strjoin(arrayfun(@(x) sprintf('%.3f', x), svmAccuracy{a}, 'UniformOutput', false), ', '));
    else
        fprintf('Warning: No data available for area %s in the multi-area file\n', areaName);
    end
end

% Check if we successfully loaded data for all areas
loadedAreas = areasToTest(~cellfun(@isempty, svmLatents(areasToTest)));
if isempty(loadedAreas)
    fprintf('Error: No SVM decoding results loaded for any areas. Please check file availability.\n');
    return;
end

fprintf('\nSuccessfully loaded results for areas: %s\n', strjoin(areas(loadedAreas), ', '));

% Calculate window size to match criticality analysis for each area
% The criticality analysis uses stepSize to move through the data
% We need to calculate how many SVM bins fit in the criticality window
svmWindowSize = cell(1, length(areas));
svmStepSize = cell(1, length(areas));

for a = loadedAreas
    svmWindowSize{a} = criticalityWindowSize / frameSize;  % Number of SVM bins per criticality window
    svmStepSize{a} = stepSize / frameSize;  % Number of SVM bins to step
    fprintf('Area %s - Criticality window size: %.1f s = %.1f SVM bins\n', ...
        areas{a}, criticalityWindowSize, svmWindowSize{a});
    fprintf('Area %s - Criticality step size: %.1f s = %.1f SVM bins\n', ...
        areas{a}, stepSize, svmStepSize{a});
end

%% ==============================================     Setup and Variables     ==============================================

% Get paths and setup
paths = get_paths;

% Monitor setup for plotting
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);



%% ==============================================     Calculate Decoding Accuracy for Each Window     ==============================================

fprintf('\n=== Calculating Decoding Accuracy for Each Criticality Window ===\n');

% Initialize storage for decoding accuracy results
decodingAccuracyNat = cell(1, length(areas));
corrMatNat_d2Decoding = cell(1, length(areas));

% For each brain area
for a = loadedAreas
    fprintf('\nProcessing area %s...\n', areas{a});
    
    % Check if we have data for this area
    if isempty(svmLatents{a})
        fprintf('No SVM data available for area %s, skipping\n', areas{a});
        continue;
    end
    
    % Get valid d2 time points
    validD2Idx = ~isnan(d2Nat{a});
    validD2Times = startSNat{a}(validD2Idx);
    
    % Initialize decoding accuracy array for each method
    decodingAccuracyNat{a} = struct();
    for m = 1:length(svmMethods{a})
        methodName = svmMethods{a}{m};
        decodingAccuracyNat{a}.(methodName) = nan(size(d2Nat{a}));
    end
    
    % Process each criticality window
    for i = 1:length(validD2Times)
        centerTime = validD2Times(i);
        
        % Calculate window boundaries (same as used for criticality)
        windowStartTime = centerTime - criticalityWindowSize/2;
        windowEndTime = centerTime + criticalityWindowSize/2;
        
        % Convert to frame indices for the SVM data
        startFrame = max(1, round(windowStartTime / frameSize));
        endFrame = min(length(svmSvmInd{a}), round(windowEndTime / frameSize));
        
        % Get the indices within this window that correspond to SVM analysis
        windowSvmIndices = svmSvmInd{a}(startFrame:endFrame);
        
        % For each method, calculate decoding accuracy
        for m = 1:length(svmMethods{a})
            methodName = svmMethods{a}{m};
            
            % Get the pre-computed predictions for this method
            methodPredictions = allPredictions{a}{m};
            methodPredictionIndices = allPredictionIndices{a}{m};
            
            % Find which predictions fall within this window
            windowPredictionMask = ismember(methodPredictionIndices, windowSvmIndices);
            windowPredictions = methodPredictions(windowPredictionMask);
            
            % Get the corresponding behavior labels for this window
            windowBehaviorLabels = svmSvmID{a}(startFrame:endFrame);
            
            % Filter out invalid behavior labels (-1)
            validIndices = windowBehaviorLabels ~= -1;
            windowBehaviorLabels = windowBehaviorLabels(validIndices);
            windowPredictions = windowPredictions(validIndices);
            
            % Calculate accuracy for this window
            if length(windowPredictions) > 5 && length(windowBehaviorLabels) > 5
                % Ensure we have the same number of predictions and labels
                nPredictions = length(windowPredictions);
                nLabels = length(windowBehaviorLabels);
                
                if nPredictions == nLabels
                    accuracy = sum(windowPredictions == windowBehaviorLabels) / nPredictions;
                    decodingAccuracyNat{a}.(methodName)(i) = accuracy;
                else
                    % If there's a mismatch, use the minimum length
                    minLength = min(nPredictions, nLabels);
                    accuracy = sum(windowPredictions(1:minLength) == windowBehaviorLabels(1:minLength)) / minLength;
                    decodingAccuracyNat{a}.(methodName)(i) = accuracy;
                end
            else
                decodingAccuracyNat{a}.(methodName)(i) = nan;
            end
        end
    end
    fprintf('Area %s completed\n', areas{a});
end

%% ==============================================     Correlation Analysis     ==============================================

% Calculate correlations between d2 and decoding accuracy for each method
fprintf('\n=== Correlation Analysis: d2 and Decoding Accuracy ===\n');
corrResults = struct();

for a = loadedAreas
    fprintf('\nProcessing correlations for area %s...\n', areas{a});
    
    % Check if we have data for this area
    if isempty(svmLatents{a})
        fprintf('No SVM data available for area %s, skipping correlations\n', areas{a});
        continue;
    end
    
    % Get minimum length for d2
    minLenNat = length(d2Nat{a});
    
    % Initialize correlation results for this area
    corrResults(a).area = areas{a};
    corrResults(a).methods = svmMethods{a};
    corrResults(a).correlations = nan(1, length(svmMethods{a}));
    corrResults(a).p_values = nan(1, length(svmMethods{a}));
    corrResults(a).n_valid_points = nan(1, length(svmMethods{a}));
    
    % Calculate correlations for each method
    for m = 1:length(svmMethods{a})
        methodName = svmMethods{a}{m};
        
        % Get decoding accuracy for this method
        decodingAcc = decodingAccuracyNat{a}.(methodName);
        
        % Ensure same length
        minLen = min(minLenNat, length(decodingAcc));
        
        % Naturalistic data
        Xnat = [d2Nat{a}(1:minLen); decodingAcc(1:minLen)]';
        validIdx = ~any(isnan(Xnat), 2);
        
        if sum(validIdx) > 10
            corrMatNat_d2Decoding{a}.(methodName) = corrcoef(Xnat(validIdx, :));
            corrResults(a).correlations(m) = corrMatNat_d2Decoding{a}.(methodName)(1, 2);
            [rho, pval] = corr(Xnat(validIdx, 1), Xnat(validIdx, 2), 'type', 'Pearson');
            corrResults(a).p_values(m) = pval;
            corrResults(a).n_valid_points(m) = sum(validIdx);
            
            fprintf('  %s: Correlation = %.3f, p = %.3f, n = %d\n', ...
                methodName, corrResults(a).correlations(m), corrResults(a).p_values(m), corrResults(a).n_valid_points(m));
        else
            corrMatNat_d2Decoding{a}.(methodName) = nan(2, 2);
            corrResults(a).correlations(m) = nan;
            corrResults(a).p_values(m) = nan;
            corrResults(a).n_valid_points(m) = 0;
            
            fprintf('  %s: Insufficient data points\n', methodName);
        end
    end
end

%% ==============================================     Plotting Results     ==============================================

% Create comparison plots for each area and method
for a = loadedAreas
    % Check if we have data for this area
    if isempty(svmLatents{a})
        fprintf('No SVM data available for area %s, skipping plots\n', areas{a});
        continue;
    end
    
    figure(200 + a); clf;
    set(gcf, 'Position', monitorTwo);
    
    % Use tight_subplot for 2x1 layout
    ha = tight_subplot(2, 1, [0.05 0.02], [0.1 0.05], [0.1 0.05]);
    
    % Plot d2
    axes(ha(1));
    hold on;
    validIdx = ~isnan(d2Nat{a});
    plot(startSNat{a}(validIdx)/60, d2Nat{a}(validIdx), '-', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('Distance to Criticality (d2)', 'FontSize', 14);
    title(sprintf('%s - Distance to Criticality (d2)', areas{a}), 'FontSize', 14);
    grid on;
    set(gca, 'XTickLabel', [], 'FontSize', 14);
    set(gca, 'YTickLabelMode', 'auto');
    xlim([collectStart/60 (collectStart+collectFor)/60]);
    
    % Plot decoding accuracy for all methods
    axes(ha(2));
    hold on;
    
    % Define colors for different methods
    methodColors = {'r', 'g', 'b', 'm', 'c', 'y', 'k', '--r', '--g', '--b'};
    
    for m = 1:length(svmMethods{a})
        methodName = svmMethods{a}{m};
        decodingAcc = decodingAccuracyNat{a}.(methodName);
        validIdx = ~isnan(decodingAcc);
        
        if sum(validIdx) > 0
            plot(startSNat{a}(validIdx)/60, decodingAcc(validIdx), '-', ...
                'Color', methodColors{mod(m-1, length(methodColors))+1}, ...
                'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', upper(methodName));
        end
    end
    
    ylabel('Decoding Accuracy', 'FontSize', 14);
    title(sprintf('%s - Decoding Accuracy (All Methods)', areas{a}), 'FontSize', 14);
    grid on;
    xlabel('Minutes', 'FontSize', 14);
    set(gca, 'YTickLabelMode', 'auto', 'FontSize', 14);
    set(gca, 'XTickLabelMode', 'auto');
    xlim([collectStart/60 (collectStart+collectFor)/60]);
    ylim([0 1]);
    legend('Location', 'best');
    
    sgtitle(sprintf('Criticality and Decoding Accuracy - %s (nDim=%d, bin=%.2f)', areas{a}, nDim, frameSize), 'FontSize', 14);
    exportgraphics(gcf, fullfile(paths.dropPath, sprintf('criticality/criticality_decoding_%s_%s_nDim%d_bin%.2f.png', areas{a}, transOrWithin, nDim, frameSize)), 'Resolution', 300);
end

% Plot correlation matrices for each area and method
figure(300); clf;
set(gcf, 'Position', monitorTwo);

nAreas = length(loadedAreas);
maxMethods = max(cellfun(@length, svmMethods(loadedAreas)));

for a = 1:nAreas
    areaIdx = loadedAreas(a);
    if isempty(svmLatents{areaIdx})
        continue;
    end
    
    for m = 1:length(svmMethods{areaIdx})
        methodName = svmMethods{areaIdx}{m};
        subplot(nAreas, maxMethods, (a-1)*maxMethods + m);
        
        if isfield(corrMatNat_d2Decoding{areaIdx}, methodName)
            imagesc(corrMatNat_d2Decoding{areaIdx}.(methodName));
            colorbar;
            title(sprintf('%s - %s', areas{areaIdx}, upper(methodName)));
            xticks(1:2); yticks(1:2);
            xticklabels({'d2','Decoding'}); yticklabels({'d2','Decoding'});
            axis square;
            caxis([-1 1]);
        else
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center');
            title(sprintf('%s - %s', areas{areaIdx}, upper(methodName)));
            axis off;
        end
    end
end
sgtitle(sprintf('Correlation Matrices: d2 and Decoding Accuracy (nDim=%d, bin=%.2f)', nDim, frameSize));

% Plot correlation summary for all methods
figure(301); clf;
set(gcf, 'Position', monitorOne);

% Prepare data for grouped bar plot
% Find the maximum number of methods across all areas
maxMethods = max(cellfun(@length, svmMethods(loadedAreas)));
corrData = nan(nAreas, maxMethods);
pData = nan(nAreas, maxMethods);

for a = 1:nAreas
    areaIdx = loadedAreas(a);
    if ~isempty(svmLatents{areaIdx})
        nMethodsArea = length(svmMethods{areaIdx});
        corrData(a, 1:nMethodsArea) = corrResults(areaIdx).correlations;
        pData(a, 1:nMethodsArea) = corrResults(areaIdx).p_values;
    end
end

% Create grouped bar plot
bar(corrData);
set(gca, 'XTickLabel', areas(loadedAreas));
ylabel('Correlation Coefficient');
title(sprintf('Correlation between d2 and Decoding Accuracy (nDim=%d, bin=%.2f)', nDim, frameSize));
legend(upper(svmMethods{loadedAreas(1)}), 'Location', 'best');
grid on;

% Add significance markers
hold on;
for a = 1:nAreas
    areaIdx = loadedAreas(a);
    if ~isempty(svmLatents{areaIdx})
        for m = 1:length(svmMethods{areaIdx})
            if pData(a, m) < 0.05
                plot(a, corrData(a, m) + 0.05, '*', 'Color', 'black', 'MarkerSize', 60);
            end
        end
    end
end

exportgraphics(gcf, fullfile(paths.dropPath, sprintf('criticality/correlation_summary_%s_nDim%d_bin%.2f.png', transOrWithin, nDim, frameSize)), 'Resolution', 300);


%% ==============================================     Save Results     ==============================================

% Save all results
results = struct();
results.areas = areas;
results.loadedAreas = loadedAreas;
results.svmMethods = svmMethods;
results.frameSize = frameSize;
results.nDim = nDim;
results.transOrWithin = transOrWithin;
results.svmAccuracy = svmAccuracy;

% Naturalistic data results
results.naturalistic.d2 = d2Nat;
results.naturalistic.decodingAccuracy = decodingAccuracyNat;
results.naturalistic.startS = startSNat;
results.naturalistic.criticalityBinSize = criticalityBinSize;
results.naturalistic.criticalityWindowSize = criticalityWindowSize;
results.naturalistic.corrMat = corrMatNat_d2Decoding;
results.naturalistic.corrResults = corrResults;

% Analysis parameters
results.params.stepSize = stepSize;
results.params.frameSize = frameSize;
results.params.nDim = nDim;
results.params.transOrWithin = transOrWithin;
results.params.svmWindowSize = svmWindowSize;
results.params.svmStepSize = svmStepSize;

% Save to file
filename = sprintf('criticality_decoding_results_%s_nDim%d_bin%.2f.mat', ...
    transOrWithin, nDim, frameSize);
save(fullfile(paths.dropPath, filename), 'results');

fprintf('\nAnalysis complete! Results saved to %s\n', filename);

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('Correlation between d2 and decoding accuracy:\n');
fprintf('Area\t\tMethod\t\tCorrelation\tp-value\t\tn\n');
fprintf('----\t\t------\t\t----------\t-------\t\t-\n');

for a = 1:length(loadedAreas)
    areaIdx = loadedAreas(a);
    if isempty(svmLatents{areaIdx})
        fprintf('%s\t\tNo data available\n', areas{areaIdx});
        continue;
    end
    
    for m = 1:length(svmMethods{areaIdx})
        methodName = svmMethods{areaIdx}{m};
        corrVal = corrResults(areaIdx).correlations(m);
        pVal = corrResults(areaIdx).p_values(m);
        nVal = corrResults(areaIdx).n_valid_points(m);
        
        if ~isnan(corrVal)
            significance = '';
            if pVal < 0.001
                significance = '***';
            elseif pVal < 0.01
                significance = '**';
            elseif pVal < 0.05
                significance = '*';
            end
            
            fprintf('%s\t\t%s\t\t%.3f%s\t\t%.3f\t\t%d\n', ...
                areas{areaIdx}, upper(methodName), corrVal, significance, pVal, nVal);
        else
            fprintf('%s\t\t%s\t\tN/A\t\tN/A\t\t%d\n', ...
                areas{areaIdx}, upper(methodName), nVal);
        end
    end
end

fprintf('\nAnalysis complete!\n'); 