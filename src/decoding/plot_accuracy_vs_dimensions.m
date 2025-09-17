%% Plot SVM Decoding Accuracy vs Dimensions for Each Method and Area
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script loads SVM decoding results across different dimensions and creates
% comparison plots showing how accuracy varies with dimensionality for each
% latent method and brain area

%% Specify parameters
% Dimensions to analyze (should match those used in svm_decoding_compare.m)
dimToTest = [4 6 8];

% Analysis conditions to analyze
transWithinToTest = {'trans', 'transPost', 'within', 'all'};

% SVM kernel function used
kernelFunction = 'polynomial';

% Frame/bin size (should match svm_decoding_compare.m)
frameSize = 0.1;

% Number of shuffles (should match svm_decoding_compare.m)
nShuffles = 2;

% Plotting options
savePlotFlag = 1;  % Save plots as PNG files
plotPermutedData = 1;  % Include permuted data in plots

% Get paths
get_paths

% Create save path
savePath = fullfile(paths.dropPath, 'decoding');
if ~exist(savePath, 'dir')
    fprintf('Results directory does not exist: %s\n', savePath);
    return;
end

% Monitor setup for plotting
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);

% Brain areas
areas = {'M23', 'M56', 'DS', 'VS'};

%% Load results for all dimension/condition combinations
fprintf('\n=== Loading SVM Decoding Results ===\n');

% Initialize storage
allCombinedResults = struct();
allCombinedResults.dimensions = dimToTest;
allCombinedResults.conditions = transWithinToTest;
allCombinedResults.areas = areas;
allCombinedResults.kernelFunction = kernelFunction;

% Storage for each condition
for condIdx = 1:length(transWithinToTest)
    condition = transWithinToTest{condIdx};
    fprintf('\n--- Loading results for condition: %s ---\n', condition);
    
    % Initialize storage for this condition
    conditionResults = struct();
    conditionResults.condition = condition;
    conditionResults.dimensions = dimToTest;
    conditionResults.areas = areas;
    
    % Load results for each dimension
    for dimIdx = 1:length(dimToTest)
        nDim = dimToTest(dimIdx);
        fprintf('Loading %dD results...\n', nDim);
        
        % Construct filename
        filename = sprintf('svm_%s_decoding_compare_multi_area_%s_nDim%d_bin%.2f_nShuffles%d.mat', ...
            kernelFunction, condition, nDim, frameSize, nShuffles);
        fullFilePath = fullfile(savePath, filename);
        
        if exist(fullFilePath, 'file')
            % Load the results
            load(fullFilePath, 'allResults');
            
            % Store results for this dimension
            conditionResults.dimensionResults{dimIdx} = allResults;
            fprintf('  Loaded %dD results for %s\n', nDim, condition);
        else
            fprintf('  Warning: File not found: %s\n', filename);
            conditionResults.dimensionResults{dimIdx} = [];
        end
    end
    
    % Store condition results
    allCombinedResults.conditionResults{condIdx} = conditionResults;
end

%% Extract accuracy data for plotting
fprintf('\n=== Extracting Accuracy Data ===\n');

% Initialize storage for plotting data
plotData = struct();
plotData.dimensions = dimToTest;
plotData.conditions = transWithinToTest;
plotData.areas = areas;

% For each condition
for condIdx = 1:length(transWithinToTest)
    condition = transWithinToTest{condIdx};
    fprintf('Processing condition: %s\n', condition);
    
    % Initialize storage for this condition
    conditionData = struct();
    conditionData.condition = condition;
    
    % For each area
    for areaIdx = 1:length(areas)
        areaName = areas{areaIdx};
        fprintf('  Processing area: %s\n', areaName);
        
        % Initialize storage for this area
        areaData = struct();
        areaData.areaName = areaName;
        areaData.methods = {};
        areaData.accuracies = [];
        areaData.accuraciesPermuted = [];
        areaData.dimensions = [];
        
        % Extract data for each dimension
        for dimIdx = 1:length(dimToTest)
            nDim = dimToTest(dimIdx);
            
            % Get results for this dimension
            if ~isempty(allCombinedResults.conditionResults{condIdx}.dimensionResults{dimIdx})
                dimResults = allCombinedResults.conditionResults{condIdx}.dimensionResults{dimIdx};
                
                % Check if this area has data
                if areaIdx <= length(dimResults.methods) && ~isempty(dimResults.methods{areaIdx})
                    methods = dimResults.methods{areaIdx};
                    accuracy = dimResults.accuracy{areaIdx};
                    accuracyPermuted = dimResults.accuracyPermuted{areaIdx};
                    
                    % Store methods (should be same across dimensions)
                    if isempty(areaData.methods)
                        areaData.methods = methods;
                    end
                    
                    % Store accuracies
                    areaData.accuracies = [areaData.accuracies; accuracy'];
                    areaData.accuraciesPermuted = [areaData.accuraciesPermuted; accuracyPermuted];
                    areaData.dimensions = [areaData.dimensions; repmat(nDim, length(accuracy), 1)];
                    
                    fprintf('    %dD: %d methods, accuracies: %s\n', nDim, length(accuracy), ...
                        sprintf('%.3f ', accuracy));
                else
                    fprintf('    %dD: No data available\n', nDim);
                end
            else
                fprintf('    %dD: No results file\n', nDim);
            end
        end
        
        % Store area data
        conditionData.areaData{areaIdx} = areaData;
    end
    
    % Store condition data
    plotData.conditionData{condIdx} = conditionData;
end

%% Create comparison plots
fprintf('\n=== Creating Comparison Plots ===\n');

% Colors for different methods (you may want to customize these)
methodColors = [
    0.0, 0.4470, 0.7410;  % Blue
    0.8500, 0.3250, 0.0980;  % Red
    0.9290, 0.6940, 0.1250;  % Yellow
    0.4940, 0.1840, 0.5560;  % Purple
    0.4660, 0.6740, 0.1880;  % Green
    0.3010, 0.7450, 0.9330;  % Cyan
    0.6350, 0.0780, 0.1840;  % Dark Red
];

% For each condition
for condIdx = 1:length(transWithinToTest)
    condition = transWithinToTest{condIdx};
    fprintf('Creating plots for condition: %s\n', condition);
    
    % Get data for this condition
    conditionData = plotData.conditionData{condIdx};
    
    % Create figure for this condition
    fig = figure(200 + condIdx);
    set(fig, 'Position', [monitorOne(1), monitorOne(2), monitorOne(3), monitorOne(4)]);
    clf;
    
    % Create subplots for each area
    nAreas = length(areas);
    for areaIdx = 1:nAreas
        areaName = areas{areaIdx};
        areaData = conditionData.areaData{areaIdx};
        
        subplot(1, nAreas, areaIdx);
        hold on;
        
        % Check if we have data for this area
        if isempty(areaData.methods) || isempty(areaData.accuracies)
            % No data - show empty plot
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 14, 'Color', 'red');
            title(sprintf('%s - No Data', areaName), 'Interpreter', 'none');
            xlim([0, 1]);
            ylim([0, 1]);
            continue;
        end
        
        % Get unique methods
        methods = areaData.methods;
        nMethods = length(methods);
        
        % Plot each method
        for methodIdx = 1:nMethods
            methodName = methods{methodIdx};
            
            % Find data for this method across dimensions
            methodAccuracies = [];
            methodAccuraciesPermuted = [];
            methodDims = [];
            
            for dimIdx = 1:length(dimToTest)
                nDim = dimToTest(dimIdx);
                
                % Find rows for this dimension
                dimMask = areaData.dimensions == nDim;
                if any(dimMask)
                    % Get accuracy for this method at this dimension
                    if methodIdx <= size(areaData.accuracies, 2)
                        acc = areaData.accuracies(dimMask, methodIdx);
                        accPerm = areaData.accuraciesPermuted(dimMask, methodIdx, :);
                        
                        if ~isnan(acc) && acc > 0
                            methodAccuracies = [methodAccuracies; acc];
                            methodAccuraciesPermuted = [methodAccuraciesPermuted; accPerm];
                            methodDims = [methodDims; nDim];
                        end
                    end
                end
            end
            
            % Plot this method if we have data
            if ~isempty(methodAccuracies)
                % Choose color for this method
                colorIdx = mod(methodIdx - 1, size(methodColors, 1)) + 1;
                color = methodColors(colorIdx, :);
                
                % Plot real data
                plot(methodDims, methodAccuracies, 'o-', 'Color', color, ...
                    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', color, ...
                    'DisplayName', upper(methodName));
                
                % Plot permuted data if requested
                if plotPermutedData && ~isempty(methodAccuraciesPermuted)
                    meanPermuted = mean(methodAccuraciesPermuted, 3);
                    stdPermuted = std(methodAccuraciesPermuted, [], 3);
                    
                    % Plot mean permuted as dashed line
                    plot(methodDims, meanPermuted, '--', 'Color', color, ...
                        'LineWidth', 1, 'DisplayName', [upper(methodName) ' (permuted)']);
                    
                    % Add error bars for permuted data
                    errorbar(methodDims, meanPermuted, stdPermuted, 'Color', color, ...
                        'LineStyle', 'none', 'LineWidth', 1);
                end
            end
        end
        
        % Customize subplot
        xlabel('Number of Dimensions');
        ylabel('Accuracy');
        title(sprintf('%s (%s)', areaName, condition), 'Interpreter', 'none');
        legend('Location', 'best', 'Interpreter', 'none');
        grid on;
        xlim([min(dimToTest) - 0.5, max(dimToTest) + 0.5]);
        ylim([0, 1]);
        
        % Set x-axis ticks to match dimensions tested
        set(gca, 'XTick', dimToTest);
    end
    
    % Add overall title
    sgtitle(sprintf('SVM Decoding Accuracy vs Dimensions - %s Condition', upper(condition)), ...
        'FontSize', 16);
    
    % Save plot if requested
    if savePlotFlag
        plotFilename = sprintf('accuracy_vs_dimensions_%s_%s_kernel.png', ...
            condition, kernelFunction);
        plotPath = fullfile(savePath, plotFilename);
        exportgraphics(fig, plotPath, 'Resolution', 300);
        fprintf('Saved plot: %s\n', plotFilename);
    end
end

%% Create summary statistics
fprintf('\n=== Summary Statistics ===\n');

% For each condition
for condIdx = 1:length(transWithinToTest)
    condition = transWithinToTest{condIdx};
    fprintf('\n--- %s Condition ---\n', upper(condition));
    
    conditionData = plotData.conditionData{condIdx};
    
    % For each area
    for areaIdx = 1:length(areas)
        areaName = areas{areaIdx};
        areaData = conditionData.areaData{areaIdx};
        
        fprintf('\n%s:\n', areaName);
        
        if isempty(areaData.methods) || isempty(areaData.accuracies)
            fprintf('  No data available\n');
            continue;
        end
        
        methods = areaData.methods;
        
        % For each method
        for methodIdx = 1:length(methods)
            methodName = methods{methodIdx};
            
            % Find data for this method across dimensions
            methodAccuracies = [];
            methodDims = [];
            
            for dimIdx = 1:length(dimToTest)
                nDim = dimToTest(dimIdx);
                dimMask = areaData.dimensions == nDim;
                
                if any(dimMask) && methodIdx <= size(areaData.accuracies, 2)
                    acc = areaData.accuracies(dimMask, methodIdx);
                    if ~isnan(acc) && acc > 0
                        methodAccuracies = [methodAccuracies; acc];
                        methodDims = [methodDims; nDim];
                    end
                end
            end
            
            if ~isempty(methodAccuracies)
                % Calculate statistics
                [maxAcc, maxIdx] = max(methodAccuracies);
                bestDim = methodDims(maxIdx);
                minAcc = min(methodAccuracies);
                meanAcc = mean(methodAccuracies);
                
                fprintf('  %s: %.3f (mean), %.3f (max at %dD), %.3f (min)\n', ...
                    upper(methodName), meanAcc, maxAcc, bestDim, minAcc);
            else
                fprintf('  %s: No valid data\n', upper(methodName));
            end
        end
    end
end

%% Save summary data
fprintf('\n=== Saving Summary Data ===\n');

summaryFilename = sprintf('accuracy_vs_dimensions_summary_%s_kernel.mat', kernelFunction);
summaryPath = fullfile(savePath, summaryFilename);

save(summaryPath, 'plotData', 'allCombinedResults', '-v7.3');
fprintf('Summary data saved to: %s\n', summaryFilename);

fprintf('\nAnalysis complete!\n');
