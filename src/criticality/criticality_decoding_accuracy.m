%%
% Criticality Decoding Accuracy Script
% Combines criticality measurements (d2) with decoding accuracy analysis
% Uses sliding window analysis on naturalistic data
% Options for PCA or UMAP dimensionality reduction
% Tests correlation between d2 and decoding accuracy

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

% collectStart = results.naturalistic.collectStart;
% collectFor = results.naturalistic.collectFor;

%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = collectStart; % 0 * 60 * 60; % seconds
opts.collectFor = collectFor; % 45 * 60; % seconds
opts.minFiringRate = .05;

paths = get_paths;

% Criticality parameter ranges for reference
tauRange = [1.2 2.5];
alphaRange = [1.5 2.2];
paramSDRange = [1.3 1.7];

% Monitor setup
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

%% ==============================================     Analysis Parameters     ==============================================

% Dimensionality reduction options
dimReductionMethod = 'pca';  % 'pca' or 'umap' (UMAP requires MATLAB toolbox)
nDim = 6;                    % Number of dimensions to use

% SVM parameters
kernelFunction = 'polynomial';
nPermutations = 1;
svmBinSize = 0.1;        % Bin size for SVM analysis (can differ from criticality bin size)
opts.frameSize = svmBinSize;

% Behavior analysis parameters
transOrWithin = 'all';   % 'trans', 'within', or 'all'
shiftSec = 0;            % Time shift between neural and behavior data


%% ==============================================     Data Loading     ==============================================

%% Naturalistic data
getDataType = 'spikes';
opts.firingRateCheckTime = 5 * 60;
get_standard_data

[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);

areas = {'M23', 'M56', 'DS', 'VS'};
idListNat = {idM23, idM56, idDS, idVS};

% Behavior labels and colors
colors = colors_for_behaviors(codes);
bhvLabels = {'investigate_1', 'investigate_2', 'investigate_3', ...
    'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
    'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
    'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};

% Initialize all area-wise cells
[decodingAccuracyNat, svmModels, predictedLabels, realLabels, corrMatNat_d2Decoding]...
    = deal(cell(1, length(areas)));


%% Choose which areas to fit (can fill in blank arrays in decodingAccuracyNat
areasToTest = 4;


%% ==============================================     Train SVM Models     ==============================================

fprintf('\n=== Training SVM Models on All Data ===\n');

% Train SVM models and predict behavior labels for all data
for a = areasToTest
    fprintf('Training SVM model for area %s...\n', areas{a});
    
    aID = idListNat{a};
    
    % Bin data for SVM analysis
    aDataMatSVM = dataMat(:, aID);
    
    % Prepare behavior labels for SVM training
    if strcmp(transOrWithin, 'trans')
        % Find transitions
        preInd = find(diff(bhvID) ~= 0);
        svmID = bhvID(preInd + 1);  % behavior ID being transitioned into
        svmInd = preInd + 1;
    elseif strcmp(transOrWithin, 'within')
        % Find within-behavior periods
        transIndLog = zeros(length(bhvID), 1);
        preInd = find(diff(bhvID) ~= 0);
        transIndLog(preInd) = 1;
        svmInd = find(~transIndLog);
        svmID = bhvID(svmInd);
    elseif strcmp(transOrWithin, 'all')
        % Use all behavior labels
        svmInd = 1:length(bhvID);
        svmID = bhvID;
    end
    
    % Only proceed if we have enough data points
    if length(unique(svmID)) > 1 && length(svmID) > 10
fprintf('Using first %d %s components...\n', nDim, dimReductionMethod);        
% Prepare data for dimensionality reduction
        if strcmp(dimReductionMethod, 'pca')    
            % PCA dimensionality reduction
            [coeff, score, ~, ~, explained] = pca(zscore(aDataMatSVM));
            projData = score(:, 1:nDim);
        elseif strcmp(dimReductionMethod, 'umap')
            % UMAP dimensionality reduction
            try
                [reduction, umap, clusterIdentifiers, extras] = run_umap(aDataMatSVM, 'n_components', nDim);
                projData = reduction;
            catch
                warning('UMAP not available, falling back to PCA');
                [coeff, score, ~, ~, explained] = pca(zscore(aDataMatSVM));
                projData = score(:, 1:nDim);
            end
        end
        
        % Get corresponding projection data
        svmProj = projData(svmInd, :);
        
        % Train SVM model on all data
        try
            t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);
            svmModels{a} = fitcecoc(svmProj, svmID, 'Learners', t);
            fprintf('SVM model trained for area %s\n', areas{a});
            
            % Predict behavior labels for all data
            allProjData = projData;  % Use all projected data
            allPredictedLabels = predict(svmModels{a}, allProjData);
            
            % Store predictions and real labels
            predictedLabels{a} = allPredictedLabels;
            realLabels{a} = svmID;
            
            fprintf('Predicted behavior labels for area %s\n', areas{a});
        catch e
            fprintf('SVM training failed for area %s: %s\n', areas{a}, e.message);
            svmModels{a} = [];
            predictedLabels{a} = [];
            realLabels{a} = [];
        end
    else
        fprintf('Insufficient data for SVM training in area %s\n', areas{a});
        svmModels{a} = [];
        predictedLabels{a} = [];
        realLabels{a} = [];
    end
end

%% ==============================================     Calculate Decoding Accuracy for Each Window     ==============================================

fprintf('\n=== Calculating Decoding Accuracy for Each Criticality Window ===\n');


for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    
    if isempty(svmModels{a}) || isempty(predictedLabels{a})
        fprintf('No SVM model or predictions available for area %s, skipping\n', areas{a});
        decodingAccuracyNat{a} = nan(size(d2Nat{a}));
        continue;
    end
    
    % Get valid d2 time points
    validD2Idx = ~isnan(d2Nat{a});
    validD2Times = startSNat{a}(validD2Idx);
    
    % Initialize decoding accuracy array
    decodingAccuracyNat{a} = nan(size(d2Nat{a}));
    
    % Process each criticality window
    for i = 1:length(validD2Times)
        centerTime = validD2Times(i);
        
        % Calculate window boundaries (same as used for criticality)
        windowStartTime = centerTime - criticalityWindowSize/2;
        windowEndTime = centerTime + criticalityWindowSize/2;
        
        % Convert to frame indices for the predicted labels
        % Note: predictedLabels{a} corresponds to the same time points as bhvID
        startFrame = max(1, round(windowStartTime / svmBinSize));
        endFrame = min(length(predictedLabels{a}), round(windowEndTime / svmBinSize));
        
        % Get predicted and real labels for this window
        windowPredictedLabels = predictedLabels{a}(startFrame:endFrame);
        windowRealLabels = realLabels{a}(startFrame:endFrame);
        
        % Filter out indices with real label values of -1
        validIndices = windowRealLabels ~= -1;
        windowPredictedLabels = windowPredictedLabels(validIndices);
        windowRealLabels = windowRealLabels(validIndices);
        
        % Calculate accuracy for this window
        if length(windowPredictedLabels) > 5
            accuracy = sum(windowPredictedLabels == windowRealLabels) / length(windowPredictedLabels);
            decodingAccuracyNat{a}(i) = accuracy;
        else
            decodingAccuracyNat{a}(i) = nan;
        end
    end
    fprintf('Area %s completed\n', areas{a});
end

%% ==============================================     Correlation Analysis     ==============================================

% Calculate correlations between d2 and decoding accuracy
fprintf('\n=== Correlation Analysis: d2 and Decoding Accuracy ===\n');
corrResults = struct();

for a = areasToTest
    % Get minimum length for d2 and decoding accuracy
    minLenNat = min(length(d2Nat{a}), length(decodingAccuracyNat{a}));
    
    % Naturalistic data
    Xnat = [d2Nat{a}(1:minLenNat); decodingAccuracyNat{a}(1:minLenNat)]';
    validIdx = ~any(isnan(Xnat), 2);
    
    if sum(validIdx) > 10
        corrMatNat_d2Decoding{a} = corrcoef(Xnat(validIdx, :));
        corrResults(a).area = areas{a};
        corrResults(a).correlation = corrMatNat_d2Decoding{a}(1, 2);
        [rho, pval] = corr(Xnat(validIdx, 1), Xnat(validIdx, 2), 'type', 'Pearson');
        corrResults(a).p_value = pval;
        corrResults(a).n_valid_points = sum(validIdx);
        
        fprintf('Area %s: Correlation = %.3f, p = %.3f, n = %d\n', ...
            areas{a}, corrResults(a).correlation, corrResults(a).p_value, corrResults(a).n_valid_points);
    else
        corrMatNat_d2Decoding{a} = nan(2, 2);
        corrResults(a).area = areas{a};
        corrResults(a).correlation = nan;
        corrResults(a).p_value = nan;
        corrResults(a).n_valid_points = 0;
        
        fprintf('Area %s: Insufficient data points\n', areas{a});
    end
end

%% ==============================================     Plotting Results     ==============================================

% Create comparison plots for each area
for a = areasToTest
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
    xlim([opts.collectStart/60 (opts.collectStart+opts.collectFor)/60]);
    
    % Plot decoding accuracy
    axes(ha(2));
    hold on;
    validIdx = ~isnan(decodingAccuracyNat{a});
    plot(startSNat{a}(validIdx)/60, decodingAccuracyNat{a}(validIdx), '-', 'Color', 'r', 'LineWidth', 2, 'MarkerSize', 4);
    ylabel('Decoding Accuracy', 'FontSize', 14);
    title(sprintf('%s - Decoding Accuracy (%s, %s)', areas{a}, upper(dimReductionMethod), transOrWithin), 'FontSize', 14);
    grid on;
    xlabel('Minutes', 'FontSize', 14);
    set(gca, 'YTickLabelMode', 'auto', 'FontSize', 14);
    set(gca, 'XTickLabelMode', 'auto');
    xlim([opts.collectStart/60 (opts.collectStart+opts.collectFor)/60]);
    ylim([0 1]);
    
    sgtitle(sprintf('Criticality and Decoding Accuracy - %s', areas{a}), 'FontSize', 14);
exportgraphics(gcf, fullfile(paths.dropPath, sprintf('criticality/criticality_decoding_%s_%s_%s.png', areas{a}, dimReductionMethod, transOrWithin)), 'Resolution', 300);
end

% Plot correlation matrices
figure(300); clf;
set(gcf, 'Position', monitorTwo);
for a = areasToTest
    subplot(1, length(areas), a);
    imagesc(corrMatNat_d2Decoding{a});
    colorbar;
    title(sprintf('%s (d2/Decoding)', areas{a}));
    xticks(1:2); yticks(1:2);
    xticklabels({'d2','Decoding'}); yticklabels({'d2','Decoding'});
    axis square;
    caxis([-1 1]);
end
sgtitle('Correlation Matrices: d2 and Decoding Accuracy');

% Plot correlation summary
figure(301); clf;
set(gcf, 'Position', monitorOne);
corrVals = [corrResults.correlation];
pVals = [corrResults.p_value];
areasPlot = {corrResults.area};

bar(corrVals);
set(gca, 'XTickLabel', areasPlot);
ylabel('Correlation Coefficient');
title('Correlation between d2 and Decoding Accuracy');
grid on;

% Add significance markers
hold on;
for i = 1:length(pVals)
    if pVals(i) < 0.05
        plot(i, corrVals(i) + 0.05, '*', 'Color', 'black', 'MarkerSize', 60);
    end
end
exportgraphics(gcf, fullfile(paths.dropPath, sprintf('criticality/correlation_summary_%s_%s.png', dimReductionMethod, transOrWithin)), 'Resolution', 300);


%% ==============================================     Save Results     ==============================================

% Save all results
results = struct();
results.areas = areas;
results.dimReductionMethod = dimReductionMethod;
results.nDim = nDim;
results.svmModels = svmModels;
results.svmBinSize = svmBinSize;
results.predictedLabels = predictedLabels;
results.realLabels = realLabels;

% Naturalistic data results
results.naturalistic.d2 = d2Nat;
results.naturalistic.decodingAccuracy = decodingAccuracyNat;
results.naturalistic.startS = startSNat;
results.naturalistic.criticalityBinSize = criticalityBinSize;
results.naturalistic.criticalityWindowSize = criticalityWindowSize;
results.naturalistic.corrMat = corrMatNat_d2Decoding;
results.naturalistic.corrResults = corrResults;

% Analysis parameters
results.params.kernelFunction = kernelFunction;
results.params.transOrWithin = transOrWithin;
results.params.shiftSec = shiftSec;
results.params.svmBinSize = svmBinSize;
results.params.stepSize = stepSize;

% Save to file
save(fullfile(paths.dropPath, sprintf('criticality_decoding_results_%s_%s.mat', dimReductionMethod, transOrWithin)), 'results');

fprintf('\nAnalysis complete! Results saved to criticality_decoding_results_%s_%s.mat\n', dimReductionMethod, transOrWithin); 