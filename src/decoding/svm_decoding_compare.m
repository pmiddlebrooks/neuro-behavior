%%                     Compare SVM decoding accuracy across different methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script compares SVM decoding performance across PCA, UMAP, PSID, and ICG
% dimensionality reduction methods using the same neural data and behavior labels

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 45 * 60; % seconds
opts.frameSize = .1;

% Get kinematics data for PSID
kinBinSize = .1;
getDataType = 'kinematics';
get_standard_data
startFrame = 1 + opts.collectStart / kinBinSize;
endFrame = startFrame - 1 + (opts.collectFor / kinBinSize);
kinData = kinData(startFrame:endFrame,:);

% Get neural data
getDataType = 'spikes';
get_standard_data
% Curate behavior labels
[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);


% Get colors for behaviors
colors = colors_for_behaviors(codes);

% Monitor setup for plotting
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

% Behavior labels
bhvLabels = {'investigate_1', 'investigate_2', 'investigate_3', ...
    'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
    'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
    'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};

%% =============================================================================
% --------    ANALYSIS PARAMETERS
% =============================================================================

% Dimensionality for all methods
nDim = 4;
forDim = nDim;

% Select which data to run analyses on
selectFrom = 'M56';
% selectFrom = 'DS';
% selectFrom = 'Both';
% selectFrom = 'VS';
% selectFrom = 'All';

switch selectFrom
    case 'M56'
        idSelect = idM56;
        figHFull = 260;
        figHModel = 270;
        figHFullModel = 280;
    case 'DS'
        idSelect = idDS;
        figHFull = 261;
        figHModel = 271;
        figHFullModel = 281;
    case 'Both'
        idSelect = [idM56, idDS];
        figHFull = 262;
        figHModel = 272;
        figHFullModel = 282;
    case 'VS'
        idSelect = idVS;
        figHFull = 263;
        figHModel = 273;
        figHFullModel = 283;
    case 'All'
        idSelect = cell2mat(idAll);
        figHFull = 264;
        figHModel = 274;
        figHFullModel = 284;
end

% Analysis type
transOrWithin = 'all';  % 'all', 'trans', 'within', 'transVsWithin'
% transOrWithin = 'trans';
% transOrWithin = 'within';

% Permutation testing
nShuffles = 3;  % Number of permutation tests

% Plotting options
plotFullMap = 1;
plotModelData = 1;
plotResults = 1;

% Figure properties
allFontSize = 12;

%% =============================================================================
% --------    GENERATE LATENTS FOR EACH METHOD
% =============================================================================

fprintf('Generating latents for each method...\n');

% Initialize storage for latents
latents = struct();

%% PCA
fprintf('Running PCA...\n');
[coeff, score, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));
latents.pca = score(:, 1:nDim);
fprintf('PCA explained variance: %.2f%%\n', sum(explained(1:nDim)));

%% UMAP
fprintf('Running UMAP...\n');
% Change to UMAP directory
if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
    cd '/Users/paulmiddlebrooks/Projects/toolboxes/umapFileExchange (4.4)/umap/'
else
    cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
end

% UMAP parameters
min_dist = 0.02;
spread = 1.3;
n_neighbors = 10;

[latents.umap, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', nDim, ...
    'randomize', true, 'verbose', 'none', 'min_dist', min_dist, ...
    'spread', spread, 'n_neighbors', n_neighbors);

%% PSID
fprintf('Running PSID...\n');
% Return to original directory
cd('E:/Projects/neuro-behavior/src/decoding');

% Prepare data for PSID
y = zscore(dataMat(:, idSelect));
z = zscore(kinData);
nx = nDim * 2;
n1 = nDim;
i = 10;

idSys = PSID(y, z, nx, n1, i);

% Predict behavior using the learned model
[zPred, yPred, xPred] = PSIDPredict(idSys, y);
latents.psid = xPred(1:nDim, :)';

%% ICG
fprintf('Running ICG...\n');
dataICG = dataMat(:, idSelect);
[activityICG, outPairID] = ICG(dataICG');

% Take the first nDim groups of most correlated neurons
latents.icg = activityICG{4}(1:nDim, :)';

fprintf('All methods completed.\n');

%% =============================================================================
% --------    PREPARE BEHAVIOR LABELS
% =============================================================================

% Shift behavior labels if needed
shiftSec = 0;
shiftFrame = 0;
if shiftSec > 0
    shiftFrame = ceil(shiftSec / opts.frameSize);
    bhvID = double(bhvID(1+shiftFrame:end));
end

% Adjust latents for shift
if shiftFrame > 0
    fieldNames = fieldnames(latents);
    for i = 1:length(fieldNames)
        latents.(fieldNames{i}) = latents.(fieldNames{i})(1:end-shiftFrame, :);
    end
end

% Find behavior transitions
preInd = find(diff(bhvID) ~= 0);
transEnd = unique([preInd; preInd+1; preInd-1]);

% Prepare behavior labels based on analysis type
switch transOrWithin
    case 'all'
        svmInd = 1:length(bhvID);
        svmID = bhvID;
        transWithinLabel = 'all';
    case 'trans'
        svmID = bhvID(transEnd + 1);
        svmInd = transEnd;
        transWithinLabel = 'transitions';
    case 'within'
        svmInd = setdiff(1:length(bhvID), transEnd);
        svmID = bhvID(svmInd);
        transWithinLabel = 'within-behavior';
    case 'transVsWithin'
        svmID = bhvID;
        svmID(diff(bhvID) ~= 0) = 9;
        svmID(diff(bhvID) == 0) = 12;
        svmID(end) = [];
        svmInd = 1:length(svmID);
        transWithinLabel = 'trans-vs-within';
end

% Remove invalid behavior labels
deleteInd = svmID == -1;
svmID(deleteInd) = [];
svmInd(deleteInd) = [];

% Get behavior codes and names for modeling
if strcmp(transOrWithin, 'transVsWithin')
    bhv2ModelCodes = [9 12];
    bhv2ModelNames = {'transitions', 'within'};
else
    bhv2ModelCodes = unique(svmID);
    bhv2ModelNames = behaviors(bhv2ModelCodes+2);
    bhv2ModelColors = colors(ismember(codes, bhv2ModelCodes), :);
end

fprintf('Modeling %s behaviors: %s\n', transWithinLabel, strjoin(bhv2ModelNames, ', '));

%% =============================================================================
% --------    PLOT FULL DATA FOR EACH METHOD
% =============================================================================

if plotFullMap
    fprintf('Plotting full data for each method...\n');
    
    fieldNames = fieldnames(latents);
    for i = 1:length(fieldNames)
        methodName = fieldNames{i};
        latentData = latents.(methodName);
        
        % Plot full time of all behaviors
        colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:});
        
        figure(figHFull + i);
        clf; hold on;
        
        if nDim > 2
            scatter3(latentData(:, 1), latentData(:, 2), latentData(:, 3), 20, colorsForPlot, 'filled', 'MarkerFaceAlpha', 0.6);
        else
            scatter(latentData(:, 1), latentData(:, 2), 20, colorsForPlot, 'filled', 'MarkerFaceAlpha', 0.6);
        end
        
        title(sprintf('%s %s %dD - All Behaviors (bin=%.2f)', selectFrom, upper(methodName), nDim, opts.frameSize));
        xlabel('D1'); ylabel('D2');
        if nDim > 2
            zlabel('D3');
        end
        grid on;
        
        % Add legend for behaviors
        uniqueBhv = unique(bhvID);
        uniqueBhv = uniqueBhv(uniqueBhv >= 0);
        for j = 1:length(uniqueBhv)
            bhvIdx = uniqueBhv(j);
            if bhvIdx < length(behaviors)
                scatter(NaN, NaN, 20, colors(bhvIdx+2,:), 'filled', 'DisplayName', behaviors{bhvIdx+2});
            end
        end
        legend('Location', 'best');
    end
end

%% =============================================================================
% --------    PLOT MODELING DATA FOR EACH METHOD
% =============================================================================

if plotModelData
    fprintf('Plotting modeling data for each method...\n');
    
    fieldNames = fieldnames(latents);
    for i = 1:length(fieldNames)
        methodName = fieldNames{i};
        latentData = latents.(methodName);
        
        % Get colors for modeled behaviors
        colorsForPlot = arrayfun(@(x) colors(x,:), svmID + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:});
        
        figure(figHModel + i);
        clf; hold on;
        
        if nDim > 2
            scatter3(latentData(svmInd, 1), latentData(svmInd, 2), latentData(svmInd, 3), 40, colorsForPlot, 'filled');
        else
            scatter(latentData(svmInd, 1), latentData(svmInd, 2), 40, colorsForPlot, 'filled');
        end
        
        title(sprintf('%s %s %dD - %s (bin=%.2f)', selectFrom, upper(methodName), nDim, transWithinLabel, opts.frameSize));
        xlabel('D1'); ylabel('D2');
        if nDim > 2
            zlabel('D3');
        end
        grid on;
        
        % Add legend for modeled behaviors
        for j = 1:length(bhv2ModelCodes)
            bhvIdx = bhv2ModelCodes(j);
            if bhvIdx < length(behaviors)
                scatter(NaN, NaN, 40, colors(bhvIdx+2,:), 'filled', 'DisplayName', behaviors{bhvIdx+2});
            end
        end
        legend('Location', 'best');
    end
end

%% =============================================================================
% --------    RUN SVM DECODING FOR EACH METHOD
% =============================================================================

fprintf('Running SVM decoding for each method...\n');

% Initialize storage for results
methods = fieldnames(latents);
nMethods = length(methods);
accuracy = zeros(nMethods, 1);
accuracyPermuted = zeros(nMethods, nShuffles);

% SVM parameters
kernelFunction = 'polynomial';

% Cross-validation setup
cv = cvpartition(svmID, 'HoldOut', 0.2);

for m = 1:nMethods
    methodName = methods{m};
    fprintf('\n--- %s ---\n', upper(methodName));
    
    % Get latent data for this method
    latentData = latents.(methodName);
    
    % Prepare data for SVM
    svmProj = latentData(svmInd, :);
    trainData = svmProj(training(cv), :);
    testData = svmProj(test(cv), :);
    trainLabels = svmID(training(cv));
    testLabels = svmID(test(cv));
    
    % Train SVM model
    tic;
    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);
    svmModel = fitcecoc(trainData, trainLabels, 'Learners', t);
    
    % Test model
    predictedLabels = predict(svmModel, testData);
    accuracy(m) = sum(predictedLabels == testLabels) / length(testLabels);
    
    fprintf('Real data accuracy: %.4f\n', accuracy(m));
    fprintf('Model training time: %.2f seconds\n', toc);
    
    % Permutation tests
    fprintf('Running %d permutation tests...\n', nShuffles);
    tic;
    
    for s = 1:nShuffles
        % Circular shuffle of training labels
        rng('shuffle');
        randShift = randi([1, length(trainLabels)]);
        shuffledLabels = circshift(trainLabels, randShift);
        
        % Train model on shuffled data
        svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);
        
        % Test on real data
        predictedLabelsPermuted = predict(svmModelPermuted, testData);
        accuracyPermuted(m, s) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
        
        fprintf('  Permutation %d: %.4f\n', s, accuracyPermuted(m, s));
    end
    
    fprintf('Permutation testing time: %.2f seconds\n', toc);
    fprintf('Mean permuted accuracy: %.4f\n', mean(accuracyPermuted(m, :)));
end

%% =============================================================================
% --------    PLOT RESULTS COMPARISON
% =============================================================================

if plotResults
    fprintf('Plotting results comparison...\n');
    
    % Create comparison figure
    fig = figure(100);
    set(fig, 'Position', [monitorOne(1), monitorOne(2), monitorOne(3)/2, monitorOne(4)/2]);
    clf; hold on;
    
    % Bar plot of accuracies
    x = 1:nMethods;
    barWidth = 0.35;
    
    % Real data bars
    b1 = bar(x - barWidth/2, accuracy, barWidth, 'FaceColor', 'blue', 'DisplayName', 'Real Data');
    
    % Permuted data bars
    b2 = bar(x + barWidth/2, mean(accuracyPermuted, 2), barWidth, 'FaceColor', 'red', 'DisplayName', 'Permuted (mean)');
    
    % Add error bars for permuted data
    errorbar(x + barWidth/2, mean(accuracyPermuted, 2), std(accuracyPermuted, [], 2), 'k.', 'LineWidth', 1.5, 'DisplayName', 'Permuted (std)');
    
    % Customize plot
    set(gca, 'XTick', x, 'XTickLabel', upper(methods));
    ylabel('Accuracy');
    title(sprintf('SVM Decoding Accuracy Comparison - %s (%s)', selectFrom, transWithinLabel));
    legend('Location', 'best');
    grid on;
    ylim([0, 1]);
    
    % Add significance indicators
    for m = 1:nMethods
        % Simple significance test: if real accuracy > 95th percentile of permuted
        permSorted = sort(accuracyPermuted(m, :));
        threshold95 = permSorted(ceil(0.95 * nShuffles));
        
        if accuracy(m) > threshold95
            text(m, accuracy(m) + 0.02, '*', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
        end
    end
    
    % Add accuracy values as text
    for m = 1:nMethods
        text(m - barWidth/2, accuracy(m) + 0.01, sprintf('%.3f', accuracy(m)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
        text(m + barWidth/2, mean(accuracyPermuted(m, :)) + 0.01, sprintf('%.3f', mean(accuracyPermuted(m, :))), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    
    % Summary statistics
    fprintf('\n=== SUMMARY ===\n');
    fprintf('Method\t\tReal\t\tPermuted (mean ± std)\t\tDifference\n');
    fprintf('-----\t\t----\t\t----------------------\t\t----------\n');
    for m = 1:nMethods
        diff = accuracy(m) - mean(accuracyPermuted(m, :));
        fprintf('%s\t\t%.3f\t\t%.3f ± %.3f\t\t%.3f\n', ...
            upper(methods{m}), accuracy(m), mean(accuracyPermuted(m, :)), ...
            std(accuracyPermuted(m, :)), diff);
    end
    
    % Find best method
    [bestAcc, bestIdx] = max(accuracy);
    fprintf('\nBest performing method: %s (accuracy: %.3f)\n', upper(methods{bestIdx}), bestAcc);
end

fprintf('\nAnalysis complete!\n');
