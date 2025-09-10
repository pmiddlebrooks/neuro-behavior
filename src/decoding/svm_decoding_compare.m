%%                     Compare SVM decoding accuracy across different methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script compares SVM decoding performance across PCA, UMAP, PSID, and ICG
% dimensionality reduction methods using the same neural data and behavior labels


%% Specify main parameters

% Dimensionality for all methods
nDim = 6;

% Analysis type
transOrWithin = 'all';  % 'trans',transPost 'within', 'all'

% Frame/bin size
frameSize = .1;

% Create full path
savePath = fullfile(paths.dropPath, 'decoding');
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 45 * 60; % seconds
opts.frameSize = frameSize;

% Get kinematics data for PSID
getDataType = 'kinematics';
get_standard_data




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

% Define brain areas to test
areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 2:4;  % Test M56, DS, VS


% Analysis control flags
runAllMethods = false;  % Set to true to run all methods (time intensive)
runAdditionalPSID = true;  % Set to true to run psidKin and psidBhv remaining dimensions
loadExistingResults = true;  % Set to true to load and append to existing results

% Permutation testing
nShuffles = 2;  % Number of permutation tests

% Plotting options
plotFullMap = 0;
plotModelData = 1;
plotResults = 1;
savePlotFlag = 1;  % Save plots as PNG files

% Figure properties
allFontSize = 12;

% Initialize storage for all areas
allResults = struct();
allResults.areas = areas;
allResults.areasToTest = areasToTest;
allResults.parameters = struct();
allResults.parameters.nDim = nDim;
allResults.parameters.frameSize = opts.frameSize;
allResults.parameters.nShuffles = nShuffles;
allResults.parameters.kernelFunction = 'polynomial';
allResults.parameters.collectStart = opts.collectStart;
allResults.parameters.collectFor = opts.collectFor;
allResults.parameters.minActTime = opts.minActTime;
allResults.parameters.transOrWithin = transOrWithin;

% Only initialize empty cells if we're not doing additional PSID analysis
% (to avoid overriding existing loaded data)
if ~(runAdditionalPSID && ~runAllMethods)
    % Initialize results storage for each area
    allResults.latents = cell(1, length(areas));
    allResults.accuracy = cell(1, length(areas));
    allResults.accuracyPermuted = cell(1, length(areas));
    allResults.methods = cell(1, length(areas));
    allResults.bhv2ModelCodes = cell(1, length(areas));
    allResults.bhv2ModelNames = cell(1, length(areas));
    allResults.svmInd = cell(1, length(areas));
    allResults.svmID = cell(1, length(areas));
    allResults.idSelect = cell(1, length(areas));
    allResults.bhvMapping = cell(1, length(areas));
    allResults.svmModels = cell(1, length(areas));
    allResults.allPredictions = cell(1, length(areas));
    allResults.allPredictionIndices = cell(1, length(areas));
end

%% =============================================================================
% --------    LOAD EXISTING RESULTS (IF REQUESTED)
% =============================================================================

% Check if we should load existing results
if loadExistingResults
    fprintf('\n=== Checking for Existing SVM Decoding Results ===\n');

    % Look for existing results file
    existingFilename = sprintf('svm_decoding_compare_multi_area_%s_nDim%d_bin%.2f_nShuffles%d.mat', ...
        transOrWithin, nDim, opts.frameSize, nShuffles);
    existingPath = fullfile(paths.dropPath, 'decoding', existingFilename);

    if exist(existingPath, 'file')
        fprintf('Loading existing SVM decoding results from: %s\n', existingFilename);
        load(existingPath, 'allResults');

        % Display what methods are already analyzed
        fprintf('Existing methods analyzed:\n');
        for a = areasToTest
            if ~isempty(allResults.methods{a})
                fprintf('  Area %s: %s\n', areas{a}, strjoin(allResults.methods{a}, ', '));
            end
        end

        fprintf('Analysis flags: runAllMethods=%s, runAdditionalPSID=%s\n', ...
            string(runAllMethods), string(runAdditionalPSID));
    else
        fprintf('No existing SVM decoding results found. Will create new analysis.\n');
    end
end

%% =============================================================================
% --------    LOOP THROUGH BRAIN AREAS
% =============================================================================

fprintf('\n=== Processing Multiple Brain Areas ===\n');

for areaIdx = areasToTest
    areaName = areas{areaIdx};
    fprintf('\n=======================   --- Processing Area: %s ---  ==================\n', areaName);

    % Check if we're doing additional PSID analysis and if methods already exist
    if runAdditionalPSID && ~runAllMethods && ~isempty(allResults.latents{areaIdx})
        % existingMethods = fieldnames(allResults.latents{areaIdx});
        % if ismember('psidKin_nonBhv', existingMethods) && ismember('psidBhv_nonBhv', existingMethods)
        %     fprintf('Area %s already has additional PSID methods. Skipping.\n', areaName);
        %     continue;
        % end
    end

    % Select data for this area
    switch areaName
        case 'M23'
            idSelect = idM23;
        case 'M56'
            idSelect = idM56;
        case 'DS'
            idSelect = idDS;
        case 'VS'
            idSelect = idVS;
        otherwise
            fprintf('Unknown area: %s, skipping\n', areaName);
    end

    % Store the selected IDs
    allResults.idSelect{areaIdx} = idSelect;

    fprintf('Selected %d neurons for area %s\n', length(idSelect), areaName);

    %% =============================================================================
    % --------    GENERATE LATENTS FOR EACH METHOD
    % =============================================================================

    % Determine which methods to run
    methodsToRun = {};
    if runAllMethods
        methodsToRun = {'pca', 'umap', 'psidKin', 'psidKin_nonBhv', 'psidBhv', 'psidBhv_nonBhv', 'icg'};
        fprintf('Running all methods: %s\n', strjoin(methodsToRun, ', '));
    elseif runAdditionalPSID
        methodsToRun = {'psidKin_nonBhv', 'psidBhv_nonBhv'};
        fprintf('Running additional PSID methods: %s\n', strjoin(methodsToRun, ', '));
    else
        fprintf('No methods specified to run. Set runAllMethods=true or runAdditionalPSID=true\n');
        continue;
    end

    fprintf('Generating latents for specified methods...\n');

    % Initialize storage for latents
    if isempty(allResults.latents{areaIdx})
        latents = struct();
    else
        latents = allResults.latents{areaIdx};
    end

    %% PCA
    if ismember('pca', methodsToRun)
        fprintf('Running PCA...\n');
        [coeff, score, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));
        latents.pca = score(:, 1:nDim);
        fprintf('PCA explained variance: %.2f%%\n', sum(explained(1:nDim)));
    end

    %% UMAP
    if ismember('umap', methodsToRun)
        fprintf('Running UMAP...\n');
        % Change to UMAP directory
        cd(fullfile(paths.homePath, '/toolboxes/umapFileExchange (4.4)/umap/'))

        % UMAP parameters
        min_dist = 0.1;
        spread = 1.3;
        n_neighbors = 10;

        [latents.umap, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', nDim, ...
            'randomize', true, 'verbose', 'none', 'min_dist', min_dist, ...
            'spread', spread, 'n_neighbors', n_neighbors);
    end

    %% PSID with Kinematics (psidKin)
    if ismember('psidKin', methodsToRun) || ismember('psidKin_nonBhv', methodsToRun)
        fprintf('Running PSID with kinematics...\n');
        % Return to original directory
        cd(fullfile(paths.homePath, 'neuro-behavior/src/decoding'));

        % Prepare data for PSID with kinematics
        y = zscore(dataMat(:, idSelect));
        z = zscore(kinData);
        nx = nDim * 2;
        n1 = nDim;
        i = 10;

        idSys = PSID(y, z, nx, n1, i);

        % Predict behavior using the learned model
        [zPred, yPred, xPred] = PSIDPredict(idSys, y);

        if ismember('psidKin', methodsToRun)
            latents.psidKin = xPred(:, 1:nDim);
        end
        if ismember('psidKin_nonBhv', methodsToRun)
            % Use remaining dimensions (beyond nDim)
            remainingDims = (nDim+1):size(xPred, 2);
            if ~isempty(remainingDims)
                latents.psidKin_nonBhv = xPred(:, remainingDims);
                fprintf('PSID Kin remaining dimensions: %d\n', length(remainingDims));
            else
                fprintf('Warning: No remaining dimensions available for PSID Kin\n');
                latents.psidKin_nonBhv = [];
            end
        end
    end


    %% PSID with Behavior Labels (psidBhv)
    if ismember('psidBhv', methodsToRun) || ismember('psidBhv_nonBhv', methodsToRun)
        fprintf('Running PSID with behavior labels...\n');

        % Create one-hot encoded behavior labels
        uniqueBhv = unique(bhvID);
        uniqueBhv = uniqueBhv(uniqueBhv >= 0);  % Remove invalid labels (-1)
        nBehaviors = length(uniqueBhv);

        % Create one-hot encoding
        bhvOneHot = zeros(length(bhvID), nBehaviors);
        for b = 1:nBehaviors
            bhvOneHot(:, b) = (bhvID == uniqueBhv(b));
        end

        % Store behavior mapping for reference
        bhvMapping = struct();
        bhvMapping.uniqueBhv = uniqueBhv;
        bhvMapping.nBehaviors = nBehaviors;
        bhvMapping.behaviorNames = behaviors(uniqueBhv + 2);  % +2 because behaviors array starts at index 1

        % Prepare data for PSID with behavior labels
        y = zscore(dataMat(:, idSelect));
        z = zscore(bhvOneHot);  % Use one-hot encoded behavior labels
        nx = nDim * 2;
        n1 = nDim;
        i = 10;

        idSys = PSID(y, z, nx, n1, i);

        % Predict behavior using the learned model
        [zPred, yPred, xPred] = PSIDPredict(idSys, y);

        if ismember('psidBhv', methodsToRun)
            latents.psidBhv = xPred(:, 1:nDim);

            % Store behavior mapping in results
            allResults.bhvMapping{areaIdx} = bhvMapping;
        end
        if ismember('psidBhv_nonBhv', methodsToRun)
            % Use remaining dimensions (beyond nDim)
            remainingDims = (nDim+1):size(xPred, 2);
            if ~isempty(remainingDims)
                latents.psidBhv_nonBhv = xPred(:, remainingDims);
                fprintf('PSID Bhv remaining dimensions: %d\n', length(remainingDims));
            else
                fprintf('Warning: No remaining dimensions available for PSID Bhv\n');
                latents.psidBhv_nonBhv = [];
            end
        end
    end


    %% ICG
    if ismember('icg', methodsToRun)
        fprintf('Running ICG...\n');
        dataICG = dataMat(:, idSelect);
        [activityICG, outPairID] = ICG(dataICG');

        if nDim == 4
        % Take the first nDim groups of most correlated neurons
        % latents.icg = activityICG{4}(1:nDim, :)';
        elseif nDim == 6
        % Take the first nDim groups of most correlated neurons
        warning('Changed ICG population')
        latents.icg = zscore(activityICG{3}(1:9, :)');
        end
    end

    fprintf('All methods completed for area %s.\n', areaName);

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
    preIdx = find(diff(bhvID) ~= 0);

    % Prepare behavior labels based on analysis type
    switch transOrWithin
        case 'all'
            svmInd = 1:length(bhvID);
            svmID = bhvID;
            transWithinLabel = 'all';
        case 'trans'
            svmID = bhvID(preIdx + 1);
            svmInd = preIdx;
            transWithinLabel = 'transitions: Pre';
        case 'transPost'
            svmID = bhvID(preIdx + 1);
            svmInd = preIdx + 1;
            transWithinLabel = 'transitions: Post';
        case 'within'
            svmInd = setdiff(1:length(bhvID), preIdx);
            svmID = bhvID(svmInd);
            transWithinLabel = 'within-behavior';
    end

    % Remove invalid behavior labels
    deleteInd = svmID == -1;
    svmID(deleteInd) = [];
    svmInd(deleteInd) = [];

    % Get behavior codes and names for modeling
    bhv2ModelCodes = unique(svmID);
    bhv2ModelNames = behaviors(bhv2ModelCodes+2);
    bhv2ModelColors = colors(ismember(codes, bhv2ModelCodes), :);

    fprintf('Modeling %s behaviors: %s\n', transWithinLabel, strjoin(bhv2ModelNames, ', '));

    %% =============================================================================
    % --------    RUN SVM DECODING FOR EACH METHOD
    % =============================================================================

    fprintf('Running SVM decoding for specified methods...\n');

    % Get methods that were actually run
    methods = fieldnames(latents);
    
    % When doing additional PSID analysis, only count new methods
    if runAdditionalPSID && ~runAllMethods && ~isempty(allResults.methods{areaIdx})
        existingMethods = allResults.methods{areaIdx};
        % Filter to only include new methods
        newMethods = {};
        for i = 1:length(methods)
            if ~ismember(methods{i}, existingMethods)
                newMethods{end+1} = methods{i};
            end
        end
        methods = newMethods;
    end
    
    nMethods = length(methods);

    % Initialize or load existing results
    if isempty(allResults.accuracy{areaIdx})
        accuracy = zeros(nMethods, 1);
        accuracyPermuted = zeros(nMethods, nShuffles);
        svmModels = cell(1, nMethods);
        allPredictions = cell(1, nMethods);
        allPredictionIndices = cell(1, nMethods);
        existingMethods = {};
    else
        % Load existing results
        accuracy = allResults.accuracy{areaIdx};
        accuracyPermuted = allResults.accuracyPermuted{areaIdx};
        svmModels = allResults.svmModels{areaIdx};
        allPredictions = allResults.allPredictions{areaIdx};
        allPredictionIndices = allResults.allPredictionIndices{areaIdx};
        existingMethods = allResults.methods{areaIdx};
        
        % Extend arrays for new methods only
        if nMethods > 0  % Only extend if there are new methods
            accuracy = [accuracy; zeros(nMethods, 1)];
            accuracyPermuted = [accuracyPermuted; zeros(nMethods, nShuffles)];
            svmModels = [svmModels, cell(1, nMethods)];
            allPredictions = [allPredictions, cell(1, nMethods)];
            allPredictionIndices = [allPredictionIndices, cell(1, nMethods)];
        end
    end

    % SVM parameters
    kernelFunction = 'polynomial';

    % Cross-validation setup
    cv = cvpartition(svmID, 'HoldOut', 0.2);

    % Only process if there are new methods
    if nMethods > 0
        % Calculate starting index for new methods (after existing methods)
        if isempty(allResults.accuracy{areaIdx})
            startIdx = 1;  % First methods
        else
            startIdx = length(allResults.methods{areaIdx}) + 1;  % After existing methods
        end
        
        for m = 1:nMethods
            methodName = methods{m};
            % Calculate the correct index in the full arrays
            fullIdx = startIdx + m - 1;
            
            fprintf('\n--- %s ---\n', upper(methodName));

            % Get latent data for this method
            latentData = latents.(methodName);
            
            % Check if latent data is valid
            if isempty(latentData)
                fprintf('Warning: No latent data for method %s\n', methodName);
                continue;
            end

            % Prepare data for SVM
            svmProj = latentData(svmInd, :);
            trainData = svmProj(training(cv), :);
            testData = svmProj(test(cv), :);
            trainLabels = svmID(training(cv));
            testLabels = svmID(test(cv));

            % Train SVM model on all relevant data (not just training set)
            tic;
            t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);
            svmModelFull = fitcecoc(svmProj, svmID, 'Learners', t);
            
            % Store the full model at the correct index
            svmModels{fullIdx} = svmModelFull;

            % Generate predictions for all relevant indices
            allPredictions{fullIdx} = predict(svmModelFull, svmProj);
            allPredictionIndices{fullIdx} = svmInd;  % Store the indices that were predicted
            
            % For cross-validation accuracy calculation, use the holdout approach
            svmModelCV = fitcecoc(trainData, trainLabels, 'Learners', t);
            predictedLabels = predict(svmModelCV, testData);
            accuracy(fullIdx) = sum(predictedLabels == testLabels) / length(testLabels);

            fprintf('Real data accuracy: %.4f\n', accuracy(fullIdx));
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
                accuracyPermuted(fullIdx, s) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);

                fprintf('  Permutation %d: %.4f\n', s, accuracyPermuted(fullIdx, s));
            end

            fprintf('Permutation testing time: %.2f seconds\n', toc);
            fprintf('Mean permuted accuracy: %.4f\n', mean(accuracyPermuted(fullIdx, :)));
        end
    else
        fprintf('No new methods to process for area %s\n', areaName);
    end

    % Store results for this area
    allResults.latents{areaIdx} = latents;

    % Update methods list to include new methods
    if isempty(allResults.methods{areaIdx})
        allResults.methods{areaIdx} = methods;
    else
        % Append new methods to existing list
        allResults.methods{areaIdx} = [allResults.methods{areaIdx}; methods'];
    end

    allResults.accuracy{areaIdx} = accuracy;
    allResults.accuracyPermuted{areaIdx} = accuracyPermuted;
    allResults.bhv2ModelCodes{areaIdx} = bhv2ModelCodes;
    allResults.bhv2ModelNames{areaIdx} = bhv2ModelNames;
    allResults.svmInd{areaIdx} = svmInd;
    allResults.svmID{areaIdx} = svmID;
    allResults.svmModels{areaIdx} = svmModels;
    allResults.allPredictions{areaIdx} = allPredictions;
    allResults.allPredictionIndices{areaIdx} = allPredictionIndices;

    fprintf('\nArea %s completed.\n', areaName);
end

% load handel
% sound(y,Fs)

% =============================================================================
% --------    PLOT FULL DATA FOR EACH METHOD
% =============================================================================
for areaIdx = areasToTest
    areaName = areas{areaIdx};

    figHFull = 270;
    if plotFullMap
        fprintf('Plotting full data for each method...\n');

        fieldNames = fieldnames(allResults.latents{areaIdx});
        for i = 1:length(fieldNames)
            methodName = fieldNames{i};
            latentData = allResults.latents{areaIdx}.(methodName);

            % Plot full time of all behaviors
            colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + 2, 'UniformOutput', false);
            colorsForPlot = vertcat(colorsForPlot{:});

            figure(figHFull + i);
            clf; hold on;

            if nDim > 2
                scatter3(latentData(:, 1), latentData(:, 2), latentData(:, 3), 20, colorsForPlot, 'filled', 'MarkerFaceAlpha', 0.6);
                % Set view angle to show all three axes with depth
                view(45, 30);
            else
                scatter(latentData(:, 1), latentData(:, 2), 20, colorsForPlot, 'filled', 'MarkerFaceAlpha', 0.6);
            end

            title(sprintf('%s %s %dD - All Behaviors (bin=%.2f)', areaName, upper(methodName), nDim, opts.frameSize), 'Interpreter','none');
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
            legend('Location', 'best', 'Interpreter', 'none');

            % Save plot if flag is set
            if savePlotFlag
                plotFilename = sprintf('full_data_%s_%s_%dD_bin%.2f.png', ...
                    areaName, methodName, nDim, opts.frameSize);
                plotPath = fullfile(savePath, plotFilename);
                exportgraphics(figure(figHFull + i), plotPath, 'Resolution', 300);
                fprintf('Saved plot: %s\n', plotFilename);
            end
        end
    end

    % =============================================================================
    % --------    PLOT MODELING DATA FOR EACH METHOD
    % =============================================================================
    figHModel = 280;
    if plotModelData
        fprintf('Plotting modeling data for each method...\n');

        fieldNames = fieldnames(allResults.latents{areaIdx});
        for i = 1:length(fieldNames)
            methodName = fieldNames{i};
            latentData = allResults.latents{areaIdx}.(methodName);

            % Get colors for modeled behaviors
            colorsForPlot = arrayfun(@(x) colors(x,:), allResults.svmID{areaIdx} + 2, 'UniformOutput', false);
            colorsForPlot = vertcat(colorsForPlot{:});

            figure(figHModel + i);
            clf; hold on;

            if nDim > 2
                scatter3(latentData(allResults.svmInd{areaIdx}, 1), latentData(allResults.svmInd{areaIdx}, 2), latentData(allResults.svmInd{areaIdx}, 3), 40, colorsForPlot, 'filled');
                % Set view angle to show all three axes with depth
                view(45, 30);
            else
                scatter(latentData(allResults.svmInd{areaIdx}, 1), latentData(allResults.svmInd{areaIdx}, 2), 40, colorsForPlot, 'filled');
            end

            title(sprintf('%s %s %dD - %s (bin=%.2f)', areaName, upper(methodName), nDim, transWithinLabel, opts.frameSize), 'Interpreter','none');
            xlabel('D1'); ylabel('D2');
            if nDim > 2
                zlabel('D3');
            end
            grid on;

            % Add legend for modeled behaviors
            for j = 1:length(allResults.bhv2ModelCodes{areaIdx})
                bhvIdx = allResults.bhv2ModelCodes{areaIdx}(j);
                if bhvIdx < length(behaviors)
                    scatter(NaN, NaN, 40, colors(bhvIdx+2,:), 'filled', 'DisplayName', behaviors{bhvIdx+2});
                end
            end
            legend('Location', 'best');

            % Save plot if flag is set
            if savePlotFlag
                plotFilename = sprintf('modeling_data_%s_%s_%s_%dD_bin%.2f.png', ...
                    areaName, methodName, transOrWithin, nDim, opts.frameSize);
                plotPath = fullfile(savePath, plotFilename);
                exportgraphics(figure(figHModel + i), plotPath, 'Resolution', 300);
                fprintf('Saved plot: %s\n', plotFilename);
            end
        end
    end
end

% =============================================================================
% --------    PLOT RESULTS COMPARISON FOR ALL AREAS
% =============================================================================

if plotResults
    fprintf('Plotting results comparison for all areas...\n');

    % Create comparison figure for all areas
    fig = figure(100);
    set(fig, 'Position', [monitorOne(1), monitorOne(2), monitorOne(3), monitorOne(4)]);
    clf;

    nAreas = length(areasToTest);
    % Get the actual number of methods with data (not all available methods)
    nMethods = length(allResults.methods{areasToTest(1)});

    % Create subplots for each area
    for a = 1:nAreas
        areaIdx = areasToTest(a);
        areaName = areas{areaIdx};

        subplot(1, nAreas, a);
        hold on;

        % Get data for this area
        accuracy = allResults.accuracy{areaIdx};
        accuracyPermuted = allResults.accuracyPermuted{areaIdx};

        % Bar plot of accuracies
        x = 1:nMethods;
        barWidth = 0.35;

        % Real data bars
        b1 = bar(x, accuracy(:), barWidth, 'FaceColor', 'blue', 'DisplayName', 'Real Data');

        % Permuted data bars
        b2 = bar(x + barWidth/2, mean(accuracyPermuted, 2), barWidth, 'FaceColor', 'red', 'DisplayName', 'Permuted (mean)');

        % Add error bars for permuted data
        errorbar(x + barWidth/2, mean(accuracyPermuted, 2), std(accuracyPermuted, [], 2), 'k.', 'LineWidth', 1.5, 'DisplayName', 'Permuted (std)');

        % Customize plot
        methods = allResults.methods{areaIdx};
        set(gca, 'XTick', x, 'XTickLabel', upper(methods), 'TickLabelInterpreter', 'none');
        ylabel('Accuracy');
        title(sprintf('%s (%s)', areaName, transWithinLabel), 'Interpreter','none');
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
    end

    sgtitle('SVM Decoding Accuracy Comparison - All Areas', 'FontSize', 16);

    % Save comparison plot if flag is set
    if savePlotFlag
        plotFilename = sprintf('results_comparison_all_areas_%s_%dD_bin%.2f_nShuffles%d.png', ...
            transOrWithin, nDim, opts.frameSize, nShuffles);
        plotPath = fullfile(savePath, plotFilename);
        exportgraphics(fig, plotPath, 'Resolution', 300);
        fprintf('Saved comparison plot: %s\n', plotFilename);
    end

    % Summary statistics for all areas
    fprintf('\n=== SUMMARY FOR ALL AREAS ===\n');
    fprintf('Area\t\tMethod\t\tReal\t\tPermuted (mean ± std)\t\tDifference\n');
    fprintf('----\t\t-----\t\t----\t\t----------------------\t\t----------\n');

    for a = 1:nAreas
        areaIdx = areasToTest(a);
        areaName = areas{areaIdx};
        accuracy = allResults.accuracy{areaIdx};
        accuracyPermuted = allResults.accuracyPermuted{areaIdx};
        methods = allResults.methods{areaIdx};

        for m = 1:length(methods)
            diffAcc = accuracy(m) - mean(accuracyPermuted(m, :));
            fprintf('%s\t\t%s\t\t%.3f\t\t%.3f ± %.3f\t\t%.3f\n', ...
                areaName, upper(methods{m}), accuracy(m), mean(accuracyPermuted(m, :)), ...
                std(accuracyPermuted(m, :)), diffAcc);
        end

        % Find best method for this area
        [bestAcc, bestIdx] = max(accuracy);
        fprintf('Best method for %s: %s (accuracy: %.3f)\n', areaName, upper(methods{bestIdx}), bestAcc);
        fprintf('\n');
    end
end

% =============================================================================
% --------    SAVE RESULTS
% =============================================================================

fprintf('Saving results...\n');

% Create filename without brain area name
filename = sprintf('svm_decoding_compare_multi_area_%s_nDim%d_bin%.2f_nShuffles%d.mat', ...
    transOrWithin, nDim, opts.frameSize, nShuffles);


fullFilePath = fullfile(savePath, filename);

% Save the multi-area results
save(fullFilePath, 'allResults', '-v7.3');

fprintf('Multi-area results saved to: %s\n', fullFilePath);

% Also save a summary text file
summaryFilename = strrep(filename, '.mat', '_summary.txt');
summaryPath = fullfile(savePath, summaryFilename);

fid = fopen(summaryPath, 'w');
fprintf(fid, 'SVM Decoding Comparison Results - Multi-Area\n');
fprintf(fid, '============================================\n\n');
fprintf(fid, 'Areas Tested: %s\n', strjoin(areas(areasToTest), ', '));
fprintf(fid, 'Analysis Type: %s\n', transOrWithin);
fprintf(fid, 'Number of Dimensions: %d\n', nDim);
fprintf(fid, 'Frame Size: %.2f seconds\n', opts.frameSize);
fprintf(fid, 'Number of Permutations: %d\n', nShuffles);
fprintf(fid, 'Kernel Function: %s\n', 'polynomial');
fprintf(fid, 'Data Collection Duration: %.0f seconds\n', opts.collectFor);
fprintf(fid, 'Minimum Activity Time: %.2f seconds\n', opts.minActTime);
fprintf(fid, '\n');

fprintf(fid, 'Results Summary for All Areas:\n');
fprintf(fid, 'Area\t\tMethod\t\tReal Accuracy\tPermuted (mean ± std)\tDifference\n');
fprintf(fid, '----\t\t-----\t\t-------------\t----------------------\t----------\n');

for a = 1:length(areasToTest)
    areaIdx = areasToTest(a);
    areaName = areas{areaIdx};
    accuracy = allResults.accuracy{areaIdx};
    accuracyPermuted = allResults.accuracyPermuted{areaIdx};
    methods = allResults.methods{areaIdx};

    for m = 1:length(methods)
        diffAcc = accuracy(m) - mean(accuracyPermuted(m, :));
        fprintf(fid, '%s\t\t%s\t\t%.4f\t\t%.4f ± %.4f\t\t%.4f\n', ...
            areaName, upper(methods{m}), accuracy(m), mean(accuracyPermuted(m, :)), ...
            std(accuracyPermuted(m, :)), diffAcc);
    end

    % Find best method for this area
    [bestAcc, bestIdx] = max(accuracy);
    fprintf(fid, 'Best method for %s: %s (accuracy: %.4f)\n', areaName, upper(methods{bestIdx}), bestAcc);
    fprintf(fid, '\n');
end

fclose(fid);

fprintf('Summary saved to: %s\n', summaryPath);
fprintf('\nMulti-area analysis complete!\n');

%% =============================================================================
% --------    HELPER FUNCTIONS
% =============================================================================

function downsampledData = downsample_kinematics(data, originalBinSize, targetBinSize)
% Downsamples kinematics data using boxcar averaging
% Inputs:
%   data: original kinematics data (time x features)
%   originalBinSize: original time bin size in seconds
%   targetBinSize: target time bin size in seconds

if targetBinSize <= originalBinSize
    % No downsampling needed
    downsampledData = data;
    return;
end

% Calculate downsampling factor
downsampleFactor = targetBinSize / originalBinSize;

% Ensure downsampleFactor is reasonable (not too large)
if downsampleFactor > 10
    warning('Downsampling factor is very large (%.1f). Consider using a smaller target bin size.', downsampleFactor);
end

% Calculate number of original bins per target bin
binsPerTarget = round(downsampleFactor);

% Calculate new number of time points
nOriginalBins = size(data, 1);
nTargetBins = floor(nOriginalBins / binsPerTarget);

% Initialize output
downsampledData = zeros(nTargetBins, size(data, 2));

% Apply boxcar averaging
for i = 1:nTargetBins
    startIdx = (i-1) * binsPerTarget + 1;
    endIdx = min(i * binsPerTarget, nOriginalBins);

    % Average the bins within this window
    downsampledData(i, :) = mean(data(startIdx:endIdx, :), 1);
end

fprintf('Downsampled kinematics from %.3fs to %.3fs bins (%d -> %d time points)\n', ...
    originalBinSize, targetBinSize, nOriginalBins, nTargetBins);
end

function downsampledData = downsample_kinematics_weighted(data, originalBinSize, targetBinSize)
% DOWNSAMPLE_KINEMATICS_WEIGHTED Downsamples kinematics data using proper weighted averaging
% This function handles fractional downsampling factors by giving full weight to
% original bins that are fully overlapped and partial weight to bins that are
% partially overlapped by the target bins.
%
% Inputs:
%   data - Input data matrix (time x features)
%   originalBinSize - Original bin size in seconds
%   targetBinSize - Target bin size in seconds
%
% Outputs:
%   downsampledData - Downsampled data matrix

if targetBinSize <= originalBinSize
    % No downsampling needed
    downsampledData = data;
    return;
end

% Calculate downsampling factor
downsampleFactor = targetBinSize / originalBinSize;

% Ensure downsampleFactor is reasonable (not too large)
if downsampleFactor > 10
    warning('Downsampling factor is very large (%.1f). Consider using a smaller target bin size.', downsampleFactor);
end

% Calculate number of original bins per target bin (can be fractional)
binsPerTarget = downsampleFactor;

% Calculate new number of time points
nOriginalBins = size(data, 1);
nTargetBins = floor(nOriginalBins / binsPerTarget);

% Initialize output
downsampledData = zeros(nTargetBins, size(data, 2));

% Apply weighted averaging with proper overlap handling
for i = 1:nTargetBins
    % Calculate the start and end times for this target bin
    targetStartTime = (i-1) * targetBinSize;
    targetEndTime = i * targetBinSize;

    % Find original bins that overlap with this target bin
    originalStartBin = floor(targetStartTime / originalBinSize) + 1;
    originalEndBin = ceil(targetEndTime / originalBinSize);

    % Ensure we don't exceed data bounds
    originalStartBin = max(1, originalStartBin);
    originalEndBin = min(nOriginalBins, originalEndBin);

    if originalStartBin > originalEndBin
        % No overlap - use nearest neighbor
        downsampledData(i, :) = data(min(originalStartBin, nOriginalBins), :);
        continue;
    end

    % Calculate weights for each original bin based on overlap
    weights = zeros(originalEndBin - originalStartBin + 1, 1);
    weightedSum = zeros(1, size(data, 2));
    totalWeight = 0;

    for j = 1:length(weights)
        originalBinIdx = originalStartBin + j - 1;

        % Calculate the time range of this original bin
        originalBinStartTime = (originalBinIdx - 1) * originalBinSize;
        originalBinEndTime = originalBinIdx * originalBinSize;

        % Calculate overlap with target bin
        overlapStart = max(targetStartTime, originalBinStartTime);
        overlapEnd = min(targetEndTime, originalBinEndTime);
        overlapDuration = max(0, overlapEnd - overlapStart);

        % Weight is proportional to overlap duration
        weight = overlapDuration / originalBinSize;
        weights(j) = weight;

        % Add weighted contribution
        weightedSum = weightedSum + weight * data(originalBinIdx, :);
        totalWeight = totalWeight + weight;
    end

    % Normalize by total weight
    if totalWeight > 0
        downsampledData(i, :) = weightedSum / totalWeight;
    else
        % Fallback to simple averaging if no valid weights
        downsampledData(i, :) = mean(data(originalStartBin:originalEndBin, :), 1);
    end
end

fprintf('Downsampled kinematics from %.3fs to %.3fs bins (%d -> %d time points)\n', ...
    originalBinSize, targetBinSize, nOriginalBins, nTargetBins);
end

function downsampledData = downsample_kinematics_decimate(data, originalBinSize, targetBinSize)
% Downsamples kinematics data using decimation with anti-aliasing filter
% This is more sophisticated than simple averaging

if targetBinSize <= originalBinSize
    downsampledData = data;
    return;
end

downsampleFactor = targetBinSize / originalBinSize;

% Apply anti-aliasing filter first
% Use a low-pass filter to prevent aliasing
cutoffFreq = 1 / (2 * targetBinSize);  % Nyquist frequency for target rate
sampleRate = 1 / originalBinSize;

% Design low-pass filter
[b, a] = butter(4, cutoffFreq / (sampleRate/2), 'low');

% Apply filter to each feature
filteredData = zeros(size(data));
for feat = 1:size(data, 2)
    filteredData(:, feat) = filtfilt(b, a, data(:, feat));
end

% Decimate
downsampledData = downsample(filteredData, round(downsampleFactor));
end
