%%                     Compare SVM decoding accuracy across different methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script compares SVM decoding performance across PCA, UMAP, PSID, and ICG
% dimensionality reduction methods using the same neural data and behavior labels


%% Specify main parameters
paths = get_paths;

% Data type selection
dataType = 'reach';  % 'reach' or 'naturalistic'

% Dimensionality for all methods
dimToTest = [4 6 8];
dimToTest = 8;

% Analysis type
transWithinToTest = {'trans','transPost' 'within', 'all'};
transWithinToTest = {'trans'};
% transOrWithin = 'trans';  % 'trans','transPost' 'within', 'all'

% Frame/bin size
frameSize = .1;

% SVM parameters
kernelFunction = 'polynomial'; % 'linear' polynomial


% Create full path
savePath = fullfile(paths.dropPath, 'decoding');
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

%%

% Data loading based on data type
if strcmp(dataType, 'reach')
    reachCode = 2; % Behavior label for reaching indices
    % Load reach data
    reachDataFile = fullfile(paths.reachDataPath, 'Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');

    dataR = load(reachDataFile);

    opts = neuro_behavior_options;
    opts.firingRateCheckTime = 5 * 60;
    opts.collectStart = 0;
    opts.collectFor = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
    opts.minFiringRate = .1;
    opts.maxFiringRate = 70;
    opts.frameSize = frameSize;

    [dataMat, idLabels, areaLabels] = neural_matrix_mark_data(dataR, opts);
    areas = {'M23', 'M56', 'DS', 'VS'};
    idM23 = find(strcmp(areaLabels, 'M23'));
    idM56 = find(strcmp(areaLabels, 'M56'));
    idDS = find(strcmp(areaLabels, 'DS'));
    idVS = find(strcmp(areaLabels, 'VS'));
    idList = {idM23, idM56, idDS, idVS};

    % Get behavior labels for reach task
    bhvOpts = struct();
    bhvOpts.frameSize = frameSize;
    bhvOpts.collectStart = opts.collectStart;
    bhvOpts.collectFor = opts.collectFor;
    bhvID = define_reach_bhv_labels(reachDataFile, bhvOpts);

    disp('Modeling reach vs non-reach')
    bhvID(bhvID ~= reachCode) = 6;


    fprintf('Loaded reach data: %d neurons, %d time points\n', size(dataMat, 2), size(dataMat, 1));

elseif strcmp(dataType, 'naturalistic')
    % Load naturalistic data
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

    % Set up areas for naturalistic data
    areas = {'M23', 'M56', 'DS', 'VS'};
    idList = {idM23, idM56, idDS, idVS};

    fprintf('Loaded naturalistic data: %d neurons, %d time points\n', size(dataMat, 2), size(dataMat, 1));
else
    error('Invalid dataType. Must be ''reach'' or ''naturalistic''');
end


% Get colors for behaviors (set up for reach task)
if strcmp(dataType, 'reach')
    % For reach task: define behavior labels
    behaviors = {'pre-reach', 'reach', 'pre-reward', 'reward', 'post-reward', 'intertrial'};
    bhvLabels = behaviors;

    % Create colors for each behavior (simple color scheme for reach behaviors)
    nBehaviors = length(behaviors);
    colors = lines(nBehaviors);
    colorsAdjust = 0;
else
    % For naturalistic data: use existing color system
    colors = colors_for_behaviors(codes);
    colorsAdjust = 2; % Add 2 b/c color indices offset from bhvID values
    bhvLabels = {'investigate_1', 'investigate_2', 'investigate_3', ...
        'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
        'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
        'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};
end

% Monitor setup for plotting
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

%% =============================================================================
% --------    ANALYSIS PARAMETERS & INITIALIZATIONS
% =============================================================================

% Define brain areas to test
areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 2:4;  % Test M56, DS, VS (append ICG to existing results)


% Analysis control flags
loadExistingResults = false;  % Set to true to load and append to existing results

% Determine which methods to run
methodsToRun = {'pca', 'umap', 'psidKin', 'psidKin_nonBhv', 'psidBhv', 'psidBhv_nonBhv', 'icg'}';
methodsToRun = {'pca', 'umap'}';
fprintf('Running methods: %s\n', strjoin(methodsToRun, ', '));

% Permutation testing
nShuffles = 2;  % Number of permutation tests

% Plotting options
plotFullMap = 0;
plotModelData = 1;
plotResults = 1;
savePlotFlag = 1;  % Save plots as PNG files

% Figure properties
allFontSize = 12;





% =============================================================================
% --------    NESTED LOOPS FOR ALL CONDITIONS
% =============================================================================

ttime = tic;

fprintf('\n=== Processing All Combinations ===\n');
fprintf('Dimensions to test: %s\n', mat2str(dimToTest));
fprintf('Trans/Within conditions to test: %s\n', strjoin(transWithinToTest, ', '));
fprintf('Areas to test: %s\n', strjoin(areas(areasToTest), ', '));
% Loop through all dimensions
% for dimIdx = 1:length(dimToTest)
dimIdx = 1;
nDim = dimToTest(dimIdx);
fprintf('\n\n=======================   DIMENSION: %dD   =======================\n', nDim);

% Loop through all trans/within conditions
% for transIdx = 1:length(transWithinToTest)
transIdx = 1;
transOrWithin = transWithinToTest{transIdx};


fprintf('\n=======================   CONDITION: %s   =======================\n', upper(transOrWithin));





% =============================================================================
% --------    LOAD EXISTING RESULTS (IF REQUESTED)
% =============================================================================

% % Load existing results for this combination
% filename = sprintf('svm_%s_decoding_compare_multi_area_%s_nDim%d_bin%.2f_nShuffles%d.mat', ...
%     kernelFunction, transOrWithin, nDim, opts.frameSize, nShuffles);
% fullFilePath = fullfile(savePath, filename);
% 
% if exist(fullFilePath, 'file')
%     fprintf('Loading existing results from: %s\n', filename);
%     load(fullFilePath, 'allResults');
%     fprintf('Loaded existing results for %s %dD\n', transOrWithin, nDim);
% else
    fprintf('No existing results found for %s %dD. Creating new analysis.\n', transOrWithin, nDim);
    % Initialize storage for all areas
    allResults = struct();
    allResults.areas = areas;
    allResults.areasToTest = areasToTest;
    allResults.parameters = struct();
    allResults.parameters.frameSize = opts.frameSize;
    allResults.parameters.nShuffles = nShuffles;
    allResults.parameters.kernelFunction = kernelFunction;
    allResults.parameters.collectStart = opts.collectStart;
    allResults.parameters.collectFor = opts.collectFor;
    allResults.parameters.minActTime = opts.minActTime;
    allResults.parameters.transOrWithin = transOrWithin;

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
% end

% Update parameters for this combination
allResults.parameters.nDim = nDim;
allResults.parameters.transOrWithin = transOrWithin;
allResults.parameters.kernelFunction = kernelFunction;


for areaIdx = areasToTest
    areaName = areas{areaIdx};
    fprintf('\n=======================   --- Processing Area: %s ---  ==================\n', areaName);


    % Select data for this area using idList
    idSelect = idList{areaIdx};

    % Store the selected IDs
    allResults.idSelect{areaIdx} = idSelect;

    fprintf('Selected %d neurons for area %s\n', length(idSelect), areaName);

    %% =============================================================================
    % --------    GENERATE LATENTS FOR EACH METHOD
    % =============================================================================


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
        min_dist = 0.3;
        spread = 1;
        n_neighbors = 15;

        % Add parameters to prevent GUI from appearing
        [latents.umap, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', nDim, ...
            'randomize', true, 'verbose', 'none', 'min_dist', min_dist, ...
            'spread', spread, 'n_neighbors', n_neighbors);

        % Return to original directory
        cd(fullfile(paths.homePath, 'neuro-behavior/src/decoding'));
    end

    %% PSID with Kinematics (psidKin)
    if (ismember('psidKin', methodsToRun) || ismember('psidKin_nonBhv', methodsToRun)) && strcmp(dataType, 'naturalistic')
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
    elseif (ismember('psidKin', methodsToRun) || ismember('psidKin_nonBhv', methodsToRun)) && strcmp(dataType, 'reach')
        fprintf('Skipping PSID with kinematics (kinData not available for reach task)...\n');
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
        if strcmp(dataType, 'reach')
            bhvMapping.behaviorNames = behaviors(uniqueBhv);  % Direct mapping for reach behaviors
        else
            bhvMapping.behaviorNames = behaviors(uniqueBhv + 2);  % +2 because behaviors array starts at index 1
        end

        % Prepare data for PSID with behavior labels
        y = zscore(dataMat(:, idSelect));
        z = zscore(bhvOneHot);  % Use one-hot encoded behavior labels
        nx = nDim * 2;
        n1 = nDim;
        i = 10;

        % Debug: Check dimensions
        fprintf('PSID inputs - y: %dx%d, z: %dx%d\n', size(y,1), size(y,2), size(z,1), size(z,2));

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

        % For M23 (few neurons), use few correlated neurons
        if areaIdx == 1
            latents.icg = zscore(activityICG{3}(1:3,:)');
        else
            if nDim == 4
                % Take the first nDim groups of most correlated neurons
                latents.icg = zscore(activityICG{4}(1:nDim, :)');
            elseif nDim >= 6
                % Take the first nDim groups of most correlated neurons
                warning('Changed ICG population')
                latents.icg = zscore(activityICG{3}(1:9, :)');
            end
        end
    end

    fprintf('All methods completed for area %s.\n', areaName);


end





%% =============================================================================
% --------    CHOOSE WHICH DATA POINTS TO MODEL
% =============================================================================
switch dataType
    case 'naturalistic'
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
    case 'reach'
        switch transOrWithin
            case 'all'
                svmInd = 1:length(bhvID);
                svmID = bhvID;
                transWithinLabel = 'all';
            case 'trans'
                % Build windows around reach starts: [-2s, +3s] relative to each reach start
                % Reach starts are the first indices where bhvID transitions into 2
                reachMask = (bhvID == reachCode);
                reachStarts = find([reachMask(1); diff(reachMask) == 1]);
                reachStops = find([reachMask(1); diff(reachMask) == -1]);

                preSec = 1;  % seconds before reach start
                postSec = 0; % seconds after reach start
                preFrames = round(preSec / opts.frameSize);
                postFrames = round(postSec / opts.frameSize);

                keepMask = false(length(bhvID), 1);
                for r = 1:length(reachStarts)
                    winStart = reachStarts(r) - preFrames;
                    winEnd = max(min(length(bhvID), reachStarts(r) + postFrames), reachStops(r));
                    keepMask(winStart:winEnd) = true;
                end

                svmInd = find(keepMask);
                svmID = bhvID(svmInd);
                transWithinLabel = 'peri-reach (-2s to +2s)';
        end
end
% Get behavior codes and names for modeling
bhv2ModelCodes = unique(svmID);
if strcmp(dataType, 'reach')
    bhv2ModelNames = behaviors(bhv2ModelCodes);
else
    bhv2ModelNames = behaviors(bhv2ModelCodes+colorsAdjust);
    bhv2ModelColors = colors(ismember(codes, bhv2ModelCodes), :);
end

% For reach task, assign colors based on behavior codes
if strcmp(dataType, 'reach')
    bhv2ModelColors = colors(bhv2ModelCodes, :);
end


    fprintf('Modeling %s behaviors: %s\n', transWithinLabel, strjoin(bhv2ModelNames, ', '));



%% =============================================================================
% --------    RUN SVM DECODING FOR EACH METHOD
% =============================================================================
% Parallel processing setup
nWorkers = min(4,length(methodsToRun));  % Number of parallel workers to use

fprintf('Setting up parallel pool with %d workers...\n', nWorkers);
if isempty(gcp('nocreate'))
    parpool('local', nWorkers);
else
    fprintf('Parallel pool already exists with %d workers\n', gcp('nocreate').NumWorkers);
end

for areaIdx = areasToTest
    areaName = areas{areaIdx};

    fprintf('Running SVM decoding for specified methods...\n');

    % Get methods that were actually run
    methods = methodsToRun;


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

        % Process methods in parallel (each method is independent)
        % Pre-allocate temporary storage for parfor results
        tempAccuracy = zeros(nMethods, 1);
        tempAccuracyPermuted = cell(nMethods, 1);  % Use cell array for variable-length results
        tempSvmModels = cell(nMethods, 1);
        tempAllPredictions = cell(nMethods, 1);
        tempAllPredictionIndices = cell(nMethods, 1);

        parfor m = 1:nMethods
            % for m = 1:nMethods
            methodName = methods{m};


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

            % Store the full model in temporary variable
            tempSvmModels{m} = svmModelFull;

            % Generate predictions for all relevant indices
            tempAllPredictions{m} = predict(svmModelFull, svmProj);
            tempAllPredictionIndices{m} = svmInd;  % Store the indices that were predicted

            % For cross-validation accuracy calculation, use the holdout approach
            svmModelCV = fitcecoc(trainData, trainLabels, 'Learners', t);
            predictedLabels = predict(svmModelCV, testData);
            tempAccuracy(m) = sum(predictedLabels == testLabels) / length(testLabels);

            fprintf('Real data accuracy: %.4f\n', tempAccuracy(m));
            fprintf('Model training time: %.2f seconds\n', toc);

            % Permutation tests - collect results in a temporary array
            fprintf('Running %d permutation tests...\n', nShuffles);
            tic;

            % Pre-allocate permutation results for this method
            methodAccuracyPermuted = zeros(nShuffles, 1);

            for s = 1:nShuffles
                % Circular shuffle of training labels
                rng('shuffle');
                randShift = randi([1, length(trainLabels)]);
                shuffledLabels = circshift(trainLabels, randShift);

                % Train model on shuffled data
                svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);

                % Test on real data
                predictedLabelsPermuted = predict(svmModelPermuted, testData);
                methodAccuracyPermuted(s) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);

                fprintf('  Permutation %d: %.4f\n', s, methodAccuracyPermuted(s));
            end

            % Store permutation results for this method
            tempAccuracyPermuted{m} = methodAccuracyPermuted;

            fprintf('Permutation testing time: %.2f seconds\n', toc);
            fprintf('Mean permuted accuracy: %.4f\n', mean(methodAccuracyPermuted));
        end

        % Assign temporary results to final arrays
        for m = 1:nMethods
            fullIdx = startIdx + m - 1;
            accuracy(fullIdx) = tempAccuracy(m);
            accuracyPermuted(fullIdx, :) = tempAccuracyPermuted{m}';  % Convert cell to row vector
            svmModels{fullIdx} = tempSvmModels{m};
            allPredictions{fullIdx} = tempAllPredictions{m};
            allPredictionIndices{fullIdx} = tempAllPredictionIndices{m};
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
        allResults.methods{areaIdx} = [allResults.methods{areaIdx}; methods];
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
delete(gcp('nocreate'));
fprintf('Parallel pool closed.\n');

%%
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
            if strcmp(dataType, 'reach')
                colorsForPlot = arrayfun(@(x) colors(x,:), bhvID, 'UniformOutput', false);
            else
                colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + colorsAdjust, 'UniformOutput', false);
            end
            colorsForPlot = vertcat(colorsForPlot{:});

            figure(figHFull + i);
            % Set figure position using monitor dimensions (3/4 height, 1/2 width)
            figWidth = monitorOne(3) * 0.5;  % Half width
            figHeight = monitorOne(4) * 0.75; % Three-quarters height
            figX = monitorOne(1) + (monitorOne(3) - figWidth) / 2;  % Center horizontally
            figY = monitorOne(2) + (monitorOne(4) - figHeight) / 2; % Center vertically
            set(figHFull + i, 'Position', [figX, figY, figWidth, figHeight]);
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
                    scatter(NaN, NaN, 20, colors(bhvIdx+colorsAdjust,:), 'filled', 'DisplayName', behaviors{bhvIdx+colorsAdjust});
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
            colorsForPlot = arrayfun(@(x) colors(x,:), allResults.svmID{areaIdx} + colorsAdjust, 'UniformOutput', false);
            colorsForPlot = vertcat(colorsForPlot{:});

            figure(figHModel + i);
            % Set figure position using monitor dimensions (3/4 height, 1/2 width)
            figWidth = monitorOne(3) * 0.5;  % Half width
            figHeight = monitorOne(4) * 0.75; % Three-quarters height
            figX = monitorOne(1) + (monitorOne(3) - figWidth) / 2;  % Center horizontally
            figY = monitorOne(2) + (monitorOne(4) - figHeight) / 2; % Center vertically
            set(figHModel + i, 'Position', [figX, figY, figWidth, figHeight]);
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
                    scatter(NaN, NaN, 40, colors(bhvIdx+colorsAdjust,:), 'filled', 'DisplayName', behaviors{bhvIdx+colorsAdjust});
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

    % Plot all areas regardless of which ones are being analyzed
    allAreasToPlot = 1:4;  % M23, M56, DS, VS
    nAreas = length(allAreasToPlot);

    % Find the first area that has data to determine number of methods
    nMethods = 0;
    for checkArea = allAreasToPlot
        if ~isempty(allResults.methods{checkArea})
            nMethods = length(allResults.methods{checkArea});
            break;
        end
    end

    if nMethods == 0
        fprintf('No data found for any area. Skipping plot.\n');
        % continue;
    end

    % Create subplots for each area
    for a = 1:nAreas
        areaIdx = allAreasToPlot(a);
        areaName = areas{areaIdx};

        subplot(1, nAreas, a);
        hold on;

        % Check if this area has data
        if isempty(allResults.methods{areaIdx}) || isempty(allResults.accuracy{areaIdx})
            % No data for this area - show empty plot
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 14, 'Color', 'red');
            title(sprintf('%s (%s) - No Data', areaName, transWithinLabel), 'Interpreter','none');
            ylim([0, 1]);
            xlim([0, 1]);
            continue;
        end

        % Get data for this area
        accuracy = allResults.accuracy{areaIdx};
        accuracyPermuted = allResults.accuracyPermuted{areaIdx};
        methods = allResults.methods{areaIdx};

        % Use the actual number of methods for this area
        areaMethods = length(methods);

        % Bar plot of accuracies
        x = 1:areaMethods;
        barWidth = 0.35;

        % Real data bars
        b1 = bar(x, accuracy(:), barWidth, 'FaceColor', 'blue', 'DisplayName', 'Real Data');

        % Permuted data bars
        b2 = bar(x + barWidth/2, mean(accuracyPermuted, 2), barWidth, 'FaceColor', 'red', 'DisplayName', 'Permuted (mean)');

        % Add error bars for permuted data
        errorbar(x + barWidth/2, mean(accuracyPermuted, 2), std(accuracyPermuted, [], 2), 'k.', 'LineWidth', 1.5, 'DisplayName', 'Permuted (std)');

        % Customize plot
        set(gca, 'XTick', x, 'XTickLabel', upper(methods), 'TickLabelInterpreter', 'none');
        ylabel('Accuracy');
        title(sprintf('%s (%s)', areaName, transWithinLabel), 'Interpreter','none');
        legend('Location', 'best');
        grid on;
        ylim([0, 1]);

        % Add significance indicators
        for m = 1:areaMethods
            % Simple significance test: if real accuracy > 95th percentile of permuted
            permSorted = sort(accuracyPermuted(m, :));
            threshold95 = permSorted(ceil(0.95 * nShuffles));

            if accuracy(m) > threshold95
                text(m, accuracy(m) + 0.02, '*', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
            end
        end

        % Add accuracy values as text
        for m = 1:areaMethods
            text(m - barWidth/2, accuracy(m) + 0.01, sprintf('%.3f', accuracy(m)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10);
            text(m + barWidth/2, mean(accuracyPermuted(m, :)) + 0.01, sprintf('%.3f', mean(accuracyPermuted(m, :))), ...
                'HorizontalAlignment', 'center', 'FontSize', 10);
        end
    end

    sgtitle('SVM Decoding Accuracy Comparison - All Areas', 'FontSize', 16);

    % Save comparison plot if flag is set
    if savePlotFlag
        plotFilename = sprintf('svm_%s_results_comparison_all_areas_%s_%dD_bin%.2f_nShuffles%d.png', ...
            kernelFunction, transOrWithin, nDim, opts.frameSize, nShuffles);
        plotPath = fullfile(savePath, plotFilename);
        exportgraphics(fig, plotPath, 'Resolution', 300);
        fprintf('Saved comparison plot: %s\n', plotFilename);
    end

    % Summary statistics for all areas
    fprintf('\n=== SUMMARY FOR ALL AREAS ===\n');
    fprintf('Area\t\tMethod\t\tReal\t\tPermuted (mean ± std)\t\tDifference\n');
    fprintf('----\t\t-----\t\t----\t\t----------------------\t\t----------\n');

    for a = 1:nAreas
        areaIdx = allAreasToPlot(a);
        areaName = areas{areaIdx};

        % Check if this area has data
        if isempty(allResults.methods{areaIdx}) || isempty(allResults.accuracy{areaIdx})
            fprintf('%s\t\tNo Data Available\n', areaName);
            continue;
        end

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

fprintf('Saving updated results with M23 data for %s %dD...\n', transOrWithin, nDim);

% Create filename without brain area name
filename = sprintf('svm_%s_decoding_compare_multi_area_%s_nDim%d_bin%.2f_nShuffles%d.mat', ...
    kernelFunction, transOrWithin, nDim, opts.frameSize, nShuffles);


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

for a = 1:4  % All areas: M23, M56, DS, VS
    areaIdx = a;
    areaName = areas{areaIdx};

    % Check if this area has data
    if isempty(allResults.methods{areaIdx}) || isempty(allResults.accuracy{areaIdx})
        fprintf(fid, '%s\t\tNo Data Available\n', areaName);
        continue;
    end

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
fprintf('\nM23 analysis complete for %s %dD! Results added to existing file.\n', transOrWithin, nDim);

%     end  % End transWithinToTest loop
% end  % End dimToTest loop

% Clean up parallel pool
fprintf('Closing parallel pool...\n');
delete(gcp('nocreate'));
fprintf('Parallel pool closed.\n');

fprintf('\n\nTotal analysis time: %.2f hours\n', toc(ttime)/60/60);

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
