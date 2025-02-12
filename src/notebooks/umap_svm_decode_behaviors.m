%%                     Compare neuro-behavior in UMAP spaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .2;
% opts.shiftAlignFactor = .05; % I want spike counts xxx ms peri-behavior label

opts.minFiringRate = .5;

getDataType = 'spikes';
get_standard_data
% getDataType = 'behavior';
% get_standard_data

[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);



colors = colors_for_behaviors(codes);

% for plotting consistency
%
monitorPositions = get(0, 'MonitorPositions');
if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
    monitorPositions = flipud(monitorPositions);
end
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one


%
bhvLabels = {'investigate_1', 'investigate_2', 'investigate_3', ...
    'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
    'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
    'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};












%% =============================================================================
% --------    RUN UMAP SVM FITS FOR VARIOUS CONDITIONS ON VARIOUS DATA
% =============================================================================

% Select which data to run analyses on, UMAP dimensions, etc

% forDim = 4:2:8; % Loop through these dimensions to fit UMAP
forDim = 4; % Loop through these dimensions to fit UMAP
% lowDModel = 'umap';
lowDModel = 'tsne';
newLowDModel = 1; % Do we need to get a new umap model to analyze (or did you tweak some things that come after umap?)
umapTransOnly = 0;

% Change these (and check their sections below) to determine which
% variables to test
% ==========================


% Apply to all:
% -------------
analyzePredictions = 1;
plotFullMap = 0;
plotFullModelData = 1;
plotModelData = 1;
plotTransPred = 0; % Predicting transitions based on whatever model you fit
changeBhvLabels = 0;

% Transition or within variables
% -------------------------
transOrWithin = 'trans';
transOrWithin = 'within';
% transOrWithin = 'all';
% transOrWithin = 'transVsWithin';
firstNFrames = 0;
matchTransitionCount = 0;
minFramePerBout = 0;

% Apply to all:
% --------------
collapseBhv = 0;
minBoutNumber = 0;
downSampleBouts = 0;
minTotalFrames = 0;
downSampleFrames = 0;


selectFrom = 'M56';
% selectFrom = 'DS';
% selectFrom = 'Both';
% selectFrom = 'M23';
% selectFrom = 'VS';
% selectFrom = 'All';
switch selectFrom
    case 'M23'
        idSelect = idM23;
        figHFull = 260;
        figHModel = 270;
        figHFullModel = 280;
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


% idSelect = idDS(fStatZ > 25);

% Some figure properties
allFontSize = 12;

switch lowDModel
    case 'umap'
        min_dist = (.1 : .1 : .5);
        min_dist = (.01 : .03 : .1);
        min_dist = .02;
        spread = 1.3;
        n_neighbors = [6 8 10 12 15];
        n_neighbors = 10;

        if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
            cd '/Users/paulmiddlebrooks/Projects/toolboxes/umapFileExchange (4.4)/umap/'
        else
            cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
        end

    case 'tsne'
        exaggeration = [60 90 120];
        perplexity = [10 30 40 50];
end

modelAccuracy = zeros(length(forDim), length(min_dist), length(spread), length(n_neighbors));
permAccuracy = zeros(length(forDim), length(min_dist), length(spread), length(n_neighbors));







% % PCA testing....
% [coeff, score, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));
% expVarThresh = [15 20 30 45 65 90];
% pcaDim = zeros(1, length(expVarThresh));
% expVar = zeros(1, length(expVarThresh));
% for p = 1:length(expVar)
%     pcaDim(p) = find(cumsum(explained) > expVarThresh(p), 1);
%     expVar(p) = sum(explained(1:pcaDim(p)));
% end

% Modeling variables
nPermutations = 1; % How many random permutations to run to compare with best fit model?
accuracy = zeros(length(forDim), 1);
accuracyPermuted = zeros(length(forDim), nPermutations);



    for k = 1:length(forDim)
    % for k = 1:length(pcaDim)


        iDim = forDim(1);
        fitType = [lowDModel, ' ', num2str(iDim), 'D'];
        % fitType = 'NeuralSpace';


        % for x = 1:length(exaggeration)
        %     for y = 1:length(perplexity)
        for x = 1:length(min_dist)
            for y = 1:length(spread)
                for z = 1:length(n_neighbors)
                    disp('=================================================================')
                    %% Run UMAPto get projections in low-D space
                    if newLowDModel && ~umapTransOnly
                        switch lowDModel
                            case 'umap'
                                fprintf('\n%s %s min_dist=%.2f spread=%.1f n_n=%d\n\n', selectFrom, fitType, min_dist(x), spread(y), n_neighbors(z));
                                umapFrameSize = opts.frameSize;
                                rng(1);

                                warning('Function: %s, Line: %d - Using PCA before UMAP.', mfilename, dbstack(1).line);
                                % [coeff, score, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));
                                % expVarThresh = 25;
                                % dimExplained = find(cumsum(explained) > expVarThresh, 1);
                                % disp(['PCA: Exp Var = ', num2str(expVar(k)), ' nComponents = ', num2str(pcaDim(k))])
                                % [projSelect, ~, ~, ~] = run_umap(score(:, 1:pcaDim(k)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
                                %     'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));

                                [projSelect, ~, ~, ~] = run_umap(zscore(dataMat(:, idSelect)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
                                    'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));

                                % minFrames = 2;
                                %                     [stackedActivity, stackedLabels] = datamat_stacked_means(dataMat, bhvID, minFrames);
                                %                     [~, umap, ~, ~] = run_umap(zscore(stackedActivity(:, idSelect)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
                                %                         'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));
                                %                     projSelect = umap.transform(zscore(dataMat(:, idSelect)));
                                pause(2); close
                            case 'tsne'
                                fprintf('\n%s %s exagg=%d perplx=%d \n\n', selectFrom, fitType, exaggeration(x), perplexity(y));
                                projSelect = tsne(zscore(dataMat(:, idSelect)),'Exaggeration', exaggeration(x), 'Perplexity', perplexity(y), 'NumDimensions',iDim);
                        end
                    end



                    %% --------------------------------------------
                    % Shift behavior label w.r.t. neural to account for neuro-behavior latency
                    shiftSec = 0;
                    shiftFrame = 0;
                    if shiftSec > 0
                        shiftFrame = ceil(shiftSec / opts.frameSize);
                        projSelect = projSelect(1:end-shiftFrame, :); % Remove shiftFrame frames from projections to accoun for time shift in bhvID
                    end
                    bhvID = double(bhvID(1+shiftFrame:end)); % Shift bhvID to account for frame shift




                    %% --------------------------------------------
                    % Plot FULL TIME OF ALL BEHAVIORS
                    if plotFullMap && ~umapTransOnly
                        colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + 2, 'UniformOutput', false);
                        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
                        % colorsForPlot = [.2 .2 .2];
                        figH = figHFull;
                        plotPos = [monitorOne(1), 1, monitorOne(3)/2, monitorOne(4)];
                        % titleM = sprintf('%s %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, opts.frameSize, min_dist(x), spread(y), n_neighbors(z));
                        titleM = sprintf('%s %s bin=%.2f', selectFrom, fitType, opts.frameSize);
                        % titleM = [selectFrom, ' ', fitType, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
                        plotFrames = 1:length(bhvID);
                        plot_3d_scatter
                    end








                    %% --------------------------------------------------------------------------------------------------------------
                    % -----------------     TWEAK THE BEHAVIOR LABELS IF YOU WANT       -----------------
                    % --------------------------------------------------------------------------------------------------------------
                    if changeBhvLabels
                        % bhvToChange = [1, 15];
                        % newBhvCode = 16;
                        % minDurFrames = 4;
                        %
                        % for iBhv = 1 : length(bhvToChange)
                        %
                        %     valIndices = (bhvID == bhvToChange(iBhv));
                        %
                        %     % Identify the start and end of stretches of the target value
                        %     diffIndices = [false; diff(valIndices) ~= 0]; % Find where the value changes
                        %     startIndices = find(diffIndices & valIndices); % Starts of stretches
                        %     endIndices = find(diffIndices & ~valIndices) - 1; % Ends of stretches
                        %
                        %     % Handle case where the last stretch runs until the end of the vector
                        %     if isempty(endIndices) || endIndices(end) < numel(bhvID) && valIndices(end)
                        %         endIndices = [endIndices; numel(bhvID)];
                        %     end
                        %
                        %     % Calculate the durations of each stretch
                        %     durations = endIndices - startIndices + 1;
                        %
                        %     % Get the indices of stretches with durations <= 2
                        %     shortBhvs = find(durations <= minDurFrames);
                        %
                        %     % Optionally, get the specific indices for these stretches
                        %     shortIndices = [];
                        %     for kBhv = 1:length(shortBhvs)
                        %         shortIndices = [shortIndices, startIndices(shortBhvs(kBhv)):endIndices(shortBhvs(kBhv))];
                        %     end
                        %
                        %     bhvID(shortIndices) = newBhvCode;
                        % end
                        %
                        % % figure(2343)
                        % % histogram(durations)
                        %
                        % behaviors = [behaviors, 'explore'];



                        minDurFrames = 6;
                        valIdx = bhvID == 15;
                        % Identify the start and end of stretches of the target value
                        diffIndices = [false; diff(valIdx) ~= 0]; % Find where the value changes
                        startIndices = find(diffIndices & valIdx); % Starts of stretches
                        endIndices = find(diffIndices & ~valIdx) - 1; % Ends of stretches

                        % Handle case where the last stretch runs until the end of the vector
                        if isempty(endIndices) || endIndices(end) < numel(bhvID) && valIdx(end)
                            endIndices = [endIndices; numel(bhvID)];
                        end

                        % Calculate the durations of each stretch
                        durations = endIndices - startIndices + 1;

                        % Get the indices of stretches with durations <= 2
                        switchBhvs = find(durations <= minDurFrames);
                        % switchBhvs = find(durations > minDurFrames);

                        % Optionally, get the specific indices for these stretches
                        switchIdx = [];
                        for kBhv = 1:length(switchBhvs)
                            switchIdx = [switchIdx, startIndices(switchBhvs(kBhv)):endIndices(switchBhvs(kBhv))];
                        end

                        bhvID(switchIdx) = 1;


                    end




                    %% --------------------------------------------------------------------------------------------------------------
                    % ---------------   TRANSITIONS OR WITHIN-BEHAVIOR     ----------------------------------------------------------
                    % --------------------------------------------------------------------------------------------------------------
                    % Find all time bins preceding all behavior transitions:

                    preInd = find(diff(bhvID) ~= 0); % 1 frame prior to all behavior transitions


                    switch transOrWithin
                        case 'trans'
                            %% TRANSITIONS of all behaviors (for now, include behaviors that last one frame)
                            % svmID: vector of category labels for each data point to be analyzed/fit

                            svmID = bhvID(preInd + 1);  % behavior ID being transitioned into

                            % Pre and/or Post: Adjust which bin(s) to plot (and train SVN on below)
                            svmInd = preInd; % + 1; % First bin before or after (+1) transition

                            % Pre & Post: Comment/uncomment to use more than one bin
                            % svmID = repelem(svmID, 2);
                            % svmInd = sort([svmInd - 1; svmInd]); % two bins before transition
                            % svmInd = sort([svmInd; svmInd + 1]); % Last bin before transition and first bin after

                            transWithinLabel = 'transitions pre';
                            % transWithinLabel = 'transitions 200ms pre';
                            % transWithinLabel = 'transitions post';
                            % transWithinLabel = 'transitions pre & post';
                            % transWithinLabel = ['transitions pre minBout ', num2str(nMinFrames)];



                            if umapTransOnly && newLowDModel
                                % now use only the transition data to fit umap (and decode svm)
                                disp('You are running a low-d model and fitting decoder only on transition data');
                                switch lowDModel
                                    case 'umap'

                                        umapFrameSize = opts.frameSize;

                                        % rng(1);
                                        % [projSelect, ~, ~, ~] = run_umap(dataMat(:, idSelect), 'n_components', iDim, 'randomize', false);
                                        [projSelect, ~, ~, ~] = run_umap(zscore(dataMat(svmInd, idSelect)), 'n_components', iDim, 'randomize', false, 'verbose', 'none', ...
                                            'min_dist', min_dist(x), 'spread', spread(y), 'n_neighbors', n_neighbors(z));
                                    case 'tsne'
                                        projSelect = tsne(zscore(dataMat(svmInd, idSelect)),'Exaggeration',90, 'NumDimensions',iDim);

                                end
                                % whittled down the matrix to transition-only frames, so now fit all the
                                % data:
                                svmInd = 1:size(projSelect, 1);
                                pause(4); close
                            end


                            %% WITHIN-BEHAVIOR of all behaviors (for now, include behaviors that last one frame)
                        case 'within'

                            transIndLog = zeros(length(bhvID), 1);
                            transIndLog(preInd) = 1;

                            % If you want to remove another pre-behavior onset bin, do this:
                            vec = find(transIndLog);
                            transIndLog(vec-1) = 1;

                            % If you want to remove a bin after behavior onset, do this:
                            % transIndLog(vec+1) = 1;

                            svmInd = find(~transIndLog);
                            svmID = bhvID(svmInd);

                            % choose correct title
                            transWithinLabel = 'within-behavior';


                            %
                            if changeBhvLabels
                                % transWithinLabel = [transWithinLabel, ' short inv2-loco as new bhv'];
                                transWithinLabel = [transWithinLabel, ' short loco as inv2'];
                                % transWithinLabel = [transWithinLabel, ' long loco as inv2'];
                                % transWithinLabel = 'within-behavior xxxxx';
                            end
                        case 'all'
                            svmInd = 1:length(bhvID);
                            svmID = bhvID;

                            % choose correct title
                            transWithinLabel = 'all';
                        case 'transVsWithin'
                            svmID = bhvID;
                            svmID(diff(bhvID) ~= 0) = 9;
                            svmID(diff(bhvID) == 0) = 12;
                            svmID(end) = [];
                            svmInd = 1 : length(svmID);
                            transWithinLabel = 'trans-vs-within';

                    end

                    %% If you set any behavior/data curation flags:
                    modify_data_frames_to_model


                    %% Get rid of sleeping/in_nest/irrelavents
                    deleteInd = svmID == -1;
                    svmID(deleteInd) = [];
                    svmInd(deleteInd) = [];



                    %% Keep track of the behavior IDs you end up using
                    if strcmp(transOrWithin, 'transVsWithin')
                        bhv2ModelCodes = [9 12];
                        bhv2ModelNames = {'transitions', 'within'};
                        % bhv2ModelColors = [0 0 1; 1 .33 0];
                    else
                        bhv2ModelCodes = unique(svmID);
                        bhv2ModelNames = behaviors(bhv2ModelCodes+2);
                        bhv2ModelColors = colors(ismember(codes, bhv2ModelCodes), :);
                    end


                    %% --------------------------------------------
                    % Plot Full time of all behaviors that we are modeling
                    if plotFullModelData
                        allBhvModeled = ismember(bhvID, bhv2ModelCodes);

                        colorsForPlot = arrayfun(@(x) colors(x,:), bhvID(allBhvModeled) + 2, 'UniformOutput', false);
                        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

                        figH = figHFullModel;
                        % Plot on second monitor, half-width
                        plotPos = [monitorTwo(1), 1, monitorTwo(3)/2, monitorTwo(4)];
                        % titleM = sprintf('%s %s All Frames %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, transWithinLabel, opts.frameSize, min_dist(x), spread(y), n_neighbors(z));
                        titleM = sprintf('%s %s All Frames %s bin=%.2f', selectFrom, fitType, transWithinLabel, opts.frameSize);
                        % titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' All Frames' ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
                        plotFrames = allBhvModeled;
                        plot_3d_scatter
                    end
                    copy_figure_to_clipboard

                    %% Plot data to model
                    if plotModelData
                        colorsForPlot = arrayfun(@(x) colors(x,:), svmID + 2, 'UniformOutput', false);
                        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

                        figH = figHModel;
                        % Plot on second monitor, half-width
                        plotPos = [monitorTwo(1) + monitorTwo(3)/2, 1, monitorTwo(3)/2, monitorTwo(4)];
                        % titleM = sprintf('%s %s Modeled Data %s bin=%.2f min_dist=%.2f spread=%.1f nn=%d', selectFrom, fitType, transWithinLabel, opts.frameSize, min_dist(x), spread(y), n_neighbors(z));
                        titleM = sprintf('%s %s Modeled Data %s bin=%.2f', selectFrom, fitType, transWithinLabel, opts.frameSize);
                        % titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
                        plotFrames = svmInd;
                        plot_3d_scatter

                    end






                    %%                  SVM classifier to predict behavior ID

                    %{
% Script to train an SVM model with different kernels and select the best model
nDim = 8;

tic

% Define different kernel functions to try
% kernelFunctions = {'linear', 'gaussian', 'polynomial', 'rbf'};
kernelFunctions = {'polynomial'};

% Initialize variables to store results
bestCVAccuracy = 0;
bestKernel = '';

% Perform cross-validation for each kernel
for i = 1:length(kernelFunctions)
    % Set SVM template with the current kernel
    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunctions{i});

    % Train the SVM model using cross-validation
    svmModel = fitcecoc(projSelect(svmInd,1:nDim), svmID, 'Learners', t, 'KFold', 5);
    % svmModel = fitcecoc(dataMat(svmInd,:), svmID, 'Learners', t, 'KFold', 5);

    % Compute cross-validation accuracy
    cvAccuracy = 1 - kfoldLoss(svmModel, 'LossFun', 'ClassifError');

    % Display the best model and its kernel function
    fprintf('Kernel: %s\n', kernelFunctions{i});
    fprintf('Cross-Validation Accuracy: %.2f%%\n', cvAccuracy * 100);

    % Check if this model is the best one so far
    if cvAccuracy > bestCVAccuracy
        bestCVAccuracy = cvAccuracy;
        bestModel = svmModel;
        bestKernel = kernelFunctions{i};
    end
end

% Display the best model and its kernel function
fprintf('%s Best Kernel for %s: %s\n', selectFrom, transWithinLabel, bestKernel);
fprintf('%s Best Cross-Validation %s Accuracy: %.2f%%\n', selectFrom, transWithinLabel, bestCVAccuracy * 100);

% The bestModel variable now contains the trained model with the best kernel
toc
load handel
sound(y(1:3*Fs),Fs)


%% Train model on all data - how well does it fit? (without cross-validation)
tic
nDim = 8;

% Define different kernel functions to try
% kernelFunctions = {'linear', 'gaussian', 'polynomial', 'rbf'};
kernelFunctions = {'polynomial'};
% kernelFunctions = {'rbf'};

% Choose which data to model
svmProj = projSelect(svmInd, 1:nDim);
% svmProj = dataMat(svmInd, idSelect);

% Train model
for i = 1:length(kernelFunctions)
    % Set SVM template with the current kernel
    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunctions{i});

    % Train the SVM model using cross-validation
    svmModel = fitcecoc(svmProj, svmID, 'Learners', t);
end
toc

predictedLabels = predict(svmModel, svmProj);

% Calculate and display the overall accuracy
accuracy = sum(predictedLabels == svmID) / length(svmID) * 100;
fprintf('%s %s Overall Accuracy: %.2f%%\n', selectFrom, transWithinLabel, accuracy);

load handel
sound(y(1:3*Fs),Fs)
                    %}
                    %% Train and test model on single hold-out set
                    appendModelName = selectFrom;


                    tic


                    % Split data into training (80%) and testing (20%) sets
                    cv = cvpartition(svmID, 'HoldOut', 0.2);


                    % UMAP dimension version
                    fprintf('\n\n%s %s %s %d Dim\n\n', selectFrom, lowDModel, transWithinLabel, iDim)  % UMAP Dimensions
                    % Choose which data to model
                    svmProj = projSelect(svmInd, :);
                    trainData = svmProj(training(cv), :);  % UMAP Dimensions
                    testData = svmProj(test(cv), :); % UMAP Dimensions


                    % % % Neural space version
                    % fprintf('\n\n%s %s Neural Space\n\n', selectFrom, transWithinLabel)  % Neural Space
                    % svmProj = zscore(dataMat(svmInd, idSelect));
                    % trainData = svmProj(training(cv), :);  % Neural Space
                    % testData = svmProj(test(cv), :); % Neural Space



                    trainLabels = svmID(training(cv));
                    testLabels = svmID(test(cv));


                    % Define different kernel functions to try
                    % kernelFunctions = {'linear', 'gaussian', 'polynomial', 'rbf'};
                    kernelFunction = 'polynomial';

                    % Train model
                    % Set SVM template with the current kernel
                    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

                    % Train the SVM model using cross-validation
                    svmModel = fitcecoc(trainData, trainLabels, 'Learners', t);
                    modelName = ['svmModel', appendModelName];
                    % Reassign the value of modelName to the new variable name using eval
                    eval([modelName, ' = svmModel;']);


                    predictedLabels = predict(svmModel, testData);

                    % Calculate and display the overall accuracy
                    accuracy(k) = sum(predictedLabels == testLabels) / length(testLabels);
                    fprintf('%s %s Overall Accuracy: %.4f\n', selectFrom, transWithinLabel, accuracy(k));

                    fprintf('Model fit took %.2f min\n', toc/60)

                    tic
                    % Randomize labels and Train model on single hold-out set
                    % tic
                    if nPermutations > 0
rng('shuffle'); % Seeds based on the current time

                        shuffleInd = zeros(length(trainLabels), nPermutations);
                        for iPerm = 1:nPermutations


                            % Shuffle the labels
                            % shuffledLabels = trainLabels(randperm(length(trainLabels)));

                            iRandom = 1 : length(trainLabels);

                            randShift = randi([1 length(trainLabels)]);
                            randShift = randShift + iPerm * 17;
                            % Shuffle the data by moving the last randShift elements to the front
                            lastNElements = iRandom(end - randShift + 1:end);  % Extract the last randShift elements
                            iRandom(randShift+1:end) = iRandom(1:end-randShift); % Shift the remaining elements to the end
                            iRandom(1:randShift) = lastNElements; % Place the last n elements at the beginning
                            shuffleInd(:, iPerm) = iRandom;
                            shuffledLabels = trainLabels(shuffleInd(:, iPerm));


                            % Set SVM template with the current kernel
                            t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

                            % Train the SVM model on shuffled training data
                            svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);

                            % Predict the labels using observed test data
                            predictedLabelsPermuted = predict(svmModelPermuted, testData);

                            % Calculate the permuted accuracy
                            accuracyPermuted(k, iPerm) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
                            fprintf('Permuted %s %s Overall Accuracy permutation %d: %.4f\n', selectFrom, transWithinLabel, iPerm, accuracyPermuted(k, iPerm));

                        end
                        modelName = ['svmModelPermuted', appendModelName];
                        % Reassign the value of modelName to the new variable name using eval
                        eval([modelName, ' = svmModelPermuted;']);

                    end

fprintf('Mean Permuted = %.4f\n', mean(accuracyPermuted(k, :)));
% Get the elapsed time
                    fprintf('Permutation model fit(s) took %.2f min\n', toc/60)

                    %     load handel
                    % sound(y(1:3*Fs),Fs)

                    modelAccuracy(k, x, y, z) = accuracy(k);
                    if nPermutations > 0
                        permAccuracy(k, x, y, z) = accuracyPermuted(k, iPerm);
                    end


                    %% Analzyze the predictions vs observed
                    if analyzePredictions
                        analyze_model_predictions
                    end

                    %% This needs to be updated  --------------------------------------------
                    % % Plot predictions
                    % colorsForPlot = arrayfun(@(x) colors(x,:), predictedLabels + 1, 'UniformOutput', false);
                    % colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
                    %
                    %
                    % figure(821); clf; hold on;
                    % titleD = ['Predicted UMAP ', selectFrom,' ', num2str(nUmapDim), 'D ', transWithinLabel, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
                    % title(titleD)
                    % if nUmapDim > 2
                    %     scatter3(projSelect(svmInd, dimPlot(1)), projSelect(svmInd, dimPlot(2)), projSelect(svmInd, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 2)
                    % elseif nUmapDim == 2
                    %     scatter(projSelect(svmInd, dimPlot(1)), projSelect(svmInd, dimPlot(2)), 60, projSelect, 'LineWidth', 2)
                    % end
                    % grid on;
                    % xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])
                    % saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')


                    % %% Randomize data to get chance level SVM accuracy
                    % svmIDRand = svmID(randperm(length(svmID)));
                    % % svmIndRand = svmInd(randperm(length(svmID)));
                    % kernelFunction = 'polynomial';
                    % % kernelFunction = 'rbf';
                    %
                    % t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);
                    %
                    % % Choose which data to model
                    % % svmProj = projSelect(svmInd, 1:3);
                    % % svmProj = projSelect(svmInd, :);
                    % svmProj = dataMat(svmInd, idSelect);
                    %
                    % % Train the SVM model
                    % svmModel = fitcecoc(svmProj, svmIDRand, 'Learners', t);
                    %
                    % % Use the best model to make predictions
                    % % svmID = svmID(svmIndRand);
                    %
                    % predictedLabels = predict(svmModel, svmProj);
                    %
                    % % Calculate and display the overall accuracy
                    % accuracy = sum(predictedLabels == svmIDRand) / length(svmIDRand) * 100;
                    % fprintf('%s Randomized %s Overall Accuracy: %.2f%%\n', selectFrom, transWithinLabel, accuracy);
                    % load handel
                    % sound(y(1:3*Fs),Fs)
                    % %% Expected accuracy for randomly choosing
                    %
                    % randomPredict = svmID(randperm(length(svmID)));
                    %
                    % accuracy = sum(randomPredict == svmID) / length(randomPredict) * 100
                    %
                    % % Count the occurrences of each unique number in the vector
                    % counts = histcounts(svmID, unique(svmID));
                    % %  Calculate the total number of elements
                    % totalElements = length(svmID);
                    %
                    % % Calculate the proportions
                    % proportions = counts / totalElements;
                    %
                    % % Calculate the expected accuracy
                    % expectedAccuracy = sum(proportions .^ 2);
                    % % Display the result
                    % fprintf('Expected Accuracy (Random Guessing): %.2f%%\n', expectedAccuracy * 100);






















                    %% =======================================================================================
                    %                   USE A PARTICULAR MODEL TO ANALYZE/PREDICT PARTICULAR
                    %                   TIME BINS (GOING INTO AND OUT OF BEHAVIORS)
                    % ========================================================================================

                    % Get data (top of script) if needed

                    %% Go back up top to Choose which area to select data from (to analyze clusters in that area, and/or to project same time points in another area


                    %% Go back up top if needed to (Re-)Train the particular model you want to test on other data


                    %% Use the model to predict single frames going into and out of behavior transitions
                    % This tests the transistions on ALL transitions into behaviors (so if
                    % you subsampled, downsampled, etc., it will test beyond only the data
                    % points used to generate the model. Thus, the accuracy my differ (be
                    % lower than in many cases) than the model tested on the test set.

                    if plotTransPred
                        fprintf('\nPredicting each frame going into transitions:\n')
                        frames = -2 : 4; % 2 frames pre to two frames post transition

                        svmIDTest = bhvID(preInd + 1);  % behavior ID being transitioned into
                        svmIndTest = preInd + 1;

                        % Get rid of behaviors you didn't model
                        rmvBhv = setdiff(unique(bhvID), bhv2ModelCodes);
                        deleteInd = ismember(svmIDTest, rmvBhv);
                        svmIDTest(deleteInd) = [];
                        svmIndTest(deleteInd) = [];


                        accuracyTrans = zeros(length(frames), 1);
                        accuracyPermutedTrans = zeros(length(frames), 1);
                        for iFrame = 1 : length(frames)
                            % Get relevant frame to test (w.r.t. transition frame)
                            iSvmInd = svmIndTest + frames(iFrame);
                            testData = projSelect(iSvmInd, 1:iDim); % UMAP Dimensions

                            % Calculate and display the overall accuracy (make sure it matches the
                            % original fits to ensure we're modeling the same way)
                            predictedLabels = predict(svmModel, testData);
                            accuracyTrans(iFrame) = sum(predictedLabels == svmIDTest) / length(svmIDTest);
                            fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(iFrame), accuracyTrans(iFrame));

                            % Calculate and display the overall accuracy (make sure it matches the
                            % original fits to ensure we're modeling the same way)
                            predictedLabelsPermuted = predict(svmModelPermuted, testData);
                            accuracyPermutedTrans(iFrame) = sum(predictedLabelsPermuted == svmIDTest) / length(svmIDTest);
                            % fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(iFrame), accuracyPermuted(iFrame));

                        end

                        goingInPosition = [monitorOne(3)/2, monitorOne(2), monitorOne(3)/3, monitorOne(4)/2.2];
                        fig = figure(67);
                        set(fig, 'Position', goingInPosition); clf; hold on;
                        % bar(codes(2:end), f1Score);
                        % xticks(0:codes(end))
                        plot(frames, accuracyTrans, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
                        plot(frames, accuracyPermutedTrans, '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
                        set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
                        set(findall(fig,'-property','Box'),'Box','off') % optional
                        xticks(frames)
                        ylim([0 1])
                        ylabel('Accuracy')
                        xlabel('Frames relative to transition')
                        xline(0)
                        titleE = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' Accuracy into transitions'];
                        title(titleE)
                        % saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')
                        print('-dpdf', fullfile(paths.figurePath, [titleE, '.pdf']), '-bestfit')

                    end
                    % %% Use the model to predict all within-behavior frames
                    %
                    % transIndLog = zeros(length(bhvID), 1);
                    % transIndLog(preInd) = 1;
                    %
                    % % If you want to remove another pre-behavior onset bin, do this:
                    % vec = find(transIndLog);
                    % transIndLog(vec-1) = 1;
                    %
                    % % If you want to remove a bin after behavior onset, do this:
                    % % transIndLog(vec+1) = 1;
                    %
                    % svmIndTest = find(~transIndLog);
                    % svmIDTest = bhvID(svmIndTest);
                    %
                    % % Get rid of behaviors you didn't model
                    % rmvBhv = setdiff(unique(bhvID), bhv2ModelCodes);
                    % deleteInd = ismember(svmIDTest, rmvBhv);
                    % svmIDTest(deleteInd) = [];
                    % svmIndTest(deleteInd) = [];
                    %
                    %
                    % testData = projSelect(svmIndTest, 1:iDim); % UMAP Dimensions
                    %
                    % predictedLabels = predict(svmModel, testData);
                    %
                    % % Calculate and display the overall accuracy (make sure it matches the
                    % % original fits to ensure we're modeling the same way)
                    % accuracy = sum(predictedLabels == svmIDTest) / length(svmIDTest);
                    % fprintf('%s Test Within-Behavior Overall Accuracy: %.4f%%\n', selectFrom, accuracy);




                end
            end
        end

    end



% figure(92); clf; hold on;
% plot(expVar, accuracy, '-b');
% plot(expVar, mean(accuracyPermuted, 1), '-r');
% plot(expVar, accuracy - mean(accuracyPermuted, 1), '--k');
% legend({'Accuracy', 'Permuted', 'Difference'})








% Plot the accuracy results per dimension



if analyzePredictions
    perDimPosition = [monitorOne(3)/2, monitorOne(4)/2.2, monitorOne(3)/3, monitorOne(4)/2.2];
    fig = figure(65);
    set(fig, 'Position', perDimPosition); clf; hold on;
    plot(forDim, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
    plot(forDim, mean(accuracyPermuted, 2), '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
    set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
    set(findall(fig,'-property','Box'),'Box','off') % optional
    xticks(forDim)
    ylim([0 1])
    ylabel('Accuracy')
    xlabel('Dimensions')
    titleE = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' Accuracy per dimensions'];
    title(titleE);
    % saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')
    print('-dpdf', fullfile(paths.figurePath, [titleE, '.pdf']), '-bestfit')
end


% load handel
% sound(y(1:round(2.2*Fs)),Fs)



slack_code_done












































%% --------------------------------------------
% Plot just the TRANSITIONS with individual behavior labels

behaviorsPlot = {'investigate_1', 'head_groom'};
behaviorsPlot = {'contra_itch', 'paw_groom'};
behaviorsPlot = {'locomotion', 'contra_orient', 'ipsi_orient'};
behaviorsPlot = {'contra_itch', 'rear'};
behaviorsPlot = {'investigate_2'};
% behaviorsPlot = {'paw_groom'};
behaviorsPlot = {'locomotion'};
% behaviorsPlot = {'face_groom_1'};
behaviorsPlot = {'contra_itch'};
% behaviorsPlot = {'investigate_2'};


colors = colors_for_behaviors(codes);
periEventTime = -opts.frameSize : opts.frameSize : 0; % seconds around onset
% periEventTime = -opts.frameSize : opts.frameSize : opts.frameSize; % seconds around onset
dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

figure(230); clf; hold on;
titleM = ['UMAP M56 ', num2str(iDim), 'D,  bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleM)
grid on;
xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])

figure(231); clf; hold on;
titleD = ['UMAP DS ', num2str(iDim), 'D, bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleD)
grid on;
xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])

for i = 1 : length(behaviorsPlot)
    bhvPlot = find(strcmp(behaviors, behaviorsPlot{i})) - 2;

    allInd = bhvID == bhvPlot; % all labeled target behaviors
    firstInd = 1 + find(diff(allInd) == 1); % first frames of all target behaviors

    transitionsInd = zeros(length(dataWindow) * length(firstInd), 1);
    for j = 1 : length(firstInd)
        % Calculate the start index in the expanded array
        startIndex = (j-1) * length(dataWindow) + 1;

        % Add dataWindow to the current element of firstInd and store it in the correct position
        transitionsInd(startIndex:startIndex + length(dataWindow) - 1) = firstInd(j) + dataWindow;
    end

    figure(230)
    if iDim > 2
        scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), projectionsM56(transitionsInd, dimPlot(3)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    elseif iDim == 2
        scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
        % plot3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, '-', 'color', [.6 .6 .6])
        % scatter(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    end

    figure(231)
    if iDim > 2
        scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), projectionsDS(transitionsInd, dimPlot(3)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    elseif iDim == 2
        scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
        % plot3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), '-', 'color', [.6 .6 .6])
        % scatter(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
    end
end

figure(230)
% saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')
figure(231)
% saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')


%% --------------------------------------------
% Compare, for a single behavior, the frame before and after behavior start
% time (frame)

behaviorsPlot = {'investigate_1', 'head_groom'};
behaviorsPlot = {'contra_itch', 'paw_groom'};
behaviorsPlot = {'locomotion', 'contra_orient', 'ipsi_orient'};
behaviorsPlot = {'contra_itch', 'rear'};
behaviorsPlot = {'investigate_2'};
% behaviorsPlot = {'paw_groom'};
behaviorsPlot = {'locomotion'};
behaviorsPlot = {'contra_itch'};


colors = [0 0 0; 1 0 0];
frameShift = [0 opts.frameSize];
periEventTime = -opts.frameSize : opts.frameSize : 0; % seconds around onset

figure(230); clf; hold on;
titleM = ['UMAP M56 ', num2str(iDim), 'D,  bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleM)
grid on;
xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])

figure(231); clf; hold on;
titleD = ['UMAP DS ', num2str(iDim), 'D, bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleD)
grid on;
xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])

for k = 1 : length(frameShift)
    periEventTime = periEventTime + frameShift(k);
    dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)


    for i = 1 : length(behaviorsPlot)
        bhvPlot = find(strcmp(behaviors, behaviorsPlot{i})) - 2;

        allInd = bhvID == bhvPlot; % all labeled target behaviors
        firstInd = 1 + find(diff(allInd) == 1); % first frames of all target behaviors

        transitionsInd = zeros(length(dataWindow) * length(firstInd), 1);
        for j = 1 : length(firstInd)
            % Calculate the start index in the expanded array
            startIndex = (j-1) * length(dataWindow) + 1;

            % Add dataWindow to the current element of firstInd and store it in the correct position
            transitionsInd(startIndex:startIndex + length(dataWindow) - 1) = firstInd(j) + dataWindow;
        end

        figure(230)
        if iDim > 2
            scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), projectionsM56(transitionsInd, dimPlot(3)), 60, colors(k,:), 'LineWidth', 2)
        elseif iDim == 2
            scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colors(k,:), 'LineWidth', 2)
            % plot3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, '-', 'color', [.6 .6 .6])
        end

        figure(231)
        if iDim > 2
            scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), projectionsDS(transitionsInd, dimPlot(3)), 60, colors(k,:), 'LineWidth', 2)
        elseif iDim == 2
            scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colors(k,:), 'LineWidth', 2)
            % plot3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), '-', 'color', [.6 .6 .6])
        end
    end
end
figure(230)
% saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')
figure(231)
% saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')


















%% Use HDBSCAN to select data points for analaysis. Play with this until you're happy with the groups
nDimUse = dimPlot; % use the first nDimUse dimensions for HDBSCAN

% Classify using HDBSCAN
clusterer = HDBSCAN(projSelect(transitionsInd, nDimUse));
clusterer.minpts = 2; %tends to govern cluster number  %was 3? with all neurons
clusterer.minclustsize = 10; %governs accuracy  %was 4? with all neurons

clusterer.fit_model(); 			% trains a cluster hierarchy
clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
clusterer.get_membership();		% assigns cluster labels to the points in X
figure(823);
h = clusterer.plot_clusters();
% Change appearance if you want
% h.Marker = 'o'
% set( h.Parent,'tickdir','out','box','off','color','k','xcolor','w','ycolor','w' );

% title(['umap HDBSCAN ,' area, ' binSize = ', num2str(binSize)])
% saveas(gcf, fullfile(paths.figurePath, ['umap HDBSCAN ,' area, ' binsize ', num2str(binSize), '.png']), 'png')

%% HDBSCAN defined some groups. Now select those group data from the umap projections
nGroup = single(max(clusterer.labels));
bhvFrame = cell(nGroup, 1);
plotTime = -3 : opts.frameSize : 3; % seconds around onset (to plot PSTHs below)
plotFrame = round(plotTime(1:end-1) / opts.frameSize);
matSelect = cell(nGroup, 1);

for i = 1 : nGroup

    % Choose only cluster of interest
    bhvFrame{i} = transitionsInd(clusterer.labels == i);
    % Get rid of instances too close to beginning/end (to enable plotting
    % them)
    bhvFrame{i}((bhvFrame{i} < -plotFrame(1)) | bhvFrame{i} > size(dataMat, 1) - plotFrame(end)) = [];

    % Collect the neural data peri-bout start
    nBout = length(bhvFrame{i});
    iMatSelect = zeros(length(idSelect), length(plotFrame), nBout);
    for n = 1 : nBout
        iMatSelect(:, :, n) = dataMat((bhvFrame{i}(n) - dataWindow) + plotFrame, idSelect)';
    end
    matSelect{i} = iMatSelect;

end



%% Alternative to HDBSCAN: Do it by hand.  Use a circle or sphere to select data points for analysis
clear center radius
% center{1} = [-.25 -.5];
% center{2} = [-4.5 1];
% center{3} = [.5 4];
center{1} = [2.25 1 7];
center{2} = [2 -.7 3.5];
center{3} = [0 1.5 5.7];
radius{1} = 1;
radius{2} = 1;
radius{3} = 1;
% center{1} = [-2.5 -8];
% center{2} = [0 0];
% center{3} = [1 -6];
% radius{1} = 2;
% radius{2} = 2;
% radius{3} = 2;

nGroup = length(center);

bhvFrame = cell(nGroup, 1);
prevBhv = cell(nGroup, 1);

% Number of points to define the circle
theta = linspace(0, 2*pi, 100);  % Generate 100 points along the circle

plotTime = -3 : opts.frameSize : 3; % seconds around onset (to plot PSTHs below)
plotFrame = round(plotTime(1:end-1) / opts.frameSize);
matSelect = cell(nGroup, 1);

for i = 1 : nGroup
    if strcmp(selectFrom, 'M56')
        figure(230)
    else
        figure(231)
    end

    if length(center{1}) == 2
        % Parametric equation of the circle
        x = center{i}(1) + radius{i} * cos(theta);
        y = center{i}(2) + radius{i} * sin(theta);
        plot(x, y, 'color', colors(i,:), 'lineWidth', 4);  % Plot the circle with a blue line

        % Calculate the Euclidean distance between the point and the center of the circle
        iDistance = sqrt((projSelect(:, 1) - center{i}(1)).^2 + (projSelect(:, 2) - center{i}(2)).^2);
    elseif length(center{1}) == 3
        [x, y, z] = sphere(50);
        x = x * radius{i} + center{i}(1);  % Scale and translate the sphere
        y = y * radius{i} + center{i}(2);  % Scale and translate the sphere
        z = z * radius{i} + center{i}(3);  % Scale and translate the sphere

        surf(x, y, z, 'FaceColor', colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');  % Translucent sphere

        % Calculate the Euclidean distance between the point and the center of the circle
        iDistance = sqrt((projSelect(:, 1) - center{i}(1)).^2 + (projSelect(:, 2) - center{i}(2)).^2 + (projSelect(:, 3) - center{i}(3)).^2);
    end

    % Determine if the point is inside the circle (including on the boundary)
    isInside = iDistance <= radius{i};

    % Choose only behavior of interest
    bhvInd = zeros(length(projSelect), 1);
    bhvInd(transitionsInd) = 1;


    bhvFrame{i} = find(isInside & bhvInd);
    % Get rid of instances too close to beginning/end (to enable plotting
    % them)
    bhvFrame{i}((bhvFrame{i} < -plotFrame(1)) | bhvFrame{i} > size(dataMat, 1) - plotFrame(end)) = [];

    prevBhv{i} = bhvID(bhvFrame{i});

    % Plot the data in the other area
    if strcmp(selectFrom, 'M56')
        figure(231)
    else
        figure(230)
    end
    scatter3(projProject(bhvFrame{i}, dimPlot(1)), projProject(bhvFrame{i}, dimPlot(2)), opts.frameSize * (bhvFrame{i}-1), 60, colors(i,:), 'LineWidth', 2)

    % Collect the neural data peri-bout start
    nBout = length(bhvFrame{i});
    iMatSelect = zeros(length(idSelect), length(plotFrame), nBout);
    for n = 1 : nBout
        iMatSelect(:, :, n) = dataMat((bhvFrame{i}(n) - dataWindow) + plotFrame, idSelect)';
    end
    matSelect{i} = iMatSelect;
end

n = 1:10;
% t = seconds([bhvTime1(n) bhvTime2(n) bhvTime3(n) bhvTime4(n) bhvTime5(n)] .* opts.frameSize);
t = seconds([bhvFrame{1}(n) bhvFrame{2}(n) bhvFrame{3}(n)] .* opts.frameSize);
t.Format = 'hh:mm:ss.S'





%% Want to get all the behaviors just before this one, for each group in UMAP space
figure(11)
for i = 1 : nGroup
    [uniqueElements, ~, idx] = unique(prevBhv{i});
    counts = accumarray(idx, 1);

    % Create the bar graph
    subplot(1, nGroup, i);  % Open a new figure window
    bar(uniqueElements, counts, 'FaceColor', colors(i,:));  % Create a bar plot
end


%% Plot PSTHS of the various groups
figure(3235);

allMat = [];
for i = 1 : nGroup
    allMat = cat(3, allMat, matSelect{i});
end
meanMat = mean(allMat, 3);
meanWindow = mean(meanMat, 2);
stdWindow = std(meanMat, [], 2);

zMat = cell(nGroup, 1);
zMeanMat = cell(nGroup, 1);
for i = 1 : nGroup

    % z-score the psth data
    zMat{i} = (matSelect{i} - meanWindow) ./ stdWindow;
    zMeanMat{i} = mean(zMat{i}, 3);
    subplot(1, nGroup, i)
    % imagesc(mean(matSelect{i}, 3))
    imagesc(zMeanMat{i})
    colormap(bluewhitered_custom([-4 4]))
    title(num2str(i))

end
% colorbar

figure(423);
diffMat12 = zMeanMat{1} - zMeanMat{2};
diffMat13 = zMeanMat{1} - zMeanMat{3};
diffMat23 = zMeanMat{2} - zMeanMat{3};
% diffMat12 = mean(matSelect{1}, 3) - mean(matSelect{2}, 3);
% diffMat13 = mean(matSelect{1}, 3) - mean(matSelect{3}, 3);
% diffMat23 = mean(matSelect{2}, 3) - mean(matSelect{3}, 3);
colorBarSpan = [-4 4];
subplot(1,3,1)
imagesc(diffMat12)
colormap(bluewhitered_custom(colorBarSpan))
title('1 - 2')
subplot(1,3,2)
imagesc(diffMat13)
colormap(bluewhitered_custom(colorBarSpan))
title('1 - 3')
subplot(1,3,3)
imagesc(diffMat23)
colormap(bluewhitered_custom(colorBarSpan))
title('2 - 3')


testFrame = (-plotFrame(1) + 1) + (-1:0);
zTest1 = mean(zMeanMat{1}(:, testFrame), 2);
zTest2 = mean(zMeanMat{2}(:, testFrame), 2);
zTest3 = mean(zMeanMat{3}(:, testFrame), 2);
[h,p] = ttest(zTest1, zTest2)
[h,p] = ttest(zTest1, zTest3)
[h,p] = ttest(zTest2, zTest3)














%%                  RUN UMAP / HDBSCAN clusters in GPFA
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build a set of neural matrices for training gpfa

% Go through the data and get all the relevant indices in dataBhv
nTrial = min(cellfun(@length, bhvFrame)); % Use same amount of trials per "condition"
dataBhvInd = cell(length(bhvFrame), 1);
for i = 1 : length(bhvFrame)
    bhvFrame{i} = bhvFrame{i}(randperm(length(bhvFrame{i}), nTrial));
    dataBhvInd{i} = zeros(nTrial, 1);
    for j = 1 : nTrial
        % Find the dataBhv index for this particular start time
        dataBhvInd{i}(j) = find(strcmp(dataBhv.Name, behaviorsPlot) & dataBhv.StartFrame == (bhvFrame{i}(j)+1));
    end
end

%% Get a new dataMat and dataBvh for gpfa
opts.frameSize = .001;
get_standard_data
cd('E:/Projects/toolboxes/gpfa_v0203/')
startup


%% Find the start times of the behaviors in .001 frame-time
clear dat
preTime = .2;
postTime = .2;
periEventTime = -preTime : opts.frameSize : postTime; % seconds around onset
dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)

for i = 1 : length(bhvFrame)
    for j = 1 : nTrial
        % Find the dataBhv index for this particular start time
        jStartFrame = floor(dataBhv.StartTime(dataBhvInd{i}(j)) / opts.frameSize);
        trialIdx = (i-1) * nTrial + j;
        dat(trialIdx).trialId = i;
        dat(trialIdx).spikes = dataMat(jStartFrame + dataWindow, idSelect)';
    end
end


%%
method = 'gpfa';
runIdx = 1;
binWidth = 20; % ms
startInd = ceil(1000* preTime / binWidth); % When the behavior starts (first frame)

% Select number of latent dimensions
xDim = 8;
kernSD = 30;

% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', binWidth);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.



%% Plot the gpfa fits
figure(430); clf; grid on; hold on
xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])
for i = 1 : length(seqTrain)
    trialID = seqTrain(i).trialId;
    plot3(seqTrain(i).xorth(1,:), seqTrain(i).xorth(2,:), seqTrain(i).xorth(3,:), '.-', 'Color', colors(trialID,:), 'LineWidth', 2, 'MarkerSize', 10')
    scatter3(seqTrain(i).xorth(1,1), seqTrain(i).xorth(2,1), seqTrain(i).xorth(3,1), 40, colors(trialID,:), 'filled')
    scatter3(seqTrain(i).xorth(1,startInd), seqTrain(i).xorth(2,startInd), seqTrain(i).xorth(3,startInd), 100, colors(trialID,:), 'filled')
end





























%% ====================================================================================
% =====           Test clusters of within-behavior frames             ===============
% ====================================================================================

% Plot all behaviors in light gray, full time of a behavior in black, and transitions into that behavior in green (or some other color)

behaviorsPlot = {'locomotion'};
% behaviorsPlot = {'contra_itch'};
% behaviorsPlot = {'investigate_2'};
% behaviorsPlot = {'face_groom_2'};

% i = 1;
% for i = 2 : length(behaviors)
% behaviorsPlot = behaviors(i);

% --------------------------------------------
% Plot FULL TIME OF ALL BEHAVIORS - in light gray, sampling every X
dimPlot = [1 2 3];

colorsForPlot = [.8 .8 .8];
colorsForPlot = [.5 .5 .5];
alphaVal = .3;
propSample = .5; % subsample the data to keep the plot clean (but still sample the whole space)
randInd = randperm(length(bhvID), floor(length(bhvID) * propSample));

figure(520); clf; hold on;
titleM = [behaviorsPlot{1}, '  UMAP M56 ', num2str(iDim), 'D,  bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleM, 'interpreter', 'none')
if iDim > 2
    scatter3(projectionsM56(randInd, dimPlot(1)), projectionsM56(randInd, dimPlot(2)), projectionsM56(randInd, dimPlot(3)), 40, colorsForPlot, 'MarkerEdgeAlpha', alphaVal)
elseif iDim == 2
    scatter(projectionsM56(randInd, dimPlot(1)), projectionsM56(randInd, dimPlot(2)), 60, colorsForPlot, 'LineWidth', 1, 'MarkerEdgeAlpha', alphaVal)
end
grid on;
xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])
% saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')


% figure(221+i); clf; hold on;
figure(521); clf; hold on;
titleD = [behaviorsPlot{1}, '  UMAP DS ', num2str(iDim), 'D, bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
title(titleD, 'interpreter', 'none')
if iDim > 2
    scatter3(projectionsDS(randInd, dimPlot(1)), projectionsDS(randInd, dimPlot(2)), projectionsDS(randInd, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 1, 'MarkerEdgeAlpha', alphaVal)
elseif iDim == 2
    scatter(projectionsDS(randInd, dimPlot(1)), projectionsDS(randInd, dimPlot(2)), 40, colorsForPlot, 'LineWidth', 1, 'MarkerEdgeAlpha', alphaVal)
end
grid on;
xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])
% saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')
%


%

% --------------------------------------------
% Plot FULL TIME OF Single behavior

colors = colors_for_behaviors(codes);
bhvColor = find(strcmp(behaviors, behaviorsPlot));
fullColor = [0 0 0];

bhvPlot = find(strcmp(behaviors, behaviorsPlot)) - 2;
allInd = bhvID == bhvPlot; % all labeled target behaviors

figure(520);
if iDim > 2
    scatter3(projectionsM56(allInd, dimPlot(1)), projectionsM56(allInd, dimPlot(2)), projectionsM56(allInd, dimPlot(3)), 60, fullColor, 'LineWidth', 2)
elseif iDim == 2
    scatter(projectionsM56(allInd, dimPlot(1)), projectionsM56(allInd, dimPlot(2)), 60, fullColor, 'LineWidth', 2)
end
% saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')


figure(521);
if iDim > 2
    scatter3(projectionsDS(allInd, dimPlot(1)), projectionsDS(allInd, dimPlot(2)), projectionsDS(allInd, dimPlot(3)), 60, fullColor, 'LineWidth', 2)
elseif iDim == 2
    scatter(projectionsDS(allInd, dimPlot(1)), projectionsDS(allInd, dimPlot(2)), 60, fullColor, 'LineWidth', 2)
end
% saveas(gcf, fullfile(paths.figurePath, [titleD, '.png']), 'png')


% --------------------------------------------
% Plot just the TRANSITIONS with individual behavior labels

startColor = [1 0 .5];
startColor = [0 .9 0];
periEventTime = -opts.frameSize : opts.frameSize : 0; % seconds around onset
dataWindow = floor(periEventTime(1:end-1) / opts.frameSize); % frames around onset (remove last frame)


firstInd = 1 + find(diff(allInd) == 1); % first frames of all target behaviors

transitionsInd = zeros(length(dataWindow) * length(firstInd), 1);
for j = 1 : length(firstInd)
    % Calculate the start index in the expanded array
    startIndex = (j-1) * length(dataWindow) + 1;

    % Add dataWindow to the current element of firstInd and store it in the correct position
    transitionsInd(startIndex:startIndex + length(dataWindow) - 1) = firstInd(j) + dataWindow;
end
colorsForPlot = arrayfun(@(x) colors(x,:), bhvID(transitionsInd) + 2, 'UniformOutput', false);
colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

figure(520)
if iDim > 2
    scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), projectionsM56(transitionsInd, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 2)
elseif iDim == 2
    scatter3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colorsForPlot, 'LineWidth', 2)
    % plot3(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, '-', 'color', [.6 .6 .6])
    % scatter(projectionsM56(transitionsInd, dimPlot(1)), projectionsM56(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
end

figure(521)
if iDim > 2
    scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), projectionsDS(transitionsInd, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 2)
elseif iDim == 2
    scatter3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), 60, colorsForPlot, 'LineWidth', 2)
    % plot3(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), opts.frameSize * (transitionsInd-1), '-', 'color', [.6 .6 .6])
    % scatter(projectionsDS(transitionsInd, dimPlot(1)), projectionsDS(transitionsInd, dimPlot(2)), 60, colors(bhvPlot+2,:), 'LineWidth', 2)
end

% end

%% Choose which area to select data from (to analyze clusters in that area, and/or to project same time points in another area
selectFrom = 'M56';
selectFrom = 'DS';
switch selectFrom
    case 'M56'
        projSelect = projectionsM56;
        projProject = projectionsDS;
        idSelect = idM56;
    case 'DS'
        projSelect = projectionsDS;
        projProject = projectionsM56;
        idSelect = idDS;
end


%%                  SVM classifier to predict transitions from all behaviors into single behavior X
% Get rid of "irrelevant/in-nest"
% svmInd = withinID ~= -1;
svmInd = transitionsInd(bhvID(transitionsInd) ~= -1);

% Define different kernel functions to try
% kernelFunctions = {'linear', 'gaussian', 'polynomial', 'rbf'};
kernelFunctions = {'polynomial', 'rbf'};
% kernelFunctions = {'polynomial'};

% Initialize variables to store results
bestCVAccuracy = 0;
bestKernel = '';

% Perform cross-validation for each kernel
for i = 1:length(kernelFunctions)
    % Set SVM template with the current kernel
    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunctions{i});

    % Train the SVM model using cross-validation
    svmModel = fitcecoc(projSelect(svmInd,:), bhvID(svmInd), 'Learners', t, 'KFold', 5);

    % Compute cross-validation accuracy
    cvAccuracy = 1 - kfoldLoss(svmModel, 'LossFun', 'ClassifError');

    % Display the best model and its kernel function
    fprintf('Kernel: %s\n', kernelFunctions{i});
    fprintf('Cross-Validation Accuracy: %.2f%%\n', cvAccuracy * 100);

    % Check if this model is the best one so far
    if cvAccuracy > bestCVAccuracy
        bestCVAccuracy = cvAccuracy;
        bestModel = svmModel;
        bestKernel = kernelFunctions{i};
    end
end

% Display the best model and its kernel function
fprintf('Best Kernel for Within Behaviors: %s\n', bestKernel);
fprintf('Best Cross-Validation Accuracy: %.2f%%\n', bestCVAccuracy * 100);



%% Use HDBSCAN to select data points for analaysis. Play with this until you're happy with the groups
nDimUse = dimPlot; % use the first nDimUse dimensions for HDBSCAN

% Classify using HDBSCAN
clusterer = HDBSCAN(projSelect(allInd, 1:8));
clusterer.minpts = 2; %tends to govern cluster number  %was 3? with all neurons
clusterer.minclustsize = 100; %governs accuracy  %was 4? with all neurons
clusterer.minClustNum = 2; % force at least this many clusters

clusterer.fit_model(); 			% trains a cluster hierarchy
clusterer.get_best_clusters(); 	% finds the optimal "flat" clustering scheme
clusterer.get_membership();		% assigns cluster labels to the points in X
figure(823);
h = clusterer.plot_clusters();
% Change appearance if you want
% h.Marker = 'o'
% set( h.Parent,'tickdir','out','box','off','color','k','xcolor','w','ycolor','w' );

% title(['umap HDBSCAN ,' area, ' binSize = ', num2str(binSize)])
% saveas(gcf, fullfile(paths.figurePath, ['umap HDBSCAN ,' area, ' binsize ', num2str(binSize), '.png']), 'png')


%% HDBSCAN defined some groups. Now select those group data from the umap projections
nGroup = single(max(clusterer.labels));
clusterFrame = cell(nGroup, 1);
clusterSpikes = cell(nGroup, 1);
% plotTime = -3 : opts.frameSize : 3; % seconds around onset (to plot PSTHs below)
% plotFrame = round(plotTime(1:end-1) / opts.frameSize);
matSelect = cell(nGroup, 1);

figure(92); clf; hold on;
for i = 1 : nGroup

    % Choose only cluster of interest
    iFrames = find(allInd);
    clusterFrame{i} = iFrames(clusterer.labels == i);
    clusterSpikes{i} = dataMat(clusterFrame{i}, idSelect);
    % scatter(clusterFrame{i}, mean(clusterSpikes{i}(:)))
    scatter(1, mean(clusterSpikes{i}(:)))

end




%% Alternative: Do it by hand.  Use a circle or sphere to select data points for analysis (instead of HDBSCAN)
clear center radius
% center{1} = [-.25 -.5];
% center{2} = [-4.5 1];
% center{3} = [.5 4];
center{1} = [.5 -1.5 -6.5];
center{2} = [-1.5 .5 -5.5];
center{3} = [1 -1.5 -3.25];
% center{1} = [-2.5 -8];
% center{2} = [0 0];
% center{3} = [1 -6];
radius{1} = 1.2;
radius{2} = .7;
radius{3} = 1;

% Choose colors: alter this to change how many colors (clusters) you want to use
colors = three_color_heatmap([1 0 0],[0 .7 0], [0 0 1], 3);

nGroup = length(center);

clusterFrame = cell(nGroup, 1);
clusterSpikes = cell(nGroup, 1);

% Number of points to define the circle
theta = linspace(0, 2*pi, 100);  % Generate 100 points along the circle

matSelect = cell(nGroup, 1);

for i = 1 : nGroup
    if strcmp(selectFrom, 'M56')
        figure(520)
    else
        figure(521)
    end

    if length(center{1}) == 2
        % Parametric equation of the circle
        x = center{i}(1) + radius{i} * cos(theta);
        y = center{i}(2) + radius{i} * sin(theta);
        plot(x, y, 'color', colors(i,:), 'lineWidth', 4);  % Plot the circle with a blue line

        % Calculate the Euclidean distance between the point and the center of the circle
        iDistance = sqrt((projSelect(:, 1) - center{i}(1)).^2 + (projSelect(:, 2) - center{i}(2)).^2);
    elseif length(center{1}) == 3
        [x, y, z] = sphere(50);
        x = x * radius{i} + center{i}(1);  % Scale and translate the sphere
        y = y * radius{i} + center{i}(2);  % Scale and translate the sphere
        z = z * radius{i} + center{i}(3);  % Scale and translate the sphere

        surf(x, y, z, 'FaceColor', colors(i,:), 'FaceAlpha', 0.5, 'EdgeColor', 'none');  % Translucent sphere

        % Calculate the Euclidean distance between the point and the center of the circle
        iDistance = sqrt((projSelect(:, 1) - center{i}(1)).^2 + (projSelect(:, 2) - center{i}(2)).^2 + (projSelect(:, 3) - center{i}(3)).^2);
    end

    % Determine if the point is inside the circle (including on the boundary)
    isInside = iDistance <= radius{i};

    % Choose only behavior of interest
    bhvInd = zeros(length(projSelect), 1);
    bhvInd(allInd) = 1;

    clusterFrame{i} = find(isInside & bhvInd);
    clusterSpikes{i} = dataMat(clusterFrame{i}, idSelect);

    % Plot the data in the other area
    if strcmp(selectFrom, 'M56')
        figure(521)
    else
        figure(520)
    end
    scatter3(projProject(bhvFrame{i}, dimPlot(1)), projProject(bhvFrame{i}, dimPlot(2)), opts.frameSize * (bhvFrame{i}-1), 60, colors(i,:), 'LineWidth', 2)

end




%% Repeated measures ANOVA across groups
meanSpikes = cellfun(@mean, clusterSpikes, 'UniformOutput', false);
stdSpikes = cellfun(@std, clusterSpikes, 'UniformOutput', false);

% spikeData = [];
% clusterID = [];
% neuronID = [];
% for i = 1 : length(meanSpikes)
%     spikeData = [spikeData; meanSpikes{i}'];
%     clusterID = [clusterID; i * ones(length(meanSpikes{i}), 1)];
%     neuronID = [neuronID; idSelect'];
% end

% figure(987); clf; hold on;
% for i = 1 : length(meanSpikes)
% plot(meanSpikes{i})
% end

%% Correlation structures
edges = -.6 : .02 : .6;
binCenters = (edges(1:end-1) + edges(2:end)) / 2;
returnIdx = tril(true(length(idSelect)), -1);
figure(54); clf; hold on;
colors = {'r', 'g', 'b'};
for i = 1 : length(meanSpikes)
    rho{i} = corr(clusterSpikes{i});
    rhoPop{i} = rho{i}(returnIdx);
    N = histcounts(rhoPop{i}, edges, 'Normalization', 'pdf');
    bar(binCenters, N, colors{i}, 'FaceAlpha', .5);%, 'hist')
end
cellfun(@mean, rhoPop)

%%
dataTbl = table([1:length(rhoPop{1})]', ...
    'VariableNames', {'neuronID'});
for i = 1 : length(meanSpikes)
    iName = ['cluster', num2str(i)];
    dataTbl.(iName) = rhoPop{i};
end

% Define the within-subject design
within = table([1:length(meanSpikes)]', ...
    'VariableNames', {'clusterID'});

varNames = dataTbl.Properties.VariableNames;
modelStr = [varNames{2},'-',varNames{end},'~1'];
% Fit the repeated measures model
rm = fitrm(dataTbl, modelStr, 'WithinDesign', within);

% Perform the repeated measures ANOVA
ranovatbl = ranova(rm);

% Display the results
disp(ranovatbl);

%%
dataTbl = table(idSelect', ...
    'VariableNames', {'neuronID'});
for i = 1 : length(meanSpikes)
    iName = ['cluster', num2str(i)];
    % dataTbl.(iName) = meanSpikes{i}';
    dataTbl.(iName) = stdSpikes{i}';
end

% Define the within-subject design
within = table([1:length(meanSpikes)]', ...
    'VariableNames', {'clusterID'});

varNames = dataTbl.Properties.VariableNames;
modelStr = [varNames{2},'-',varNames{end},'~1'];
% Fit the repeated measures model
rm = fitrm(dataTbl, modelStr, 'WithinDesign', within);

% Perform the repeated measures ANOVA
ranovatbl = ranova(rm);

% Display the results
disp(ranovatbl);







%%                  RUN UMAP / clusters (HDBSCAN or by hand) in GPFA
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build a set of neural matrices for training gpfa

%% Get a new dataMat and dataBvh for gpfa
opts.frameSize = .001;
get_standard_data
cd('E:/Projects/toolboxes/gpfa_v0203/')
startup


%% Concatenate frames into gpfa rate (.001s per frame)
clear dat
trialDur = .4; % How long to make each "trial" for GPFA
nTrial = floor(min(cellfun(@length, clusterFrame)) * umapFrameSize / trialDur); % Use same amount of trials per "condition"
% nTrial = min(40, nTrial);
umapFramePerTrial = trialDur / umapFrameSize;

dataBhvInd = cell(length(clusterFrame), 1);
for i = 1 : length(clusterFrame)
    clusterFrameSample{i} = clusterFrame{i}(randperm(length(clusterFrame{i}), nTrial));

    for j = 1 : nTrial
        trialData = [];
        for k = 1 : umapFramePerTrial
            % convert UMAP frame to .001 s frame for gpfa
            gpfaFrame = clusterFrameSample{i}(j) * umapFrameSize / .001;
            trialData = [trialData; dataMat(gpfaFrame : gpfaFrame + umapFrameSize / .001 - 1, idSelect)];
        end
        trialIdx = (i-1) * nTrial + j;
        dat(trialIdx).trialId = i;
        dat(trialIdx).spikes = trialData';


    end
end





%%
method = 'gpfa';
runIdx = 1;
binWidth = 20; % ms

% Select number of latent dimensions
xDim = 8;
kernSD = 30;

% Extract neural trajectories
result = neuralTraj(runIdx, dat, 'method', method, 'xDim', xDim,...
    'kernSDList', kernSD, 'seglength', 50, 'binWidth', binWidth);
% NOTE: This function does most of the heavy lifting.

% Orthonormalize neural trajectories
[estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
% NOTE: The importance of orthnormalization is described on
%       pp.621-622 of Yu et al., J Neurophysiol, 2009.



%% Plot the gpfa fits
figure(430); clf; grid on; hold on
xlabel(['D', num2str(dimPlot(1))]); ylabel(['D', num2str(dimPlot(2))]); zlabel(['D', num2str(dimPlot(3))])
for i = 1 : length(seqTrain)
    trialID = seqTrain(i).trialId;
    plot3(seqTrain(i).xorth(dimPlot(1),:), seqTrain(i).xorth(dimPlot(2),:), seqTrain(i).xorth(dimPlot(3),:), '.-', 'Color', colors(trialID,:), 'LineWidth', 2, 'MarkerSize', 10')
end





%%
i = 3
figure(13);
% plot(diff(bhvFrame{i}))
plot(bhvFrame{i})

sum(diff(bhvFrame{i}) > 1)
sum(diff(bhvFrame{i}) == 1)









