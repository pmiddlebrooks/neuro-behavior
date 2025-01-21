%%                     Compare neuro-behavior in PCA spaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 60 * 60; % seconds
opts.frameSize = .1;

getDataType = 'spikes';
get_standard_data

colors = colors_for_behaviors(codes);

[dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);


%% for plotting consistency
%
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



%%
bhvLabels = {'investigate_1', 'investigate_2', 'investigate_3', ...
    'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
    'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
    'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};












%% =============================================================================
% --------    RUN PCA SVM FITS FOR VARIOUS CONDITIONS ON VARIOUS DATA
% =============================================================================

% Select which data to run analyses on, pca dimensions, etc

forDim = 3:8; % Loop through these dimensions to fit pca
forDim = [3 8]; % Loop through these dimensions to fit UMAP
newPcaModel = 0; % Do we need to get a new pca model to analyze (or did you tweak some things that come after pca?)
usePCAFromMeans = 1;

% Change these (and check their sections below) to determine which
% variables to test
% ==========================

% Modeling variables
nPermutations = 2; % How many random permutations to run to compare with best fit model?
accuracy = zeros(length(forDim), 1);
accuracyPermuted = zeros(length(forDim), nPermutations);

% Apply to all:
% -------------
plotFullMap = 0;
plotFullModelData = 0;
plotModelData = 1;
plotTransitions = 0;
changeBhvLabels = 0;

% Transition or within variables
% -------------------------
% transOrWithin = 'trans';
transOrWithin = 'within';
% transOrWithin = 'transVsWithin';
matchTransitionCount = 0;
minFramePerBout = 0;

% Apply to all:
% --------------
collapseBhv = 0;
minBoutNumber = 0;
downSampleBouts = 0;
minFrames = 0;
downSampleFrames = 0;


selectFrom = 'M56';
% selectFrom = 'DS';
% selectFrom = 'Both';
% selectFrom = 'VS';
% selectFrom = 'All';
switch selectFrom
    case 'M56'
        % projSelect = projectionsM56;
        % projProject = projectionsDS;
        idSelect = idM56;
        figHFull = 260;
        figHModel = 270;
        figHFullModel = 280;
    case 'DS'
        % projSelect = projectionsDS;
        % projProject = projectionsM56;
        idSelect = idDS;
        figHFull = 261;
        figHModel = 271;
        figHFullModel = 281;
    case 'Both'
        % projSelect = [projectionsM56; projectionsDS];
        idSelect = [idM56, idDS];
        figHFull = 262;
        figHModel = 272;
        figHFullModel = 282;
    case 'VS'
        % projSelect = projectionsVS;
        % projProject = projectionsDS;
        idSelect = idVS;
        figHFull = 263;
        figHModel = 273;
        figHFullModel = 283;
    case 'All'
        % projSelect = projectionsAll;
        % projProject = projectionsDS;
        idSelect = cell2mat(idAll);
        figHFull = 264;
        figHModel = 274;
        figHFullModel = 284;
end


% Some figure properties
allFontSize = 12;


% Run PCA to get projections in low-D space
if newPcaModel
    % rng(1);
    [coeff, score, ~, ~, explained] = pca(dataMat(:, idSelect));
    % [coeff, score, ~, ~, explained] = pca(zscore(dataMat(:, idSelect)));
end


%%
for k = 1:length(forDim)
    iDim = forDim(k);
    fitType = ['PCA ', num2str(iDim), 'D'];
    % fitType = 'NeuralSpace';

if usePCAFromMeans
    warning('You are using PCA projections from mean neural activity within each bout')
    projSelect = dataMat(:,idSelect) * coeff;
projSelect = projSelect(:,1:iDim);
else
        projSelect = score(:, 1:iDim);
end


    %% --------------------------------------------
    % Shift behavior label w.r.t. neural to account for neuro-behavior latency
    shiftSec = 0;
    shiftFrame = ceil(shiftSec / opts.frameSize);
    bhvID = double(bhvID(1+shiftFrame:end)); % Shift bhvIDMat to account for time shift


    projSelect = projSelect(1:end-shiftFrame, :); % Remove shiftFrame frames from projections to accoun for time shift in bhvIDMat





    %% --------------------------------------------
    % Plot FULL TIME OF ALL BEHAVIORS
    if plotFullMap
        colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
        % colorsForPlot = [.2 .2 .2];
        figH = figHFull;
        plotPos = [monitorOne(1), 1, monitorOne(3)/2, monitorOne(4)];
        titleM = [selectFrom, ' ', fitType, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
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
            svmInd = preInd;% + 1; % First bin after transition

            % Pre & Post: Comment/uncomment to use more than one bin
            % svmID = repelem(svmID, 2);
            % svmInd = sort([svmInd - 1; svmInd]); % two bins before transition
            % svmInd = sort([svmInd; svmInd + 1]); % Last bin before transition and first bin after

            transWithinLabel = 'transitions pre';
            % transWithinLabel = 'transitions 200ms pre';
            % transWithinLabel = 'transitions post';
            % transWithinLabel = 'transitions pre & post';
            % transWithinLabel = ['transitions pre minBout ', num2str(nMinFrames)];


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



            if changeBhvLabels
                % transWithinLabel = [transWithinLabel, ' short inv2-loco as new bhv'];
                transWithinLabel = [transWithinLabel, ' short loco as inv2'];
                % transWithinLabel = [transWithinLabel, ' long loco as inv2'];
                % transWithinLabel = 'within-behavior xxxxx';
            end
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
        titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' All Frames' ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
        plotFrames = allBhvModeled;
        plot_3d_scatter
    end

    %% Plot data to model
    if plotModelData
        colorsForPlot = arrayfun(@(x) colors(x,:), svmID + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

        figH = figHModel;
        % Plot on second monitor, half-width
        plotPos = [monitorTwo(1) + monitorTwo(3)/2, 1, monitorTwo(3)/2, monitorTwo(4)];
        titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];

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

    disp('=================================================================')

    % pca dimension version
    fprintf('\n\n%s %s DIMENSIONS %d Explained %.2f\n\n', selectFrom, transWithinLabel, iDim, sum(explained(1:iDim)))  % pca Dimensions
    % Choose which data to model
    svmProj = projSelect(svmInd, :);
    trainData = svmProj(training(cv), :);  % pca Dimensions
    testData = svmProj(test(cv), :); % pca Dimensions


    % % Neural space version
    % fprintf('\n\n%s %s Neural Space\n\n', selectFrom, transWithinLabel)  % Neural Space
    % svmProj = dataMat(svmInd, idSelect);
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
    fprintf('%s %s Overall Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy(k));

    fprintf('Model fit took %.2f min\n', toc/60)

    tic
    % Randomize labels and Train model on single hold-out set
    % tic
    shuffleInd = zeros(length(trainLabels), nPermutations);
    for iPerm = 1:nPermutations


        % Shuffle the labels
        % shuffledLabels = trainLabels(randperm(length(trainLabels)));

        iRandom = 1 : length(trainLabels);

        randShift = randi([1 length(trainLabels)]);
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
        fprintf('Permuted %s %s Overall Accuracy permutation %d: %.4f%%\n', selectFrom, transWithinLabel, k, accuracyPermuted(k, iPerm));

    end
    modelName = ['svmModelPermuted', appendModelName];
    % Reassign the value of modelName to the new variable name using eval
    eval([modelName, ' = svmModelPermuted;']);



    % Get the elapsed time
    fprintf('Permutation model fit(s) took %.2f min\n', toc/60)

    %     load handel
    % sound(y(1:3*Fs),Fs)




    %% Analzyze the predictions vs observed

    analyze_model_predictions


    %% This needs to be updated  --------------------------------------------
    % % Plot predictions
    % colorsForPlot = arrayfun(@(x) colors(x,:), predictedLabels + 1, 'UniformOutput', false);
    % colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
    %
    %
    % figure(821); clf; hold on;
    % titleD = ['Predicted pca ', selectFrom,' ', num2str(niDim), 'D ', transWithinLabel, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
    % title(titleD)
    % if niDim > 2
    %     scatter3(projSelect(svmInd, dimPlot(1)), projSelect(svmInd, dimPlot(2)), projSelect(svmInd, dimPlot(3)), 60, colorsForPlot, 'LineWidth', 2)
    % elseif niDim == 2
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

    if plotTransitions
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
            testData = projSelect(iSvmInd, 1:iDim); % pca Dimensions

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
    % testData = projSelect(svmIndTest, 1:iDim); % pca Dimensions
    %
    % predictedLabels = predict(svmModel, testData);
    %
    % % Calculate and display the overall accuracy (make sure it matches the
    % % original fits to ensure we're modeling the same way)
    % accuracy = sum(predictedLabels == svmIDTest) / length(svmIDTest);
    % fprintf('%s Test Within-Behavior Overall Accuracy: %.4f%%\n', selectFrom, accuracy);





end

























%% Get mean of within-trial behaivors and check how low-D it is (averaging removes bout-by-bout "noise")

% Test normal PCA (on all data) to get that explained variance:
idSelect = idM56;

preInd = [diff(bhvID) ~= 0; 0]; % 1 frame prior to all behavior transitions
withinInd = ~preInd & ~[0; preInd(1:end-1)];
modelID = bhvID(find(preInd)+1);

bhvCheck = 15;
bhvInd = withinInd & bhvID == bhvCheck;

[coeff, score, ~, ~, explained] = pca(dataMat(bhvInd, idSelect));

%%
[stackedActivity, stackedLabels] = datamat_stacked_means(dataMat, bhvID);


%
[coeff, score, ~, ~, explained] = pca(zscore(stackedActivity(:,idSelect)));

figure(); plot(cumsum(explained));

% modelData = nanmean(neuralDataByBout(1:10, idSelect, :), 3);
% 
% [coeff, score, ~, ~, explained] = pca(zscore(modelData));
