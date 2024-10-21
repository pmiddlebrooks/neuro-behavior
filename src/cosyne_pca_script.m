%%                     Compare neuro-behavior in PCA spaces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 210 * 60; % seconds
opts.frameSize = .1;

getDataType = 'all';
get_standard_data

colors = colors_for_behaviors(codes);

%%
third = size(dataMat, 1) / 3;
frames1 = 1:third;
frames2 = third + 1:third*2;
frames3 = third*2 + 1:size(dataMat, 1);
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
%%
[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);

bhvID = double(bhvIDMat); % Shift bhvIDMat to account for time shift











%% =============================================================================
% --------    RUN PCA SVM FITS FOR VARIOUS CONDITIONS ON VARIOUS DATA
% =============================================================================

% Select which data to run analyses on, pca dimensions, etc

forDim = 3:8; % Loop through these dimensions to fit pca
forDim = 8; % Loop through these dimensions to fit UMAP
newPcaModel = 1; % Do we need to get a new pca model to analyze (or did you tweak some things that come after pca?)


% Change these (and check their sections below) to determine which
% variables to test
% ==========================

% Modeling variables
nPermutations = 2; % How many random permutations to run to compare with best fit model?
accuracy = zeros(length(forDim), 1);
accuracyPermuted = zeros(length(forDim), nPermutations);

% Apply to all:
% -------------
plotFullMap = 1;
plotFullModelData = 0;
plotModelData = 1;
plotTransitions = 0;
changeBhvLabels = 0;

% Transition or within variables
% -------------------------
transOrWithin = 'trans';
% transOrWithin = 'within';
% transOrWithin = 'transVsWithin';
matchTransitionCount = 0;
minFramePerBout = 0;

% Apply to all:
% --------------
collapseBhv = 0;
minBoutNumber = 0;
downSampleBouts = 0;
minFrames = 1;
downSampleFrames = 1;


selectFrom = 'M56';
selectFrom = 'DS';
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
end


% Some figure properties
allFontSize = 12;


% Run PCA to get projections in low-D space for the first third
if newPcaModel
    % rng(1);
    [coeff, score, ~, ~, explained] = pca(dataMat(frames1, idSelect));
end

k = 1;
iDim = 8;
% for k = 1:length(forDim)
% iDim = forDim(k);
fitType = ['PCA ', num2str(iDim), 'D'];
% fitType = 'NeuralSpace';


projSelect1 = score(:, 1:iDim);
projSelect2 = dataMat(frames2, idSelect) * coeff(:,1:iDim);
projSelect3 = dataMat(frames3, idSelect) * coeff(:,1:iDim);
bhvID1 = bhvID(frames1);
bhvID2 = bhvID(frames2);
bhvID3 = bhvID(frames3);



%
% %% --------------------------------------------
%  % Plot FULL TIME OF ALL BEHAVIORS
%  if plotFullMap
%      colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + 2, 'UniformOutput', false);
%      colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix
%      % colorsForPlot = [.2 .2 .2];
%      figH = figHFull;
%      plotPos = [monitorOne(1), 1, monitorOne(3)/2, monitorOne(4)];
%      titleM = [selectFrom, ' ', fitType, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
%      plotFrames = 1:length(bhvID);
%      plot_3d_scatter
%  end
%
%




%% --------------------------------------------------------------------------------------------------------------
% ---------------   TRANSITIONS OR WITHIN-BEHAVIOR     ----------------------------------------------------------
% --------------------------------------------------------------------------------------------------------------
% Find all time bins preceding all behavior transitions:

preInd1 = find(diff(bhvID1) ~= 0); % 1 frame prior to all behavior transitions
preInd2 = find(diff(bhvID2) ~= 0); % 1 frame prior to all behavior transitions
preInd3 = find(diff(bhvID3) ~= 0); % 1 frame prior to all behavior transitions


switch transOrWithin
    case 'trans'
        % TRANSITIONS of all behaviors (for now, include behaviors that last one frame)
        % svmID: vector of category labels for each data point to be analyzed/fit

        svmID1 = bhvID1(preInd1 + 1);  % behavior ID being transitioned into
        svmID2 = bhvID2(preInd2 + 1);  % behavior ID being transitioned into
        svmID3 = bhvID3(preInd3 + 1);  % behavior ID being transitioned into

        % Pre and/or Post: Adjust which bin(s) to plot (and train SVN on below)
        svmInd1 = preInd1;% + 1; % First bin after transition
        svmInd2 = preInd2;% + 1; % First bin after transition
        svmInd3 = preInd3;% + 1; % First bin after transition

        % Pre & Post: Comment/uncomment to use more than one bin
        svmID1 = repelem(svmID1, 2);
        svmID2 = repelem(svmID2, 2);
        svmID3 = repelem(svmID3, 2);
        % svmInd1 = sort([svmInd1 - 1; svmInd1]); % two bins before transition
        % svmInd2 = sort([svmInd2 - 1; svmInd1]); % two bins before transition
        % svmInd3 = sort([svmInd3 - 1; svmInd3]); % two bins before transition
        svmInd1 = sort([svmInd1; svmInd1 + 1]); % Last bin before transition and first bin after
        svmInd2 = sort([svmInd2; svmInd2 + 1]); % Last bin before transition and first bin after
        svmInd3 = sort([svmInd3; svmInd3 + 1]); % Last bin before transition and first bin after

        % transWithinLabel = 'transitions pre';
        % transWithinLabel = 'transitions 200ms pre';
        % transWithinLabel = 'transitions post';
        transWithinLabel = 'transitions pre & post';
        % transWithinLabel = ['transitions pre minBout ', num2str(nMinFrames)];


        % WITHIN-BEHAVIOR of all behaviors (for now, include behaviors that last one frame)
    case 'within'

        transIndLog1 = zeros(length(bhvID1), 1);
        transIndLog1(preInd1) = 1;
        transIndLog2 = zeros(length(bhvID2), 1);
        transIndLog2(preInd2) = 1;
        transIndLog3 = zeros(length(bhvID3), 1);
        transIndLog3(preInd3) = 1;

        % If you want to remove another pre-behavior onset bin, do this:
        vec = find(transIndLog1);
        transIndLog1(vec-1) = 1;
        vec = find(transIndLog2);
        transIndLog2(vec-1) = 1;
        vec = find(transIndLog3);
        transIndLog3(vec-1) = 1;

        % If you want to remove a bin after behavior onset, do this:
        % transIndLog1(vec+1) = 1;
        % transIndLog2(vec+1) = 1;
        % transIndLog3(vec+1) = 1;

        svmInd1 = find(~transIndLog1);
        svmID1 = bhvID1(svmInd1);
        svmInd2 = find(~transIndLog2);
        svmID2 = bhvID2(svmInd2);
        svmInd3 = find(~transIndLog3);
        svmID3 = bhvID3(svmInd3);

        % choose correct title
        transWithinLabel = 'within-behavior';

end


% Get rid of sleeping/in_nest/irrelavents
deleteInd1 = svmID1 == -1;
svmID1(deleteInd1) = [];
svmInd1(deleteInd1) = [];
deleteInd2 = svmID2 == -1;
svmID2(deleteInd2) = [];
svmInd2(deleteInd2) = [];
deleteInd3 = svmID3 == -1;
svmID3(deleteInd3) = [];
svmInd3(deleteInd3) = [];

warning('Some indices may be multiply labeled, b/c some behaviors last only one frame')

% Loop of nModel models/predictions
nModel = 20;
for m = 1 : nModel
    svmID1Model = svmID1;
    svmInd1Model = svmInd1;
    svmID2Model = svmID2;
    svmInd2Model = svmInd2;
    svmID3Model = svmID3;
    svmInd3Model = svmInd3;

    %% REMOVE OF ENTIRE BEHAVIORS WITH UNDER A MINIMUM NUMBER OF FRAMES/DATA POINTS
    if minFrames
        nMinFrames = 300;

        [uniqueVals, ~, idx] = unique(svmID1Model); % Find unique integers and indices
        bhvDataCount1 = accumarray(idx, 1); % Count occurrences of each unique integer
        [uniqueVals, ~, idx] = unique(svmID2Model); % Find unique integers and indices
        bhvDataCount2 = accumarray(idx, 1); % Count occurrences of each unique integer
        [uniqueVals, ~, idx] = unique(svmID3Model); % Find unique integers and indices
        bhvDataCount3 = accumarray(idx, 1); % Count occurrences of each unique integer

        % bhvDataCount = histcounts(svmID, (min(bhvID)-0.5):(max(bhvID)+0.5));
        rmvBehaviors = uniqueVals(bhvDataCount1 < nMinFrames | bhvDataCount2 < nMinFrames | bhvDataCount3 < nMinFrames);

        rmvBhvInd1 = find(ismember(svmID1, rmvBehaviors));
        rmvBhvInd2 = find(ismember(svmID2, rmvBehaviors));
        rmvBhvInd3 = find(ismember(svmID3, rmvBehaviors));

        svmID1Model(rmvBhvInd1) = [];
        svmInd1Model(rmvBhvInd1) = [];
        svmID2Model(rmvBhvInd2) = [];
        svmInd2Model(rmvBhvInd2) = [];
        svmID3Model(rmvBhvInd3) = [];
        svmInd3Model(rmvBhvInd3) = [];

    %     transWithinLabel = [transWithinLabel, ', minTotalFrames ', num2str(nMinFrames)];
    end
    %% Subsample frames so all behaviors have same # of data points
    if downSampleFrames
        % subsampling to match single frame transition number
        [uniqueVals, ~, idx] = unique(svmID1Model); % Find unique integers and indices
        frameCounts = accumarray(idx, 1); % Count occurrences of each unique integer
        downSample = min(frameCounts(frameCounts > 0));
        for iBhv = 1 : length(frameCounts)
            iBhvInd = find(svmID1Model == uniqueVals(iBhv));
            if ~isempty(iBhvInd)
                nRemove = length(iBhvInd) - downSample;
                rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
                svmID1Model(rmvBhvInd) = [];
                svmInd1Model(rmvBhvInd) = [];
            end
        end
        % transWithinLabel = [transWithinLabel, ', downsample to ', num2str(downSample), ' frames'];

        % subsampling to match single frame transition number
        [uniqueVals, ~, idx] = unique(svmID2Model); % Find unique integers and indices
        frameCounts = accumarray(idx, 1); % Count occurrences of each unique integer
        downSample = min(frameCounts(frameCounts > 0));
        for iBhv = 1 : length(frameCounts)
            iBhvInd = find(svmID2Model == uniqueVals(iBhv));
            if ~isempty(iBhvInd)
                nRemove = length(iBhvInd) - downSample;
                rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
                svmID2Model(rmvBhvInd) = [];
                svmInd2Model(rmvBhvInd) = [];
            end
        end

        % subsampling to match single frame transition number
        [uniqueVals, ~, idx] = unique(svmID3Model); % Find unique integers and indices
        frameCounts = accumarray(idx, 1); % Count occurrences of each unique integer
        downSample = min(frameCounts(frameCounts > 0));
        for iBhv = 1 : length(frameCounts)
            iBhvInd = find(svmID3Model == uniqueVals(iBhv));
            if ~isempty(iBhvInd)
                nRemove = length(iBhvInd) - downSample;
                rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
                svmID3Model(rmvBhvInd) = [];
                svmInd3Model(rmvBhvInd) = [];
            end
        end

    end



    %% If you set any behavior/data curation flags:
    % modify_data_frames_to_model





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



    %% Plot data to model
    if plotModelData
        colorsForPlot = arrayfun(@(x) colors(x,:), svmID1Model + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

        figH = figHModel;
        % Plot on second monitor, half-width
        plotPos = [monitorTwo(1) + monitorTwo(3)/2, 1, monitorTwo(3)/2, monitorTwo(4)];
        titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' bin=', num2str(opts.frameSize), ' shift=', num2str(shiftSec)];
projSelect = projSelect1;
        plotFrames = svmInd1Model;
        plot_3d_scatter

    end






    %%                  SVM classifier to predict behavior ID


    %% Train and test model on single hold-out set
    appendModelName = selectFrom;


    tic


    % Split data into training (80%) and testing (20%) sets
    cv = cvpartition(svmID1Model, 'HoldOut', 0.2);

    disp('=================================================================')

    % pca dimension version
    fprintf('\n\n%s %s DIMENSIONS %d Explained %.2f\n\n', selectFrom, transWithinLabel, iDim, sum(explained(1:iDim)))  % pca Dimensions
    % Choose which data to model
    svmProj = projSelect1(svmInd1Model, :);
    trainData = svmProj(training(cv), :);  % pca Dimensions
    testData = svmProj(test(cv), :); % pca Dimensions

testData(1:10)

    % % Neural space version
    % fprintf('\n\n%s %s Neural Space\n\n', selectFrom, transWithinLabel)  % Neural Space
    % svmProj = dataMat(svmInd, idSelect);
    % trainData = svmProj(training(cv), :);  % Neural Space
    % testData = svmProj(test(cv), :); % Neural Space



    trainLabels = svmID1Model(training(cv));
    testLabels = svmID1Model(test(cv));

testLabels(1:10)
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


    predictedLabels1 = predict(svmModel, testData);
    predictedLabels2 = predict(svmModel, projSelect2(svmInd2Model, :));
    predictedLabels3 = predict(svmModel, projSelect3(svmInd3Model, :));

    % Calculate and display the overall accuracy
    accuracy1(m) = sum(predictedLabels1 == testLabels) / length(testLabels);
    accuracy2(m) = sum(predictedLabels2 == svmID2Model) / length(svmID2Model);
    accuracy3(m) = sum(predictedLabels3 == svmID3Model) / length(svmID3Model);
    fprintf('%s %s First 3rd Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy1(m));
    fprintf('%s %s Second 3rd Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy2(m));
    fprintf('%s %s Third 3rd Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy3(m));

    fprintf('Model fit took %.2f min\n', toc/60)

    % tic
    % % Randomize labels and Train model on single hold-out set
    % % tic
    % shuffleInd = zeros(length(trainLabels), nPermutations);
    % for iPerm = 1:nPermutations
    %
    %
    %     % Shuffle the labels
    %     % shuffledLabels = trainLabels(randperm(length(trainLabels)));
    %
    %     iRandom = 1 : length(trainLabels);
    %
    %     randShift = randi([1 length(trainLabels)]);
    %     % Shuffle the data by moving the last randShift elements to the front
    %     lastNElements = iRandom(end - randShift + 1:end);  % Extract the last randShift elements
    %     iRandom(randShift+1:end) = iRandom(1:end-randShift); % Shift the remaining elements to the end
    %     iRandom(1:randShift) = lastNElements; % Place the last n elements at the beginning
    %     shuffleInd(:, iPerm) = iRandom;
    %     shuffledLabels = trainLabels(shuffleInd(:, iPerm));
    %
    %
    %     % Set SVM template with the current kernel
    %     t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);
    %
    %     % Train the SVM model on shuffled training data
    %     svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);
    %
    %     % Predict the labels using observed test data
    %     predictedLabelsPermuted = predict(svmModelPermuted, testData);
    %
    %     % Calculate the permuted accuracy
    %     accuracyPermuted(k, iPerm) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
    %     fprintf('Permuted %s %s Overall Accuracy permutation %d: %.4f%%\n', selectFrom, transWithinLabel, k, accuracyPermuted(k, iPerm));
    %
    % end
    % modelName = ['svmModelPermuted', appendModelName];
    % % Reassign the value of modelName to the new variable name using eval
    % eval([modelName, ' = svmModelPermuted;']);
    %
    %
    %
    % % Get the elapsed time
    % fprintf('Permutation model fit(s) took %.2f min\n', toc/60)

    %     load handel
    % sound(y(1:3*Fs),Fs)
end



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





% end
