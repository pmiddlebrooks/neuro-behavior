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

% for plotting consistency
%
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



%
bhvLabels = {'investigate_1', 'investigate_2', 'investigate_3', ...
    'rear', 'dive_scrunch', 'paw_groom', 'face_groom_1', 'face_groom_2', ...
    'head_groom', 'contra_body_groom', 'ipsi_body groom', 'contra_itch', ...
    'ipsi_itch_1', 'contra_orient', 'ipsi_orient', 'locomotion'};
%
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
nModel = 20;

% Apply to all:
% -------------
plotFullMap = 1;
plotFullModelData = 0;
plotModelData = 0;
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


third = size(dataMat, 1) / 3;
frames1 = 1:third;
frames2 = third + 1:third*2;
frames3 = third*2 + 1:size(dataMat, 1);

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



projSegment{1} = score(:, 1:iDim);
projSegment{2} = dataMat(frames2, idSelect) * coeff(:,1:iDim);
projSegment{3} = dataMat(frames3, idSelect) * coeff(:,1:iDim);
bhvIDSegment{1} = bhvID(frames1);
bhvIDSegment{2} = bhvID(frames2);
bhvIDSegment{3} = bhvID(frames3);


accuracy = zeros(nModel, length(bhvIDSegment));
accuracyPermuted = accuracy;


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
for i = 1 : length(bhvIDSegment)

    preInd{i} = find(diff(bhvIDSegment{i}) ~= 0); % 1 frame prior to all behavior transitions

    switch transOrWithin
        case 'trans'
            % TRANSITIONS of all behaviors (for now, include behaviors that last one frame)
            % svmID: vector of category labels for each data point to be analyzed/fit

            svmID{i} = bhvIDSegment{i}(preInd{i} + 1);  % behavior ID being transitioned into

            % Pre and/or Post: Adjust which bin(s) to plot (and train SVN on below)
            svmInd{i} = preInd{i};% + 1; % First bin after transition

            % Pre & Post: Comment/uncomment to use more than one bin
            svmID{i} = repelem(svmID{i}, 2);
            % svmInd{i} = sort([svmInd{i} - 1; svmInd{i}]); % two bins before transition
            svmInd{i} = sort([svmInd{i}; svmInd{i} + 1]); % Last bin before transition and first bin after

            % transWithinLabel = 'transitions pre';
            % transWithinLabel = 'transitions 200ms pre';
            % transWithinLabel = 'transitions post';
            transWithinLabel = 'transitions pre & post';
            % transWithinLabel = ['transitions pre minBout ', num2str(nMinFrames)];

            warning('Some indices may be multiply labeled, b/c some behaviors last only one frame')

            % WITHIN-BEHAVIOR of all behaviors (for now, include behaviors that last one frame)
        case 'within'

            transIndLog{i} = zeros(length(bhvIDSegment{i}), 1);
            transIndLog{i}(preInd{i}) = 1;

            % If you want to remove another pre-behavior onset bin, do this:
            vec = find(transIndLog{i});
            transIndLog{i}(vec-1) = 1;

            % If you want to remove a bin after behavior onset, do this:
            % transIndLog{i}(vec+1) = 1;

            svmInd{i} = find(~transIndLog{i});
            svmID{i} = bhvIDSegment{i}(svmInd{i});

            % choose correct title
            transWithinLabel = 'within-behavior';

    end


    % Get rid of sleeping/in_nest/irrelavents
    deleteInd1 = svmID{i} == -1;
    svmID{i}(deleteInd1) = [];
    svmInd{i}(deleteInd1) = [];





    svmIDModel{i} = svmID{i};
    svmIndModel{i} = svmInd{i};

end



% REMOVE OF ENTIRE BEHAVIORS WITH UNDER A MINIMUM NUMBER OF FRAMES/DATA POINTS
if minFrames
    nMinFrames = 200;
    % nMinFrames = 400;

    for i = 1 : length(bhvIDSegment)
        [uniqueVals, ~, idx] = unique(svmIDModel{i}); % Find unique integers and indices
        bhvDataCount{i} = accumarray(idx, 1); % Count occurrences of each unique integer
    end

    rmvBehaviors =  uniqueVals(any(cell2mat(bhvDataCount) < nMinFrames, 2));

    for i = 1 : length(bhvIDSegment)
        iRmvBhv = find(ismember(svmID{i}, rmvBehaviors));

        svmIDModel{i}(iRmvBhv) = [];
        svmIndModel{i}(iRmvBhv) = [];
    end
    %     transWithinLabel = [transWithinLabel, ', minTotalFrames ', num2str(nMinFrames)];
end




% Loop of nModel models/predictions
for m = 1 : nModel
    % Subsample frames so all behaviors have same # of data points
    % This randomizes the set of data points to model for all but the most
    % prevalent behavior
    if downSampleFrames
        for i = 1 : length(bhvIDSegment)
            % subsampling to match single frame transition number
            [uniqueVals, ~, idx] = unique(svmIDModel{i}); % Find unique integers and indices
            frameCounts = accumarray(idx, 1); % Count occurrences of each unique integer
            downSample = min(frameCounts(frameCounts > 0));
            for iBhv = 1 : length(frameCounts)
                iBhvInd = find(svmIDModel{i} == uniqueVals(iBhv));
                if ~isempty(iBhvInd)
                    nRemove = length(iBhvInd) - downSample;
                    rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
                    svmIDModel{i}(rmvBhvInd) = [];
                    svmIndModel{i}(rmvBhvInd) = [];
                end
            end
            % transWithinLabel = [transWithinLabel, ', downsample to ', num2str(downSample), ' frames'];
        end
    end




    %% Plot data to model
    if plotModelData
        colorsForPlot = arrayfun(@(x) colors(x,:), svmIDModel{1} + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

        figH = figHModel;
        % Plot on second monitor, half-width
        plotPos = [monitorTwo(1) + monitorTwo(3)/2, 1, monitorTwo(3)/2, monitorTwo(4)];
        titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' bin=', num2str(opts.frameSize), ' shift=', num2str(shiftSec)];
        projSelect = projSegment{1};
        plotFrames = svmIndModel{1};
        plot_3d_scatter

    end






    %%                  SVM classifier to predict behavior ID

    % Train and test model on first segment of data, then predict on the other segments
    appendModelName = selectFrom;


    tic

    % Split data into training (80%) and testing (20%) sets
    cv = cvpartition(svmIDModel{1}, 'HoldOut', 0.2);

    disp('=================================================================')
fprintf('\t\tModel iteration %d\n', m)

    % pca dimension version
    fprintf('\n\n%s %s DIMENSIONS %d Explained %.2f\n\n', selectFrom, transWithinLabel, iDim, sum(explained(1:iDim)))  % pca Dimensions
    % Choose which data to model
    svmProj = projSegment{1}(svmIndModel{1}, :);
    trainData = svmProj(training(cv), :);  % pca Dimensions
    testData = svmProj(test(cv), :); % pca Dimensions


    trainLabels = svmIDModel{1}(training(cv));
    testLabels = svmIDModel{1}(test(cv));

    % Define different kernel functions to try
    kernelFunction = 'polynomial';

    % Train model
    % Set SVM template with the current kernel
    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

    % Train the SVM model using cross-validation
    svmModel = fitcecoc(trainData, trainLabels, 'Learners', t);


    % Calculate and display the overall accuracy
    predictedLabels{1} = predict(svmModel, testData);
    accuracy(m, 1) = sum(predictedLabels{1} == testLabels) / length(testLabels);
    fprintf('%s %s Segment 1 Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy(m, 1));
    for i = 2 : length(bhvIDSegment)
        predictedLabels{i} = predict(svmModel, projSegment{i}(svmIndModel{i}, :));
        accuracy(m, i) = sum(predictedLabels{i} == svmIDModel{i}) / length(svmIndModel{i});
        % accuracy(m, i) = sum(predictedLabels{i} == svmIndModel{i}) / length(svmIndModel{i});
        fprintf('%s %s Segment %d Accuracy: %.4f%%\n', selectFrom, transWithinLabel, i, accuracy(m, i));
    end

    fprintf('Model fit took %.2f min\n', toc/60)



tic
    % Permute the data from the first segment, fit the permuted data, and
    % predict the other segments
    % Shuffle the labels from each third and make a prediction
    iRandom = 1 : length(svmIDModel{1});

    randShift = randi([1 length(svmIndModel{1})]);
    % Shuffle the labels by moving the last randShift elements to the front
    lastNElements = iRandom(end - randShift + 1:end);  % Extract the last randShift elements
    iRandom(randShift+1:end) = iRandom(1:end-randShift); % Shift the remaining elements to the end
    iRandom(1:randShift) = lastNElements; % Place the last n elements at the beginning
    shuffleInd = iRandom;
    svmID1Shuffle = svmIDModel{1}(shuffleInd);

    trainLabelsPerm = svmID1Shuffle(training(cv));
    testLabelsPerm = svmID1Shuffle(test(cv));

    % Set SVM template with the current kernel
    t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

    % Train the SVM model on shuffled training data
    svmModelPermuted = fitcecoc(trainData, trainLabelsPerm, 'Learners', t);

    % Predict the labels using observed test data
    predictedLabelsPermuted{1} = predict(svmModelPermuted, testData);

    % Calculate the permuted accuracy
    accuracyPermuted(m, 1) = sum(predictedLabelsPermuted{1} == testLabelsPerm) / length(testLabelsPerm);
    % randomBhvAll = ones(length(svmIDModel{i}), 1) * 15;
    % accuracyPermuted(m, 1) = sum(predictedLabelsPermuted == randomBhvAll) / length(svmID1Shuffle);
    fprintf('Permuted %s %s Segment 1 Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracyPermuted(m, 1));
    for i = 2 : length(bhvIDSegment)
        predictedLabelsPermuted{i} = predict(svmModelPermuted, projSegment{i}(svmIndModel{i}, :));
        accuracyPermuted(m, i) = sum(predictedLabelsPermuted{i} == svmIDModel{i}) / length(svmIndModel{i});
        fprintf('Permuted %s %s Segment %d Accuracy: %.4f%%\n', selectFrom, transWithinLabel, i, accuracyPermuted(m, i));
    end

    fprintf('Permutation fit took %.2f min\n', toc/60)









end























%%
% accuracyDS = accuracy;
% accuracyPermutedDS = accuracyPermuted;
% %
accuracyMS = accuracy;
accuracyPermutedMS = accuracyPermuted;

%%
accuracy = accuracyDS;
accuracyPermuted = accuracyPermutedDS;
selectFrom = 'DS';
%%
accuracy = accuracyMS;
accuracyPermuted = accuracyPermutedMS;
selectFrom = 'M56'

%% Plot the accuracy per recording duration
fun = @sRGB_to_OKLab;
colors = maxdistcolor(size(accuracy, 1),fun);

figure(44); clf; hold on;
for m = 1 : nModel
    % plot(1:3, accuracy(m, :), 'or', 'color', colors(m,:), 'lineWidth', 3, 'markerSize', 10)
    scatter(1:3, accuracy(m, :), 120, 'MarkerFaceColor', 'r', 'lineWidth', 1, 'markerEdgeColor', 'k', 'MarkerFaceAlpha', .5)
    scatter(1:3, accuracyPermuted(m, :), 120, 'MarkerFaceColor',  [.7 .7 .7], 'lineWidth', 1, 'markerEdgeColor', 'k', 'MarkerFaceAlpha', .5)
    % plot(1:3, accuracyPermuted(m, :), '.', 'color', [.7 .7 .7], 'lineWidth', 2, 'markerSize', 25)
end
% yline(1/16, '--', 'lineWidth', 3)
ylabel('Accuracy')
xlabel('Session Thirds')
xlim([.8 3.2])
ylim([0 .35])
xticks(1:3)
yticks(0:.1:.3)
figure_pretty_things
titleSave = [selectFrom, ' ', transWithinLabel, ' PCA Decoding Accuracy'];
title(titleSave)
saveas(gcf, fullfile(paths.figurePath, titleSave) , 'pdf');
legend('Predicted', 'Shuffled')



