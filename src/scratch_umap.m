%%

preInd = find(diff(bhvID) ~= 0); % 1 frame prior to all behavior transitions


% % TRANSITIONS of all behaviors (for now, include behaviors that last one frame)
% % svmID: vector of category labels for each data point to be analyzed/fit
% 
% svmID = bhvID(preInd + 1);  % behavior ID being transitioned into
% 
% % Pre and/or Post: Adjust which bin(s) to plot (and train SVN on below)
% svmInd = preInd; % + 1; % First bin after transition
% 
% % Pre & Post: Comment/uncomment to use more than one bin
% % svmID = [svmID; svmID];
% % svmInd = [svmInd - 1; svmInd]; % two bins before transition
% % svmInd = [svmInd; svmInd + 1]; % Last bin before transition and first bin after
% 
% transWithinLabel = 'transitions pre';
% % transWithinLabel = 'transitions 200ms pre';
% % transWithinLabel = 'transitions pre & post';



% WITHIN-BEHAVIOR of all behaviors (for now, include behaviors that last one frame)

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





% Get rid of sleeping/in_nest/irrelavents
deleteInd = svmID == -1;
svmID(deleteInd) = [];
svmInd(deleteInd) = [];



% Keep track of the behavior IDs you end up using
bhv2ModelCodes = unique(svmID);
bhv2ModelNames = behaviors(bhv2ModelCodes+2);







% Choose which area to select data from (to analyze clusters in that area, and/or to project same time points in another area
selectFrom = 'M56';
switch selectFrom
    case 'M56'
        projSelect = projectionsM56;
        projProject = projectionsDS;
        idSelect = idM56;
end



% Train and test model on single hold-out set
appendModelName = selectFrom;

dimToModel = 2:2:8;
nPermutations = 1;

accuracy = zeros(length(dimToModel), 1);
accuracyPermuted = zeros(length(dimToModel), nPermutations);

for n = 1 : length(dimToModel)
    tic


    % Split data into training (80%) and testing (20%) sets
    cv = cvpartition(svmID, 'HoldOut', 0.2);

    disp('=================================================================')

    % UMAP dimension version
    fprintf('\n\n%s %s DIMENSIONS 1 - %d\n\n', selectFrom, transWithinLabel, dimToModel(n))  % UMAP Dimensions
    % Choose which data to model
    svmProj = projSelect(svmInd, 1:dimToModel(n));
    trainData = svmProj(training(cv), :);  % UMAP Dimensions
    testData = svmProj(test(cv), :); % UMAP Dimensions


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
    accuracy(n) = sum(predictedLabels == testLabels) / length(testLabels);
    fprintf('%s %s Overall Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy(n));

    toc/60

    % Randomize labels and Train model on single hold-out set
    tic
    for i = 1:nPermutations

        % Shuffle the labels
        shuffledLabels = trainLabels(randperm(length(trainLabels)));

        % Set SVM template with the current kernel
        t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

        % Train the SVM model on shuffled training data
        svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);

        % Predict the labels using observed test data
        predictedLabelsPermuted = predict(svmModelPermuted, testData);

        % Calculate the permuted accuracy
        accuracyPermuted(n, i) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
        fprintf('Permuted %s %s Overall Accuracy permutation %d: %.4f%%\n', selectFrom, transWithinLabel, i, accuracyPermuted(n, i));

    end
    modelName = ['svmModelPermuted', appendModelName];
    % Reassign the value of modelName to the new variable name using eval
    eval([modelName, ' = svmModelPermuted;']);



    % Get the elapsed time
    toc/60
end
load handel
sound(y,Fs)



% Plot the accuracy results per dimension
figure(65); clf; hold on;
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
plot(dimToModel, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
plot(dimToModel, mean(accuracyPermuted, 2), '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
xticks(dimToModel)
ylim([0 1])
ylabel('Accuracy')
xlabel('Dimensions')
title(['Accuracy across dimensions', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy across dimensions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Analzyze the predictions vs observed


monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(54); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(bhv2ModelCodes);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.04 .02], .1);

edges = -.5 : 1 : bhv2ModelCodes(end)+1;
iAccuracy = zeros(length(bhv2ModelCodes), 1);
for i = 1 : length(bhv2ModelCodes)
    iObsInd = testLabels == bhv2ModelCodes(i);
    iObs = testLabels(iObsInd);
    iPred = predictedLabels(iObsInd);
    iAccuracy(i) = sum(iObs == iPred) / length(iObs);
    iPredWrong = iPred(iObs ~= iPred);
    N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
    barCodes = 0:bhv2ModelCodes(end);

    axes(ax(i))
    bar(barCodes, N, 'b', 'FaceAlpha', .5);
    iTitle = sprintf('%s (%d): %.2f', bhv2ModelNames{i}, bhv2ModelCodes(i), iAccuracy(i));
    title(iTitle, 'interpreter', 'none');
end
iTitle = sprintf('Errors due to other behaviors: %s %s %s', selectFrom, fitType, transWithinLabel);
sgtitle(iTitle)
titleW = [selectFrom, ' ',fitType, ' ', transWithinLabel,' mislabeled as others'];
saveas(gcf, fullfile(paths.figurePath, [titleW, '.png']), 'png')

% Accuracies and F-scores

figure(55);
bar(bhv2ModelCodes, iAccuracy);
xticks(bhv2ModelCodes)
ylim([0 1])
% bvhLabels = behaviors(2:end)
% xticklabels(bhvLabels)
% xtickangle(45);
title(['Accuracy for each behavior- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy for each behavior- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')

% F1 score
% Initialize arrays to store precision, recall, and F1 scores for each label
% observedLabels = svmID;
uniqueLabels = unique(testLabels);
precision = zeros(size(uniqueLabels));
recall = zeros(size(uniqueLabels));
f1Score = zeros(size(uniqueLabels));

% Calculate F1 score for each unique label
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);

    % True positives (TP)
    tp = sum((predictedLabels == label) & (testLabels == label));

    % False positives (FP)
    fp = sum((predictedLabels == label) & (testLabels ~= label));

    % False negatives (FN)
    fn = sum((predictedLabels ~= label) & (testLabels == label));

    % Precision
    if (tp + fp) > 0
        precision(i) = tp / (tp + fp);
    else
        precision(i) = 0;
    end

    % Recall
    if (tp + fn) > 0
        recall(i) = tp / (tp + fn);
    else
        recall(i) = 0;
    end

    % F1 Score
    if (precision(i) + recall(i)) > 0
        f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    else
        f1Score(i) = 0;
    end
end

% % Display the results
% resultsTable = table(uniqueLabels', precision', recall', f1Score', ...
%                      'VariableNames', {'Label', 'Precision', 'Recall', 'F1_Score'});
% disp(resultsTable);

figure(56);
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
bar(uniqueLabels, f1Score);
xticks(uniqueLabels)
ylim([0 1])
title(['F1 scores- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['F1 scores- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Use the model to predict single frames going into and out of behavior transitions
fprintf('\nPredicting each frame going into transitions:\n')
frames = -2 : 4; % 2 frames pre to two frames post transition

svmIDTest = bhvID(preInd + 1);  % behavior ID being transitioned into
svmIndTest = preInd + 1;

% Get rid of behaviors you didn't model
rmvBhv = setdiff(unique(bhvID), bhv2ModelCodes);
deleteInd = ismember(svmIDTest, rmvBhv);
svmIDTest(deleteInd) = [];
svmIndTest(deleteInd) = [];

accuracy = zeros(length(frames), 1);
accuracyPermuted = zeros(length(frames), 1);
for i = 1 : length(frames)
    % Get relevant frame to test (w.r.t. transition frame)
    iSvmInd = svmIndTest + frames(i);
    testData = projSelect(iSvmInd, 1:dimToModel(n)); % UMAP Dimensions

    % Calculate and display the overall accuracy (make sure it matches the
    % original fits to ensure we're modeling the same way)
    predictedLabels = predict(svmModel, testData);
    accuracy(i) = sum(predictedLabels == svmIDTest) / length(svmIDTest);
    fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(i), accuracy(i));

    % Calculate and display the overall accuracy (make sure it matches the
    % original fits to ensure we're modeling the same way)
    predictedLabelsPermuted = predict(svmModelPermuted, testData);
    accuracyPermuted(i) = sum(predictedLabelsPermuted == svmIDTest) / length(svmIDTest);
    % fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(i), accuracyPermuted(i));

end

figure(67); clf; hold on;
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
plot(frames, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
plot(frames, accuracyPermuted, '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
xticks(frames)
ylim([0 1])
ylabel('Accuracy')
xlabel('Frames relative to transition')
xline(0)
title(['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')










% Choose which area to select data from (to analyze clusters in that area, and/or to project same time points in another area
selectFrom = 'DS';
switch selectFrom
    case 'DS'
        projSelect = projectionsDS;
        projProject = projectionsM56;
        idSelect = idDS;
        % idSelect = cell2mat(idAll);
end



% Train and test model on single hold-out set
appendModelName = selectFrom;

dimToModel = 2:2:8;
nPermutations = 1;

accuracy = zeros(length(dimToModel), 1);
accuracyPermuted = zeros(length(dimToModel), nPermutations);

for n = 1 : length(dimToModel)
    tic


    % Split data into training (80%) and testing (20%) sets
    cv = cvpartition(svmID, 'HoldOut', 0.2);

    disp('=================================================================')

    % UMAP dimension version
    fprintf('\n\n%s %s DIMENSIONS 1 - %d\n\n', selectFrom, transWithinLabel, dimToModel(n))  % UMAP Dimensions
    % Choose which data to model
    svmProj = projSelect(svmInd, 1:dimToModel(n));
    trainData = svmProj(training(cv), :);  % UMAP Dimensions
    testData = svmProj(test(cv), :); % UMAP Dimensions


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
    accuracy(n) = sum(predictedLabels == testLabels) / length(testLabels);
    fprintf('%s %s Overall Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy(n));

    toc/60

    % Randomize labels and Train model on single hold-out set
    tic
    for i = 1:nPermutations

        % Shuffle the labels
        shuffledLabels = trainLabels(randperm(length(trainLabels)));

        % Set SVM template with the current kernel
        t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

        % Train the SVM model on shuffled training data
        svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);

        % Predict the labels using observed test data
        predictedLabelsPermuted = predict(svmModelPermuted, testData);

        % Calculate the permuted accuracy
        accuracyPermuted(n, i) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
        fprintf('Permuted %s %s Overall Accuracy permutation %d: %.4f%%\n', selectFrom, transWithinLabel, i, accuracyPermuted(n, i));

    end
    modelName = ['svmModelPermuted', appendModelName];
    % Reassign the value of modelName to the new variable name using eval
    eval([modelName, ' = svmModelPermuted;']);



    % Get the elapsed time
    toc/60
end
load handel
sound(y,Fs)



% Plot the accuracy results per dimension
figure(65); clf; hold on;
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
plot(dimToModel, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
plot(dimToModel, mean(accuracyPermuted, 2), '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
xticks(dimToModel)
ylim([0 1])
ylabel('Accuracy')
xlabel('Dimensions')
title(['Accuracy across dimensions', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy across dimensions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Analzyze the predictions vs observed


monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(54); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(bhv2ModelCodes);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.04 .02], .1);

edges = -.5 : 1 : bhv2ModelCodes(end)+1;
iAccuracy = zeros(length(bhv2ModelCodes), 1);
for i = 1 : length(bhv2ModelCodes)
    iObsInd = testLabels == bhv2ModelCodes(i);
    iObs = testLabels(iObsInd);
    iPred = predictedLabels(iObsInd);
    iAccuracy(i) = sum(iObs == iPred) / length(iObs);
    iPredWrong = iPred(iObs ~= iPred);
    N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
    barCodes = 0:bhv2ModelCodes(end);

    axes(ax(i))
    bar(barCodes, N, 'b', 'FaceAlpha', .5);
    iTitle = sprintf('%s (%d): %.2f', bhv2ModelNames{i}, bhv2ModelCodes(i), iAccuracy(i));
    title(iTitle, 'interpreter', 'none');
end
iTitle = sprintf('Errors due to other behaviors: %s %s %s', selectFrom, fitType, transWithinLabel);
sgtitle(iTitle)
titleW = [selectFrom, ' ',fitType, ' ', transWithinLabel,' mislabeled as others'];
saveas(gcf, fullfile(paths.figurePath, [titleW, '.png']), 'png')

% Accuracies and F-scores

figure(55);
bar(bhv2ModelCodes, iAccuracy);
xticks(bhv2ModelCodes)
ylim([0 1])
% bvhLabels = behaviors(2:end)
% xticklabels(bhvLabels)
% xtickangle(45);
title(['Accuracy for each behavior- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy for each behavior- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')

% F1 score
% Initialize arrays to store precision, recall, and F1 scores for each label
% observedLabels = svmID;
uniqueLabels = unique(testLabels);
precision = zeros(size(uniqueLabels));
recall = zeros(size(uniqueLabels));
f1Score = zeros(size(uniqueLabels));

% Calculate F1 score for each unique label
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);

    % True positives (TP)
    tp = sum((predictedLabels == label) & (testLabels == label));

    % False positives (FP)
    fp = sum((predictedLabels == label) & (testLabels ~= label));

    % False negatives (FN)
    fn = sum((predictedLabels ~= label) & (testLabels == label));

    % Precision
    if (tp + fp) > 0
        precision(i) = tp / (tp + fp);
    else
        precision(i) = 0;
    end

    % Recall
    if (tp + fn) > 0
        recall(i) = tp / (tp + fn);
    else
        recall(i) = 0;
    end

    % F1 Score
    if (precision(i) + recall(i)) > 0
        f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    else
        f1Score(i) = 0;
    end
end

% % Display the results
% resultsTable = table(uniqueLabels', precision', recall', f1Score', ...
%                      'VariableNames', {'Label', 'Precision', 'Recall', 'F1_Score'});
% disp(resultsTable);

figure(56);
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
bar(uniqueLabels, f1Score);
xticks(uniqueLabels)
ylim([0 1])
title(['F1 scores- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['F1 scores- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Use the model to predict single frames going into and out of behavior transitions
fprintf('\nPredicting each frame going into transitions:\n')
frames = -2 : 4; % 2 frames pre to two frames post transition

svmIDTest = bhvID(preInd + 1);  % behavior ID being transitioned into
svmIndTest = preInd + 1;

% Get rid of behaviors you didn't model
rmvBhv = setdiff(unique(bhvID), bhv2ModelCodes);
deleteInd = ismember(svmIDTest, rmvBhv);
svmIDTest(deleteInd) = [];
svmIndTest(deleteInd) = [];

accuracy = zeros(length(frames), 1);
accuracyPermuted = zeros(length(frames), 1);
for i = 1 : length(frames)
    % Get relevant frame to test (w.r.t. transition frame)
    iSvmInd = svmIndTest + frames(i);
    testData = projSelect(iSvmInd, 1:dimToModel(n)); % UMAP Dimensions

    % Calculate and display the overall accuracy (make sure it matches the
    % original fits to ensure we're modeling the same way)
    predictedLabels = predict(svmModel, testData);
    accuracy(i) = sum(predictedLabels == svmIDTest) / length(svmIDTest);
    fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(i), accuracy(i));

    % Calculate and display the overall accuracy (make sure it matches the
    % original fits to ensure we're modeling the same way)
    predictedLabelsPermuted = predict(svmModelPermuted, testData);
    accuracyPermuted(i) = sum(predictedLabelsPermuted == svmIDTest) / length(svmIDTest);
    % fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(i), accuracyPermuted(i));

end

figure(67); clf; hold on;
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
plot(frames, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
plot(frames, accuracyPermuted, '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
xticks(frames)
ylim([0 1])
ylabel('Accuracy')
xlabel('Frames relative to transition')
xline(0)
title(['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



























%% IF YOU WANT, GET RID OF DATA FOR WHICH THE BOUTS ARE UNDER A MINIMUM NUMBER OF FRAMES
transWithinLabel = 'within-behavior';


% Define the minimum number of frames
nMinFrames = 6;  % The minimum number of consecutive repetitions

% Find all unique integers in the vector
uniqueInts = unique(bhvID);

% Initialize a structure to hold the result indices for each unique integer
rmvIndices = zeros(length(bhvID), 1);

% Loop through each unique integer
for i = 1:length(uniqueInts)
    targetInt = uniqueInts(i);  % The current integer to check

    % Find the indices where the target integer appears
    indices = find(bhvID == targetInt);

    % Loop through the found indices and check for repetitions
    startIdx = 1;
    while startIdx <= length(indices)
        endIdx = startIdx;

        % Check how long the sequence of consecutive targetInt values is
        while endIdx < length(indices) && indices(endIdx + 1) == indices(endIdx) + 1
            endIdx = endIdx + 1;
        end

        % If the sequence is shorter than nMinFrames, add all its indices to tempIndices
        if (endIdx - startIdx + 1) < nMinFrames
            rmvIndices(indices(startIdx:endIdx)) = 1;
        end

        % Move to the next sequence
        startIdx = endIdx + 1;
    end
end
rmvIndices = find(rmvIndices);

% Remove any frames of behaviors that lasted less than nMinFrames
rmvSvmInd = intersect(svmInd, rmvIndices);
svmInd = setdiff(svmInd, rmvSvmInd);
svmID = bhvID(svmInd);

transWithinLabel = [transWithinLabel, ', minBoutDur ', num2str(nMinFrames)];


% IF YOU WANT, GET RID OF ENTIRE BEHAVIORS WITH UNDER A MINIMUM NUMBER OF BOUTS
nMinBouts = 45;

% Remove consecutive repetitions
noRepsVec = svmID([true; diff(svmID) ~= 0]);

% Count instances of each unique integer (each bout)
[bhvDataCount, ~] = histcounts(noRepsVec, (min(bhvID)-0.5):(max(bhvID)+0.5));

% bhvBoutCount = histcounts(noRepsVec);
rmvBehaviors = find(bhvDataCount < nMinBouts) - 2;

rmvBhvInd = find(ismember(bhvID, rmvBehaviors));
rmvSvmInd = intersect(svmInd, rmvBhvInd);
svmInd = setdiff(svmInd, rmvSvmInd);
svmID = bhvID(svmInd);

transWithinLabel = [transWithinLabel, ', minBouts ', num2str(nMinBouts)];
[codes'; bhvDataCount]
% IF YOU WANT, GET RID OF ENTIRE BEHAVIORS WITH UNDER A MINIMUM NUMBER OF FRAMES/DATA POINTS
nMinDataPoints = 500;

bhvDataCount = histcounts(svmID, (min(bhvID)-0.5):(max(bhvID)+0.5));
rmvBehaviors = find(bhvDataCount < nMinDataPoints) - 2;

rmvBhvInd = find(ismember(bhvID, rmvBehaviors));
rmvSvmInd = intersect(svmInd, rmvBhvInd);
svmInd = setdiff(svmInd, rmvSvmInd);
svmID = bhvID(svmInd);

transWithinLabel = [transWithinLabel, ', minTotalFrames ', num2str(nMinDataPoints)];
[codes'; bhvDataCount]





% Choose which area to select data from (to analyze clusters in that area, and/or to project same time points in another area
selectFrom = 'M56';
switch selectFrom
    case 'M56'
        projSelect = projectionsM56;
        projProject = projectionsDS;
        idSelect = idM56;
end



% Train and test model on single hold-out set
appendModelName = selectFrom;

dimToModel = 2:2:8;
nPermutations = 1;

accuracy = zeros(length(dimToModel), 1);
accuracyPermuted = zeros(length(dimToModel), nPermutations);

for n = 1 : length(dimToModel)
    tic


    % Split data into training (80%) and testing (20%) sets
    cv = cvpartition(svmID, 'HoldOut', 0.2);

    disp('=================================================================')

    % UMAP dimension version
    fprintf('\n\n%s %s DIMENSIONS 1 - %d\n\n', selectFrom, transWithinLabel, dimToModel(n))  % UMAP Dimensions
    % Choose which data to model
    svmProj = projSelect(svmInd, 1:dimToModel(n));
    trainData = svmProj(training(cv), :);  % UMAP Dimensions
    testData = svmProj(test(cv), :); % UMAP Dimensions


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
    accuracy(n) = sum(predictedLabels == testLabels) / length(testLabels);
    fprintf('%s %s Overall Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy(n));

    toc/60

    % Randomize labels and Train model on single hold-out set
    tic
    for i = 1:nPermutations

        % Shuffle the labels
        shuffledLabels = trainLabels(randperm(length(trainLabels)));

        % Set SVM template with the current kernel
        t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

        % Train the SVM model on shuffled training data
        svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);

        % Predict the labels using observed test data
        predictedLabelsPermuted = predict(svmModelPermuted, testData);

        % Calculate the permuted accuracy
        accuracyPermuted(n, i) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
        fprintf('Permuted %s %s Overall Accuracy permutation %d: %.4f%%\n', selectFrom, transWithinLabel, i, accuracyPermuted(n, i));

    end
    modelName = ['svmModelPermuted', appendModelName];
    % Reassign the value of modelName to the new variable name using eval
    eval([modelName, ' = svmModelPermuted;']);



    % Get the elapsed time
    toc/60
end
load handel
sound(y,Fs)



% Plot the accuracy results per dimension
figure(65); clf; hold on;
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
plot(dimToModel, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
plot(dimToModel, mean(accuracyPermuted, 2), '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
xticks(dimToModel)
ylim([0 1])
ylabel('Accuracy')
xlabel('Dimensions')
title(['Accuracy across dimensions', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy across dimensions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Analzyze the predictions vs observed


monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(54); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(bhv2ModelCodes);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.04 .02], .1);

edges = -.5 : 1 : bhv2ModelCodes(end)+1;
iAccuracy = zeros(length(bhv2ModelCodes), 1);
for i = 1 : length(bhv2ModelCodes)
    iObsInd = testLabels == bhv2ModelCodes(i);
    iObs = testLabels(iObsInd);
    iPred = predictedLabels(iObsInd);
    iAccuracy(i) = sum(iObs == iPred) / length(iObs);
    iPredWrong = iPred(iObs ~= iPred);
    N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
    barCodes = 0:bhv2ModelCodes(end);

    axes(ax(i))
    bar(barCodes, N, 'b', 'FaceAlpha', .5);
    iTitle = sprintf('%s (%d): %.2f', bhv2ModelNames{i}, bhv2ModelCodes(i), iAccuracy(i));
    title(iTitle, 'interpreter', 'none');
end
iTitle = sprintf('Errors due to other behaviors: %s %s %s', selectFrom, fitType, transWithinLabel);
sgtitle(iTitle)
titleW = [selectFrom, ' ',fitType, ' ', transWithinLabel,' mislabeled as others'];
saveas(gcf, fullfile(paths.figurePath, [titleW, '.png']), 'png')

% Accuracies and F-scores

figure(55);
bar(bhv2ModelCodes, iAccuracy);
xticks(bhv2ModelCodes)
ylim([0 1])
% bvhLabels = behaviors(2:end)
% xticklabels(bhvLabels)
% xtickangle(45);
title(['Accuracy for each behavior- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy for each behavior- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')

% F1 score
% Initialize arrays to store precision, recall, and F1 scores for each label
% observedLabels = svmID;
uniqueLabels = unique(testLabels);
precision = zeros(size(uniqueLabels));
recall = zeros(size(uniqueLabels));
f1Score = zeros(size(uniqueLabels));

% Calculate F1 score for each unique label
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);

    % True positives (TP)
    tp = sum((predictedLabels == label) & (testLabels == label));

    % False positives (FP)
    fp = sum((predictedLabels == label) & (testLabels ~= label));

    % False negatives (FN)
    fn = sum((predictedLabels ~= label) & (testLabels == label));

    % Precision
    if (tp + fp) > 0
        precision(i) = tp / (tp + fp);
    else
        precision(i) = 0;
    end

    % Recall
    if (tp + fn) > 0
        recall(i) = tp / (tp + fn);
    else
        recall(i) = 0;
    end

    % F1 Score
    if (precision(i) + recall(i)) > 0
        f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    else
        f1Score(i) = 0;
    end
end

% % Display the results
% resultsTable = table(uniqueLabels', precision', recall', f1Score', ...
%                      'VariableNames', {'Label', 'Precision', 'Recall', 'F1_Score'});
% disp(resultsTable);

figure(56);
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
bar(uniqueLabels, f1Score);
xticks(uniqueLabels)
ylim([0 1])
title(['F1 scores- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['F1 scores- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Use the model to predict single frames going into and out of behavior transitions
fprintf('\nPredicting each frame going into transitions:\n')
frames = -2 : 4; % 2 frames pre to two frames post transition

svmIDTest = bhvID(preInd + 1);  % behavior ID being transitioned into
svmIndTest = preInd + 1;

% Get rid of behaviors you didn't model
rmvBhv = setdiff(unique(bhvID), bhv2ModelCodes);
deleteInd = ismember(svmIDTest, rmvBhv);
svmIDTest(deleteInd) = [];
svmIndTest(deleteInd) = [];

accuracy = zeros(length(frames), 1);
accuracyPermuted = zeros(length(frames), 1);
for i = 1 : length(frames)
    % Get relevant frame to test (w.r.t. transition frame)
    iSvmInd = svmIndTest + frames(i);
    testData = projSelect(iSvmInd, 1:dimToModel(n)); % UMAP Dimensions

    % Calculate and display the overall accuracy (make sure it matches the
    % original fits to ensure we're modeling the same way)
    predictedLabels = predict(svmModel, testData);
    accuracy(i) = sum(predictedLabels == svmIDTest) / length(svmIDTest);
    fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(i), accuracy(i));

    % Calculate and display the overall accuracy (make sure it matches the
    % original fits to ensure we're modeling the same way)
    predictedLabelsPermuted = predict(svmModelPermuted, testData);
    accuracyPermuted(i) = sum(predictedLabelsPermuted == svmIDTest) / length(svmIDTest);
    % fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(i), accuracyPermuted(i));

end

figure(67); clf; hold on;
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
plot(frames, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
plot(frames, accuracyPermuted, '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
xticks(frames)
ylim([0 1])
ylabel('Accuracy')
xlabel('Frames relative to transition')
xline(0)
title(['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')





% Choose which area to select data from (to analyze clusters in that area, and/or to project same time points in another area
selectFrom = 'DS';
switch selectFrom
    case 'DS'
        projSelect = projectionsDS;
        projProject = projectionsM56;
        idSelect = idDS;
end



% Train and test model on single hold-out set
appendModelName = selectFrom;

dimToModel = 2:2:8;
nPermutations = 1;

accuracy = zeros(length(dimToModel), 1);
accuracyPermuted = zeros(length(dimToModel), nPermutations);

for n = 1 : length(dimToModel)
    tic


    % Split data into training (80%) and testing (20%) sets
    cv = cvpartition(svmID, 'HoldOut', 0.2);

    disp('=================================================================')

    % UMAP dimension version
    fprintf('\n\n%s %s DIMENSIONS 1 - %d\n\n', selectFrom, transWithinLabel, dimToModel(n))  % UMAP Dimensions
    % Choose which data to model
    svmProj = projSelect(svmInd, 1:dimToModel(n));
    trainData = svmProj(training(cv), :);  % UMAP Dimensions
    testData = svmProj(test(cv), :); % UMAP Dimensions


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
    accuracy(n) = sum(predictedLabels == testLabels) / length(testLabels);
    fprintf('%s %s Overall Accuracy: %.4f%%\n', selectFrom, transWithinLabel, accuracy(n));

    toc/60

    % Randomize labels and Train model on single hold-out set
    tic
    for i = 1:nPermutations

        % Shuffle the labels
        shuffledLabels = trainLabels(randperm(length(trainLabels)));

        % Set SVM template with the current kernel
        t = templateSVM('Standardize', true, 'KernelFunction', kernelFunction);

        % Train the SVM model on shuffled training data
        svmModelPermuted = fitcecoc(trainData, shuffledLabels, 'Learners', t);

        % Predict the labels using observed test data
        predictedLabelsPermuted = predict(svmModelPermuted, testData);

        % Calculate the permuted accuracy
        accuracyPermuted(n, i) = sum(predictedLabelsPermuted == testLabels) / length(testLabels);
        fprintf('Permuted %s %s Overall Accuracy permutation %d: %.4f%%\n', selectFrom, transWithinLabel, i, accuracyPermuted(n, i));

    end
    modelName = ['svmModelPermuted', appendModelName];
    % Reassign the value of modelName to the new variable name using eval
    eval([modelName, ' = svmModelPermuted;']);



    % Get the elapsed time
    toc/60
end
load handel
sound(y,Fs)



% Plot the accuracy results per dimension
figure(65); clf; hold on;
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
plot(dimToModel, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
plot(dimToModel, mean(accuracyPermuted, 2), '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
xticks(dimToModel)
ylim([0 1])
ylabel('Accuracy')
xlabel('Dimensions')
title(['Accuracy across dimensions', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy across dimensions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Analzyze the predictions vs observed


monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(54); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(bhv2ModelCodes);
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.04 .02], .1);

edges = -.5 : 1 : bhv2ModelCodes(end)+1;
iAccuracy = zeros(length(bhv2ModelCodes), 1);
for i = 1 : length(bhv2ModelCodes)
    iObsInd = testLabels == bhv2ModelCodes(i);
    iObs = testLabels(iObsInd);
    iPred = predictedLabels(iObsInd);
    iAccuracy(i) = sum(iObs == iPred) / length(iObs);
    iPredWrong = iPred(iObs ~= iPred);
    N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
    barCodes = 0:bhv2ModelCodes(end);

    axes(ax(i))
    bar(barCodes, N, 'b', 'FaceAlpha', .5);
    iTitle = sprintf('%s (%d): %.2f', bhv2ModelNames{i}, bhv2ModelCodes(i), iAccuracy(i));
    title(iTitle, 'interpreter', 'none');
end
iTitle = sprintf('Errors due to other behaviors: %s %s %s', selectFrom, fitType, transWithinLabel);
sgtitle(iTitle)
titleW = [selectFrom, ' ',fitType, ' ', transWithinLabel,' mislabeled as others'];
saveas(gcf, fullfile(paths.figurePath, [titleW, '.png']), 'png')

% Accuracies and F-scores

figure(55);
bar(bhv2ModelCodes, iAccuracy);
xticks(bhv2ModelCodes)
ylim([0 1])
% bvhLabels = behaviors(2:end)
% xticklabels(bhvLabels)
% xtickangle(45);
title(['Accuracy for each behavior- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy for each behavior- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')

% F1 score
% Initialize arrays to store precision, recall, and F1 scores for each label
% observedLabels = svmID;
uniqueLabels = unique(testLabels);
precision = zeros(size(uniqueLabels));
recall = zeros(size(uniqueLabels));
f1Score = zeros(size(uniqueLabels));

% Calculate F1 score for each unique label
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);

    % True positives (TP)
    tp = sum((predictedLabels == label) & (testLabels == label));

    % False positives (FP)
    fp = sum((predictedLabels == label) & (testLabels ~= label));

    % False negatives (FN)
    fn = sum((predictedLabels ~= label) & (testLabels == label));

    % Precision
    if (tp + fp) > 0
        precision(i) = tp / (tp + fp);
    else
        precision(i) = 0;
    end

    % Recall
    if (tp + fn) > 0
        recall(i) = tp / (tp + fn);
    else
        recall(i) = 0;
    end

    % F1 Score
    if (precision(i) + recall(i)) > 0
        f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    else
        f1Score(i) = 0;
    end
end

% % Display the results
% resultsTable = table(uniqueLabels', precision', recall', f1Score', ...
%                      'VariableNames', {'Label', 'Precision', 'Recall', 'F1_Score'});
% disp(resultsTable);

figure(56);
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
bar(uniqueLabels, f1Score);
xticks(uniqueLabels)
ylim([0 1])
title(['F1 scores- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['F1 scores- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Use the model to predict single frames going into and out of behavior transitions
fprintf('\nPredicting each frame going into transitions:\n')
frames = -2 : 4; % 2 frames pre to two frames post transition

svmIDTest = bhvID(preInd + 1);  % behavior ID being transitioned into
svmIndTest = preInd + 1;

% Get rid of behaviors you didn't model
rmvBhv = setdiff(unique(bhvID), bhv2ModelCodes);
deleteInd = ismember(svmIDTest, rmvBhv);
svmIDTest(deleteInd) = [];
svmIndTest(deleteInd) = [];

accuracy = zeros(length(frames), 1);
accuracyPermuted = zeros(length(frames), 1);
for i = 1 : length(frames)
    % Get relevant frame to test (w.r.t. transition frame)
    iSvmInd = svmIndTest + frames(i);
    testData = projSelect(iSvmInd, 1:dimToModel(n)); % UMAP Dimensions

    % Calculate and display the overall accuracy (make sure it matches the
    % original fits to ensure we're modeling the same way)
    predictedLabels = predict(svmModel, testData);
    accuracy(i) = sum(predictedLabels == svmIDTest) / length(svmIDTest);
    fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(i), accuracy(i));

    % Calculate and display the overall accuracy (make sure it matches the
    % original fits to ensure we're modeling the same way)
    predictedLabelsPermuted = predict(svmModelPermuted, testData);
    accuracyPermuted(i) = sum(predictedLabelsPermuted == svmIDTest) / length(svmIDTest);
    % fprintf('%s %d Frames Overall Accuracy: %.4f%%\n', selectFrom, frames(i), accuracyPermuted(i));

end

figure(67); clf; hold on;
% bar(codes(2:end), f1Score);
% xticks(0:codes(end))
plot(frames, accuracy, '-o', 'color', 'blue', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'blue', 'LineWidth', 1.5);
plot(frames, accuracyPermuted, '-o', 'color', 'red', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'red', 'LineWidth', 1.5);
xticks(frames)
ylim([0 1])
ylabel('Accuracy')
xlabel('Frames relative to transition')
xline(0)
title(['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')






























% ============================================================================================
% ============================================================================================
% ============================================================================================
% ============================================================================================
% ============================================================================================
% ============================================================================================
% ============================================================================================
% ============================================================================================
% ============================================================================================











%% Get UMAP for first and last hour of session

%
if exist('/Users/paulmiddlebrooks/Projects/', 'dir')
    cd '/Users/paulmiddlebrooks/Projects/toolboxes/umapFileExchange (4.4)/umap/'
else
    cd 'E:/Projects/toolboxes/umapFileExchange (4.4)/umap/'
end

% for plotting consistency
monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



opts = neuro_behavior_options;
opts.minActTime = .16;
opts.frameSize = .1;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectEnd = 4 * 60 * 60; % seconds

getDataType = 'all';
get_standard_data


[dataBhv, bhvIDMat] = curate_behavior_labels(dataBhv, opts);

firstHourSpan = 1 : length(bhvIDMat) / 4;
hourToTestSpan = 1 + 3*length(bhvIDMat)/4 : length(bhvIDMat);
hourToTestSpan = 1 + 3*length(bhvIDMat)/4 : length(bhvIDMat);

%% =============================================================================
% --------    RUN UMAP SVM FITS FOR VARIOUS CONDITIONS ON VARIOUS DATA
% =============================================================================

% Select which data to run analyses on, UMAP dimensions, etc

forDim = 4:2:8; % Loop through these dimensions to fit UMAP
forDim = 8; % Loop through these dimensions to fit UMAP
newUmapModel = 1; % Do we need to get a new umap model to analyze (or did you tweak some things that come after umap?)


% Change these (and check their sections below) to determine which
% variables to test
% ==========================

% Modeling variables
nPermutations = 1; % How many random permutations to run to compare with best fit model?
accuracy = zeros(length(forDim), 1);
accuracyPermuted = zeros(length(forDim), nPermutations);

% Apply to all:
% -------------
plotFullMap = 1;
plotFullModelData = 1;
plotModelData = 1;
changeBhvLabels = 0;

% Transition or within variables
% -------------------------
% transOrWithin = 'trans';
transOrWithin = 'within';
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
end


% Some figure properties
allFontSize = 12;


for k = 1:length(forDim)
    iDim = forDim(k);
    nUmapDim = iDim;
    fitType = ['UMAP ', num2str(iDim), 'D'];


    %% Run UMAP to get projections in low-D space
    % if runAll
% % umapBhvIDMat = [bhvIDMat(1:length(bhvIDMat)/4); bhvIDMat(1 + 3*length(bhvIDMat)/4) : length(bhvIDMat)];
% umapDataMat = [dataMat(firstHourSpan, idSelect); dataMat(lastHourSpan, idSelect)];
    if newUmapModel
        umapFrameSize = opts.frameSize;

        % rng(1);
        [projSelect, ~, ~, ~] = run_umap(dataMat(:, idSelect), 'n_components', nUmapDim, 'randomize', false);
        pause(3); close
    end
    % end

    %% --------------------------------------------
    % Shift behavior label w.r.t. neural to account for neuro-behavior latency
    shiftSec = 0;
    shiftFrame = ceil(shiftSec / opts.frameSize);
    bhvID = double(bhvIDMat(1+shiftFrame:end)); % Shift bhvIDMat to account for time shift

    projSelect = projSelect(1:end-shiftFrame, :); % Remove shiftFrame frames from projections to accoun for time shift in bhvIDMat




    %% --------------------------------------------
    % Plot FULL TIME OF ALL BEHAVIORS
    if plotFullMap

        colors = colors_for_behaviors(codes);
        colorsForPlot = arrayfun(@(x) colors(x,:), bhvID + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

        % colorsForPlot = [.2 .2 .2];

        % Plot on second monitor, half-width
        plotPos = [monitorOne(1), 1, monitorOne(3)/2, monitorOne(4)];
        fig = figure(figHFull);
        set(fig, 'Position', plotPos); clf; hold on;
        titleM = [selectFrom, ' ', fitType, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
        title(titleM)
        if nUmapDim > 2
            scatter3(projSelect(:, iDim-2), projSelect(:, iDim-1), projSelect(:, iDim), 60, colorsForPlot, 'LineWidth', 2)
            % Variable to set the viewing angle
            azimuth = 60;  % Angle for rotation around the z-axis
            elevation = 20;  % Angle for elevation
            % Set the viewing angle
            view(azimuth, elevation);
            % Set axes ranges based on the data
            xlim([min(projSelect(:, iDim-2)), max(projSelect(:, iDim-2))]);
            ylim([min(projSelect(:, iDim-1)), max(projSelect(:, iDim-1))]);
            zlim([min(projSelect(:, iDim)), max(projSelect(:, iDim))]);
            set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
            set(findall(fig,'-property','Box'),'Box','off') % optional
        xlabel(['D', num2str(iDim-2)]); ylabel(['D', num2str(iDim-1)]); zlabel(['D', num2str(iDim)])

        elseif nUmapDim == 2
            scatter(projSelect(:, 1), projSelect(:, 2), 60, colorsForPlot, 'LineWidth', 2)
        xlabel(['D', num2str(1)]); ylabel(['D', num2str(2)])
        end
        grid on;
        % saveas(gcf, fullfile(paths.figurePath, [titleM, '.png']), 'png')
        % print('-dpdf', fullfile(paths.figurePath, [titleM, '.pdf']), '-bestfit')
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
            % svmID = [svmID; svmID];
            % svmInd = [svmInd - 1; svmInd]; % two bins before transition
            % svmInd = [svmInd; svmInd + 1]; % Last bin before transition and first bin after

            transWithinLabel = 'transitions pre';
            % transWithinLabel = 'transitions 200ms pre';
            % transWithinLabel = 'transitions post';
            % transWithinLabel = 'transitions pre & post';
            % transWithinLabel = ['transitions pre minBout ', num2str(nMinFrames)];


        case 'within'%% WITHIN-BEHAVIOR of all behaviors (for now, include behaviors that last one frame)

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



            if matchTransitionCount
                frameCounts = histcounts(bhvID(preInd + 1));

                for iBhv = 1 : length(frameCounts)
                    iBhvInd = find(svmID == iBhv - 2);
                    if length(iBhvInd) > frameCounts(iBhv)
                        nRemove = length(iBhvInd) - frameCounts(iBhv);
                        rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
                        svmID(rmvBhvInd) = [];
                        svmInd(rmvBhvInd) = [];
                    end
                end
                transWithinLabel = [transWithinLabel, ' match transitions'];
            end




            %% IF YOU WANT, GET RID OF DATA FOR WHICH THE BOUTS ARE UNDER A MINIMUM NUMBER OF FRAMES
            if minFramePerBout
                % Define the minimum number of frames
                nMinFrames = 6;  % The minimum number of consecutive repetitions

                % Find all unique integers in the vector
                uniqueInts = unique(bhvID);

                % Initialize a structure to hold the result indices for each unique integer
                rmvIndices = zeros(length(bhvID), 1);

                % Loop through each unique integer
                for iBhv = 1:length(uniqueInts)
                    targetInt = uniqueInts(iBhv);  % The current integer to check

                    % Find the indices where the target integer appears
                    indices = find(bhvID == targetInt);

                    % Loop through the found indices and check for repetitions
                    startIdx = 1;
                    while startIdx <= length(indices)
                        endIdx = startIdx;

                        % Check how long the sequence of consecutive targetInt values is
                        while endIdx < length(indices) && indices(endIdx + 1) == indices(endIdx) + 1
                            endIdx = endIdx + 1;
                        end

                        % If the sequence is shorter than nMinFrames, add all its indices to tempIndices
                        if (endIdx - startIdx + 1) < nMinFrames
                            rmvIndices(indices(startIdx:endIdx)) = 1;
                        end

                        % Move to the next sequence
                        startIdx = endIdx + 1;
                    end
                end
                rmvIndices = find(rmvIndices);

                % Remove any frames of behaviors that lasted less than nMinFrames
                rmvSvmInd = intersect(svmInd, rmvIndices);
                svmInd = setdiff(svmInd, rmvSvmInd);
                svmID = bhvID(svmInd);

                transWithinLabel = [transWithinLabel, ', minBoutDur ', num2str(nMinFrames)];
            end


    end
    %% TWEAK THE BEHAVIOR LABELS IF YOU WANT
    if collapseBhv
        % IF YOU WANT, COLLAPSE MULTIPLE BEHAVIOR LABELS INTO A SINGLE LABEL
        % bhvID(bhvID == 2) = 15;

        % Remove all investigate/locomotion/orients
        rmvIndices = find(svmID == 0 | svmID == 1 | svmID == 2 | svmID == 13 | svmID == 14 | svmID == 15);
        svmID(rmvIndices) = [];
        svmInd(rmvIndices) = [];
        % svmInd
        % rmvSvmInd = intersect(svmInd, rmvIndices);
        % svmInd = setdiff(svmInd, rmvSvmInd);
        % svmID = bhvID(svmInd);

        transWithinLabel = [transWithinLabel, ', remove loco-invest-orients'];
    end
    %% IF YOU WANT, GET RID OF ENTIRE BEHAVIORS WITH UNDER A MINIMUM NUMBER OF BOUTS
    if minBoutNumber
        nMinBouts = 100;

        % Remove consecutive repetitions
        noRepsVec = svmID([true; diff(svmID) ~= 0]);

        % Count instances of each unique integer (each bout)
        [bhvDataCount, ~] = histcounts(noRepsVec, (min(bhvID)-0.5):(max(bhvID)+0.5));

        % bhvBoutCount = histcounts(noRepsVec);
        rmvBehaviors = find(bhvDataCount < nMinBouts) - 2;

        rmvBhvInd = find(ismember(bhvID, rmvBehaviors)); % find bhvID indices to remove
        rmvSvmInd = intersect(svmInd, rmvBhvInd); % find those indices that are also in svmInd
        svmInd = setdiff(svmInd, rmvSvmInd); % the remaining svmInd are the ones to keep (from the original bhvInd)
        svmID = bhvID(svmInd);

        transWithinLabel = [transWithinLabel, ', minBouts ', num2str(nMinBouts)];
        [codes'; bhvDataCount]
    end
    %% IF YOU WANT, DOWNSAMPLE TO A CERTAIN NUMBER OF BOUTS (THE BEHAVIOR WITH THE MINUMUM NUMBER OF BOUTS)
    if downSampleBouts

        % Remove consecutive repetitions
        noRepsVec = svmID([true; diff(svmID) ~= 0]);

        % Count instances of each unique integer (each bout)
        [bhvDataCount, ~] = histcounts(noRepsVec, (min(bhvID)-0.5):(max(bhvID)+0.5));

        % subsampling to match single frame transition number
        downSample = min(bhvDataCount(bhvDataCount > 0));
        for iBhv = 1 : length(bhvDataCount)
            iBhvInd = find(svmID == iBhv - 2);
            if ~isempty(iBhvInd)
                nRemove = length(iBhvInd) - downSample;
                rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
                svmID(rmvBhvInd) = [];
                svmInd(rmvBhvInd) = [];
            end
        end
        transWithinLabel = [transWithinLabel, ', downsample to ', num2str(downSample), ' bouts'];
    end

     %% IF YOU WANT, GET RID OF ENTIRE BEHAVIORS WITH UNDER A MINIMUM NUMBER OF FRAMES/DATA POINTS
    if minFrames
        nMinFrames = 500;

[uniqueVals, ~, idx] = unique(svmID); % Find unique integers and indices
bhvDataCount = accumarray(idx, 1); % Count occurrences of each unique integer
        % bhvDataCount = histcounts(svmID, (min(bhvID)-0.5):(max(bhvID)+0.5));
        rmvBehaviors = find(bhvDataCount < nMinFrames) - 2;

        rmvBhvInd = find(ismember(bhvID, rmvBehaviors));
        rmvSvmInd = intersect(svmInd, rmvBhvInd);
        svmInd = setdiff(svmInd, rmvSvmInd);
        svmID = bhvID(svmInd);

        transWithinLabel = [transWithinLabel, ', minTotalFrames ', num2str(nMinFrames)];
        [uniqueVals'; bhvDataCount']
    end

    %% IF YOU WANT, DOWNSAMPLE TO A CERTAIN NUMBER OF DATA POINTS
    if downSampleFrames
        %     cutoff = 1000;
        % frameCounts = histcounts(svmID);
        % for iBhv = 2 : length(frameCounts)
        %     iBhvInd = find(svmID == iBhv - 2);
        %     if length(iBhvInd) > cutoff
        %         nRemove = length(iBhvInd) - cutoff;
        %         rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
        %         svmID(rmvBhvInd) = [];
        %         svmInd(rmvBhvInd) = [];
        %     end
        % end
        % transWithinLabel = ['within-behavior max frames ', num2str(cutoff)];

        % subsampling to match single frame transition number
[uniqueVals, ~, idx] = unique(svmID); % Find unique integers and indices
frameCounts = accumarray(idx, 1); % Count occurrences of each unique integer
        downSample = min(frameCounts(frameCounts > 0));
        for iBhv = 1 : length(frameCounts)
            iBhvInd = find(svmID == iBhv - 2);
            if ~isempty(iBhvInd)
                nRemove = length(iBhvInd) - downSample;
                rmvBhvInd = iBhvInd(randperm(length(iBhvInd), nRemove));
                svmID(rmvBhvInd) = [];
                svmInd(rmvBhvInd) = [];
            end
        end
        transWithinLabel = [transWithinLabel, ', downsample to ', num2str(downSample), ' data points'];
    end





    %% Get rid of sleeping/in_nest/irrelavents
    deleteInd = svmID == -1;
    svmID(deleteInd) = [];
    svmInd(deleteInd) = [];

%%
svmFitModelID = svmID(svmInd <= firstHourSpan(end));
svmFitModelIdx = svmInd(svmInd <= firstHourSpan(end));

    %% Keep track of the behavior IDs you end up using
    bhv2ModelCodes = unique(svmID);
    bhv2ModelNames = behaviors(bhv2ModelCodes+2);


    colors = colors_for_behaviors(codes);
    bhv2ModelColors = colors(ismember(codes, bhv2ModelCodes), :);



    %% --------------------------------------------
    % Plot Full time of all behaviors that we are modeling
    if plotFullModelData
        allBhvModeled = ismember(bhvID(firstHourSpan), bhv2ModelCodes);

        colors = colors_for_behaviors(codes);
        colorsForPlot = arrayfun(@(x) colors(x,:), bhvID(allBhvModeled) + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

        % colorsForPlot = [.2 .2 .2];

        % Plot on second monitor, half-width
        plotPos = [monitorTwo(1), 1, monitorTwo(3)/2, monitorTwo(4)];
        fig = figure(figHFullModel);
        set(fig, 'Position', plotPos); clf; hold on;
        titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' All Frames' ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
        title(titleM)
        if nUmapDim > 2
            scatter3(projSelect(allBhvModeled, iDim-2), projSelect(allBhvModeled, iDim-1), projSelect(allBhvModeled, iDim), 60, colorsForPlot, 'LineWidth', 2)
            % Variable to set the viewing angle
            azimuth = 60;  % Angle for rotation around the z-axis
            elevation = 20;  % Angle for elevation
            % Set the viewing angle
            view(azimuth, elevation);
            % Set axes ranges based on the data
            xlim([min(projSelect(:, iDim-2)), max(projSelect(:, iDim-2))]);
            ylim([min(projSelect(:, iDim-1)), max(projSelect(:, iDim-1))]);
            zlim([min(projSelect(:, iDim)), max(projSelect(:, iDim))]);
            set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
            set(findall(fig,'-property','Box'),'Box','off') % optional
        xlabel(['D', num2str(iDim-2)]); ylabel(['D', num2str(iDim-1)]); zlabel(['D', num2str(iDim)])

        elseif nUmapDim == 2
            scatter(projSelect(allBhvModeled, 1), projSelect(allBhvModeled, 2), 60, colorsForPlot, 'LineWidth', 2)
        xlabel(['D', num2str(iDim-1)]); ylabel(['D', num2str(iDim)])
        end
        grid on;
        % print('-dpdf', fullfile(paths.figurePath, [titleM, '.pdf']), '-bestfit')
    end

    %% Plot data to model
    if plotModelData
        colors = colors_for_behaviors(codes);
        colorsForPlot = arrayfun(@(x) colors(x,:), svmFitModelID + 2, 'UniformOutput', false);
        colorsForPlot = vertcat(colorsForPlot{:}); % Convert cell array to a matrix

        plotPos = [monitorTwo(1) + monitorTwo(3)/2, 1, monitorTwo(3)/2, monitorTwo(4)];
        fig = figure(figHModel);
        set(fig, 'Position', plotPos); clf; hold on;
        titleM = [selectFrom, ' ', fitType, ' ', transWithinLabel, ' bin = ', num2str(opts.frameSize), ' shift = ', num2str(shiftSec)];
        title(titleM)
        if nUmapDim > 2
            scatter3(projSelect(svmFitModelIdx, iDim-2), projSelect(svmFitModelIdx, iDim-1), projSelect(svmFitModelIdx, iDim), 60, colorsForPlot, 'LineWidth', 2)
            % Variable to set the viewing angle
            azimuth = 60;  % Angle for rotation around the z-axis
            elevation = 20;  % Angle for elevation
            % Set the viewing angle
            view(azimuth, elevation);
            % Set axes ranges based on the data
            xlim([min(projSelect(:, iDim-2)), max(projSelect(:, iDim-2))]);
            ylim([min(projSelect(:, iDim-1)), max(projSelect(:, iDim-1))]);
            zlim([min(projSelect(:, iDim)), max(projSelect(:, iDim))]);
            set(findall(fig,'-property','FontSize'),'FontSize',allFontSize) % adjust fontsize to your document
            set(findall(fig,'-property','Box'),'Box','off') % optional
        xlabel(['D', num2str(iDim-2)]); ylabel(['D', num2str(iDim-1)]); zlabel(['D', num2str(iDim)])
        elseif nUmapDim == 2
            scatter(projSelect(svmFitModelIdx, 1), projSelect(svmFitModelIdx, 2), 60, colorsForPlot, 'LineWidth', 2)
        xlabel(['D', num2str(iDim-1)]); ylabel(['D', num2str(iDim)])
        end
        grid on;
        % print('-dpdf', fullfile(paths.figurePath, [titleM, '.pdf']), '-bestfit')

    end


    %%                  SVM classifier to predict behavior ID
    
    % Train and test model on single hold-out set
    appendModelName = selectFrom;


    tic


    % Split data into training (80%) and testing (20%) sets
    cv = cvpartition(svmFitModelID, 'HoldOut', 0.2);

    disp('=================================================================')

    % UMAP dimension version
    fprintf('\n\n%s %s DIMENSIONS %d\n\n', selectFrom, transWithinLabel, nUmapDim)  % UMAP Dimensions
    % Choose which data to model
    svmProj = projSelect(svmFitModelIdx, :);
    trainData = svmProj(training(cv), :);  % UMAP Dimensions
    testData = svmProj(test(cv), :); % UMAP Dimensions


    % % Neural space version
    % fprintf('\n\n%s %s Neural Space\n\n', selectFrom, transWithinLabel)  % Neural Space
    % svmProj = dataMat(svmInd, idSelect);
    % trainData = svmProj(training(cv), :);  % Neural Space
    % testData = svmProj(test(cv), :); % Neural Space



    trainLabels = svmFitModelID(training(cv));
    testLabels = svmFitModelID(test(cv));


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
    for iPerm = 1:nPermutations

        % Shuffle the labels
        shuffledLabels = trainLabels(randperm(length(trainLabels)));

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


end

% Use the fit SVM model to test the last hour of data
%

for iHour = 1 : 3
    hourToTestSpan = 1 + iHour*length(bhvIDMat)/4 : (iHour+1)*length(bhvIDMat)/4;
svmLastHourID = svmID(svmInd >= hourToTestSpan(1) & svmInd <= hourToTestSpan(end));
svmLastHourIdx = svmInd(svmInd >= hourToTestSpan(1) & svmInd <= hourToTestSpan(end));

testData = projSelect(svmLastHourIdx, :);
testLabels = svmLastHourID;

    predictedLabels = predict(svmModel, testData);

    % Calculate and display the overall accuracy
    accuracyLast = sum(predictedLabels == testLabels) / length(testLabels);
    fprintf('%s %s Hour %d Overall Accuracy: %.4f%%\n', selectFrom, transWithinLabel, iHour, accuracyLast);

end

