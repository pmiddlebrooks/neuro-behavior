%%

preInd = find(diff(bhvID) ~= 0); % 1 frame prior to all behavior transitions


% TRANSITIONS of all behaviors (for now, include behaviors that last one frame)
% svmID: vector of category labels for each data point to be analyzed/fit

svmID = bhvID(preInd + 1);  % behavior ID being transitioned into

% Pre and/or Post: Adjust which bin(s) to plot (and train SVN on below)
svmInd = preInd; % + 1; % First bin after transition

% Pre & Post: Comment/uncomment to use more than one bin
svmID = [svmID; svmID];
% svmInd = [svmInd - 1; svmInd]; % two bins before transition
svmInd = [svmInd; svmInd + 1]; % Last bin before transition and first bin after

% transWithinLabel = 'transitions pre';
% transWithinLabel = 'transitions 200ms pre';
transWithinLabel = 'transitions pre & post';



% % WITHIN-BEHAVIOR of all behaviors (for now, include behaviors that last one frame)
% subsample = 0;
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
% svmInd = find(~transIndLog);
% svmID = bhvID(svmInd);
% 
% % choose correct title
% transWithinLabel = 'within-behavior';



% Get rid of sleeping/in_nest/irrelavents
deleteInd = svmID == -1;
svmID(deleteInd) = [];
svmInd(deleteInd) = [];



% Keep track of the behavior IDs you end up using
bhv2ModelCodes = unique(svmID);







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

dimToModel = 2:8;
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
title(['Accuracy for dimensions', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Analzyze the predictions vs observed

% Use above code to establish model to analyze

% fitType = 'UMAP 1-3';
fitType = ['UMAP 1-', num2str(dimToModel(n))];
% fitType = 'NeuralSpace';


monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(54); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(codes) - 1;
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.04 .02], .1);

edges = -.5 : 1 : codes(end)+1;
for i = 2 : length(codes)
    % iObs = svmID == codes(i);
    iObsInd = testLabels == codes(i);
    iObs = testLabels(iObsInd);
    iPred = predictedLabels(iObsInd);
    % iAccuracy(i-1) = sum(svmID(iObs) == iPred) / sum(iObs);
    iAccuracy(i-1) = sum(iObs == iPred) / length(iObs);
    iPredWrong = iPred(iObs ~= iPred);
    N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
    % barCodes = codes(codes ~= codes(i)) + 1;
    barCodes = 0:codes(end);

    axes(ax(i-1))
    bar(barCodes, N, 'b', 'FaceAlpha', .5);
    iTitle = sprintf('%s (%d): %.2f', behaviors{i}, codes(i), iAccuracy(i-1));
    title(iTitle, 'interpreter', 'none');
end
iTitle = sprintf('Errors due to other behaviors: %s %s %s', selectFrom, fitType, transWithinLabel);
sgtitle(iTitle)
titleW = [selectFrom, ' ',fitType, ' ', transWithinLabel,' mislabeled as others'];
saveas(gcf, fullfile(paths.figurePath, [titleW, '.png']), 'png')

% Accuracies and F-scores

figure(55);
bar(codes(2:end), iAccuracy);
xticks(0:codes(end))
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
observedLabels = testLabels;
uniqueLabels = unique(observedLabels);
precision = zeros(size(uniqueLabels));
recall = zeros(size(uniqueLabels));
f1Score = zeros(size(uniqueLabels));

% Calculate F1 score for each unique label
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);

    % True positives (TP)
    tp = sum((predictedLabels == label) & (observedLabels == label));

    % False positives (FP)
    fp = sum((predictedLabels == label) & (observedLabels ~= label));

    % False negatives (FN)
    fn = sum((predictedLabels ~= label) & (observedLabels == label));

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
nDim = 8;

dimToModel = 2:8;
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
    trainData = svmProj(training(cv), 1:dimToModel(n));  % UMAP Dimensions
    testData = svmProj(test(cv), 1:dimToModel(n)); % UMAP Dimensions


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
title(['Accuracy for dimensions', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Analzyze the predictions vs observed

% Use above code to establish model to analyze

% fitType = 'UMAP 1-3';
fitType = ['UMAP 1-', num2str(dimToModel(n))];
% fitType = 'NeuralSpace';


monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(54); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(codes) - 1;
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.04 .02], .1);

edges = -.5 : 1 : codes(end)+1;
for i = 2 : length(codes)
    % iObs = svmID == codes(i);
    iObsInd = testLabels == codes(i);
    iObs = testLabels(iObsInd);
    iPred = predictedLabels(iObsInd);
    % iAccuracy(i-1) = sum(svmID(iObs) == iPred) / sum(iObs);
    iAccuracy(i-1) = sum(iObs == iPred) / length(iObs);
    iPredWrong = iPred(iObs ~= iPred);
    N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
    % barCodes = codes(codes ~= codes(i)) + 1;
    barCodes = 0:codes(end);

    axes(ax(i-1))
    bar(barCodes, N, 'b', 'FaceAlpha', .5);
    iTitle = sprintf('%s (%d): %.2f', behaviors{i}, codes(i), iAccuracy(i-1));
    title(iTitle, 'interpreter', 'none');
end
iTitle = sprintf('Errors due to other behaviors: %s %s %s', selectFrom, fitType, transWithinLabel);
sgtitle(iTitle)
titleW = [selectFrom, ' ',fitType, ' ', transWithinLabel,' mislabeled as others'];
saveas(gcf, fullfile(paths.figurePath, [titleW, '.png']), 'png')

% Accuracies and F-scores

figure(55);
bar(codes(2:end), iAccuracy);
xticks(0:codes(end))
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
observedLabels = testLabels;
uniqueLabels = unique(observedLabels);
precision = zeros(size(uniqueLabels));
recall = zeros(size(uniqueLabels));
f1Score = zeros(size(uniqueLabels));

% Calculate F1 score for each unique label
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);

    % True positives (TP)
    tp = sum((predictedLabels == label) & (observedLabels == label));

    % False positives (FP)
    fp = sum((predictedLabels == label) & (observedLabels ~= label));

    % False negatives (FN)
    fn = sum((predictedLabels ~= label) & (observedLabels == label));

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

dimToModel = 2:8;
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
    trainData = svmProj(training(cv), 1:dimToModel(n));  % UMAP Dimensions
    testData = svmProj(test(cv), 1:dimToModel(n)); % UMAP Dimensions


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
title(['Accuracy for dimensions', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Analzyze the predictions vs observed

% Use above code to establish model to analyze

% fitType = 'UMAP 1-3';
fitType = ['UMAP 1-', num2str(dimToModel(n))];
% fitType = 'NeuralSpace';


monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(54); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(codes) - 1;
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.04 .02], .1);

edges = -.5 : 1 : codes(end)+1;
for i = 2 : length(codes)
    % iObs = svmID == codes(i);
    iObsInd = testLabels == codes(i);
    iObs = testLabels(iObsInd);
    iPred = predictedLabels(iObsInd);
    % iAccuracy(i-1) = sum(svmID(iObs) == iPred) / sum(iObs);
    iAccuracy(i-1) = sum(iObs == iPred) / length(iObs);
    iPredWrong = iPred(iObs ~= iPred);
    N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
    % barCodes = codes(codes ~= codes(i)) + 1;
    barCodes = 0:codes(end);

    axes(ax(i-1))
    bar(barCodes, N, 'b', 'FaceAlpha', .5);
    iTitle = sprintf('%s (%d): %.2f', behaviors{i}, codes(i), iAccuracy(i-1));
    title(iTitle, 'interpreter', 'none');
end
iTitle = sprintf('Errors due to other behaviors: %s %s %s', selectFrom, fitType, transWithinLabel);
sgtitle(iTitle)
titleW = [selectFrom, ' ',fitType, ' ', transWithinLabel,' mislabeled as others'];
saveas(gcf, fullfile(paths.figurePath, [titleW, '.png']), 'png')

% Accuracies and F-scores

figure(55);
bar(codes(2:end), iAccuracy);
xticks(0:codes(end))
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
observedLabels = testLabels;
uniqueLabels = unique(observedLabels);
precision = zeros(size(uniqueLabels));
recall = zeros(size(uniqueLabels));
f1Score = zeros(size(uniqueLabels));

% Calculate F1 score for each unique label
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);

    % True positives (TP)
    tp = sum((predictedLabels == label) & (observedLabels == label));

    % False positives (FP)
    fp = sum((predictedLabels == label) & (observedLabels ~= label));

    % False negatives (FN)
    fn = sum((predictedLabels ~= label) & (observedLabels == label));

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





%% Choose which area to select data from (to analyze clusters in that area, and/or to project same time points in another area
selectFrom = 'DS';
switch selectFrom
    case 'DS'
        projSelect = projectionsDS;
        projProject = projectionsM56;
        idSelect = idDS;
end



% Train and test model on single hold-out set
appendModelName = selectFrom;

dimToModel = 2:8;
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
    trainData = svmProj(training(cv), 1:dimToModel(n));  % UMAP Dimensions
    testData = svmProj(test(cv), 1:dimToModel(n)); % UMAP Dimensions


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
title(['Accuracy for dimensions', selectFrom, ' ',fitType, ' ', transWithinLabel])
titleE = ['Accuracy going into transitions- ', selectFrom, ' ',fitType, ' ', transWithinLabel];
saveas(gcf, fullfile(paths.figurePath, [titleE, '.png']), 'png')



% Analzyze the predictions vs observed

% Use above code to establish model to analyze

% fitType = 'UMAP 1-3';
fitType = ['UMAP 1-', num2str(dimToModel(n))];
% fitType = 'NeuralSpace';


monitorPositions = get(0, 'MonitorPositions');
secondMonitorPosition = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one

% Create a maximized figure on the second monitor
fig = figure(54); clf
set(fig, 'Position', secondMonitorPosition);
nPlot = length(codes) - 1;
[ax, pos] = tight_subplot(2, ceil(nPlot/2), [.04 .02], .1);

edges = -.5 : 1 : codes(end)+1;
for i = 2 : length(codes)
    % iObs = svmID == codes(i);
    iObsInd = testLabels == codes(i);
    iObs = testLabels(iObsInd);
    iPred = predictedLabels(iObsInd);
    % iAccuracy(i-1) = sum(svmID(iObs) == iPred) / sum(iObs);
    iAccuracy(i-1) = sum(iObs == iPred) / length(iObs);
    iPredWrong = iPred(iObs ~= iPred);
    N = histcounts(iPredWrong, edges, 'Normalization', 'pdf');
    % barCodes = codes(codes ~= codes(i)) + 1;
    barCodes = 0:codes(end);

    axes(ax(i-1))
    bar(barCodes, N, 'b', 'FaceAlpha', .5);
    iTitle = sprintf('%s (%d): %.2f', behaviors{i}, codes(i), iAccuracy(i-1));
    title(iTitle, 'interpreter', 'none');
end
iTitle = sprintf('Errors due to other behaviors: %s %s %s', selectFrom, fitType, transWithinLabel);
sgtitle(iTitle)
titleW = [selectFrom, ' ',fitType, ' ', transWithinLabel,' mislabeled as others'];
saveas(gcf, fullfile(paths.figurePath, [titleW, '.png']), 'png')

% Accuracies and F-scores

figure(55);
bar(codes(2:end), iAccuracy);
xticks(0:codes(end))
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
observedLabels = testLabels;
uniqueLabels = unique(observedLabels);
precision = zeros(size(uniqueLabels));
recall = zeros(size(uniqueLabels));
f1Score = zeros(size(uniqueLabels));

% Calculate F1 score for each unique label
for i = 1:length(uniqueLabels)
    label = uniqueLabels(i);

    % True positives (TP)
    tp = sum((predictedLabels == label) & (observedLabels == label));

    % False positives (FP)
    fp = sum((predictedLabels == label) & (observedLabels ~= label));

    % False negatives (FN)
    fn = sum((predictedLabels ~= label) & (observedLabels == label));

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








