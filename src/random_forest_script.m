%% Go to spiking_script and get behavioral and neural data


%% Settings for building the neural and behavioral matrices for the random forest tree
matchBouts = 0;

%% Make a vector of behavioral IDs for each neural matrix frame
dataBhv.DurFrame(dataBhv.DurFrame == 0) = 1;

%% This matrix is unordered w.r.t. real time. It concatenates in order of behaviors 
% (so you can't plot ongoing behavioral changes when predicting the random
% forest model)
behaviorID = [];
neuralMatrix = [];
for iBhv = 1 : length(analyzeCodes)
    dataBhvInd = dataBhv.ID == analyzeCodes(iBhv) & validBhv(:, codes == analyzeCodes(iBhv));
    iStartFrames = dataBhv.StartFrame(dataBhvInd);
    iDurFrames = dataBhv.DurFrame(dataBhvInd);
    if matchBouts
        iRand = randperm(length(iStartFrames));
        iStartFrames = iStartFrames(iRand(1:nSample));
        iDurFrames = iDurFrames(iRand(1:nSample)); 
    end
    for jStart = 1 : length(iStartFrames)
        % Concatenated the neural matrix and behavior ID
        dataWindow = 1 : iDurFrames(jStart);
        behaviorID = [behaviorID; analyzeCodes(iBhv) * ones(length(dataWindow), 1)];
        neuralMatrix = [neuralMatrix; dataMatZ(iStartFrames(jStart) + dataWindow, :)];
    end
end

%% Make area-specific neural matrices
neuralM56 = neuralMatrix(:,idM56);
neuralDS = neuralMatrix(:,idDS);


%% This matrix keeps real time in order. So you can plot predictions in their order
dataBhvValid = dataBhv(dataBhv.Valid,:);

behaviorIDOrdered = [];
neuralMatrixOrdered = [];
for i = 1 : size(dataBhvValid, 1)
    iInd = dataBhvValid.StartFrame(i) : dataBhvValid.StartFrame(i) + dataBhvValid.DurFrame(i) - 1;
    neuralMatrixOrdered = [neuralMatrixOrdered; dataMatZ(iInd, :)];
    behaviorIDOrdered = [behaviorIDOrdered; dataBhvValid.ID(i) * ones(length(iInd), 1)];
end





%% SMOTE: This matrix is balanced using a maximum number of bouts then performing smote to increase undersampled behaviors
behaviorID = [];
neuralMatrix = [];
maxBout = 200;
for iBhv = 1 : length(analyzeCodes)
    dataBhvInd = dataBhv.ID == analyzeCodes(iBhv) & validBhv(:, codes == analyzeCodes(iBhv));
    iStartFrames = dataBhv.StartFrame(dataBhvInd);
    iDurFrames = dataBhv.DurFrame(dataBhvInd);
    if sum(dataBhvInd) > maxBout
        iRand = randperm(length(iStartFrames));
        iStartFrames = iStartFrames(iRand(1:maxBout));
        iDurFrames = iDurFrames(iRand(1:maxBout));
    end
    for jStart = 1 : length(iStartFrames)
        % Concatenated the neural matrix and behavior ID
        dataWindow = 1 : iDurFrames(jStart);
        behaviorID = [behaviorID; analyzeCodes(iBhv) * ones(length(dataWindow), 1)];
        neuralMatrix = [neuralMatrix; dataMatZ(iStartFrames(jStart) + dataWindow, :)];
    end
end

% Form the upsampled (smote) neural matrix
nC = groupcounts(behaviorID);
N = max(nC)./nC-1;
k = ceil(N);
k(k < 5) = 5;
[neuralSmote, bhvSmote, Xn, Cn] = smote(neuralMatrix, N, k, 'Class', behaviorID);

%% Hold out 20% of smoted matrix for testing
testProp = .2;
nTest = floor(length(bhvSmote) * testProp);
permInd = randperm(length(bhvSmote));
testInd = permInd(1 : nTest);
trainInd = permInd(nTest + 1 : end);

neuralSmoteTrain = neuralSmote(trainInd,:);
bhvSmoteTrain = bhvSmote(trainInd);
neuralSmoteTest = neuralSmote(testInd,:);
bhvSmoteTest = bhvSmote(testInd);




%% Random Forest Classification using TreeBagger

numTrees = 100; % Define the number of trees
% randomForestModel = TreeBagger(numTrees, neuralM56, behaviorID, 'OOBPrediction', 'On');
randomForestModel = TreeBagger(numTrees, neuralSmoteTrain(:, idM56), bhvSmoteTrain, ...
    OOBPrediction='On', OOBPredictorImportance='On');

% Calculate OOB Error
oobErrorVal = oobError(randomForestModel);

% Plot OOB Error as a function of the number of grown trees
figure(402);
plot(oobErrorVal);
title('Out-of-Bag Error as a Function of Number of Trees');
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');

% Predict Accuracy using oobPredict
[oobPredictions, scores] = oobPredict(randomForestModel);
accuracy = sum(str2double(oobPredictions) == bhvSmoteTrain) / length(oobPredictions);
disp(['OOB Prediction Accuracy: ', num2str(accuracy)]);
  
% % Plot a text description of the trees
% view(randomForestModel.Trees{1}, 'Mode', 'text');
% 
% % Plot a graphical version of one of the trees (e.g., the first tree)
% view(randomForestModel.Trees{1}, 'Mode', 'graph');


%% Which neurons are more/less important?
importanceVal = randomForestModel.OOBPermutedPredictorDeltaError;

figure;
bar(importanceVal);
title('Predictor Importances');
xlabel('Predictors');
ylabel('Importance');
xticks(1:length(importanceVal));
xticklabels(randomForestModel.PredictorNames);




%% You have a fitted RF model. Use it to predict behaviors: confusion matrix
neuralMat = neuralM56;
bhvID = behaviorID;
predBhvID = rf_predict_behavior(randomForestModel, neuralMat, bhvID);




 %% Plot the confusion matrix
 [confMat, C_norm] = plot_confusion_matrix(bhvID, predBhvID, analyzeBhv);
 % [confMat, order] = plot_confusion_matrix(bhvSmote, predictedBehaviorID)


 %% Cross-validation to get training accuracies (not out-of-bag)
 % accuracies = perform_rf_classification(neuralM56, behaviorID);
 accuracies = perform_rf_classification(neuralSmote(:,idM56), bhvSmote)

























%%                  FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function accuracies = perform_rf_classification(neuralMatrix, behaviorID)
    % Function to perform random forest classification with 5-fold cross-validation

    % Number of trees for the random forest
    numTrees = 100;
    
    % Number of folds for cross-validation
    numFolds = 5;
    
    % Initialize accuracies array
    accuracies = zeros(numFolds, 1);
    
    % Cross-validation
    cv = cvpartition(length(behaviorID), 'KFold', numFolds);
    
    for i = 1:numFolds
        % Indices for training and test set
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        
        % Create random forest model
        randomForestModel = TreeBagger(numTrees, neuralMatrix(trainIdx, :), behaviorID(trainIdx));
        
        % Predict on test set
        predicted = predict(randomForestModel, neuralMatrix(testIdx, :));
        predicted = str2double(predicted); % Convert to numerical values if necessary
        
        % Calculate accuracy
        accuracies(i) = sum(predicted == behaviorID(testIdx)) / length(testIdx);
    end
end


function predBhvID = rf_predict_behavior(rfModel, neuralMatrix, bhvID)
% Predict behaviorID using the random forest model
behaviors = unique(bhvID);
colors = colors_for_behaviors(behaviors);
predBhvID = predict(rfModel, neuralMatrix);
predBhvID = str2double(predBhvID); % Convert to numerical values if necessary

% Create 2 x 1 subplot
figure(43);
clf

for i = 1 : length(behaviors)
subplot(2, 1, 1)
hold on
bhvX = find(bhvID == behaviors(i));
bhvY = bhvID(bhvID == behaviors(i));
plot(bhvX, bhvY, '.', 'color', colors(i,:)); % Plot actual behaviorID
% plot(bhvID, '.'); % Plot actual behaviorID

subplot(2, 1, 2)
hold on
bhvX = find(predBhvID == behaviors(i));
bhvY = predBhvID(predBhvID == behaviors(i));
plot(bhvX, bhvY, '.', 'color', colors(i,:)); % Plot actual behaviorID
% plot(predBhvID, '.'); % Plot predicted behaviorID
end
subplot(2, 1, 1);
title('B-Soid Behavior ID');
xlabel('Index');
ylabel('Behavior ID');

subplot(2, 1, 2);
title('RF-Predicted Behavior ID');
xlabel('Index');
ylabel('Behavior ID');

accuracy = sum(predBhvID == bhvID) / length(predBhvID);
disp(['Training Prediction Accuracy: ', num2str(accuracy)]);
end



function [C, C_norm] = plot_confusion_matrix(predBhvID, obsBhvID, analyzeBhv)
    % Function to plot two confusion matrices: absolute values and normalized values

    % Create confusion matrix
    [C, order] = confusionmat(obsBhvID, predBhvID);
    
    % Normalize confusion matrix
    C_norm = C ./ sum(C, 2);

    % Create figure
    figure(83);
    clf
    set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
    % Plot 1: Absolute values using confusionchart
    subplot(1, 2, 1);
    confusionchart(C, order);
    % set(gca, 'XTickLabel', analyzeBhv);
    % set(gca, 'YTickLabel', analyzeBhv);
    title('Confusion Matrix (Absolute Values)');
    
    % Plot 2: Normalized values in grayscale
    subplot(1, 2, 2);
    colormap('gray'); % Set colormap to grayscale
    colormap(flipud(gray)); % Set colormap to grayscale
    imagesc(C_norm); % Display the normalized matrix as an image
    % C_norm2 = imcomplement(C_norm);
    % imagesc(C_norm2)
    colorbar; % Show color scale
    axis square; % Make the plot square
    
    % Setting axes and labels for the grayscale plot
    set(gca, 'XTick', 1:length(order), 'XTickLabel', analyzeBhv);
    set(gca, 'YTick', 1:length(order), 'YTickLabel', analyzeBhv);
    % set(gca, 'XTick', 1:length(order), 'XTickLabel', order);
    % set(gca, 'YTick', 1:length(order), 'YTickLabel', order);
    title('Normalized Confusion Matrix (Grayscale)');
    xlabel('Predicted Class');
    ylabel('True Class');
end
