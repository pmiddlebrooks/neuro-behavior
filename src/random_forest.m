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

%% This matrix keeps real time in order. So you can plot predictions in their order
dataBhvValid = dataBhv(allValid,:);

behaviorIDOrdered = [];
neuralMatrixOrdered = [];
for i = 1 : size(dataBhvValid, 1)
    iInd = dataBhvValid.StartFrame(i) : dataBhvValid.StartFrame(i) + dataBhvValid.DurFrame(i) - 1;
    neuralMatrixOrdered = [neuralMatrixOrdered; dataMatZ(iInd, :)];
    behaviorIDOrdered = [behaviorIDOrdered; dataBhvValid.ID(i) * ones(length(iInd), 1)];
end

%% Fit a random forest classifier, using out of bag samples to predict the accuracy
neuralM56 = neuralMatrix(:,idM56);
neuralDS = neuralMatrix(:,idDS);


%% This matrix is balanced using a maximum number of bouts then performing smote to increase undersampled behaviors
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
%% Fit a random forest classifier, using out of bag samples to predict the accuracy
neuralM56 = neuralMatrix(:,idM56);
neuralDS = neuralMatrix(:,idDS);



%%
nC = groupcounts(behaviorID);
N = max(nC)./nC-1;
k = ceil(N);
k(k < 5) = 5;
[neuralSmote,bhvSmote,Xn,Cn] = smote(neuralMatrix, N, k, 'Class', behaviorID);



%%

% Random Forest Classification using TreeBagger
numTrees = 100; % Define the number of trees
% randomForestModel = TreeBagger(numTrees, neuralM56, behaviorID, 'OOBPrediction', 'On');
randomForestModel = TreeBagger(numTrees, neuralSmote(:, idM56), bhvSmote, OOBPrediction='On', OOBPredictorImportance='On');

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
accuracy = sum(str2double(oobPredictions) == bhvSmote) / length(bhvSmote);
disp(['OOB Prediction Accuracy: ', num2str(accuracy)]);

%%  
% % Plot a text description of the trees
% view(randomForestModel.Trees{1}, 'Mode', 'text');
% 
% % Plot a graphical version of one of the trees (e.g., the first tree)
% view(randomForestModel.Trees{1}, 'Mode', 'graph');




%%
% Assuming randomForestModel is already trained and a new neuralMatrix is available

% Predict behaviorID using the random forest model
predictedBehaviorID = predict(randomForestModel, neuralM56);
predictedBehaviorID = str2double(predictedBehaviorID); % Convert to numerical values if necessary

% Create 2 x 1 subplot
figure(43);
clf
subplot(2, 1, 1);
plot(behaviorID, '.'); % Plot actual behaviorID
title('Actual Behavior ID');
xlabel('Index');
ylabel('Behavior ID');

subplot(2, 1, 2);
plot(predictedBehaviorID, '.'); % Plot predicted behaviorID
title('Predicted Behavior ID');
xlabel('Index');
ylabel('Behavior ID');

accuracy = sum(predictedBehaviorID == behaviorID) / length(behaviorID);
disp(['Training Prediction Accuracy: ', num2str(accuracy)]);




%% Cross-validation to get training accuracies (not out-of-bag)
 accuracies = perform_rf_classification(neuralM56, behaviorID);
 accuracies = perform_rf_classification(neuralSmote(:,idM56), bhvSmote)


 %%
 [confMat, order] = plot_confusion_matrix(behaviorID, predictedBehaviorID)

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



function [confMat, order] = plot_confusion_matrix(obsBehaviorID, predBehaviorID)
    % Function to plot confusion matrix for given predicted and observed values

    % Create confusion matrix
    [confMat, order] = confusionmat(obsBehaviorID, predBehaviorID);

    % Plot confusion matrix
    figure(83);
    confusionchart(obsBehaviorID, order);
    title('Confusion Matrix');
    xlabel('Predicted Class');
    ylabel('True Class');
end