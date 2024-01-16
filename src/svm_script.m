

%%
[svmModels, meanValidationAccuracies] = multiClassSVM(neuralMatrix, behaviorID)


%%

% function svmModel = multiClassSVM(neuralMatrix, behaviorID)
%     % Convert behaviorID to categorical if it's not already
%     if ~iscategorical(behaviorID)
%         behaviorID = categorical(behaviorID);
%     end
% 
%     % % Set options for hyperparameter optimization
%     % opts = struct('Optimizer', 'bayesopt', 'ShowPlots', false, 'CVPartition', cvpartition(behaviorID, 'KFold', 5), ...
%     %               'AcquisitionFunctionName', 'expected-improvement-plus');
%     % 
%     % % Multi-class SVM model with hyperparameter optimization
%     % svmModel = fitcecoc(neuralMatrix, behaviorID, ...
%     %                     'Learners', 'svm', 'Coding', 'onevsall', ...
%     %                     'OptimizeHyperparameters', 'auto', ...
%     %                     'HyperparameterOptimizationOptions', opts);
% 
% 
% 
%     % Define the template for SVM with hyperparameter optimization
%     t = templateSVM('Standardize', true, 'KernelFunction', 'gaussian', ...
%                     'OptimizeHyperparameters', 'auto', ...
%                     'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
%                     'expected-improvement-plus'));
%     Define the template for SVM with hyperparameter optimization
%     % t = templateSVM('Standardize', true, 'KernelFunction', 'gaussian');
% 
%     % Multi-class SVM model with hyperparameter optimization
%     svmModel = fitcecoc(neuralMatrix, behaviorID, 'Learners', t, 'Coding', 'onevsall');
% 
%     % Optional: Cross-validation (can be commented out if not needed)
%     % cvsSVM = crossval(svmModel, 'KFold', 10); % 10-fold cross-validation
%     % validationAccuracy = 1 - kfoldLoss(cvsSVM, 'LossFun', 'ClassifError')
% 
%     % Save the model
%     % save('OptimizedSVMModel.mat', 'svmModel');
% end



% function svmModel = multiClassSVM(neuralMatrix, behaviorID)
%     % Convert behaviorID to categorical if it's not already
%     if ~iscategorical(behaviorID)
%         behaviorID = categorical(behaviorID);
%     end
% 
% 
%     % Partition the data for k-fold cross-validation
%     c = cvpartition(behaviorID, 'KFold', 10); % 10-fold cross-validation
% 
%     % Initialize variables to store cross-validation results
%     kFoldModel = cell(c.NumTestSets, 1);
%     validationAccuracy = zeros(c.NumTestSets, 1);
% 
%     % Perform k-fold cross-validation
%     for i = 1:c.NumTestSets
%         trainingIdx = training(c, i);
%         testIdx = test(c, i);
% 
%             % Set options for hyperparameter optimization
%     opts = struct('Optimizer', 'bayesopt', 'ShowPlots', false, 'CVPartition', cvpartition(behaviorID(trainingIdx), 'KFold', 5), ...
%                   'AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 20);
% 
%         % Train the model on the training set
%         svmModel = fitcecoc(neuralMatrix(trainingIdx, :), behaviorID(trainingIdx), ...
%                             'Learners', 'svm', ...
%                             'OptimizeHyperparameters', 'auto', ...
%                             'HyperparameterOptimizationOptions', opts);
% 
%         % Save the trained model for each fold
%         kFoldModel{i} = svmModel;
% 
%         % Test the model on the test set and calculate accuracy
%         predictions = predict(svmModel, neuralMatrix(testIdx, :));
%         validationAccuracy(i) = sum(predictions == behaviorID(testIdx)) / length(testIdx)
%     end
% 
%     % Calculate the average accuracy over all folds
%     meanValidationAccuracy = mean(validationAccuracy)
% 
%     % Optionally, you can choose to save the k-fold models and the accuracies
%     save('kFoldSVMModels.mat', 'kFoldModel', 'validationAccuracy', 'meanValidationAccuracy');
% end


function [svmModels, meanValidationAccuracies] = multiClassSVM(neuralMatrix, behaviorID)
    % Convert behaviorID to categorical if it's not already
    if ~iscategorical(behaviorID)
        behaviorID = categorical(behaviorID);
    end

    % Define the kernel functions to test
    kernelFunctions = {'linear', 'polynomial', 'rbf'}; % RBF is Gaussian

    % Initialize storage for results
    svmModels = cell(length(kernelFunctions), 1);
    meanValidationAccuracies = zeros(length(kernelFunctions), 1);

    % Iterate over each kernel function
    for k = 1:length(kernelFunctions)
kernelFunctions{k}
        % Partition the data for k-fold cross-validation
        c = cvpartition(behaviorID, 'KFold', 10); % 10-fold cross-validation

        % Initialize variables to store cross-validation results
        kFoldModel = cell(c.NumTestSets, 1);
        validationAccuracy = zeros(c.NumTestSets, 1);

        % Perform k-fold cross-validation
        for i = 1:c.NumTestSets
            trainingIdx = training(c, i);
            testIdx = test(c, i);

        % Set options for hyperparameter optimization
        opts = struct('Optimizer', 'bayesopt', 'ShowPlots', false, 'CVPartition', cvpartition(behaviorID(trainingIdx), 'KFold', 5), ...
                      'AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 15);

        % Train the model on the training set with specified kernel
            svmModel = fitcecoc(neuralMatrix(trainingIdx, :), behaviorID(trainingIdx), ...
                                'Learners', templateSVM('KernelFunction', kernelFunctions{k}), ...
                                'Coding', 'onevsall', ...
                                'OptimizeHyperparameters', 'auto', ...
                                'HyperparameterOptimizationOptions', opts);

            % Save the trained model for each fold
            kFoldModel{i} = svmModel;

            % Test the model on the test set and calculate accuracy
            predictions = predict(svmModel, neuralMatrix(testIdx, :));
            validationAccuracy(i) = sum(predictions == behaviorID(testIdx)) / length(testIdx);
        end

        % Calculate the average accuracy over all folds for the current kernel
        meanValidationAccuracy = mean(validationAccuracy);

        % Store the model and accuracy
        svmModels{k} = kFoldModel;
        meanValidationAccuracies(k) = meanValidationAccuracy;
    end

    % Save the results
    % save('KernelComparisonResults.mat', 'svmModels', 'meanValidationAccuracies', 'kernelFunctions');
end
