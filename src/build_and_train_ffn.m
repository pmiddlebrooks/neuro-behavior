% function [model, history] = build_and_train_ffn(dataMat, bhvIDMat, input_dim, output_dim, hidden_layers, activation, epochs, batch_size, learning_rate)
%     % Build and train a feedforward neural network with 3-fold cross-validation to predict bhvIDMat from dataMat.
%     %
%     % Parameters:
%     % dataMat: matrix, input data matrix (samples x features)
%     % bhvIDMat: matrix, target labels (samples x output_dim)
%     % input_dim: scalar, number of features in the input data
%     % output_dim: scalar, number of output neurons (e.g., number of classes or targets)
%     % hidden_layers: vector, number of neurons in each hidden layer
%     % activation: string, activation function for hidden layers ('relu', 'sigmoid', etc.)
%     % epochs: scalar, number of training epochs
%     % batch_size: scalar, batch size for training
%     % learning_rate: scalar, learning rate for the optimizer
%     %
%     % Returns:
%     % model: Trained MATLAB neural network model (from the best fold)
%     % history: Training history object (from the best fold)
% 
%     % Create a feedforward neural network
%     layers = [ 
%         featureInputLayer(input_dim, 'Normalization', 'zscore', 'Name', 'input')
%     ];
% 
%     % Add hidden layers
%     for i = 1:length(hidden_layers)
%         layers = [layers; 
%             fullyConnectedLayer(hidden_layers(i), 'Name', sprintf('fc%d', i));
%             batchNormalizationLayer('Name', sprintf('batchNorm%d', i));
%             reluLayer('Name', sprintf('relu%d', i))
%         ];
%     end
% 
%     % Add output layer
%     if output_dim > 1
%         % For classification problems
%         layers = [layers; 
%             fullyConnectedLayer(output_dim, 'Name', 'output');
%             softmaxLayer('Name', 'softmax');
%             classificationLayer('Name', 'classificationoutput')
%         ];
%     else
%         % For regression problems
%         layers = [layers; 
%             fullyConnectedLayer(1, 'Name', 'output');
%             regressionLayer('Name', 'regression')
%         ];
%     end
% 
%     % Initialize cross-validation
%     k = 3; % Number of folds
%     cv = cvpartition(size(dataMat, 1), 'KFold', k);
% 
%     bestAccuracy = -Inf;
%     bestModel = [];
%     bestHistory = [];
% 
%     for fold = 1:k
%         fprintf('Training fold %d/%d\n', fold, k);
% 
%         % Split data into training and test sets for the current fold
%         trainIdx = training(cv, fold);
%         testIdx = test(cv, fold);
% 
%         dataTrain = dataMat(trainIdx, :);
%         bhvTrain = bhvIDMat(trainIdx);
%         dataTest = dataMat(testIdx, :);
%         bhvTest = bhvIDMat(testIdx);
% 
%         % Specify training options
%         options = trainingOptions('adam', ...
%             'InitialLearnRate', learning_rate, ...
%             'MaxEpochs', epochs, ...
%             'MiniBatchSize', batch_size, ...
%             'Plots', 'none', ...
%             'Verbose', false);
% 
%         % Train the network
%         if output_dim > 1
%             bhvTrain = categorical(bhvTrain);
%             bhvTest = categorical(bhvTest);
%         end
%         tempModel = trainNetwork(dataTrain, bhvTrain, layers, options);
% 
%         % Evaluate the model
%         predictions = predict(tempModel, dataTest);
%         if output_dim > 1
%             uniqueClasses = unique(bhvIDMat);
%             [~, predictedIndices] = max(predictions, [], 2);
%             predictedLabels = categorical(uniqueClasses(predictedIndices));
%             % [~, predictedLabels] = max(predictions, [], 2);
%             accuracy = sum(predictedLabels == bhvTest) / numel(bhvTest);
%         else
%             accuracy = -mean((predictions - bhvTest).^2); % Negative MSE for comparison
%         end
% 
%         % Save the best model
%         if accuracy > bestAccuracy
%             bestAccuracy = accuracy;
%             bestModel = tempModel;
%             bestHistory = tempModel; % MATLAB does not provide a direct 'history' object
%         end
%     end
% 
%     fprintf('Best accuracy across folds: %.2f\n', bestAccuracy);
% 
%     % Return the best model and its history
%     model = bestModel;
%     history = bestHistory;
% end
% 
% % % Example usage
% % if exist('OCTAVE_VERSION', 'builtin')
% %     warning('This script is designed for MATLAB and may not be compatible with Octave.');
% % else
% %     % Dummy example data
% %     rng(42);
% %     dataMat = rand(100, 20);  % 100 samples, 20 features
% %     bhvIDMat = randi([0, 1], 100, 1);  % Binary classification (100 samples)
% % 
% %     input_dim = size(dataMat, 2);
% %     output_dim = 1;  % Change to number of classes for classification
% % 
% %     % Train the feedforward network with cross-validation
% %     [model, history] = build_and_train_ffn(dataMat, bhvIDMat, input_dim, output_dim, [64, 32], 'relu', 20, 16, 0.001);
% % end


function [bestModel, bestHistory, hyperparamResults] = build_and_train_ffn(dataMat, bhvIDMat, input_dim, output_dim, max_trials, epochs, batch_size)
    % Optimize hyperparameters and train a feedforward neural network with dropout and 3-fold cross-validation.
    %
    % Parameters:
    % dataMat: matrix, input data matrix (samples x features)
    % bhvIDMat: matrix, target labels (samples x output_dim)
    % input_dim: scalar, number of features in the input data
    % output_dim: scalar, number of output neurons (e.g., number of classes or targets)
    % max_trials: scalar, number of hyperparameter combinations to evaluate
    % epochs: scalar, number of training epochs
    % batch_size: scalar, batch size for training
    %
    % Returns:
    % bestModel: Trained MATLAB neural network model with optimal hyperparameters
    % bestHistory: Training history object for the best model
    % hyperparamResults: Table of hyperparameter configurations and their corresponding accuracies

    % Define hyperparameter ranges
    hiddenLayerOptions = [1, 2, 3]; % Number of hidden layers
    unitsOptions = [32, 64, 128, 256]; % Units per layer
    dropoutOptions = [0.0, 0.2, 0.5]; % Dropout rates
    learningRateOptions = logspace(-4, -2, 5); % Learning rates

    % Initialize cross-validation
    k = 3; % Number of folds
    cv = cvpartition(size(dataMat, 1), 'KFold', k);

    bestAccuracy = -Inf;
    bestModel = [];
    bestHistory = [];
    results = []; % Store hyperparameter results

    % Random search over hyperparameters
    for trial = 1:max_trials
        fprintf('Trial %d/%d\n', trial, max_trials);

        % Randomly sample hyperparameters
        numHiddenLayers = randsample(hiddenLayerOptions, 1);
        unitsPerLayer = randsample(unitsOptions, numHiddenLayers, true);
        dropoutRate = randsample(dropoutOptions, 1);
        learningRate = randsample(learningRateOptions, 1);

        % Build the network layers
        layers = [ 
            featureInputLayer(input_dim, 'Normalization', 'zscore', 'Name', 'input')
        ];

        for i = 1:numHiddenLayers
            layers = [layers; 
                fullyConnectedLayer(unitsPerLayer(i), 'Name', sprintf('fc%d', i));
                batchNormalizationLayer('Name', sprintf('batchNorm%d', i));
                reluLayer('Name', sprintf('relu%d', i));
                dropoutLayer(dropoutRate, 'Name', sprintf('dropout%d', i))
            ];
        end

        % Add output layer
        if output_dim > 1
            layers = [layers; 
                fullyConnectedLayer(output_dim, 'Name', 'output');
                softmaxLayer('Name', 'softmax');
                classificationLayer('Name', 'classificationoutput')
            ];
        else
            layers = [layers; 
                fullyConnectedLayer(1, 'Name', 'output');
                regressionLayer('Name', 'regression')
            ];
        end

        % Cross-validation for current hyperparameters
        trialAccuracy = 0;
        for fold = 1:k
            fprintf('  Fold %d/%d\n', fold, k);

            % Split data into training and test sets for the current fold
            trainIdx = training(cv, fold);
            testIdx = test(cv, fold);

            dataTrain = dataMat(trainIdx, :);
            bhvTrain = bhvIDMat(trainIdx, :);
            dataTest = dataMat(testIdx, :);
            bhvTest = bhvIDMat(testIdx, :);

            % Specify training options
            options = trainingOptions('adam', ...
                'InitialLearnRate', learningRate, ...
                'MaxEpochs', epochs, ...
                'MiniBatchSize', batch_size, ...
                'Plots', 'none', ...
                'Verbose', false);

            % Train the network
            if output_dim > 1
                bhvTrain = categorical(bhvTrain);
                bhvTest = categorical(bhvTest);
            end
            tempModel = trainNetwork(dataTrain, bhvTrain, layers, options);

            % Evaluate the model
            predictions = predict(tempModel, dataTest);
            if output_dim > 1
                uniqueClasses = unique(bhvIDMat);
                [~, predictedIndices] = max(predictions, [], 2);
                predictedLabels = uniqueClasses(predictedIndices);
                foldAccuracy = sum(categorical(predictedLabels) == bhvTest) / numel(bhvTest);
            else
                foldAccuracy = -mean((predictions - bhvTest).^2); % Negative MSE for comparison
            end

            trialAccuracy = trialAccuracy + foldAccuracy;
        end

        % Average accuracy across folds
        trialAccuracy = trialAccuracy / k;

        % Record hyperparameters and accuracy
        results = [results; table(numHiddenLayers, {unitsPerLayer}, dropoutRate, learningRate, trialAccuracy, ...
            'VariableNames', {'NumHiddenLayers', 'UnitsPerLayer', 'DropoutRate', 'LearningRate', 'Accuracy'})];

        % Save the best model
        if trialAccuracy > bestAccuracy
            bestAccuracy = trialAccuracy;
            bestModel = tempModel;
            bestHistory = tempModel; % MATLAB does not provide a direct 'history' object
            fprintf('  New best accuracy: %.2f\n', bestAccuracy);
        end
    end

    % Output the results table
    fprintf('Best accuracy after optimization: %.2f\n', bestAccuracy);
    hyperparamResults = results;
    disp(results);
end

