

%% Test out different kernels (linear, polynomial, and rbf (gaussian)).

[bestModels, allAccuracies] = multiClassSVM(neuralMatrix, behaviorID)


%% The polynomial model fit best (cross-validated). Try out different polynomial orders (matlab default is 3)

[bestModelsPoly, allAccuraciesPoly] = polynomialKernelSVM(neuralMatrix, behaviorID)


%% Notes about hyperparameters:
%
% BoxConstraint:  (summary: Small/big ~ high/low bias ~ more/less misclassification)
%  - This hyperparameter, also known as the C parameter in some contexts,
% controls the penalty for misclassified data points in an SVM. It's a regularization parameter.
%  - A smaller BoxConstraint value allows more misclassifications but tries to
% ensure that the decision boundary is as simple as possible, which can
% help in avoiding overfitting. This might be beneficial in cases where the data is noisy.
%  - A larger BoxConstraint value, on the other hand, puts more emphasis on
% correctly classifying all training points, which can lead to a more
% complex decision boundary. However, this can also make the model more
% prone to overfitting, especially if the training data is not
% representative of the general population of data.    z
%  - Essentially, BoxConstraint is a trade-off between finding a decision
% boundary that generalizes well to new data (lower values) and one that
% performs well on the training data (higher values).
%
% KernelScale:  (summary: Small/big ~ low/high bias ~ less/more misclassification)
%  - This hyperparameter is specific to kernelized SVMs. It's used to scale
%  the data before applying the kernel function. For the Radial Basis
%  Function (RBF) or Gaussian kernel, KernelScale is particularly important.
%  - The KernelScale parameter can be thought of as a way to control the
%  spread or reach of the kernel function. A small KernelScale means that
%  the kernel function will decay rapidly, making the SVM sensitive to the
%  data points that are close to the decision boundary (leading to a more
%  complex, wiggly boundary). This can capture finer details in the data but may lead to overfitting.
%  - A larger KernelScale value results in a smoother decision boundary as
%  the kernel function's effect spreads out more, which might be better for
%  generalizing to new data but can smooth over important details in the data.
%  - Choosing the right KernelScale is about balancing the need to capture
%  the complexity in the data against the risk of overfitting to the
%  training data.
%%

function [bestModels, allAccuracies] = multiClassSVM(neuralMatrix, behaviorID)
% Convert behaviorID to categorical if it's not already
if ~iscategorical(behaviorID)
    behaviorID = categorical(behaviorID);
end

% Define some constants
nObjectiveEval = 25;
nKFold = 5;

% Define the kernel functions to test
kernelFunctions = {'linear', 'polynomial', 'rbf'}; % RBF is Gaussian

% Set options for hyperparameter optimization
opts = struct('Optimizer', 'bayesopt', 'ShowPlots', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', nObjectiveEval);

% Initialize storage for results
bestModels = cell(length(kernelFunctions), 1);
allAccuracies = cell(length(kernelFunctions), 1);

% Iterate over each kernel function
for k = 1:length(kernelFunctions)
    kernelFunctions(k)
    % Train the best model using all data with specified kernel
    t = templateSVM('KernelFunction', kernelFunctions{k});

    bestModel = fitcecoc(neuralMatrix, behaviorID, 'Learners', t, ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', opts);

    % Extract optimized hyperparameters from the first binary learner
    % Trying without BoxConstraint to see if it works ok
    kernelScale = bestModel.BinaryLearners{1}.KernelParameters.Scale;
    codingName = bestModel.CodingName;

    % Store the best model
    bestModels{k} = bestModel;

    % Partition the data for k-fold cross-validation
    c = cvpartition(behaviorID, 'KFold', nKFold); % 5-fold cross-validation

    % Initialize array to store accuracies for each fold
    foldAccuracies = zeros(c.NumTestSets, 1);

    % Perform cross-validation
    for i = 1:c.NumTestSets
        trainingIdx = training(c, i);
        testIdx = test(c, i);

        % Train the model on the training set using the optimized parameters
        t = templateSVM('KernelFunction', kernelFunctions{k}, ...
            'KernelScale', kernelScale);
        svmModel = fitcecoc(neuralMatrix(trainingIdx, :), behaviorID(trainingIdx), 'Learners', t, ...
            'Coding', codingName);

        % Test the model on the test set and calculate accuracy
        predictions = predict(svmModel, neuralMatrix(testIdx, :));
        foldAccuracies(i) = sum(predictions == behaviorID(testIdx)) / length(testIdx);
    end

    % Store accuracies for each kernel
    allAccuracies{k} = foldAccuracies;
end
end



function [bestModels, allAccuracies] = polynomialKernelSVM(neuralMatrix, behaviorID)
% Convert behaviorID to categorical if it's not already
if ~iscategorical(behaviorID)
    behaviorID = categorical(behaviorID);
end

% Define some constants
nObjectiveEval = 25;
nKFold = 5;


% Define the range of polynomial orders to test
polynomialOrders = 3:5;

% Set options for hyperparameter optimization
opts = struct('Optimizer', 'bayesopt', 'ShowPlots', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', nObjectiveEval);

% Initialize storage for results
bestModels = cell(length(polynomialOrders), 1);
allAccuracies = cell(length(polynomialOrders), 1);

% Iterate over each polynomial order
for k = 1:length(polynomialOrders)
    polynomialOrders(k)
    % Train the best model using all data with specified polynomial order
    t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', polynomialOrders(k));
    bestModel = fitcecoc(neuralMatrix, behaviorID, 'Learners', t, ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', opts);

    kernelScale = bestModel.BinaryLearners{1}.KernelParameters.Scale;
    codingName = bestModel.CodingName;

    % Store the best model
    bestModels{k} = bestModel;

    % Partition the data for 5-fold cross-validation
    c = cvpartition(behaviorID, 'KFold', nKFold);

    % Initialize array to store accuracies for each fold
    foldAccuracies = zeros(c.NumTestSets, 1);

    % Perform cross-validation
    for i = 1:c.NumTestSets
        trainingIdx = training(c, i);
        testIdx = test(c, i);

        % Train the model on the training set using the optimized parameters
        t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', polynomialOrders(k), ...
            'KernelScale', kernelScale);
        svmModel = fitcecoc(neuralMatrix(trainingIdx, :), behaviorID(trainingIdx), 'Learners', t, ...
            'Coding', codingName);

        % Test the model on the test set and calculate accuracy
        predictions = predict(svmModel, neuralMatrix(testIdx, :));
        foldAccuracies(i) = sum(predictions == behaviorID(testIdx)) / length(testIdx);
    end

    % Store accuracies for each polynomial order
    allAccuracies{k} = foldAccuracies;
end
end

