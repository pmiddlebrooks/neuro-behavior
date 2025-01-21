%% Goal is to determine how much of the dynamics in the data is generated internally
% versus how much is input from external source.
% Fit a model with the neural data


idSelect = idM56;
bhvIDModel = bhvID;

%% Get data based on which model you want to use
modelType = 'glm'; % 'glm'
switch modelType
    case 'glm'
        dataModel = dataMat(:, idSelect);
    case 'svr'
dataModel = zscore(dataMat(:, idSelect));
end        

%% Or use low-d (pca) projected data (get pca data from pca_svm_decode_behaviors.m)
    projSelect = dataMat(:,idSelect) * coeff;
    nDim = find(cumsum(explained) > 90, 1);
dataModel = projSelect(:,1:nDim);
%% fitlm was taking too long. Trying GLM (fitglm) or SVR (support vector regression)
% Inputs:
% dataModel: Neural activity matrix (samples x neurons)

% Parameters
maxLag = 5; % Maximum lag for the autoregressive model
nSamples = size(dataModel, 1); % Number of samples
nNeurons = size(dataModel, 2); % Number of neurons
kernelFunction = 'rbf'; % Radial Basis Function kernel for SVR

% Preallocate results
varianceInternal = zeros(1, nNeurons); % Variance explained by internal dynamics
varianceAugmented = zeros(1, nNeurons); % Variance explained by augmented model

% SVR options
svrOptions = struct(...
    'KernelFunction', kernelFunction, ...
    'BoxConstraint', 1, ...        % Regularization parameter (adjust as needed)
    'Epsilon', 0.1);               % Epsilon-tube width for regression

% Parallel loop for computation
for neuronIdx = 1:nNeurons
    tic
    % parfor neuronIdx = 1:nNeurons
    % Extract the target neuron activity
    targetNeuron = dataModel(:, neuronIdx);

    % Create lagged matrix for autoregressive modeling
    laggedMatrix = lagmatrix(dataModel, 1:maxLag); % Lags for all neurons
    laggedMatrix = laggedMatrix(maxLag+1:end, :); % Remove NaNs due to lagging
    targetNeuronLagged = targetNeuron(maxLag+1:end); % Align target neuron activity

    % Exclude the target neuron's own data from the lagged matrix
    targetNeuronMask = true(1, nNeurons); % Create a mask
    targetNeuronMask(neuronIdx) = false; % Exclude the current neuron
    laggedMatrix = laggedMatrix(:, targetNeuronMask); % Keep only other neurons' data

    % Generate external predictors (e.g., random noise or other sources)
    externalPredictors = randn(size(targetNeuronLagged, 1), 2); % Two external predictors as example

    switch modelType
        case 'glm'
            % Model 1: Internal dynamics only
            mdlInternal = fitglm(laggedMatrix, targetNeuronLagged, 'Distribution', 'poisson'); % GLM with Gaussian
            varianceInternal(neuronIdx) = 1 - mdlInternal.Rsquared.Ordinary; % Residual variance explained

            % Model 2: Internal dynamics + external predictors
            mdlAugmented = fitglm([laggedMatrix, externalPredictors], targetNeuronLagged, 'Distribution', 'poisson');
            varianceAugmented(neuronIdx) = 1 - mdlAugmented.Rsquared.Ordinary; % Residual variance explained

        case 'svr'
            % Model 1: Internal dynamics only
            mdlInternal = fitrsvm(laggedMatrix, targetNeuronLagged, ...
                'KernelFunction', svrOptions.KernelFunction, ...
                'BoxConstraint', svrOptions.BoxConstraint, ...
                'Epsilon', svrOptions.Epsilon);
            predictedInternal = predict(mdlInternal, laggedMatrix);
            residualsInternal = targetNeuronLagged - predictedInternal;
            varianceInternal(neuronIdx) = 1 - (var(residualsInternal) / var(targetNeuronLagged)); % Variance explained

            % Model 2: Internal dynamics + external predictors
            mdlAugmented = fitrsvm([laggedMatrix, externalPredictors], targetNeuronLagged, ...
                'KernelFunction', svrOptions.KernelFunction, ...
                'BoxConstraint', svrOptions.BoxConstraint, ...
                'Epsilon', svrOptions.Epsilon);
            predictedAugmented = predict(mdlAugmented, [laggedMatrix, externalPredictors]);
            residualsAugmented = targetNeuronLagged - predictedAugmented;
            varianceAugmented(neuronIdx) = 1 - (var(residualsAugmented) / var(targetNeuronLagged)); % Variance explained
    end

    disp(toc)
end
% Compare variance explained
totalVariance = varianceInternal; % Total variance explained by internal dynamics
externalVariance = varianceAugmented - totalVariance; % Variance explained by external input

% Output results
disp('Variance explained by internal dynamics:');
disp(mean(totalVariance));
disp('Variance explained by external input:');
disp(mean(externalVariance));

% Plot the variance explained comparison
figure;
bar([mean(totalVariance), mean(externalVariance)]);
set(gca, 'XTickLabel', {'Internal Dynamics', 'External Input'});
ylabel('Variance Explained');
title('Variance Partitioning with GLM');


