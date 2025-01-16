%%
idSelect = idM56;
dataModel = zscore(dataMat(:, idSelect));
% dataModel = zscore(dataMat(:, idSelect(1:3)));
bhvIDModel = bhvID;

% Inputs:
% dataModel: Neural activity matrix (samples x neurons)

% Parameters
maxLag = 5; % Maximum lag for the autoregressive model
nSamples = size(dataModel, 1); % Number of samples
nNeurons = size(dataModel, 2); % Number of neurons

% Preallocate results
varianceInternal = zeros(1, nNeurons); % Variance explained by internal dynamics
varianceAugmented = zeros(1, nNeurons); % Variance explained by augmented model
externalInputCell = cell(1, nNeurons); % Temporary storage for external inputs

% Optimization options
optimOptions = optimoptions('fminunc', 'Display', 'off', 'Algorithm', 'quasi-newton');

poolID = parpool(4, 'IdleTimeout', Inf);
% Parallel loop for computation
parfor neuronIdx = 1:nNeurons
    tic
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
    
    % Define the loss function for optimization
    lossFunction = @(u) computeModelLoss(targetNeuronLagged, laggedMatrix, u);
    
    % Initialize the external input (random guess)
    uInit = randn(size(targetNeuronLagged));
    
    % Optimize the external input
    [uOptimized, ~] = fminunc(lossFunction, uInit, optimOptions);
    
    % Store results in temporary variables
    externalInputCell{neuronIdx} = uOptimized; % Optimized external input
    
    % Model 1: Internal dynamics only
    mdlInternal = fitlm(laggedMatrix, targetNeuronLagged); % Linear model
    varianceInternal(neuronIdx) = 1 - mdlInternal.Rsquared.Ordinary; % Residual variance explained
    
    % Model 2: Internal dynamics + estimated external input
    mdlAugmented = fitlm([laggedMatrix, uOptimized], targetNeuronLagged);
    varianceAugmented(neuronIdx) = 1 - mdlAugmented.Rsquared.Ordinary; % Residual variance explained
    toc
end
delete(poolID)

% Combine results from the cell array
externalInputEstimated = NaN(nSamples, nNeurons);
for neuronIdx = 1:nNeurons
    externalInputEstimated(maxLag+1:end, neuronIdx) = externalInputCell{neuronIdx};
end

% Compare variance explained
totalVariance = 1 - varianceInternal; % Total variance explained by internal dynamics
externalVariance = totalVariance - (1 - varianceAugmented); % Variance explained by external input

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
title('Variance Partitioning with Estimated External Input');

% Subfunction to compute model loss
function loss = computeModelLoss(target, laggedMatrix, externalInput)
    % Fit a linear model with internal dynamics and external input
    mdl = fitlm([laggedMatrix, externalInput], target);
    % Compute the residual sum of squares (RSS) as the loss
    residuals = target - mdl.Fitted;
    loss = sum(residuals.^2);
end

