%%
idSelect = idM56;
dataModel = zscore(dataMat(:, idSelect));
% dataModel = zscore(dataMat(:, idSelect(1:3)));
bhvIDModel = bhvID;

% Inputs:
% dataModel: Neural activity matrix (samples x neurons)

% Parameters
maxLag = 2; % Maximum lag for the autoregressive model
nSamples = size(dataModel, 1); % Number of samples
nNeurons = size(dataModel, 2); % Number of neurons

% Preallocate results
varianceInternal = zeros(1, nNeurons); % Variance explained by internal dynamics
varianceAugmented = zeros(1, nNeurons); % Variance explained by augmented model
externalInputCell = cell(1, nNeurons); % Temporary storage for external inputs



% poolID = parpool(4, 'IdleTimeout', Inf);
% Parallel loop for computation
% parfor neuronIdx = 1:nNeurons
for neuronIdx = 1
    tic
    % Extract the target neuron activity and create a lagged version
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
    



    % Trying alternatives to speed up fitting the external input

    % Optimize the external input
% Optimization options
% optimOptions = optimoptions('fminunc', 'Display', 'off', 'Algorithm', 'quasi-newton','MaxIterations',20);
    % [uOptimized, ~] = fminunc(lossFunction, uInit, optimOptions);

%     optimOptions = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
%                             'Display', 'off', 'MaxIterations', 100);
% lb = -10 * ones(size(uInit)); % Lower bound
% ub = 10 * ones(size(uInit)); % Upper bound
% [uOptimized, ~] = fmincon(lossFunction, uInit, [], [], [], [], lb, ub, [], optimOptions);


[uOptimized, ~] = fminsearch(lossFunction, uInit, optimset('Display', 'off'));

    toc






    % Store results in temporary variables
    externalInputCell{neuronIdx} = uOptimized; % Optimized external input
    
    % Model 1: Internal dynamics only
    mdlInternal = fitlm(laggedMatrix, targetNeuronLagged); % Linear model
    varianceInternal(neuronIdx) = 1 - mdlInternal.Rsquared.Ordinary; % Residual variance explained
    toc
    % Model 2: Internal dynamics + estimated external input
    mdlAugmented = fitlm([laggedMatrix, uOptimized], targetNeuronLagged);
    varianceAugmented(neuronIdx) = 1 - mdlAugmented.Rsquared.Ordinary; % Residual variance explained
    toc
end
% delete(poolID)

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

