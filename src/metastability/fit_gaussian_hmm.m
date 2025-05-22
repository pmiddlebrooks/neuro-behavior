function [bestModel, bestNumStates, stateSeq, allModels, allLogL, allBIC] = fit_gaussian_hmm(data, stateRange, numReps, numFolds)
% FIT_GAUSSIAN_HMM Fits Gaussian HMMs using block-wise k-fold cross-validation and selects by BIC.
% 
% Inputs:
%   data        : [T x D] continuous data (e.g. PCA scores)
%   stateRange  : Vector of state numbers to try (e.g., 2:10)
%   numReps     : Number of EM initializations per state count
%   numFolds    : Number of contiguous blocks for cross-validation
%
% Outputs:
%   bestModel     : Fitted model with best BIC
%   bestNumStates : Number of states in the best model
%   stateSeq      : Inferred state sequence for the best model
%   allModels     : Cell array of models for each state count
%   allLogL       : Cross-validated log-likelihoods for each state count
%   allBIC        : BIC scores for each state count

T = size(data, 1);
D = size(data, 2);
blockSize = floor(T / numFolds);

% Initialize storage
numStateOptions = numel(stateRange);
allModels = cell(numStateOptions, 1);
allLogL = -Inf(numStateOptions, 1);
allBIC = Inf(numStateOptions, 1);

% Fit models across state numbers
for idx = 1:numStateOptions
    numStates = stateRange(idx);
    disp(['Fitting ', num2str(numStates), ' states'])
    foldLogL = zeros(numFolds, 1);
    bestGMM = [];

    for fold = 1:numFolds
        testIdx = false(T,1);
        testStart = (fold-1)*blockSize + 1;
        testEnd = min(fold*blockSize, T);
        testIdx(testStart:testEnd) = true;
        trainIdx = ~testIdx;

        trainData = data(trainIdx, :);
        testData = data(testIdx, :);

        bestNegLogL = Inf;
        bestFoldGMM = [];

        for rep = 1:numReps
            try
                options = statset('MaxIter', 500, 'Display', 'off');
                gmm = fitgmdist(trainData, numStates, 'CovarianceType', 'full', ...
                    'RegularizationValue', 1e-6, 'Replicates', 1, 'Options', options);

                if gmm.NegativeLogLikelihood < bestNegLogL
                    bestNegLogL = gmm.NegativeLogLikelihood;
                    bestFoldGMM = gmm;
                end
            catch
                continue;
            end
        end

        if ~isempty(bestFoldGMM)
            foldLogL(fold) = sum(log(pdf(bestFoldGMM, testData)));  % true log-likelihood
        else
            foldLogL(fold) = -Inf;
        end

        % Use model from last fold for BIC and state sequence
        if fold == numFolds && ~isempty(bestFoldGMM)
            bestGMM = bestFoldGMM;
        end
    end

    avgLogL = mean(foldLogL);
    allLogL(idx) = avgLogL;
    allModels{idx} = bestGMM;

    if ~isempty(bestGMM)
        numParams = (numStates - 1) + numStates * D + numStates * D * (D + 1) / 2;
        allBIC(idx) = -2 * avgLogL + numParams * log(T);
    else
        allBIC(idx) = Inf;
    end
    allBIC(1:idx)
    figure(44); plot(stateRange(1):stateRange(idx), allBIC(1:idx));
end

% Select best model based on lowest BIC
[~, bestIdx] = min(allBIC);
bestModel = allModels{bestIdx};
bestNumStates = stateRange(bestIdx);

% Decode state sequence using the best model
if ~isempty(bestModel)
    stateSeq = cluster(bestModel, data);
else
    stateSeq = NaN(size(data,1), 1);
    warning('No valid HMM model found.');
end

% Plot model selection curves
figure;
subplot(2,1,1);
plot(stateRange, allLogL, '-o', 'LineWidth', 2);
xlabel('Number of States'); ylabel('CV Log-Likelihood');
title('Cross-validated Log-Likelihood');

subplot(2,1,2);
plot(stateRange, allBIC, '-o', 'LineWidth', 2);
xlabel('Number of States'); ylabel('BIC');
title('Bayesian Information Criterion');
end
