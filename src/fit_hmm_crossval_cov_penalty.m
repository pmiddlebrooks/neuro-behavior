function [bestNumStates, stateEstimates, hmmModels, penalizedLikelihoods] = fit_hmm_crossval_cov_penalty(featureMatrix, maxStates, numFolds, lambda)
% FIT_HMM_CROSSVAL_COV_PENALTY Fits HMM and determines the optimal number of states using penalized log-likelihood.
%
% Inputs:
%   binnedBandPowers - Struct with binned band power data (alpha, beta, lowGamma, highGamma).
%   maxStates - Maximum number of states to evaluate.
%
% Outputs:
%   bestNumStates - Optimal number of states based on penalized likelihood.
%   stateEstimates - State assignments for the best model.
%   hmmModels - Cell array of trained HMMs for each number of states.
featureMatrix = zscore(featureMatrix);
numBins = size(featureMatrix, 1);
foldSize = floor(numBins / numFolds);

% Initialize storage
testLogLikelihood = nan(maxStates, 1);
topEigenvalue = nan(maxStates, 1);
hmmModels = cell(maxStates, 1);

for numStates = 2:maxStates
    foldLikelihoods = zeros(numFolds, 1);
    iTopEigenvalue = zeros(numFolds, 1);
    iTestLogLikelihood = zeros(numFolds, 1);
    for fold = 1:numFolds
        % Split data into training and test sets
        testIdx = (1:foldSize) + (fold-1)*foldSize;
        trainIdx = setdiff(1:numBins, testIdx);

        trainData = featureMatrix(trainIdx, :);
        testData = featureMatrix(testIdx, :);

        % Train HMM on training data
        options = statset('MaxIter', 500);

        hmm = fitgmdist(trainData, numStates, 'Replicates', 10, 'CovarianceType', 'full', 'Options', options);
        hmmModels{numStates} = hmm;

        % Evaluate log-likelihood on test data
        iTestLogLikelihood(fold) = sum(log(pdf(hmm, testData)));

        % Compute similarity as the top eigenvalue of the state definition
        % matrix
        % Extract state definition matrix
        stateDefinitionMatrix = hmm.mu; % Size: numStates x numFeatures
        % Compute the covariance matrix of the state definition matrix
        covMatrix = cov(stateDefinitionMatrix);

        % Compute the top eigenvalue
        iTopEigenvalue(fold) = max(eig(covMatrix));

    end

    % Average likelihoods and top eigenvalues
    % penalizedLikelihoods(numStates) = mean(foldLikelihoods);
    testLogLikelihood(numStates) = mean(iTestLogLikelihood);
    topEigenvalue(numStates) = mean(iTopEigenvalue);
end

normLogLike = normalizeMetric(testLogLikelihood);
normTopEig = normalizeMetric(topEigenvalue);

penalizedLikelihoods = normLogLike ./ normTopEig;
[maxVal, bestNumStates] = max(penalizedLikelihoods);


% % Find the best number of states
% [~, bestNumStates] = max(penalizedLikelihoods);

% Train the best model on the full dataset
bestHMM = fitgmdist(featureMatrix, bestNumStates, 'Replicates', 10, 'CovarianceType', 'diagonal');

% State estimations
stateEstimates = cluster(bestHMM, featureMatrix);
hmmModels{bestNumStates} = bestHMM;
end



% Helper function: Normalize a metric between minVal and maxVal
function normMetric = normalizeMetric(metric)
normMetric = 2 * (metric - min(metric)) / (max(metric) - min(metric)) - 1;
end

% Helper function: Normalize rows of a matrix to sum to 1
function normalizedMatrix = normalize(matrix, dim)
normalizedMatrix = bsxfun(@rdivide, matrix, sum(matrix, dim));
end
