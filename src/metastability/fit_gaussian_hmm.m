function [bestModel, bestNumStates, stateSeq, allModels, allLogL] = fit_gaussian_hmm(data, stateRange, numReps, numFolds)
% FIT_GAUSSIAN_HMM_PCA Fits Gaussian HMMs to PCA-reduced data using k-fold cross-validation.
% 
% Inputs:
%   stateRange : Vector of state numbers to try (e.g., 2:10)
%   numReps    : Number of EM initializations per state count
%   numFolds   : Number of folds for cross-validation
%
% Outputs:
%   bestModel     : Fitted model with best average log-likelihood
%   bestNumStates : Number of states in the best model
%   stateSeq      : Inferred state sequence for the best model
%   allModels     : Cell array of models for each state count
%   allLogL       : Cross-validated log-likelihoods for each state count

% Reduce to specified number of PCA components
T = size(data, 1);

% Cross-validation partition
cv = cvpartition(T, 'KFold', numFolds);

% Initialize storage
numStateOptions = numel(stateRange);
allModels = cell(numStateOptions, 1);
allLogL = -Inf(numStateOptions, 1);

% Fit models across state numbers
for idx = 1:numStateOptions
    numStates = stateRange(idx);
    disp(['Fitting ', num2str(numStates),' states'])
    foldLogL = zeros(numFolds, 1);

    for fold = 1:numFolds
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);
        trainData = data(trainIdx, :);
        testData = data(testIdx, :);

        bestLogL = -Inf;
        bestGMM = [];

        for rep = 1:numReps
            try
                options = statset('MaxIter', 500, 'Display', 'off');
                gmm = fitgmdist(trainData, numStates, 'CovarianceType', 'full', ...
                    'RegularizationValue', 1e-6, 'Replicates', 1, 'Options', options);

                if gmm.NegativeLogLikelihood < -bestLogL
                    bestLogL = -gmm.NegativeLogLikelihood;
                    bestGMM = gmm;
                end
            catch
                continue;
            end
        end

        if ~isempty(bestGMM)
            foldLogL(fold) = sum(log(pdf(bestGMM, testData)));
        else
            foldLogL(fold) = -Inf;
        end
    end

    allLogL(idx) = mean(foldLogL);
    allModels{idx} = bestGMM;
end

% Select best model based on cross-validated log-likelihood
[~, bestIdx] = max(allLogL);
bestModel = allModels{bestIdx};
bestNumStates = stateRange(bestIdx);

% Decode state sequence using the best model
if ~isempty(bestModel)
    stateSeq = cluster(bestModel, data);
else
    stateSeq = NaN(size(data,1), 1);
    warning('No valid HMM model found.');
end

% Plot model selection curve
figure;
plot(stateRange, allLogL, '-o', 'LineWidth', 2);
xlabel('Number of States'); ylabel('Cross-validated Log-Likelihood');
title('Gaussian HMM Model Selection');
end



