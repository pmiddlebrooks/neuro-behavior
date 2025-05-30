function [bestModel, bestNumStates, stateSeq, allModels, allLogL, allBIC, allMargLik] = fit_gaussian_hmm(data, opts)
% FIT_GAUSSIAN_HMM Fits Gaussian HMMs using block-wise cross-validation.
% Supports marginal likelihood estimation via Laplace or reciprocal importance sampling (RIS).
% 
% Inputs:
%   data  : [T x D] continuous data (e.g. PCA scores)
%   opts  : struct with fields:
%             - stateRange       : vector of state numbers to try (e.g., 2:10)
%             - numReps          : number of EM initializations per state count
%             - numFolds         : number of contiguous blocks for cross-validation
%             - margLikMethod    : 'laplace', 'importance', or 'none'
%             - numSamples       : number of importance samples (only used if 'importance')
%             - selectBy         : 'margLik' or 'bic' (method to choose best model)
%             - plotFlag         : plot 1 or don't plot 0 the results
%
% Outputs:
%   bestModel     : Fitted model with best selection score
%   bestNumStates : Number of states in the best model
%   stateSeq      : Inferred state sequence for the best model
%   allModels     : Cell array of models for each state count
%   allLogL       : Cross-validated log-likelihoods for each state count
%   allBIC        : BIC scores for each state count
%   allMargLik    : Estimated marginal likelihoods (log scale)

if ~isfield(opts, 'plotFlag') || isempty(opts.plotFlag)
    opts.plotFlag = 1;
end

T = size(data, 1);
D = size(data, 2);
blockSize = floor(T / opts.numFolds);

% Initialize storage
numStateOptions = numel(opts.stateRange);
allModels = cell(numStateOptions, 1);
allLogL = -Inf(numStateOptions, 1);
allBIC = Inf(numStateOptions, 1);
allMargLik = -Inf(numStateOptions, 1);

% Fit models across state numbers
for idx = 1:numStateOptions
    tic
    numStates = opts.stateRange(idx);
    disp(['Fitting ', num2str(numStates), ' states'])
    foldLogL = zeros(opts.numFolds, 1);
    bestGMM = [];

    for fold = 1:opts.numFolds
        testIdx = false(T,1);
        testStart = (fold-1)*blockSize + 1;
        testEnd = min(fold*blockSize, T);
        testIdx(testStart:testEnd) = true;
        trainIdx = ~testIdx;

        trainData = data(trainIdx, :);
        testData = data(testIdx, :);

        bestNegLogL = Inf;
        bestFoldGMM = [];

        for rep = 1:opts.numReps
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

        if fold == opts.numFolds && ~isempty(bestFoldGMM)
            bestGMM = bestFoldGMM;
        end
    end

    avgLogL = mean(foldLogL);
    allLogL(idx) = avgLogL;
    allModels{idx} = bestGMM;

    if ~isempty(bestGMM)
        numParams = (numStates - 1) + numStates * D + numStates * D * (D + 1) / 2;
        allBIC(idx) = -2 * avgLogL + numParams * log(T);

        switch lower(opts.margLikMethod)
            case 'laplace'
                allMargLik(idx) = avgLogL - 0.5 * numParams * log(T);

            case 'importance'
                N = opts.numSamples;
                weights = zeros(N, 1);
                priorScale = 10;

                for i = 1:N
                    mu = randn(numStates, D) * priorScale;
                    Sigma = repmat(eye(D), [1, 1, numStates]);
                    for k = 1:numStates
                        A = randn(D); Sigma(:,:,k) = A'*A + eye(D);
                    end

                    ll_i = 0;
                    for t = 1:T
                        p = 0;
                        for k = 1:numStates
                            p = p + bestGMM.ComponentProportion(k) * mvnpdf(data(t,:), mu(k,:), Sigma(:,:,k));
                        end
                        ll_i = ll_i + log(p + realmin);
                    end

                    weights(i) = exp(ll_i);
                end

                allMargLik(idx) = -log(mean(1 ./ (weights + realmin)) + realmin);

            case 'none'
                allMargLik(idx) = NaN;
        end
    else
        allBIC(idx) = Inf;
        allMargLik(idx) = -Inf;
    end
    fprintf('\tTook %.1f minutes\n', toc/60)
end

% Select best model
switch lower(opts.selectBy)
    case 'bic'
        [~, bestIdx] = min(allBIC);
    case 'marglik'
        [~, bestIdx] = max(allMargLik);
    otherwise
        error('Invalid value for opts.selectBy. Use "bic" or "marglik".')
end

bestModel = allModels{bestIdx};
bestNumStates = opts.stateRange(bestIdx);

% Decode state sequence using the best model
if ~isempty(bestModel)
    stateSeq = cluster(bestModel, data);
else
    stateSeq = NaN(size(data,1), 1);
    warning('No valid HMM model found.');
end

if opts.plotFlag
% Plot model selection curves
figure;
subplot(3,1,1);
plot(opts.stateRange, allLogL, '-o', 'LineWidth', 2);
xlabel('Number of States'); ylabel('CV Log-Likelihood');
title('Cross-validated Log-Likelihood');

subplot(3,1,2);
plot(opts.stateRange, allBIC, '-o', 'LineWidth', 2);
xlabel('Number of States'); ylabel('BIC');
title('Bayesian Information Criterion');

subplot(3,1,3);
plot(opts.stateRange, allMargLik, '-o', 'LineWidth', 2);
xlabel('Number of States'); ylabel('Log Marginal Likelihood');
title('Marginal Likelihood Estimate');
end
end