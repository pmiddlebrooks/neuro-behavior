function [bestNumStates, stateEstimates, hmmModels, penalizedLikelihoods, stateProbabilities] = fit_hmm_crossval_cov_penalty(hmmMatrix, opts)
% FIT_HMM_CROSSVAL_COV_PENALTY Fits HMM and determines the optimal number of states using penalized log-likelihood.
%
% Inputs:
%   hmmMatrix - Matrix of features (rows = time bins, columns = features)
%   opts - Options structure with fields:
%       .maxNumStates - Maximum number of states to evaluate
%       .numFolds - Number of folds for cross-validation
%       .minState - Minimum state duration in seconds
%       .plotFlag - Boolean flag to plot summary statistics (optional)
%
% Outputs:
%   bestNumStates - Optimal number of states based on penalized likelihood.
%   stateEstimates - State assignments for the best model (filtered by minState).
%   hmmModels - Cell array of trained HMMs for each number of states.
%   penalizedLikelihoods - Penalized likelihood scores for each number of states.
%   stateProbabilities - Matrix of state probabilities for the best model.

% Extract parameters from options struct
maxStates = opts.maxNumStates;
numFolds = opts.numFolds;
minStateDuration = opts.minState;

% Check if plotFlag exists, default to false if not
if isfield(opts, 'plotFlag')
    plotFlag = opts.plotFlag;
else
    plotFlag = false;
end

% Start timing
tic;

hmmMatrix = zscore(hmmMatrix);
numBins = size(hmmMatrix, 1);
foldSize = floor(numBins / numFolds);

% Initialize storage
testLogLikelihood = nan(maxStates, 1);
topEigenvalue = nan(maxStates, 1);
hmmModels = cell(maxStates, 1);

fprintf('Starting HMM fitting with %d folds...\n', numFolds);

stateRange = 2:maxStates;
for numStates = stateRange
    fprintf('Testing %d states...\n', numStates);
    tic

    foldLikelihoods = zeros(numFolds, 1);
    iTopEigenvalue = zeros(numFolds, 1);
    iTestLogLikelihood = zeros(numFolds, 1);

    for fold = 1:numFolds
        % Split data into training and test sets
        testIdx = (1:foldSize) + (fold-1)*foldSize;
        trainIdx = setdiff(1:numBins, testIdx);

        trainData = hmmMatrix(trainIdx, :);
        testData = hmmMatrix(testIdx, :);

        % Validate data quality
        if any(isnan(trainData(:))) || any(isinf(trainData(:)))
            warning('Training data contains NaN or Inf values. Skipping fold %d.', fold);
            iTestLogLikelihood(fold) = NaN;
            iTopEigenvalue(fold) = NaN;
            continue;
        end

        % Check if training data has sufficient variance
        if any(var(trainData) < 1e-10)
            warning('Training data has very low variance. Skipping fold %d.', fold);
            iTestLogLikelihood(fold) = NaN;
            iTopEigenvalue(fold) = NaN;
            continue;
        end

        % Train HMM on training data with better regularization
        options = statset('MaxIter', 500, 'TolFun', 1e-6, 'TolX', 1e-6);
        
        try
            hmm = fitgmdist(trainData, numStates, 'Replicates', 10, ...
                'CovarianceType', 'diagonal', ...  % More stable than 'full'
                'SharedCovariance', false, ...
                'RegularizationValue', 0.01, ...   % Add regularization
                'Options', options);
            hmmModels{numStates} = hmm;

            % Evaluate log-likelihood on test data with numerical stability
            logProbs = log(pdf(hmm, testData));
            % Replace -Inf with a large negative number
            logProbs(isinf(logProbs)) = -1e6;
            iTestLogLikelihood(fold) = sum(logProbs);

            % Compute similarity as the top eigenvalue of the state definition matrix
            stateDefinitionMatrix = hmm.mu; % Size: numStates x numFeatures
            % Compute the covariance matrix of the state definition matrix
            covMatrix = cov(stateDefinitionMatrix);

            % Compute the top eigenvalue
            iTopEigenvalue(fold) = max(eig(covMatrix));
            
        catch ME
            warning('HMM fitting failed for fold %d: %s', fold, ME.message);
            iTestLogLikelihood(fold) = NaN;
            iTopEigenvalue(fold) = NaN;
        end

    end

    % Average likelihoods and top eigenvalues
    testLogLikelihood(numStates) = mean(iTestLogLikelihood);
    topEigenvalue(numStates) = mean(iTopEigenvalue);
    % End timing and display
    elapsedTime = toc;
    fprintf('Completed in %.2f minutes\nLogLike:\n', elapsedTime/60);
disp(testLogLikelihood(numStates))
end

% Calculate penalized likelihoods
normLogLike = normalizeMetric(testLogLikelihood);
normTopEig = normalizeMetric(topEigenvalue);

penalizedLikelihoods = normLogLike ./ normTopEig;

% Find the best number of states
[maxVal, bestIdx] = max(penalizedLikelihoods);
bestNumStates = stateRange(bestIdx);

% Train the best model on the full dataset
fprintf('Training best model with %d states on full dataset...\n', bestNumStates);
options = statset('MaxIter', 500, 'TolFun', 1e-6, 'TolX', 1e-6);
bestHMM = fitgmdist(hmmMatrix, bestNumStates, 'Replicates', 10, ...
    'CovarianceType', 'diagonal', ...
    'SharedCovariance', false, ...
    'RegularizationValue', 0.01, ...
    'Options', options);

% Get initial state estimates and probabilities
initialStateEstimates = cluster(bestHMM, hmmMatrix);
stateProbabilities = pdf(bestHMM, hmmMatrix); % Matrix: rows = time bins, columns = states

% Filter state estimates based on minimum duration requirement
stateEstimates = filterStatesByDuration(initialStateEstimates, minStateDuration, 0.005); % 0.005 seconds = 5ms bins

hmmModels{bestNumStates} = bestHMM;


% Optional plotting of summary statistics
if plotFlag
    plotSummaryStatistics(2:maxStates, testLogLikelihood(2:end), topEigenvalue(2:end), penalizedLikelihoods(2:end));
end
end

function plotSummaryStatistics(numStatesRange, testLogLikelihood, topEigenvalue, penalizedLikelihoods)
% PLOTSUMMARYSTATISTICS Plots summary statistics across different numbers of states
%
% Inputs:
%   numStatesRange - Range of states tested
%   testLogLikelihood - Test log-likelihood for each number of states
%   topEigenvalue - Top eigenvalue for each number of states
%   penalizedLikelihoods - Penalized likelihood scores for each number of states

figure(91); clf;

% Subplot 1: Test Log-Likelihood
subplot(3,1,1);
plot(numStatesRange, testLogLikelihood, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of States');
ylabel('Test Log-Likelihood');
title('Test Log-Likelihood vs Number of States');
grid on;

% Subplot 2: Top Eigenvalue
subplot(3,1,2);
plot(numStatesRange, topEigenvalue, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of States');
ylabel('Top Eigenvalue');
title('Top Eigenvalue vs Number of States');
grid on;

% Subplot 3: Penalized Likelihood
subplot(3,1,3);
plot(numStatesRange, penalizedLikelihoods, 'g-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of States');
ylabel('Penalized Likelihood');
title('Penalized Likelihood vs Number of States');
grid on;

% Find and mark the best number of states
[~, bestIdx] = max(penalizedLikelihoods);
bestNumStates = numStatesRange(bestIdx);
hold on;
plot(bestNumStates, penalizedLikelihoods(bestIdx), 'ko', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
text(bestNumStates, penalizedLikelihoods(bestIdx), sprintf('  Best: %d states', bestNumStates), 'FontSize', 12, 'FontWeight', 'bold');
hold off;
end

function filteredStates = filterStatesByDuration(stateEstimates, minDuration, binSize)
% FILTERSTATESBYDURATION Filters out state segments that are shorter than minDuration
%
% Inputs:
%   stateEstimates - Vector of state assignments
%   minDuration - Minimum duration in seconds
%   binSize - Size of each bin in seconds
%
% Outputs:
%   filteredStates - Vector with short segments set to 0

minBins = round(minDuration / binSize);
filteredStates = stateEstimates;

% Find continuous segments of each state
uniqueStates = unique(stateEstimates);
for state = uniqueStates'
    if state == 0
        continue; % Skip already filtered states
    end

    stateIndices = find(stateEstimates == state);

    % Find continuous segments
    segmentStarts = [stateIndices(1); stateIndices(find(diff(stateIndices) > 1) + 1)];
    segmentEnds = [stateIndices(find(diff(stateIndices) > 1)); stateIndices(end)];

    % Filter short segments
    for seg = 1:length(segmentStarts)
        segmentLength = segmentEnds(seg) - segmentStarts(seg) + 1;
        if segmentLength < minBins
            filteredStates(segmentStarts(seg):segmentEnds(seg)) = 0;
        end
    end
end
end


% Helper function: Normalize a metric between minVal and maxVal
function normMetric = normalizeMetric(metric)
normMetric = 2 * (metric - min(metric)) / (max(metric) - min(metric)) - 1;
end

% Helper function: Normalize rows of a matrix to sum to 1
function normalizedMatrix = normalize(matrix, dim)
normalizedMatrix = bsxfun(@rdivide, matrix, sum(matrix, dim));
end
