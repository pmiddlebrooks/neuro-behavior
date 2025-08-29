function [bestNumStates, stateEstimates, hmmModels, penalizedLikelihoods, stateProbabilities] = fit_hmm_crossval_cov_penalty(hmmMatrix, opts)
% FIT_HMM_CROSSVAL_COV_PENALTY Fits HMM and determines the optimal number of states using penalized log-likelihood.
%
% Inputs:
%   hmmMatrix - Matrix of features (rows = time bins, columns = features)
%   opts - Options structure with fields:
%       .stateRange - Vector of state numbers to evaluate (e.g., 2:16)
%       .numFolds - Number of folds for cross-validation
%       .minState - Minimum state duration in seconds
%       .plotFlag - Boolean flag to plot summary statistics (optional)
%       .penaltyType - Type of penalty for model selection:
%           'normalized' - Original approach: ratio of normalized metrics (default)
%           'aic' - Akaike Information Criterion: 2 * number of parameters
%           'bic' - Bayesian Information Criterion: log(n) * number of parameters
%
% Outputs:
%   bestNumStates - Optimal number of states based on penalized likelihood.
%   stateEstimates - State assignments for the best model (filtered by minState).
%   hmmModels - Cell array of trained HMMs for each number of states.
%   penalizedLikelihoods - Penalized likelihood scores for each number of states.
%   stateProbabilities - Matrix of state probabilities for the best model.

% Extract parameters from options struct
stateRange = opts.stateRange;
numFolds = opts.numFolds;
minStateDuration = opts.minState;

% Check if plotFlag exists, default to false if not
if isfield(opts, 'plotFlag')
    plotFlag = opts.plotFlag;
else
    plotFlag = false;
end

% Check penalty type, default to 'normalized' if not specified
if isfield(opts, 'penaltyType')
    penaltyType = opts.penaltyType;
else
    penaltyType = 'normalized'; % Default to original approach
end

% Validate penalty type
validPenaltyTypes = {'normalized', 'aic', 'bic'};
if ~ismember(penaltyType, validPenaltyTypes)
    warning('Invalid penaltyType: %s. Using default: normalized', penaltyType);
    penaltyType = 'normalized';
end

% Start timing
tic;

hmmMatrix = zscore(hmmMatrix);
numBins = size(hmmMatrix, 1);
foldSize = floor(numBins / numFolds);

% Initialize storage
testLogLikelihood = nan(length(stateRange), 1);
topEigenvalue = nan(length(stateRange), 1);
hmmModels = cell(length(stateRange), 1);

fprintf('Starting HMM fitting with %d folds...\n', numFolds);

    stateIdx = 1;
for numStates = stateRange
    fprintf('Testing %d states...\n', numStates);
    tic

    foldLikelihoods = zeros(numFolds, 1);
    iTopEigenvalue = zeros(numFolds, 1);
    iTestLogLikelihood = zeros(numFolds, 1);

    for fold = 1:numFolds
        % Split data into training and test sets using systematic sampling across segments
        % This ensures each fold samples evenly from all parts of the dataset
        % and prevents task parameter changes from affecting fold performance
        
        % Calculate segment size and test portion size
        segmentSize = floor(numBins / numFolds);
        testPortionSize = floor(segmentSize / numFolds);
        
        % Initialize test indices
        testIdx = [];
        
        % For each segment, take the appropriate portion as test data
        for segment = 1:numFolds
            segmentStart = (segment - 1) * segmentSize + 1;
            segmentEnd = min(segment * segmentSize, numBins);
            
            % Calculate which portion of this segment to use for test data
            % Rotate through portions for each fold to ensure comprehensive coverage
            portionStart = segmentStart + (fold - 1) * testPortionSize;
            portionEnd = min(portionStart + testPortionSize - 1, segmentEnd);
            
            % Add this portion to test indices if it's valid
            if portionStart <= portionEnd
                testIdx = [testIdx, portionStart:portionEnd];
            end
        end
        
        % Create training indices from remaining data
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

        % Train HMM on training data with better regularization and convergence handling
        options = statset('MaxIter', 500, 'TolFun', 1e-8, 'TolX', 1e-8);
        
        try
            hmm = fitgmdist(trainData, numStates, 'Replicates', 15, ...
                'CovarianceType', 'diagonal', ...  % More stable than 'full'
                'SharedCovariance', false, ...
                'RegularizationValue', 0.05, ...   % Increased regularization for stability
                'Options', options, ...
                'Start', 'plus'); % Use k-means++ initialization for better starting points
            hmmModels{stateIdx} = hmm;

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
    testLogLikelihood(stateIdx) = mean(iTestLogLikelihood);
    topEigenvalue(stateIdx) = mean(iTopEigenvalue);
    % End timing and display
    elapsedTime = toc;
    fprintf('Completed in %.2f minutes\nLogLike:\n', elapsedTime/60);
disp(testLogLikelihood(stateIdx))

    stateIdx = stateIdx + 1;

end

% Calculate penalized likelihoods based on selected penalty type
normLogLike = normalizeMetric(testLogLikelihood);
normTopEig = normalizeMetric(topEigenvalue);

switch penaltyType
    case 'normalized'
        % Original approach: ratio of normalized metrics
        penalizedLikelihoods = normLogLike ./ normTopEig;
        fprintf('Using normalized penalty approach\n');
        
    case 'aic'
        % AIC penalty: 2 * number of parameters
        % For GMM: numStates * (numFeatures + 1) parameters
        numParams = stateRange * (size(hmmMatrix, 2) + 1);
        aicPenalty = 2 * numParams;
        penalizedLikelihoods = normLogLike - aicPenalty';
        fprintf('Using AIC penalty approach\n');
        
    case 'bic'
        % BIC penalty: log(n) * number of parameters (stronger penalty)
        % For GMM: numStates * (numFeatures + 1) parameters
        numParams = stateRange * (size(hmmMatrix, 2) + 1);
        bicPenalty = log(size(hmmMatrix, 1)) * numParams;
        penalizedLikelihoods = normLogLike - bicPenalty';
        fprintf('Using BIC penalty approach\n');
        
    otherwise
        % Fallback to normalized approach
        penalizedLikelihoods = normLogLike ./ normTopEig;
        fprintf('Using fallback normalized penalty approach\n');
end

% Find the best number of states using elbow method for log-likelihood
if strcmp(penaltyType, 'aic') || strcmp(penaltyType, 'bic')
    % For AIC/BIC, lower values are better
    [~, bestIdx] = min(penalizedLikelihoods);
    bestNumStates = stateRange(bestIdx);
    fprintf('Selected %d states using %s penalty (minimum value)\n', bestNumStates, upper(penaltyType));
else
    % For normalized approach, use elbow method on log-likelihood
    bestNumStates = findElbowPoint(stateRange, testLogLikelihood);
    bestIdx = find(stateRange == bestNumStates);
    fprintf('Selected %d states using elbow method on log-likelihood\n', bestNumStates);
end

% Train the best model on the full dataset
fprintf('Training best model with %d states on full dataset...\n', bestNumStates);
options = statset('MaxIter', 1000, 'TolFun', 1e-8, 'TolX', 1e-8);
bestHMM = fitgmdist(hmmMatrix, bestNumStates, 'Replicates', 15, ...
    'CovarianceType', 'diagonal', ...
    'SharedCovariance', false, ...
    'RegularizationValue', 0.05, ...
    'Options', options, ...
    'Start', 'plus'); % Use k-means++ initialization for better starting points

% Get initial state estimates and probabilities
initialStateEstimates = cluster(bestHMM, hmmMatrix);
% Get probability of each state at each time point
% Returns matrix where rows are time bins and columns are probabilities for each state
stateProbabilities = posterior(bestHMM, hmmMatrix); % Matrix: numTimeBins x numStates

% Filter state estimates based on minimum duration requirement
stateEstimates = filterStatesByDuration(initialStateEstimates, minStateDuration, 0.005); % 0.005 seconds = 5ms bins

hmmModels{bestIdx} = bestHMM;

% Optional plotting of summary statistics
if plotFlag
    plotSummaryStatistics(stateRange, bestIdx, testLogLikelihood, topEigenvalue, penalizedLikelihoods, penaltyType);
end
end

function plotSummaryStatistics(numStatesRange, bestIdx, testLogLikelihood, topEigenvalue, penalizedLikelihoods, penaltyType)
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

% Adjust ylabel based on penalty type
if strcmp(penaltyType, 'aic') || strcmp(penaltyType, 'bic')
    ylabel('Penalized Likelihood (Lower is Better)');
    title(sprintf('Penalized Likelihood vs Number of States (%s)', upper(penaltyType)));
else
    ylabel('Penalized Likelihood (Higher is Better)');
    title('Penalized Likelihood vs Number of States (Normalized)');
end
grid on;

% Find and mark the best number of states
if strcmp(penaltyType, 'aic') || strcmp(penaltyType, 'bic')
    % For AIC/BIC, lower is better
    [~, bestIdx] = min(penalizedLikelihoods);
    bestNumStates = numStatesRange(bestIdx);
    markerColor = 'r';
    markerText = sprintf('  Best: %d states (%s)', bestNumStates, upper(penaltyType));
else
    % For normalized approach, use elbow method on log-likelihood
    bestNumStates = findElbowPoint(numStatesRange, testLogLikelihood);
    markerColor = 'g';
    markerText = sprintf('  Best: %d states (Elbow)', bestNumStates);
end

hold on;
plot(bestNumStates, penalizedLikelihoods(bestIdx), 'o', 'MarkerSize', 12, 'MarkerFaceColor', markerColor, 'MarkerEdgeColor', 'k');
text(bestNumStates, penalizedLikelihoods(bestIdx), markerText, 'FontSize', 12, 'FontWeight', 'bold');
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

function optimalStates = findElbowPoint(stateRange, logLikelihoods)
% FINDELBOWPOINT Finds the optimal number of states using the elbow method
%
% Inputs:
%   stateRange - Vector of state numbers tested
%   logLikelihoods - Corresponding log-likelihood values
%
% Outputs:
%   optimalStates - Optimal number of states based on elbow method
%
% The elbow method finds the point where adding more states provides
% diminishing returns in log-likelihood improvement.

% Remove any NaN values
validIdx = ~isnan(logLikelihoods);
if sum(validIdx) < 3
    warning('Insufficient valid log-likelihood values for elbow method. Using maximum.');
    [~, maxIdx] = max(logLikelihoods(validIdx));
    optimalStates = stateRange(validIdx);
    optimalStates = optimalStates(maxIdx);
    return;
end

stateRange = stateRange(validIdx);
logLikelihoods = logLikelihoods(validIdx);

% Calculate the rate of improvement (first derivative)
improvementRates = diff(logLikelihoods);

% Calculate the rate of change in improvement (second derivative)
accelerationRates = diff(improvementRates);

% Find the elbow point (maximum deceleration)
[~, elbowIdx] = max(accelerationRates);

% The elbow point is one step after the maximum deceleration
optimalStates = stateRange(elbowIdx + 1);

% Ensure we don't go beyond the range
if optimalStates > max(stateRange)
    optimalStates = max(stateRange);
end

% Additional check: if the improvement rate is very small, use fewer states
if improvementRates(end) < 0.01 * improvementRates(1)
    % Look for the last significant improvement
    significantImprovement = improvementRates > 0.1 * improvementRates(1);
    if any(significantImprovement)
        lastSignificantIdx = find(significantImprovement, 1, 'last');
        optimalStates = min(optimalStates, stateRange(lastSignificantIdx + 1));
    end
end

fprintf('Elbow method analysis:\n');
fprintf('  Improvement rates: [%s]\n', num2str(improvementRates, '%.3f '));
fprintf('  Acceleration rates: [%s]\n', num2str(accelerationRates, '%.3f '));
fprintf('  Elbow point identified at %d states\n', optimalStates);
end
