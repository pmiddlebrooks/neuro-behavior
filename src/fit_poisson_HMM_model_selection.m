    function hmm = fit_poisson_HMM_model_selection(dataMat, stateRange, numFolds, numReps)
% Build Poisson HMM with model selection via cross-validation
% Inputs:
%   dataMat: [T x N] binary spike matrix (1 ms bins)
%   stateRange: array of candidate state counts, e.g. 6:20
%   numFolds: number of cross-validation folds (default 20)
%   numReps: number of EM restarts (default 5)

if nargin < 3, numFolds = 20; end
if nargin < 4, numReps = 5; end

[T, ~] = size(dataMat);
cvIdx = crossvalind('Kfold', T, numFolds);

logL = zeros(length(stateRange), numFolds);
models = cell(length(stateRange), numFolds);

for i = 1:length(stateRange)
    M = stateRange(i);
    fprintf('Testing %d states...\n', M);
    for fold = 1:numFolds
        testIdx = (cvIdx == fold);
        trainIdx = ~testIdx;
        trainData = dataMat(trainIdx, :);
        testData = dataMat(testIdx, :);

        bestLL = -inf;
        bestModel = [];

        for rep = 1:numReps
            model = fit_poisson_HMM(trainData, M, 1);
            [~, ~, ~, ~, llTest] = fwdBwdPoisson(testData, model.pi0, model.A, model.lambda, 0.001);
            if llTest > bestLL
                bestLL = llTest;
                bestModel = model;
            end
        end

        logL(i, fold) = bestLL;
        models{i, fold} = bestModel;
    end
end

meanLL = mean(logL, 2);
dLL = diff(meanLL);
elbowIdx = find(diff(dLL) < min(dLL) * 0.5, 1);
if isempty(elbowIdx), [~, elbowIdx] = max(meanLL); end
bestM = stateRange(elbowIdx);

fprintf('Selected %d states via elbow in log-likelihood curve.\n', bestM);

% Final model on full data
bestLL = -inf;
for rep = 1:numReps
    model = fitPoissonHMM(dataMat, bestM, 1);
    if model.logL(end) > bestLL
        bestLL = model.logL(end);
        hmm = model;
    end
end

hmm.modelSelection.numStates = stateRange;
hmm.modelSelection.meanLL = meanLL;
hmm.modelSelection.bestM = bestM;
end

