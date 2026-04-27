function [bestNumStates, cvScores, bestModel, diagnostics] = gaussian_hmm_cv_select_num_states(observationMatrix, stateRange, numFolds, emOpts, numBands)
% GAUSSIAN_HMM_CV_SELECT_NUM_STATES Cross-validated sequence log-likelihood per state count.
%
% Variables:
%   - observationMatrix: [T x D] z-scored or raw features
%   - stateRange: vector of K values to compare (e.g. 2:6)
%   - numFolds: number of contiguous held-out blocks
%   - emOpts: options struct for gaussian_hmm_fit (maxIter, numRestarts, regVar, decode=false for speed)
%   - numBands: number of frequency bands used per area/layer feature block
% Goal:
%   Train HMM on training bins, evaluate mean per-bin log-likelihood on held-out bins,
%   then apply Akella et al. model selection:
%   score(K) = normalizedCVLL(K) / normalizedTopEigenvalue(K).
% Returns:
%   - bestNumStates: K with highest mean CV log-likelihood
%   - cvScores: length(stateRange) vector of mean CV log p / bin
%   - bestModel: model refit on full data at bestNumStates with decode true
%   - diagnostics: struct with fold details and model-selection diagnostics

if nargin < 5 || isempty(numBands)
    numBands = size(observationMatrix, 2);
end

observationMatrix = double(observationMatrix);
nonFiniteMask = ~isfinite(observationMatrix(:));
if any(nonFiniteMask)
    warning('gaussian_hmm_cv_select_num_states:NonFiniteInput', ...
        'Replacing %d non-finite feature entries with 0 before CV/fit.', nnz(nonFiniteMask));
    observationMatrix(nonFiniteMask) = 0;
end

T = size(observationMatrix, 1);
foldLen = floor(T / numFolds);
if foldLen < 10
    error('gaussian_hmm_cv_select_num_states:TooFewBins', ...
        'Need more time bins for %d folds (foldLen=%d).', numFolds, foldLen);
end

emOptsTrain = emOpts;
if isfield(emOptsTrain, 'decode')
    emOptsTrain.decode = false;
else
    emOptsTrain.decode = false;
end
if isfield(emOpts, 'verbose')
    verbose = logical(emOpts.verbose);
else
    verbose = true;
end
if isfield(emOpts, 'useParallel')
    useParallel = logical(emOpts.useParallel);
else
    useParallel = false;
end
if isfield(emOpts, 'numWorkers')
    numWorkers = emOpts.numWorkers;
else
    numWorkers = 4;
end

cvScores = nan(numel(stateRange), 1);
% Sliced temporaries for parfor (struct field assignment is not classifiable in parfor).
foldDetailsCell = cell(numel(stateRange), 1);
topEigenvalueVec = nan(numel(stateRange), 1);
stateDefMatrixCell = cell(numel(stateRange), 1);

if verbose
    fprintf('[HMM CV] Start: T=%d, D=%d, K candidates=%s, folds=%d\n', ...
        T, size(observationMatrix, 2), mat2str(stateRange), numFolds);
end

if useParallel
    try
        poolObj = gcp('nocreate');
        if isempty(poolObj)
            parpool('local', numWorkers);
            if verbose
                fprintf('[HMM CV] Parallel mode enabled with %d workers.\n', numWorkers);
            end
        elseif poolObj.NumWorkers ~= numWorkers
            delete(poolObj);
            parpool('local', numWorkers);
            if verbose
                fprintf('[HMM CV] Resized parallel pool to %d workers.\n', numWorkers);
            end
        elseif verbose
            fprintf('[HMM CV] Using existing parallel pool (%d workers).\n', poolObj.NumWorkers);
        end
    catch parallelErr
        useParallel = false;
        warning('gaussian_hmm_cv_select_num_states:ParallelUnavailable', ...
            'Could not start parallel pool (%s). Falling back to serial execution.', parallelErr.message);
    end
end

if useParallel
    parfor rangeIdx = 1:numel(stateRange)
        numStates = stateRange(rangeIdx);
        foldLls = nan(numFolds, 1);
        for foldIdx = 1:numFolds
            testStart = (foldIdx - 1) * foldLen + 1;
            testEnd = foldIdx * foldLen;
            if foldIdx == numFolds
                testEnd = T;
            end
            testMask = false(T, 1);
            testMask(testStart:testEnd) = true;
            trainIdx = find(~testMask);
            testIdx = find(testMask);
            if numel(trainIdx) < numStates * 5 || numel(testIdx) < 2
                foldLls(foldIdx) = nan;
                continue;
            end
            trainX = observationMatrix(trainIdx, :);
            testX = observationMatrix(testIdx, :);
            try
                [foldModel, ~] = gaussian_hmm_fit(trainX, numStates, emOptsTrain);
                if isempty(foldModel)
                    foldLls(foldIdx) = nan;
                    continue;
                end
                ll = gaussian_hmm_sequence_loglik(testX, foldModel);
                foldLls(foldIdx) = ll / size(testX, 1);
            catch
                foldLls(foldIdx) = nan;
            end
        end
        cvScores(rangeIdx) = mean(foldLls, 'omitnan');
        foldDetailsCell{rangeIdx} = foldLls;
        try
            emOptsEval = emOpts;
            emOptsEval.decode = false;
            [evalModel, ~] = gaussian_hmm_fit(observationMatrix, numStates, emOptsEval);
            stateDefMat = compute_state_definition_matrix(evalModel.mu, numBands);
            eigVals = eig(cov(stateDefMat));
            topEigenvalueVec(rangeIdx) = max(real(eigVals));
            stateDefMatrixCell{rangeIdx} = stateDefMat;
        catch
            topEigenvalueVec(rangeIdx) = nan;
            stateDefMatrixCell{rangeIdx} = [];
        end
    end
else
    for rangeIdx = 1:numel(stateRange)
    numStates = stateRange(rangeIdx);
    if verbose
        fprintf('[HMM CV] K=%d (%d/%d)\n', numStates, rangeIdx, numel(stateRange));
    end
    foldLls = nan(numFolds, 1);
    for foldIdx = 1:numFolds
        testStart = (foldIdx - 1) * foldLen + 1;
        testEnd = foldIdx * foldLen;
        if foldIdx == numFolds
            testEnd = T;
        end
        testMask = false(T, 1);
        testMask(testStart:testEnd) = true;
        trainIdx = find(~testMask);
        testIdx = find(testMask);
        if numel(trainIdx) < numStates * 5 || numel(testIdx) < 2
            foldLls(foldIdx) = nan;
            if verbose
                fprintf('[HMM CV]   fold %d/%d skipped (train=%d, test=%d)\n', ...
                    foldIdx, numFolds, numel(trainIdx), numel(testIdx));
            end
            continue;
        end
        trainX = observationMatrix(trainIdx, :);
        testX = observationMatrix(testIdx, :);
        if verbose
            fprintf('[HMM CV]   fold %d/%d fitting... ', foldIdx, numFolds);
        end
        try
            [foldModel, ~] = gaussian_hmm_fit(trainX, numStates, emOptsTrain);
            if isempty(foldModel)
                foldLls(foldIdx) = nan;
                if verbose
                    fprintf('failed (empty model)\n');
                end
                continue;
            end
            ll = gaussian_hmm_sequence_loglik(testX, foldModel);
            foldLls(foldIdx) = ll / size(testX, 1);
            if verbose
                fprintf('done (ll/bin=%.4f)\n', foldLls(foldIdx));
            end
        catch
            foldLls(foldIdx) = nan;
            if verbose
                fprintf('failed (exception)\n');
            end
        end
    end
    cvScores(rangeIdx) = mean(foldLls, 'omitnan');
    foldDetailsCell{rangeIdx} = foldLls;
    try
        emOptsEval = emOpts;
        emOptsEval.decode = false;
        [evalModel, ~] = gaussian_hmm_fit(observationMatrix, numStates, emOptsEval);
        stateDefMat = compute_state_definition_matrix(evalModel.mu, numBands);
        eigVals = eig(cov(stateDefMat));
        topEigenvalueVec(rangeIdx) = max(real(eigVals));
        stateDefMatrixCell{rangeIdx} = stateDefMat;
    catch
        topEigenvalueVec(rangeIdx) = nan;
        stateDefMatrixCell{rangeIdx} = [];
    end
    if verbose
        fprintf('[HMM CV] K=%d mean ll/bin=%.4f, lambda1=%.4f\n', ...
            numStates, cvScores(rangeIdx), topEigenvalueVec(rangeIdx));
    end
    end
end

diagnostics = struct();
diagnostics.foldDetails = foldDetailsCell;
diagnostics.topEigenvalue = topEigenvalueVec;
diagnostics.stateDefMatrix = stateDefMatrixCell;
diagnostics.selectionScore = nan(numel(stateRange), 1);
diagnostics.normCvScores = nan(numel(stateRange), 1);
diagnostics.normTopEigenvalue = nan(numel(stateRange), 1);

if useParallel && verbose
    for rangeIdx = 1:numel(stateRange)
        fprintf('[HMM CV] K=%d mean ll/bin=%.4f, lambda1=%.4f\n', ...
            stateRange(rangeIdx), cvScores(rangeIdx), diagnostics.topEigenvalue(rangeIdx));
    end
end

normCv = normalize_to_unit_range(cvScores);
normLambda = normalize_to_unit_range(diagnostics.topEigenvalue);
selectionScore = normCv ./ normLambda;
selectionScore(~isfinite(selectionScore)) = nan;
diagnostics.selectionScore = selectionScore;
diagnostics.normCvScores = normCv;
diagnostics.normTopEigenvalue = normLambda;

[scoreMax, bestIdx] = max(selectionScore, [], 'omitnan');
if isnan(scoreMax)
    bestNumStates = min(3, max(stateRange));
    warning('gaussian_hmm_cv_select_num_states:AllFoldsFailed', ...
        'CV failed for all K; defaulting to bestNumStates=%d.', bestNumStates);
else
    bestNumStates = stateRange(bestIdx);
end
if verbose
    fprintf('[HMM CV] Selected K=%d (Akella score=%.4f)\n', bestNumStates, scoreMax);
    fprintf('[HMM CV] Full-data fit for selected K=%d...\n', bestNumStates);
end

emOptsFull = emOpts;
emOptsFull.decode = true;
[bestModel, diagnostics.fullFitDiag] = gaussian_hmm_fit(observationMatrix, bestNumStates, emOptsFull);
if verbose
    fprintf('[HMM CV] Full-data fit complete.\n');
end
end

function stateDefMat = compute_state_definition_matrix(muMat, numBands)
% Variables:
%   - muMat: [numStates x nFeatures] Gaussian means
%   - numBands: number of frequency bands
% Goal:
%   Build state-definition matrix [numStates x numBands] by averaging each
%   band's means across areas/layers represented in the feature layout.

[numStates, nFeatures] = size(muMat);
assert(mod(nFeatures, numBands) == 0, ...
    'Feature count (%d) must be divisible by numBands (%d).', nFeatures, numBands);
numGroups = nFeatures / numBands;
muReshaped = reshape(muMat', numBands, numGroups, numStates);
stateDefMat = squeeze(mean(muReshaped, 2))';
end

function normVals = normalize_to_unit_range(valuesIn)
% Variables:
%   - valuesIn: numeric vector with potential NaN values
% Goal:
%   Normalize valid values to [-1, 1] while preserving NaN locations.

normVals = nan(size(valuesIn));
validMask = isfinite(valuesIn);
if ~any(validMask)
    return;
end
v = valuesIn(validMask);
vMin = min(v);
vMax = max(v);
if vMax == vMin
    normVals(validMask) = 0;
else
    normVals(validMask) = 2 * ((v - vMin) / (vMax - vMin)) - 1;
end
end
