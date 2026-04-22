function [bestNumStates, cvScores, bestModel, diagnostics] = gaussian_hmm_cv_select_num_states(observationMatrix, stateRange, numFolds, emOpts)
% GAUSSIAN_HMM_CV_SELECT_NUM_STATES Cross-validated sequence log-likelihood per state count.
%
% Variables:
%   - observationMatrix: [T x D] z-scored or raw features
%   - stateRange: vector of K values to compare (e.g. 2:6)
%   - numFolds: number of contiguous held-out blocks
%   - emOpts: options struct for gaussian_hmm_fit (maxIter, numRestarts, regVar, decode=false for speed)
% Goal:
%   Train HMM on training bins, evaluate mean per-bin log-likelihood on held-out bins; pick best K.
% Returns:
%   - bestNumStates: K with highest mean CV log-likelihood
%   - cvScores: length(stateRange) vector of mean CV log p / bin
%   - bestModel: model refit on full data at bestNumStates with decode true
%   - diagnostics: struct with foldDetails cell per K

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

cvScores = nan(numel(stateRange), 1);
% Scalar struct: struct('foldDetails', cell(n,1)) would create n structs, not one field of cells.
diagnostics = struct();
diagnostics.foldDetails = cell(numel(stateRange), 1);

for rangeIdx = 1:numel(stateRange)
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
    diagnostics.foldDetails{rangeIdx} = foldLls;
end

[cvMax, bestIdx] = max(cvScores, [], 'omitnan');
if isnan(cvMax)
    bestNumStates = min(3, max(stateRange));
    warning('gaussian_hmm_cv_select_num_states:AllFoldsFailed', ...
        'CV failed for all K; defaulting to bestNumStates=%d.', bestNumStates);
else
    bestNumStates = stateRange(bestIdx);
end

emOptsFull = emOpts;
emOptsFull.decode = true;
[bestModel, diagnostics.fullFitDiag] = gaussian_hmm_fit(observationMatrix, bestNumStates, emOptsFull);
end
