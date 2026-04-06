function report = hmm_mazz_unit_test_validate(results, groundTruthStateSeq)
% HMM_MAZZ_UNIT_TEST_VALIDATE Compare fitted HMM to synthetic ground-truth states.
%
% Variables:
%   results             - Top-level struct from hmm_mazz_analysis (.hmm_results cell).
%   groundTruthStateSeq - 1 x nBins vector of true state indices in 1..5.
%
% Goal:
%   Report alignment accuracy using MAP states from pStates (soft decode) with an
%   optimal permutation of fitted state labels vs truth (K x Ktrue confusion).

report = struct();
report.meanAccuracyArgmaxP = NaN;
report.meanAccuracyHard = NaN;
report.numBinsCompared = 0;
report.fittedNumStates = NaN;

if isempty(results) || ~isfield(results, 'hmm_results') || isempty(results.hmm_results{1})
    warning('hmm_mazz_unit_test_validate: empty results.');
    return;
end

areaRes = results.hmm_results{1};
if ~isfield(areaRes, 'continuous_results')
    warning('hmm_mazz_unit_test_validate: missing continuous_results.');
    return;
end

pStates = areaRes.continuous_results.pStates;
hardSeq = double(areaRes.continuous_results.sequence(:));
gt = double(groundTruthStateSeq(:));

nCompare = min(size(pStates, 1), numel(gt));
if nCompare < 1
    return;
end

pStates = pStates(1:nCompare, :);
hardSeq = hardSeq(1:nCompare);
gt = gt(1:nCompare);

[~, mapFromP] = max(pStates, [], 2);
kFit = size(pStates, 2);
report.fittedNumStates = kFit;
report.numBinsCompared = nCompare;

numTrueStates = max(gt);
report.meanAccuracyArgmaxP = local_accuracy_with_best_perm(mapFromP, gt, numTrueStates);

validHard = hardSeq > 0;
if any(validHard)
    report.meanAccuracyHard = local_accuracy_with_best_perm( ...
        hardSeq(validHard), gt(validHard), numTrueStates);
else
    report.meanAccuracyHard = NaN;
end

fprintf(['HMM unit-test validate: bins=%d, fittedK=%d, acc(MAP pStates vs truth, ' ...
    'best perm)=%.3f, acc(hard seq)=%.3f\n'], ...
    nCompare, kFit, report.meanAccuracyArgmaxP, report.meanAccuracyHard);

end

function acc = local_accuracy_with_best_perm(predLabels, trueLabels, numTrueStates)
% Greedy one-to-one matching from predicted labels to true labels (small K).

predLabels = predLabels(:);
trueLabels = trueLabels(:);
valid = predLabels > 0 & trueLabels > 0 & trueLabels <= numTrueStates;
predLabels = predLabels(valid);
trueLabels = trueLabels(valid);
if isempty(predLabels)
    acc = NaN;
    return;
end

uPred = unique(predLabels);
uTrue = unique(trueLabels);
numP = numel(uPred);
numT = numel(uTrue);
confusionMat = zeros(numP, numT);
for iPred = 1:numP
    for jTrue = 1:numT
        confusionMat(iPred, jTrue) = sum(predLabels == uPred(iPred) & trueLabels == uTrue(jTrue));
    end
end

% Greedy max-assignment (predicted row matched to at most one true column)
unusedCols = true(1, numT);
totalCorrect = 0;
rowOrder = zeros(1, numP);
for passIdx = 1:numP
    bestVal = -1;
    bestRow = 1;
    bestCol = 1;
    for iPred = 1:numP
        if rowOrder(iPred) > 0
            continue;
        end
        for jTrue = 1:numT
            if ~unusedCols(jTrue)
                continue;
            end
            v = confusionMat(iPred, jTrue);
            if v > bestVal
                bestVal = v;
                bestRow = iPred;
                bestCol = jTrue;
            end
        end
    end
    if bestVal < 0
        break;
    end
    totalCorrect = totalCorrect + bestVal;
    rowOrder(bestRow) = bestCol;
    unusedCols(bestCol) = false;
end

acc = totalCorrect / numel(predLabels);
end
