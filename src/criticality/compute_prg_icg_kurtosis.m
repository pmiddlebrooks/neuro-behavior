function prgOut = compute_prg_icg_kurtosis(dataMat, varargin)
% COMPUTE_PRG_ICG_KURTOSIS Real-space ICG coarse-graining with kurtosis and D_JS
%
% Variables:
%   dataMat - Binned spike counts [nTimeBins x nNeurons] for one analysis window
%   varargin - Optional name-value pairs:
%       'cutoffDivisors'     - Cluster-size divisors K = N/divisor (default [1 2 4 8 16])
%       'finalCutoffDivisor' - Report metrics at cluster size N/divisor (default 16)
%
% Goal:
%   Iteratively coarse-grain activity by pairing the most correlated units,
%   summing their activity, and renormalizing each new unit so the mean of
%   nonzero activity equals 1 (Morales et al. 2023 PNAS, Eq. 6). At each
%   coarse-graining scale, compute pooled kurtosis and Jensen-Shannon distance
%   to a Gaussian at the final scale (Cambrainha et al. 2026).
%
% Returns:
%   prgOut - Struct aligned with compute_prg_momentum_kurtosis output fields

    p = inputParser;
    addParameter(p, 'cutoffDivisors', [1, 2, 4, 8, 16], @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'finalCutoffDivisor', 16, @(x) isnumeric(x) && isscalar(x) && x > 0);
    parse(p, varargin{:});

    cutoffDivisors = unique(round(p.Results.cutoffDivisors(:)'), 'stable');
    finalCutoffDivisor = round(p.Results.finalCutoffDivisor);

    prgOut = struct( ...
        'kappaByCutoff', nan(1, numel(cutoffDivisors)), ...
        'nCutoffList', nan(1, numel(cutoffDivisors)), ...
        'kappaFinal', nan, ...
        'djsFinal', nan, ...
        'finalCutoffDivisor', finalCutoffDivisor, ...
        'popCv', nan, ...
        'nNeurons', 0, ...
        'nTimeBins', 0, ...
        'prgMethod', 'icg');

    if isempty(dataMat)
        return;
    end

    activityMat = double(dataMat);
    [nTimeBins, nNeurons] = size(activityMat);
    prgOut.nNeurons = nNeurons;
    prgOut.nTimeBins = nTimeBins;

    if nTimeBins < 2 || nNeurons < 1
        return;
    end

    popActivity = sum(activityMat, 2);
    popMean = mean(popActivity);
    if popMean > 0
        prgOut.popCv = std(popActivity, 0, 1) / popMean;
    else
        prgOut.popCv = inf;
    end

    icgLevels = run_icg_coarse_grain_normalized(activityMat.');
    if isempty(icgLevels)
        return;
    end

    nCutoffList = max(1, round(nNeurons ./ cutoffDivisors));
    nCutoffList = unique(nCutoffList, 'stable');
    kappaByCutoff = nan(1, numel(nCutoffList));

    for iCut = 1:numel(nCutoffList)
        nClustersTarget = nCutoffList(iCut);
        levelAct = icg_activity_at_cluster_count(icgLevels, nClustersTarget);
        kappaByCutoff(iCut) = pooled_kurtosis_from_activity(levelAct);
    end

    prgOut.nCutoffList = nCutoffList;
    prgOut.kappaByCutoff = kappaByCutoff;

    nFinal = max(1, round(nNeurons / finalCutoffDivisor));
    finalAct = icg_activity_at_cluster_count(icgLevels, nFinal);
    [prgOut.kappaFinal, psiValsFinal] = pooled_kurtosis_and_samples(finalAct);
    prgOut.djsFinal = compute_js_distance_psi_gaussian(psiValsFinal);
end

function icgLevels = run_icg_coarse_grain_normalized(activityNeuronsByTime)
% RUN_ICG_COARSE_GRAIN_NORMALIZED Greedy ICG pairing with nonzero-mean normalization
%
% Variables:
%   activityNeuronsByTime - [nNeurons x nTimeBins] binned activity
%
% Returns:
%   icgLevels - Cell array; level 1 is raw input, each subsequent level is ICG output

    icgLevels = {};
    if isempty(activityNeuronsByTime)
        return;
    end

    icgLevels{1} = activityNeuronsByTime;
    currentAct = activityNeuronsByTime;
    nNeurons = size(currentAct, 1);

    while nNeurons >= 2
        pairedAct = icg_pair_and_sum_normalized(currentAct);
        if isempty(pairedAct)
            break;
        end
        icgLevels{end + 1} = pairedAct; %#ok<AGROW>
        currentAct = pairedAct;
        nNeurons = size(currentAct, 1);
    end
end

function pairedAct = icg_pair_and_sum_normalized(activityMat)
% ICG_PAIR_AND_SUM_NORMALIZED One greedy ICG step with Morales normalization

    pairedAct = [];
    nNeurons = size(activityMat, 1);
    if nNeurons < 2
        return;
    end

    rho = corr(activityMat.');
    if any(~isfinite(rho(:)))
        return;
    end

    upperTriMask = triu(true(size(rho)), 1);
    upperVals = rho(upperTriMask);
    [~, sortIdx] = sort(upperVals, 'descend');
    [pairRows, pairCols] = ind2sub(size(rho), find(upperTriMask));
    pairRows = pairRows(sortIdx);
    pairCols = pairCols(sortIdx);

    numPairs = floor(nNeurons / 2);
    outAct = nan(numPairs, size(activityMat, 2));
    usedNeurons = false(nNeurons, 1);

    pairCount = 0;
    for iPair = 1:numel(pairRows)
        rowNew = pairRows(iPair);
        colNew = pairCols(iPair);
        if usedNeurons(rowNew) || usedNeurons(colNew)
            continue;
        end

        pairCount = pairCount + 1;
        summedAct = activityMat(rowNew, :) + activityMat(colNew, :);
        outAct(pairCount, :) = normalize_icg_nonzero_mean_unit(summedAct);
        usedNeurons(rowNew) = true;
        usedNeurons(colNew) = true;

        if pairCount >= numPairs
            break;
        end
    end

    if pairCount == 0
        return;
    end

    pairedAct = outAct(1:pairCount, :);
end

function levelAct = icg_activity_at_cluster_count(icgLevels, nClustersTarget)
% ICG_ACTIVITY_AT_CLUSTER_COUNT Pick ICG level closest to target cluster count

    levelAct = [];
    if isempty(icgLevels)
        return;
    end

    nClustersTarget = max(1, round(nClustersTarget));
    levelSizes = cellfun(@(x) size(x, 1), icgLevels);
    [~, bestIdx] = min(abs(levelSizes - nClustersTarget));
    levelAct = icgLevels{bestIdx};
end

function kappaVal = pooled_kurtosis_from_activity(activityMat)
% POOLED_KURTOSIS_FROM_ACTIVITY Pooled fourth-moment kurtosis over units and time

    [kappaVal, ~] = pooled_kurtosis_and_samples(activityMat);
end

function [kappaVal, sampleVals] = pooled_kurtosis_and_samples(activityMat)
% POOLED_KURTOSIS_AND_SAMPLES Kurtosis and pooled samples from [units x time]

    kappaVal = nan;
    sampleVals = [];
    if isempty(activityMat)
        return;
    end

    sampleVals = activityMat(:);
    sampleVals = sampleVals(isfinite(sampleVals));
    if numel(sampleVals) < 4
        sampleVals = [];
        return;
    end

    secondMoment = mean(sampleVals .^ 2);
    if secondMoment <= 0 || ~isfinite(secondMoment)
        sampleVals = [];
        return;
    end

    kappaVal = mean(sampleVals .^ 4) / (secondMoment ^ 2);
end
