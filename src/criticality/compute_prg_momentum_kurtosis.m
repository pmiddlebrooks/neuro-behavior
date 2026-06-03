function prgOut = compute_prg_momentum_kurtosis(dataMat, varargin)
% COMPUTE_PRG_MOMENTUM_KURTOSIS Momentum-space PRG coarse-graining and kurtosis
%
% Variables:
%   dataMat - Binned spike counts [nTimeBins x nNeurons] for one analysis window
%   varargin - Optional name-value pairs:
%       'cutoffDivisors' - Divisors of N for N_cutoff sequence (default [1 2 4 8 16])
%       'finalCutoffDivisor' - Divisor defining report kappa (default 16 -> N/16)
%
% Goal:
%   Apply momentum-space phenomenological RG (Bradde & Bialek 2017; Cambrainha et al.
%   2025 PRX Life): covariance eigenvector projection with per-unit variance
%   normalization, then kurtosis kappa = <psi^4> / <psi^2>^2 at each cutoff.
%   Does not perform real-space paired-correlation coarse-graining.
%
% Returns:
%   prgOut - Struct with fields:
%     .kappaByCutoff, .nCutoffList, .kappaFinal, .djsFinal, .finalCutoffDivisor,
%     .popCv, .nNeurons, .nTimeBins
%     .djsFinal - Jensen-Shannon distance sqrt(JSD(P(psi)||N(0,1))) at final cutoff

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
        'nTimeBins', 0);

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

    centeredMat = activityMat - mean(activityMat, 1);
    covMat = cov(centeredMat);
    if any(~isfinite(covMat(:)))
        return;
    end

    [eigVecs, eigVals] = eig((covMat + covMat.') / 2);
    [eigVals, sortIdx] = sort(real(diag(eigVals)), 'descend');
    eigVecs = eigVecs(:, sortIdx);

    nCutoffList = max(1, round(nNeurons ./ cutoffDivisors));
    nCutoffList = unique(nCutoffList, 'stable');
    kappaByCutoff = nan(1, numel(nCutoffList));

    for iCut = 1:numel(nCutoffList)
        nCutoff = nCutoffList(iCut);
        kappaByCutoff(iCut) = kurtosis_at_cutoff(centeredMat, eigVecs, nCutoff);
    end

    prgOut.nCutoffList = nCutoffList;
    prgOut.kappaByCutoff = kappaByCutoff;

    nFinal = max(1, round(nNeurons / finalCutoffDivisor));
    matchIdx = find(nCutoffList == nFinal, 1);
    if ~isempty(matchIdx)
        prgOut.kappaFinal = kappaByCutoff(matchIdx);
        [~, psiValsFinal] = normalized_psi_at_cutoff(centeredMat, eigVecs, nFinal);
    else
        [prgOut.kappaFinal, psiValsFinal] = normalized_psi_at_cutoff(centeredMat, eigVecs, nFinal);
    end

    prgOut.djsFinal = compute_js_distance_psi_gaussian(psiValsFinal);
end

function kappaVal = kurtosis_at_cutoff(centeredMat, eigVecs, nCutoff)
% KURTOSIS_AT_CUTOFF Project onto top covariance modes and compute pooled kurtosis.

    [kappaVal, ~] = normalized_psi_at_cutoff(centeredMat, eigVecs, nCutoff);
end

function [kappaVal, psiVals] = normalized_psi_at_cutoff(centeredMat, eigVecs, nCutoff)
% NORMALIZED_PSI_AT_CUTOFF Coarse-grained, variance-normalized activity at N_cutoff
%
% Variables:
%   centeredMat - Mean-centered binned activity [time x neurons]
%   eigVecs     - Covariance eigenvectors (columns ranked by eigenvalue)
%   nCutoff     - Number of top eigenmodes retained
%
% Returns:
%   kappaVal - Pooled kurtosis <psi^4>/<psi^2>^2 over all finite samples
%   psiVals  - Vector of normalized coarse-grained samples (for distribution metrics)

    kappaVal = nan;
    psiVals = [];
    nNeurons = size(centeredMat, 2);
    nCutoff = min(max(1, round(nCutoff)), nNeurons);

    projMat = eigVecs(:, 1:nCutoff) * eigVecs(:, 1:nCutoff).';
    psiMat = centeredMat * projMat.';

    for iUnit = 1:nNeurons
        unitStd = std(psiMat(:, iUnit), 0, 1);
        if unitStd > 0 && isfinite(unitStd)
            psiMat(:, iUnit) = psiMat(:, iUnit) ./ unitStd;
        end
    end

    psiVals = psiMat(:);
    psiVals = psiVals(isfinite(psiVals));
    if numel(psiVals) < 4
        psiVals = [];
        return;
    end

    secondMoment = mean(psiVals .^ 2);
    if secondMoment <= 0 || ~isfinite(secondMoment)
        psiVals = [];
        return;
    end
    kappaVal = mean(psiVals .^ 4) / (secondMoment ^ 2);
end
