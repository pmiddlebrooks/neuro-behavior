function djs = compute_js_distance_psi_gaussian(psiVals, varargin)
% COMPUTE_JS_DISTANCE_PSI_GAUSSIAN Jensen-Shannon distance of P(psi) vs N(0,1)
%
% Variables:
%   psiVals  - Normalized coarse-grained activity samples (pooled over units/time)
%   varargin - Optional name-value pairs:
%       'nBins'    - Number of histogram bins (default 100)
%       'psiRange' - [lo hi] integration range; default spans data and +/-5 sigma
%
% Goal:
%   Estimate JSD between the empirical distribution of coarse-grained activity
%   and a standard Gaussian reference, following Cambrainha et al. 2026 (Eq. A5-A7):
%     JSD(P||N) = 1/2 int P log2(P/M) dpsi + 1/2 int N log2(N/M) dpsi, M = (P+N)/2
%     D_JS = sqrt(JSD), bounded in [0, 1] with log base 2.
%
% Returns:
%   djs - Jensen-Shannon distance (sqrt of divergence); NaN if insufficient data

    p = inputParser;
    addParameter(p, 'nBins', 100, @(x) isnumeric(x) && isscalar(x) && x >= 8);
    addParameter(p, 'psiRange', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
    parse(p, varargin{:});

    nBins = round(p.Results.nBins);
    psiRange = p.Results.psiRange;

    psiVals = psiVals(isfinite(psiVals(:)));
    if numel(psiVals) < 4
        djs = nan;
        return;
    end

    if isempty(psiRange)
        dataLo = min(psiVals);
        dataHi = max(psiVals);
        span = dataHi - dataLo;
        pad = max(0.05 * span, 0.25);
        if span <= 0 || ~isfinite(span)
            pad = 0.5;
        end
        rangeLo = min(dataLo - pad, -5);
        rangeHi = max(dataHi + pad, 5);
    else
        rangeLo = min(psiRange(1), psiRange(2));
        rangeHi = max(psiRange(1), psiRange(2));
    end

    if rangeHi <= rangeLo
        djs = nan;
        return;
    end

    binEdges = linspace(rangeLo, rangeHi, nBins + 1);
    empCounts = histcounts(psiVals, binEdges);
    pEmp = empCounts / sum(empCounts);

    qGauss = diff(normcdf(binEdges, 0, 1));
    qGauss = qGauss / sum(qGauss);

    jsdVal = jensen_shannon_divergence_discrete(pEmp, qGauss);
    djs = sqrt(max(0, jsdVal));
end

function jsdVal = jensen_shannon_divergence_discrete(pDist, qDist)
% JENSEN_SHANNON_DIVERGENCE_DISCRETE Symmetric JSD for discrete probability masses
%
% Variables:
%   pDist, qDist - Probability mass vectors (same length, sum to 1)
%
% Goal:
%   JSD(P||Q) = 1/2 KL(P||M) + 1/2 KL(Q||M), M = (P+Q)/2, log base 2.

    pDist = pDist(:);
    qDist = qDist(:);
    if numel(pDist) ~= numel(qDist) || sum(pDist) <= 0 || sum(qDist) <= 0
        jsdVal = nan;
        return;
    end

    pDist = pDist / sum(pDist);
    qDist = qDist / sum(qDist);
    mDist = (pDist + qDist) / 2;

    jsdVal = 0.5 * kl_divergence_discrete(pDist, mDist) ...
        + 0.5 * kl_divergence_discrete(qDist, mDist);
end

function klVal = kl_divergence_discrete(pDist, qDist)
% KL_DIVERGENCE_DISCRETE KL(P||Q) with log base 2; 0*log(0) treated as 0.

    validIdx = pDist > 0 & qDist > 0;
    if ~any(validIdx)
        klVal = 0;
        return;
    end
    klVal = sum(pDist(validIdx) .* log2(pDist(validIdx) ./ qDist(validIdx)));
end
