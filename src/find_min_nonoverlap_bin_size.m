function [selectedBinSize, binSweepSummary] = find_min_nonoverlap_bin_size(spikeTimes, neuronIds, maxAllowedMultiSpikeProportion)
% FIND_MIN_NONOVERLAP_BIN_SIZE Select largest bin with no multi-spike bins.
%
% Variables:
%   spikeTimes       - Spike timestamps (seconds), vector.
%   neuronIds        - Neuron ids aligned with spikeTimes, vector.
%   maxAllowedMultiSpikeProportion - Maximum allowed fraction of occupied bins
%                                    with >1 spike (default: 0).
%
% Outputs:
%   selectedBinSize  - Selected bin size (seconds). Largest admissible bin in
%                      [0.001, 0.005]; falls back to 0.001 if none satisfy.
%   binSweepSummary  - Struct array with fields:
%                      .binSizeSec, .numOccupiedBins, .numMultiSpikeBins,
%                      .proportionMultiSpikeBins.
%
% Goal:
%   Sweep candidate bin sizes from 5 ms down to 1 ms, print the proportion
%   of occupied bins containing more than one spike at each candidate, and
%   return the largest candidate bin size whose multi-spike-bin proportion
%   is <= maxAllowedMultiSpikeProportion.

spikeTimes = spikeTimes(:);
neuronIds = neuronIds(:); %#ok<NASGU>

if numel(spikeTimes) ~= numel(neuronIds)
    error('spikeTimes and neuronIds must have the same number of elements.');
end
if nargin < 3 || isempty(maxAllowedMultiSpikeProportion)
    maxAllowedMultiSpikeProportion = 0;
end
if ~isscalar(maxAllowedMultiSpikeProportion) || ~isfinite(maxAllowedMultiSpikeProportion) ...
        || maxAllowedMultiSpikeProportion < 0 || maxAllowedMultiSpikeProportion > 1
    error('maxAllowedMultiSpikeProportion must be a scalar in [0, 1].');
end

candidateBinSizes = (5:-1:1) * 1e-3;
numCandidates = numel(candidateBinSizes);
binSweepSummary = repmat(struct( ...
    'binSizeSec', 0, ...
    'numOccupiedBins', 0, ...
    'numMultiSpikeBins', 0, ...
    'proportionMultiSpikeBins', 0), numCandidates, 1);

selectedBinSize = 1e-3;

if isempty(spikeTimes)
    fprintf('No spikes provided; defaulting selected bin size to %.3f ms.\n', selectedBinSize * 1e3);
    for candidateIdx = 1:numCandidates
        binSweepSummary(candidateIdx).binSizeSec = candidateBinSizes(candidateIdx);
    end
    return;
end

for candidateIdx = 1:numCandidates
    candidateBinSize = candidateBinSizes(candidateIdx);
    binIdx = floor(spikeTimes ./ candidateBinSize) + 1;
    spikesPerBin = accumarray(binIdx, 1);
    numOccupiedBins = numel(spikesPerBin);
    numMultiSpikeBins = sum(spikesPerBin > 1);
    proportionMultiSpikeBins = numMultiSpikeBins / max(numOccupiedBins, 1);

    binSweepSummary(candidateIdx).binSizeSec = candidateBinSize;
    binSweepSummary(candidateIdx).numOccupiedBins = numOccupiedBins;
    binSweepSummary(candidateIdx).numMultiSpikeBins = numMultiSpikeBins;
    binSweepSummary(candidateIdx).proportionMultiSpikeBins = proportionMultiSpikeBins;

    fprintf('Bin %.3f ms: multi-spike bins %d/%d (%.4f)\n', ...
        candidateBinSize * 1e3, numMultiSpikeBins, numOccupiedBins, proportionMultiSpikeBins);

    if proportionMultiSpikeBins <= maxAllowedMultiSpikeProportion
        selectedBinSize = candidateBinSize;
        break;
    end
end

fprintf('Selected HMM bin size: %.3f ms\n', selectedBinSize * 1e3);

end
