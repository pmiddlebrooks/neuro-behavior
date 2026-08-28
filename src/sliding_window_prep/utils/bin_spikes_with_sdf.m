function dataMat = bin_spikes_with_sdf(spikeTimes, spikeClusters, neuronIDs, timeRange, binSize, sdfSigmaMs)
% BIN_SPIKES_WITH_SDF - 1 ms raster, Gaussian SDF, downsample to analysis bins
%
% Variables:
%   spikeTimes   - Spike times (seconds)
%   spikeClusters - Neuron ID per spike
%   neuronIDs    - Neuron IDs to include (column order in output)
%   timeRange    - [startTime, endTime] in seconds
%   binSize      - Analysis bin width in seconds (must be >= 1 ms)
%   sdfSigmaMs   - Gaussian kernel sigma in milliseconds (default 10)
%
% Goal:
%   Smooth each neuron with spike_density_function on a 1 ms grid, then
%   integrate the SDF into analysis-sized bins. Output is expected spike
%   counts per analysis bin (same role as bin_spikes counts), so PCA,
%   population sums, and AR/d2 keep the same matrix shape.
%
% Returns:
%   dataMat - [nBins x nNeurons] expected spike counts (single)

if nargin < 6 || isempty(sdfSigmaMs)
    sdfSigmaMs = 10;
end

sdfBinSize = 0.001;
if binSize < sdfBinSize - 1e-12
    error('binSize (%.6f s) must be >= the 1 ms SDF raster used by spike_density_function.', binSize);
end

numBins = ceil((timeRange(2) - timeRange(1)) / binSize);
nFine = ceil((timeRange(2) - timeRange(1)) / sdfBinSize);
if nFine < 1 || numBins < 1
    error('timeRange and bin sizes must yield at least one fine and one analysis bin.');
end

dataMat = zeros(numBins, numel(neuronIDs), 'single');

validSpikes = spikeTimes >= timeRange(1) & spikeTimes < timeRange(2);
spikeTimesFiltered = spikeTimes(validSpikes);
spikeClustersFiltered = spikeClusters(validSpikes);

fineEdges = timeRange(1) + (0:nFine) * sdfBinSize;
if fineEdges(end) < timeRange(2)
    fineEdges(end) = timeRange(2);
end

kernel = struct('method', 'gaussian', 'sigma', sdfSigmaMs);
samplesPerBin = binSize / sdfBinSize;

for n = 1:numel(neuronIDs)
    neuronID = neuronIDs(n);
    neuronSpikes = spikeTimesFiltered(spikeClustersFiltered == neuronID);
    fineCounts = zeros(1, nFine);
    if ~isempty(neuronSpikes)
        spikeCounts = histcounts(neuronSpikes, fineEdges);
        nCopy = min(nFine, numel(spikeCounts));
        fineCounts(1:nCopy) = spikeCounts(1:nCopy);
    end

    if ~any(fineCounts)
        continue;
    end

    sdfHz = spike_density_function(fineCounts, kernel);
    dataMat(:, n) = downsample_sdf_to_analysis_bins(sdfHz, nFine, numBins, samplesPerBin, sdfBinSize);
end
end

function coarseCounts = downsample_sdf_to_analysis_bins(sdfHz, nFine, numBins, samplesPerBin, sdfBinSize)
% DOWNSAMPLE_SDF_TO_ANALYSIS_BINS - Integrate 1 ms SDF (Hz) into analysis bins
%
% Variables:
%   sdfHz         - 1 ms SDF in spikes/s (row or column)
%   nFine         - Number of 1 ms samples
%   numBins       - Number of analysis bins
%   samplesPerBin - Analysis binSize / 1 ms
%   sdfBinSize    - Fine bin width in seconds (0.001)
%
% Goal:
%   Convert firing rate to expected spike count per analysis bin:
%   sum(sdfHz * sdfBinSize) over the fine samples in that bin.

sdfHz = double(sdfHz(:));
if numel(sdfHz) > nFine
    sdfHz = sdfHz(1:nFine);
elseif numel(sdfHz) < nFine
    sdfHz(nFine, 1) = 0;
end

coarseCounts = zeros(numBins, 1, 'single');
ratioIsInteger = abs(samplesPerBin - round(samplesPerBin)) < 1e-9;
if ratioIsInteger
    k = round(samplesPerBin);
    nFull = min(numBins, floor(nFine / k));
    if nFull > 0
        trimmed = sdfHz(1:(nFull * k));
        reshaped = reshape(trimmed, k, nFull);
        coarseCounts(1:nFull) = single(sum(reshaped, 1)' * sdfBinSize);
    end
    if nFull < numBins
        iStart = nFull * k + 1;
        if iStart <= nFine
            coarseCounts(nFull + 1) = single(sum(sdfHz(iStart:nFine)) * sdfBinSize);
        end
    end
    return;
end

for iBin = 1:numBins
    iStart = floor((iBin - 1) * samplesPerBin) + 1;
    iEnd = min(nFine, floor(iBin * samplesPerBin));
    if iStart > nFine || iEnd < iStart
        continue;
    end
    coarseCounts(iBin) = single(sum(sdfHz(iStart:iEnd)) * sdfBinSize);
end
end
