function binSizeSec = population_mean_isi_bin_size(spikeTimes, spikeClusters, neuronIds, timeRange)
% POPULATION_MEAN_ISI_BIN_SIZE - Bin width from mean population inter-spike interval
%
% Variables:
%   spikeTimes    - Spike times (seconds)
%   spikeClusters - Neuron ID per spike
%   neuronIds     - Neuron IDs in the population
%   timeRange     - [startTime, endTime] in seconds
%
% Goal:
%   Literature-style avalanche binning: use mean ISI of the pooled population
%   spike train within timeRange as the bin size (seconds).

minBinSec = 1e-3;

if nargin < 4 || numel(timeRange) ~= 2
  error('timeRange must be [startTime, endTime].');
end

if isempty(neuronIds)
  error('neuronIds cannot be empty.');
end

neuronIds = neuronIds(:)';
inRange = spikeTimes >= timeRange(1) & spikeTimes < timeRange(2);
inPop = ismember(spikeClusters, neuronIds);
spikeList = sort(spikeTimes(inRange & inPop));

if numel(spikeList) < 2
  binSizeSec = minBinSec;
  return;
end

meanIsi = mean(diff(spikeList));
if ~isfinite(meanIsi) || meanIsi <= 0
  binSizeSec = minBinSec;
else
  binSizeSec = max(meanIsi, minBinSec);
end
end
