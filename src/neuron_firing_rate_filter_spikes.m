function keepMask = neuron_firing_rate_filter_spikes(opts, neuronIDs, spikeTimes, spikeClusters, timeStart, timeEnd)
% NEURON_FIRING_RATE_FILTER_SPIKES - Keep mask from firing rate on spike trains
%
% Variables:
%   opts           - options with minFiringRate, maxFiringRate, firingRateCheckTime
%   neuronIDs      - vector of neuron/cluster ids
%   spikeTimes     - all spike times (seconds)
%   spikeClusters  - cluster id per spike
%   timeStart      - start of session/window for rate calculation (seconds)
%   timeEnd        - end of session/window for rate calculation (seconds)
%
% Returns:
%   keepMask - logical vector, true for neurons that pass criteria
%
% Goal:
%   If opts.firingRateCheckTime is empty, require min/max firing rate over the
%   full [timeStart, timeEnd] interval. Otherwise check start and end windows.

nNeurons = numel(neuronIDs);
keepMask = false(nNeurons, 1);
useWholeSession = ~isfield(opts, 'firingRateCheckTime') || isempty(opts.firingRateCheckTime);
sessionDuration = timeEnd - timeStart;
if sessionDuration <= 0
    error('neuron_firing_rate_filter_spikes: timeEnd must be greater than timeStart.');
end

for iNeuron = 1:nNeurons
    neuronID = neuronIDs(iNeuron);
    neuronSpikes = spikeTimes(spikeClusters == neuronID);
    neuronSpikes = neuronSpikes(neuronSpikes >= timeStart & neuronSpikes <= timeEnd);

    if useWholeSession
        firingRate = numel(neuronSpikes) / sessionDuration;
        keepMask(iNeuron) = firingRate >= opts.minFiringRate & firingRate <= opts.maxFiringRate;
    else
        checkTime = opts.firingRateCheckTime;
        spikesStart = sum(neuronSpikes >= timeStart & neuronSpikes <= timeStart + checkTime);
        firingRateStart = spikesStart / checkTime;
        spikesEnd = sum(neuronSpikes >= (timeEnd - checkTime) & neuronSpikes <= timeEnd);
        firingRateEnd = spikesEnd / checkTime;
        keepMask(iNeuron) = firingRateStart >= opts.minFiringRate & firingRateStart <= opts.maxFiringRate & ...
            firingRateEnd >= opts.minFiringRate & firingRateEnd <= opts.maxFiringRate;
    end
end
end
