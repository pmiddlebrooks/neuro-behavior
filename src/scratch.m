
%%
% Test helper: find_min_nonoverlap_bin_size on spontaneous spikes
paths = get_paths;
opts = neuro_behavior_options;
opts.collectStart = 0;
opts.collectEnd = 10 * 60;
opts.minActTime = 0.16;
opts.minFiringRate = 0.2;
opts.firingRateCheckTime = 5 * 60;
opts.maxFiringRate = 100;
opts.removeSome = true;

% Get a session from choose_task_and_session.m
exampleSessionName = sessionName;


fprintf('Loading spontaneous session: %s\n', exampleSessionName);
spikeDataStruct = load_spike_times('spontaneous', paths, exampleSessionName, opts);

areaName = 'DS'; % Choose area to test: M23, M56, DS, VS
areaNeuronMask = strcmp(spikeDataStruct.neuronAreas, areaName);
areaNeuronIds = spikeDataStruct.neuronIDs(areaNeuronMask);
areaSpikeMask = ismember(spikeDataStruct.spikeClusters, areaNeuronIds);
areaSpikeTimes = spikeDataStruct.spikeTimes(areaSpikeMask);
areaSpikeNeuronIds = spikeDataStruct.spikeClusters(areaSpikeMask);

maxAllowedMultiSpikeProportion = 0.0; % Try values like 0.01 or 0.05
[selectedBinSize, binSweepSummary] = find_min_nonoverlap_bin_size( ...
    areaSpikeTimes, areaSpikeNeuronIds, maxAllowedMultiSpikeProportion);

fprintf('Area %s selected bin size: %.3f ms\n', areaName, selectedBinSize * 1e3);
disp(binSweepSummary);

