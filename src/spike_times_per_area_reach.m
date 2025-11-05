function [spikeData] = spike_times_per_area_reach(opts)
% SPIKE_TIMES_PER_AREA_REACH - Extract spike times, neuron IDs, and area labels for reach data
%
% Loads reach-format spike data and returns spike triplets [time, neuronId, areaNumeric]
% for neurons that pass firing-rate criteria within a collection window.
%
% INPUTS:
%   opts - Structure with fields:
%       .dataPath             - Path to the reach data
%       .collectStart         - Start time (s) of collection window (default 0)
%       .collectFor           - Number of seconds to collect (required)
%       .removeSome           - Whether to filter by firing rate (logical)
%       .firingRateCheckTime  - Seconds for start/end firing rate checks
%       .minFiringRate        - Minimum allowed firing rate (Hz)
%       .maxFiringRate        - Maximum allowed firing rate (Hz)
%
% OUTPUTS:
%   spikeData - N x 3 matrix: [spikeTimeSec, neuronId, areaNumeric]
%       areaNumeric: 1=M23, 2=M56, 3=DS, 4=VS

% Load reach-format data
data = load(opts.dataPath);

% Determine collection window
if ~isempty(opts.collectStart)
    firstSecond = opts.collectStart;
else
    firstSecond = 0;
end
if ~isempty(opts.collectEnd)
    lastSecond = firstSecond + opts.collectEnd;
else
    error('opts.collectEnd must be provided.');
end

% Map numeric area codes to 1..4 per convention: 1=M23, 2=M56, 3=DS, 4=VS
% In mark-data, area codes are already numeric in last column of idchan
% and mapped to these values downstream. We will reuse those 1..4 values.

% Select usable neurons: exclude corpus callosum (idchan(:,end) ~= 0) and keep areas 1..4
useNeuronsMask = (data.idchan(:,end) ~= 0) & ismember(data.idchan(:,4), [1 2]); % Keep neurons with good idchan.PhyRating values
idLabels = data.idchan(useNeuronsMask, 1);
areaNumeric = data.idchan(useNeuronsMask, end); % 1..4 already

% Optional firing-rate filtering
if opts.removeSome
    checkTime = opts.firingRateCheckTime;
    firingRatesStart = zeros(length(idLabels), 1);
    firingRatesEnd = zeros(length(idLabels), 1);
    
    for i = 1:length(idLabels)
        iSpikeTime = data.CSV(data.CSV(:,2) == idLabels(i), 1);
        % restrict to collection window
        iSpikeTime = iSpikeTime(iSpikeTime >= firstSecond & iSpikeTime <= lastSecond);
        
        % start window: [firstSecond, firstSecond+checkTime]
        spikesStart = sum(iSpikeTime >= firstSecond & iSpikeTime <= (firstSecond + checkTime));
        firingRatesStart(i) = spikesStart / checkTime;
        
        % end window: [lastSecond-checkTime, lastSecond]
        spikesEnd = sum(iSpikeTime >= (lastSecond - checkTime) & iSpikeTime <= lastSecond);
        firingRatesEnd(i) = spikesEnd / checkTime;
    end
    keepStart = firingRatesStart >= opts.minFiringRate & firingRatesStart <= opts.maxFiringRate;
    keepEnd = firingRatesEnd >= opts.minFiringRate & firingRatesEnd <= opts.maxFiringRate;
    keepMask = keepStart & keepEnd;
    idLabels = idLabels(keepMask);
    areaNumeric = areaNumeric(keepMask);
end

% Emit spike triplets for qualifying neurons in the collection window
spikeTimes = [];
neuronIds = [];
areas = [];
for i = 1:length(idLabels)
    iSpikeTime = data.CSV(data.CSV(:,2) == idLabels(i), 1);
    iSpikeTime = iSpikeTime(iSpikeTime >= firstSecond & iSpikeTime <= lastSecond);
    
    spikeTimes = [spikeTimes; iSpikeTime];
    neuronIds = [neuronIds; repmat(idLabels(i), length(iSpikeTime), 1)];
    areas = [areas; repmat(areaNumeric(i), length(iSpikeTime), 1)];
end

spikeData = [spikeTimes, neuronIds, areas];
end
