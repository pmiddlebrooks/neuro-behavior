function [spikeData] = spike_times_per_area_hong(opts)
% SPIKE_TIMES_PER_AREA_HONG - Extract spike times, neuron IDs, and area labels for Hong dataset
%
% Loads Hong whisker detection task spike data and returns spike triplets
% [time, neuronId, areaNumeric] for neurons that pass firing-rate criteria within
% a collection window.
%
% INPUTS:
%   opts - Structure with fields:
%       .collectStart         - Start time (s) of collection window (default 0)
%       .collectEnd           - Number of seconds to collect (required)
%       .removeSome           - Whether to filter by firing rate (logical)
%       .firingRateCheckTime  - Seconds for start/end firing rate checks
%       .minFiringRate        - Minimum allowed firing rate (Hz)
%       .maxFiringRate         - Maximum allowed firing rate (Hz)
%
%   Note: This function expects the following variables to be loaded in the workspace:
%       T - Behavior table with startTime_oe (trial start times in seconds)
%       spikeDepths - Vector of spike depths for each spike
%       sc - Spike cluster IDs (from spike_clusters.npy)
%       st - Spike times in sample units (from spike_times.npy, divide by 30000 for seconds)
%       cg - Cluster group table with cluster_id and group columns
%
% OUTPUTS:
%   spikeData - N x 3 matrix: [spikeTimeSec, neuronId, areaNumeric]
%       areaNumeric: 1=S1, 2=SC

error('Update this to match neural_matrix_hong.m')
% Check if required variables exist in workspace
if ~exist('T', 'var') || ~exist('spikeDepths', 'var') || ~exist('sc', 'var') || ~exist('st', 'var') || ~exist('cg', 'var')
    error('Required variables (T, spikeDepths, sc, st, cg) not found in workspace. Run load_data_hong.m first.');
end

% Sampling frequency (Hz)
fsSpike = 30000;

% Determine collection window
if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
    firstSecond = 0;
else
    firstSecond = opts.collectStart;
end

if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
    error('opts.collectEnd must be provided.');
end

lastSecond = firstSecond + opts.collectEnd;

% Convert spike times from sample units to seconds
spikeTimesSec = double(st) / fsSpike;

% Filter spikes to collection window
spikeMask = spikeTimesSec >= firstSecond & spikeTimesSec < lastSecond;
spikeTimesSec = spikeTimesSec(spikeMask);
spikeClusters = sc(spikeMask);
spikeDepthsFiltered = spikeDepths(spikeMask);

% Get unique clusters that are 'good' or 'mua'
validGroups = {'good', 'mua'};
validClusterMask = ismember(cg.group, validGroups);
validClusterIds = cg.cluster_id(validClusterMask);

% Filter to only include spikes from valid clusters
clusterMask = ismember(spikeClusters, validClusterIds);
spikeTimesSec = spikeTimesSec(clusterMask);
spikeClusters = spikeClusters(clusterMask);
spikeDepthsFiltered = spikeDepthsFiltered(clusterMask);

% Get unique cluster IDs that have spikes in the collection window
uniqueClusters = unique(spikeClusters);
nClusters = length(uniqueClusters);

% Determine area for each cluster based on mean spike depth
% S1: depth >= 2000, SC: depth < 2000
% Map to numeric: 1=S1, 2=SC
clusterAreaNumeric = zeros(nClusters, 1);
for i = 1:nClusters
    clusterId = uniqueClusters(i);
    clusterDepthMask = (spikeClusters == clusterId);
    meanDepth = mean(spikeDepthsFiltered(clusterDepthMask));
    
    if meanDepth >= 2000
        clusterAreaNumeric(i) = 1; % S1
    else
        clusterAreaNumeric(i) = 2; % SC
    end
end

% Optional firing-rate filtering
if isfield(opts, 'removeSome') && opts.removeSome
    checkTime = opts.firingRateCheckTime;
    firingRatesStart = zeros(nClusters, 1);
    firingRatesEnd = zeros(nClusters, 1);
    
    for i = 1:nClusters
        clusterId = uniqueClusters(i);
        iSpikeTime = spikeTimesSec(spikeClusters == clusterId);
        
        % Restrict to collection window (already filtered, but ensure)
        iSpikeTime = iSpikeTime(iSpikeTime >= firstSecond & iSpikeTime <= lastSecond);
        
        % Start window: [firstSecond, firstSecond + checkTime]
        spikesStart = sum(iSpikeTime >= firstSecond & iSpikeTime <= (firstSecond + checkTime));
        firingRatesStart(i) = spikesStart / checkTime;
        
        % End window: [lastSecond - checkTime, lastSecond]
        spikesEnd = sum(iSpikeTime >= (lastSecond - checkTime) & iSpikeTime <= lastSecond);
        firingRatesEnd(i) = spikesEnd / checkTime;
    end
    
    keepStart = firingRatesStart >= opts.minFiringRate & firingRatesStart <= opts.maxFiringRate;
    keepEnd = firingRatesEnd >= opts.minFiringRate & firingRatesEnd <= opts.maxFiringRate;
    keepMask = keepStart & keepEnd;
    
    fprintf('\nkeeping %d of %d neurons\n', sum(keepMask), length(keepMask));
    
    % Update cluster lists to only include qualifying neurons
    uniqueClusters = uniqueClusters(keepMask);
    clusterAreaNumeric = clusterAreaNumeric(keepMask);
end

% Emit spike triplets for qualifying neurons in the collection window
spikeTimes = [];
neuronIds = [];
areas = [];

for i = 1:length(uniqueClusters)
    clusterId = uniqueClusters(i);
    iSpikeTime = spikeTimesSec(spikeClusters == clusterId);
    
    % Ensure spikes are within collection window
    iSpikeTime = iSpikeTime(iSpikeTime >= firstSecond & iSpikeTime <= lastSecond);
    
    % Add spike data for this cluster
    spikeTimes = [spikeTimes; iSpikeTime];
    neuronIds = [neuronIds; repmat(clusterId, length(iSpikeTime), 1)];
    areas = [areas; repmat(clusterAreaNumeric(i), length(iSpikeTime), 1)];
end

spikeData = [spikeTimes, neuronIds, areas];

end

