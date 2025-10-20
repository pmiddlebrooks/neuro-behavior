function [spikeData] = spike_times_per_area(opts)
% SPIKE_TIMES_PER_AREA - Extracts spike times, neuron IDs, and area labels for qualifying neurons
%
% This function loads spike data from the specified path and extracts individual 
% spike times along with their corresponding neuron IDs and brain area labels. 
% It applies firing rate filtering criteria to ensure only qualifying neurons 
% are included.
%
% INPUTS:
%   opts - Structure containing the following options:
%       .dataPath        - Path to the data file
%       .useNeurons      - List of neuron IDs to include
%       .removeSome      - Flag to filter out neurons based on firing rate
%       .minFiringRate   - Minimum firing rate threshold (for removal if enabled)
%       .maxFiringRate   - Maximum firing rate threshold (for removal if enabled)
%       .firingRateCheckTime - Time period to check firing rates (for filtering)
%       .collectFor      - Number of seconds of data to collect
%
% OUTPUTS:
%   spikeData - Matrix with columns:
%       Column 1: Spike times (timestamps)
%       Column 2: Neuron IDs corresponding to each spike
%       Column 3: Brain area labels (numeric: 1=M23, 2=M56, 3=DS, 4=VS)

% Load data
data = load_data(opts, 'spikes');

% Define brain area mapping: 1=M23, 2=M56, 3=DS, 4=VS
areaMapping = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});

% Define neuron IDs
if ismember('id', data.ci.Properties.VariableNames)
    idLabels = data.ci.id(opts.useNeurons);
else
    idLabels = data.ci.cluster_id(opts.useNeurons);
end

% Initialize area labels
areaLabels = {};

% First, determine which neurons pass the firing rate criteria
rmvNeurons = [];
if opts.removeSome
    checkTime = opts.firingRateCheckTime;
    collectTime = opts.collectFor;
    
    % Calculate firing rates for each neuron directly from spike counts
    firingRatesStart = zeros(length(idLabels), 1);
    firingRatesEnd = zeros(length(idLabels), 1);
    
    for i = 1:length(idLabels)
        iSpikeTime = data.spikeTimes(data.spikeClusters == idLabels(i));
        
        % Filter spikes to only include those within collectTime
        validSpikes = iSpikeTime >= 0 & iSpikeTime <= collectTime;
        iSpikeTime = iSpikeTime(validSpikes);
        
        % Count spikes in first checkTime period
        spikesStart = sum(iSpikeTime >= 0 & iSpikeTime <= checkTime);
        firingRatesStart(i) = spikesStart / checkTime;
        
        % Count spikes in last checkTime period of the collected data
        spikesEnd = sum(iSpikeTime >= (collectTime - checkTime) & iSpikeTime <= collectTime);
        firingRatesEnd(i) = spikesEnd / checkTime;
    end
    
    % Apply firing rate filtering
    keepStart = firingRatesStart >= opts.minFiringRate & firingRatesStart <= opts.maxFiringRate;
    keepEnd = firingRatesEnd >= opts.minFiringRate & firingRatesEnd <= opts.maxFiringRate;
    rmvNeurons = ~(keepStart & keepEnd);
    fprintf('\nkeeping %d of %d neurons\n', sum(~rmvNeurons), length(rmvNeurons));
    
    % Update neuron lists to only include qualifying neurons
    idLabels(rmvNeurons) = [];
end

% Extract spike times for qualifying neurons
spikeTimes = [];
neuronIds = [];
areaLabels = [];

for i = 1:length(idLabels)
    % Get spike times for current neuron
    iSpikeTime = data.spikeTimes(data.spikeClusters == idLabels(i));
    
    % Filter spikes to only include those within collectTime
    collectTime = opts.collectFor;
    validSpikes = iSpikeTime >= 0 & iSpikeTime <= collectTime;
    iSpikeTime = iSpikeTime(validSpikes);
    
    % Get area label for current neuron
    neuronIdx = find(opts.useNeurons == idLabels(i));
    if ~isempty(neuronIdx)
        currentAreaText = data.ci.area(opts.useNeurons(neuronIdx));
        
        % Convert text area label to numeric value
        if isKey(areaMapping, currentAreaText)
            currentAreaLabel = areaMapping(currentAreaText);
        else
            warning('Unknown brain area: %s. Using default value 1 (M23)', currentAreaText);
            currentAreaLabel = 1;
        end
        
        % Add spike data for this neuron
        spikeTimes = [spikeTimes; iSpikeTime];
        neuronIds = [neuronIds; repmat(idLabels(i), length(iSpikeTime), 1)];
        areaLabels = [areaLabels; repmat(currentAreaLabel, length(iSpikeTime), 1)];
    end
end

% Combine all data into output matrix
spikeData = [spikeTimes, neuronIds, areaLabels];

end

