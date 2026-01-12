function spikeData = load_spike_times(sessionType, paths, sessionName, opts)
% LOAD_SPIKE_TIMES - Load spike times, neuron IDs, and area labels
%
% This function loads raw spike data and extracts spike times, neuron IDs,
% and area labels without creating a binned dataMat. This allows for
% on-demand binning at different bin sizes per area.
%
% Variables:
%   sessionType - Type of data: 'reach', 'spontaneous', 'schall', 'hong'
%   paths - Paths structure from get_paths
%   sessionName - Session name (format depends on sessionType)
%   opts - Options structure with firing rate filtering parameters
%
% Returns:
%   spikeData - Structure with fields:
%       .spikeTimes - Vector of all spike times (seconds)
%       .spikeClusters - Vector of neuron IDs for each spike
%       .neuronIDs - Vector of unique neuron IDs
%       .neuronAreas - Cell array of area labels for each neuron (same order as neuronIDs)
%                      Mapping: neuronAreas{i} is the area label for neuronIDs(i)
%       .idLabels - Same as neuronIDs (for compatibility)
%       .areaLabelsUnique - Cell array of unique area labels
%       .totalTime - Total recording duration (seconds)

    % Load raw data based on session type
    switch sessionType
        case 'reach'
            spikeData = load_spike_times_reach(paths, sessionName, opts);
        case 'spontaneous'
            spikeData = load_spike_times_spontaneous(paths, sessionName, opts);
        case 'schall'
            spikeData = load_spike_times_schall(paths, sessionName, opts);
        case 'hong'
            spikeData = load_spike_times_hong(paths, sessionName, opts);
        otherwise
            error('Unsupported sessionType: %s', sessionType);
    end
end

function spikeData = load_spike_times_reach(paths, sessionName, opts)
% LOAD_SPIKE_TIMES_REACH - Load spike times for reach task data
    
    % Load reach data file
    reachDataFile = fullfile(paths.reachDataPath, [sessionName, '.mat']);
    dataR = load(reachDataFile);
    
    % Set collectStart if not set
    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        opts.collectStart = 0;
    end
    
    % Set collectEnd if not set
    if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
        opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1))) / 1000);
    end
    
    % Extract spike data from CSV (CSV(:,1) is in seconds)
    spikeTimes = dataR.CSV(:,1);  % Convert from ms to seconds
    spikeClusters = dataR.CSV(:,2);
    
    % Get neuron information from idchan
    useNeurons = find(dataR.idchan(:,end) ~= 0 & ismember(dataR.idchan(:,4), [1 2]));
    neuronIDs = dataR.idchan(useNeurons, 1);
    brainAreas = dataR.idchan(useNeurons, end);
    
    % Create area labels using direct indexing (same as reach_neural_matrix.m)
    neuronAreas = cell(size(brainAreas));
    neuronAreas(brainAreas == 1) = {'M23'};
    neuronAreas(brainAreas == 2) = {'M56'};
    neuronAreas(brainAreas == 3) = {'DS'};
    neuronAreas(brainAreas == 4) = {'VS'};
    
    % Filter spikes to only include qualifying neurons and time range
    validSpikes = ismember(spikeClusters, neuronIDs) & ...
                  spikeTimes >= opts.collectStart & ...
                  spikeTimes <= opts.collectEnd;
    spikeTimes = spikeTimes(validSpikes);
    spikeClusters = spikeClusters(validSpikes);
    
    % Apply firing rate filtering if requested
    if opts.removeSome
        [spikeTimes, spikeClusters, neuronIDs, neuronAreas] = ...
            filter_by_firing_rate(spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts);
    end
    
    % Build output structure
    spikeData = struct();
    spikeData.spikeTimes = spikeTimes;
    spikeData.spikeClusters = spikeClusters;
    spikeData.neuronIDs = neuronIDs;
    spikeData.neuronAreas = neuronAreas;  % Mapping: neuronAreas{i} is the area label for neuronIDs(i)
    spikeData.idLabels = neuronIDs;  % For compatibility
    spikeData.areaLabelsUnique = unique(neuronAreas);
    spikeData.totalTime = opts.collectEnd - opts.collectStart;
    spikeData.collectStart = opts.collectStart;
    spikeData.collectEnd = opts.collectEnd;
end

function spikeData = load_spike_times_spontaneous(paths, sessionName, opts)
% LOAD_SPIKE_TIMES_SPONTANEOUS - Load spike times for spontaneous data
    
    % Set up paths
    pathParts = strsplit(sessionName, filesep);
    if ~isempty(pathParts{1}) && length(pathParts{1}) >= 2
        subDir = pathParts{1}(1:2);
    elseif length(sessionName) >= 2
        subDir = sessionName(1:2);
    else
        subDir = '';
    end
    
    if ~isempty(subDir)
        opts.dataPath = fullfile(paths.freeDataPath, subDir);
    else
        opts.dataPath = paths.freeDataPath;
    end
    opts.sessionName = sessionName;
    
    % Set collectEnd if not set
    if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
        opts.collectEnd = 10 * 60;  % Default 10 minutes
    end
    
    % Load spike data
    data = load_data(opts, 'spikes');
    
    % Find qualifying neurons
    useMulti = 1;
    if ~useMulti
        allGood = strcmp(data.ci.group, 'good');
    else
        allGood = strcmp(data.ci.group, 'good') | strcmp(data.ci.group, 'mua');
    end
    
    goodM23 = allGood & strcmp(data.ci.area, 'M23');
    goodM56 = allGood & strcmp(data.ci.area, 'M56');
    goodDS = allGood & strcmp(data.ci.area, 'DS');
    goodVS = allGood & strcmp(data.ci.area, 'VS');
    
    opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);
    
    % Get neuron IDs and areas
    if ismember('id', data.ci.Properties.VariableNames)
        neuronIDs = data.ci.id(opts.useNeurons);
    else
        neuronIDs = data.ci.cluster_id(opts.useNeurons);
    end
    neuronAreas = data.ci.area(opts.useNeurons);
    
    % Extract all spike times and clusters
    spikeTimes = data.spikeTimes;
    spikeClusters = data.spikeClusters;
    
    % Filter to qualifying neurons and time range
    validSpikes = ismember(spikeClusters, neuronIDs) & ...
                  spikeTimes >= opts.collectStart & ...
                  spikeTimes <= opts.collectEnd;
    spikeTimes = spikeTimes(validSpikes);
    spikeClusters = spikeClusters(validSpikes);
    
    % Apply firing rate filtering if requested
    if opts.removeSome
        [spikeTimes, spikeClusters, neuronIDs, neuronAreas] = ...
            filter_by_firing_rate(spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts);
    end
    
    % Build output structure
    spikeData = struct();
    spikeData.spikeTimes = spikeTimes;
    spikeData.spikeClusters = spikeClusters;
    spikeData.neuronIDs = neuronIDs;
    spikeData.neuronAreas = cell(size(neuronAreas));
    for i = 1:length(neuronAreas)
        spikeData.neuronAreas{i} = char(neuronAreas(i));
    end
    spikeData.idLabels = neuronIDs;
    spikeData.areaLabelsUnique = unique(neuronAreas);
    spikeData.totalTime = opts.collectEnd - opts.collectStart;
    spikeData.collectStart = opts.collectStart;
    spikeData.collectEnd = opts.collectEnd;
end

function spikeData = load_spike_times_schall(paths, sessionName, opts)
% LOAD_SPIKE_TIMES_SCHALL - Load spike times for Schall data
    
    % Determine subdirectory based on session name prefix
    % Extract just the filename part (in case sessionName includes subdirectory)
    [~, sessionBaseName, ~] = fileparts(sessionName);
    
    % Determine subdirectory based on prefix (case-insensitive)
    if length(sessionBaseName) >= 2 && strncmpi(sessionBaseName, 'bp', 2)
        subDir = 'broca';
    elseif length(sessionBaseName) >= 2 && strncmpi(sessionBaseName, 'jp', 2)
        subDir = 'joule';
    else
        % Default: try to extract from sessionName if it includes a path
        [parentDir, ~, ~] = fileparts(sessionName);
        if ~isempty(parentDir)
            subDir = parentDir;
        else
            % Fallback: use sessionName as-is
            subDir = '';
        end
    end
    
    % Build file path
    if ~isempty(subDir)
        schallDataFile = fullfile(paths.schallDataPath, subDir, [sessionBaseName, '.mat']);
    else
        schallDataFile = fullfile(paths.schallDataPath, [sessionBaseName, '.mat']);
    end
    
    % Load Schall data
    dataS = load(schallDataFile);
    
    % Set collectStart if not set
    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        opts.collectStart = 0;
    end
    
    
    % Extract spike data using same approach as neural_matrix_schall_fef.m
    % Get spike unit array from SessionData
    if ~isfield(dataS, 'SessionData') || ~isfield(dataS.SessionData, 'spikeUnitArray')
        error('SessionData.spikeUnitArray not found in Schall data file');
    end
    
    spikeUnitArray = dataS.SessionData.spikeUnitArray;
    nUnits = length(spikeUnitArray);
    
    % Collect all spike times and cluster IDs
    allSpikeTimes = [];
    allSpikeClusters = [];
    
    % Loop through each unit and extract spike times (matching neural_matrix_schall_fef.m)
    for i = 1:nUnits
        % Get spike times for this unit
        iSpikeTimeCell = dataS.(spikeUnitArray{i});
        
        % Convert spike times to session time (matching neural_matrix_schall_fef.m line 101)
        iSpikeTime = convert_to_session_time(iSpikeTimeCell, dataS.trialOnset) / 1000;  % Convert to seconds
        
        % Filter to collection window
        validSpikes = iSpikeTime >= opts.collectStart & iSpikeTime <= opts.collectEnd;
        iSpikeTime = iSpikeTime(validSpikes);
        
        % Append to arrays
        allSpikeTimes = [allSpikeTimes; iSpikeTime(:)];
        allSpikeClusters = [allSpikeClusters; repmat(i, length(iSpikeTime), 1)];
    end
    
    % Create neuron IDs (matching neural_matrix_schall_fef.m line 63)
    neuronIDs = 1:nUnits;
    neuronAreas = repmat({'FEF'}, nUnits, 1);  % All Schall data is FEF
    
    % Apply firing rate filtering if requested
    if opts.removeSome
        [allSpikeTimes, allSpikeClusters, neuronIDs, neuronAreas] = ...
            filter_by_firing_rate(allSpikeTimes, allSpikeClusters, neuronIDs, neuronAreas, opts);
    end
    
    % Build output structure
    spikeData = struct();
    spikeData.spikeTimes = allSpikeTimes;
    spikeData.spikeClusters = allSpikeClusters;
    spikeData.neuronIDs = neuronIDs;
    spikeData.neuronAreas = neuronAreas;
    spikeData.idLabels = neuronIDs;
    spikeData.areaLabelsUnique = {'FEF'};
    spikeData.totalTime = opts.collectEnd - opts.collectStart;
    spikeData.collectStart = opts.collectStart;
    spikeData.collectEnd = opts.collectEnd;
end

function spikeData = load_spike_times_hong(paths, sessionName, opts)
% LOAD_SPIKE_TIMES_HONG - Load spike times for Hong data
%   Note: sessionName is not used for Hong data (loads from fixed file locations)
    
    % Load Hong data files (same structure as load_hong_data)
    load(fullfile(paths.dropPath, 'hong/data', 'spikeData.mat'));
    load(fullfile(paths.dropPath, 'hong/data', 'T_allUnits2.mat'));
    
    % Set collectStart if not set
    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        opts.collectStart = 0;
    end
    
    % Set collectEnd if not set (same logic as load_hong_data, but as duration)
    if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
        load(fullfile(paths.dropPath, 'hong/data', 'behaviorTable.mat'));
        % Calculate absolute end time, then convert to duration
        absoluteEndTime = min(T.startTime_oe(end)+max(diff(T.startTime_oe)), max(sp.st));
        opts.collectEnd = absoluteEndTime - opts.collectStart;
    end
    
    % Extract spike data from sp structure (matching neural_matrix_hong.m)
    spikeTimes = sp.st;  % Spike times in seconds
    spikeClusters = sp.clu;  % Cluster IDs
    spikeDepths = sp.spikeDepths;  % Spike depths for area determination
    
    % Determine collection time window (matching neural_matrix_hong.m)
    firstSecond = opts.collectStart;
    if isempty(opts.collectEnd)
        lastSecond = double(max(spikeTimes));
    else
        lastSecond = firstSecond + opts.collectEnd;
    end
    
    % Filter spikes to collection window and valid clusters (matching neural_matrix_hong.m)
    clusterIncludeIdx = ismember(spikeClusters, sp.cids);
    spikeMask = spikeTimes >= firstSecond & spikeTimes < lastSecond & clusterIncludeIdx;
    spikeTimes = spikeTimes(spikeMask);
    spikeClusters = spikeClusters(spikeMask);
    spikeDepthsFiltered = spikeDepths(spikeMask);
    
    % Get unique cluster IDs that have spikes in the collection window
    uniqueClusters = unique(spikeClusters);
    nClusters = length(uniqueClusters);
    
    % Determine area for each cluster based on mean spike depth (matching neural_matrix_hong.m)
    % S1: depth >= 2000, SC: depth < 2000
    neuronIDs = uniqueClusters;
    neuronAreas = cell(nClusters, 1);
    for i = 1:nClusters
        clusterId = uniqueClusters(i);
        clusterDepthMask = (spikeClusters == clusterId);
        meanDepth = mean(spikeDepthsFiltered(clusterDepthMask));
        
        if meanDepth >= 2000
            neuronAreas{i} = 'S1';
        else
            neuronAreas{i} = 'SC';
        end
    end
    
    % Apply firing rate filtering if requested
    if opts.removeSome
        [spikeTimes, spikeClusters, neuronIDs, neuronAreas] = ...
            filter_by_firing_rate(spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts);
    end
    
    % Build output structure
    % Note: collectEnd is stored as absolute time for consistency with other session types
    absoluteCollectEnd = opts.collectStart + opts.collectEnd;
    
    spikeData = struct();
    spikeData.spikeTimes = spikeTimes;
    spikeData.spikeClusters = spikeClusters;
    spikeData.neuronIDs = neuronIDs;
    spikeData.neuronAreas = neuronAreas;
    spikeData.idLabels = neuronIDs;
    spikeData.areaLabelsUnique = unique(neuronAreas);
    spikeData.totalTime = opts.collectEnd;  % Duration
    spikeData.collectStart = opts.collectStart;
    spikeData.collectEnd = absoluteCollectEnd;  % Absolute time for consistency
end

function [spikeTimes, spikeClusters, neuronIDs, neuronAreas] = ...
    filter_by_firing_rate(spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts)
% FILTER_BY_FIRING_RATE - Filter neurons based on firing rate criteria
    
    checkTime = opts.firingRateCheckTime;
    collectTime = opts.collectEnd;
    
    % Calculate firing rates for each neuron
    firingRatesStart = zeros(length(neuronIDs), 1);
    firingRatesEnd = zeros(length(neuronIDs), 1);
    
    for i = 1:length(neuronIDs)
        neuronID = neuronIDs(i);
        neuronSpikes = spikeTimes(spikeClusters == neuronID);
        
        % Count spikes in first checkTime period
        spikesStart = sum(neuronSpikes >= opts.collectStart & ...
                          neuronSpikes <= opts.collectStart + checkTime);
        firingRatesStart(i) = spikesStart / checkTime;
        
        % Count spikes in last checkTime period
        spikesEnd = sum(neuronSpikes >= (collectTime - checkTime) & ...
                       neuronSpikes <= collectTime);
        firingRatesEnd(i) = spikesEnd / checkTime;
    end
    
    % Apply firing rate filtering
    keepStart = firingRatesStart >= opts.minFiringRate & ...
                firingRatesStart <= opts.maxFiringRate;
    keepEnd = firingRatesEnd >= opts.minFiringRate & ...
              firingRatesEnd <= opts.maxFiringRate;
    keepNeurons = keepStart & keepEnd;
    
    fprintf('\nKeeping %d of %d neurons after firing rate filtering\n', ...
        sum(keepNeurons), length(keepNeurons));
    
    % Update neuron lists
    neuronIDs = neuronIDs(keepNeurons);
    neuronAreas = neuronAreas(keepNeurons);
    
    % Filter spikes to only include kept neurons
    validSpikes = ismember(spikeClusters, neuronIDs);
    spikeTimes = spikeTimes(validSpikes);
    spikeClusters = spikeClusters(validSpikes);
end
