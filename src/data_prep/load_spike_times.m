function spikeData = load_spike_times(sessionType, paths, sessionName, opts)
% LOAD_SPIKE_TIMES - Load spike times, neuron IDs, and area labels
%
% This function loads raw spike data and extracts spike times, neuron IDs,
% and area labels without creating a binned dataMat. This allows for
% on-demand binning at different bin sizes per area.
%
% Variables:
%   sessionType - Type of data: 'reach', 'spontaneous', 'interval', 'semicircle', 'schall', 'hong'
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
        case 'interval'
            spikeData = load_spike_times_interval(paths, sessionName, opts);
        case 'semicircle'
            spikeData = load_spike_times_semicircle(paths, sessionName, opts);
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

    % Session end from reach events / last spike; [] omits final 180 s
    sessionEnd = round(min((dataR.R(end,1) + 5000)/1000, max(dataR.CSV(:,1))));
    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = [];
    end
    opts.collectEnd = resolve_reach_collect_end(opts.collectEnd, sessionEnd, opts.collectStart);
    
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

    if ~isfield(opts, 'subjectName') || isempty(opts.subjectName)
        error('opts.subjectName must be set before loading spontaneous spike times');
    end

    opts.dataPath = fullfile(paths.spontaneousDataPath, opts.subjectName);
    opts.sessionName = sessionName;
    
    
    % Load spike data
    data = load_data(opts, 'spikes');

    % Quality: Phy group when curated; otherwise rf_label == 'real'
    allGood = cluster_quality_mask(data.ci, opts);

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

    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        opts.collectStart = 0;
    end
    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = [];
    end
    opts.collectEnd = clamp_collect_end_to_session(opts.collectEnd, max(spikeTimes), opts.collectStart);
    
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

function spikeData = load_spike_times_interval(paths, sessionName, opts)
% LOAD_SPIKE_TIMES_INTERVAL - Load spike times for interval timing task data

    if ~isfield(opts, 'subjectName') || isempty(opts.subjectName)
        error('opts.subjectName must be set before loading interval spike times');
    end

    opts.dataPath = fullfile(paths.intervalDataPath, opts.subjectName);
    opts.sessionName = sessionName;

    data = load_data(opts, 'spikes');

    % Quality: Phy group when curated; otherwise rf_label == 'real'
    allGood = cluster_quality_mask(data.ci, opts);

    goodM23 = allGood & strcmp(data.ci.area, 'M23');
    goodM56 = allGood & strcmp(data.ci.area, 'M56');
    goodDS = allGood & strcmp(data.ci.area, 'DS');
    goodVS = allGood & strcmp(data.ci.area, 'VS');

    opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);

    if ismember('id', data.ci.Properties.VariableNames)
        neuronIDs = data.ci.id(opts.useNeurons);
    else
        neuronIDs = data.ci.cluster_id(opts.useNeurons);
    end
    neuronAreas = data.ci.area(opts.useNeurons);

    spikeTimes = data.spikeTimes;
    spikeClusters = data.spikeClusters;

    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        opts.collectStart = 0;
    end
    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = [];
    end
    opts.collectEnd = clamp_collect_end_to_session(opts.collectEnd, max(spikeTimes), opts.collectStart);

    validSpikes = ismember(spikeClusters, neuronIDs) & ...
        spikeTimes >= opts.collectStart & ...
        spikeTimes <= opts.collectEnd;
    spikeTimes = spikeTimes(validSpikes);
    spikeClusters = spikeClusters(validSpikes);

    if opts.removeSome
        [spikeTimes, spikeClusters, neuronIDs, neuronAreas] = ...
            filter_by_firing_rate(spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts);
    end

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

function spikeData = load_spike_times_semicircle(paths, sessionName, opts)
% LOAD_SPIKE_TIMES_SEMICIRCLE - Load spike times for semicircle reward task
%
% Variables:
%   paths       - Paths structure from get_paths
%   sessionName - Session .mat basename (e.g. 'AS1_0618_WellLearned')
%   opts        - Options; requires opts.subjectName (e.g. 'AS1')
%
% Goal:
%   Load CSV spike times and IdChan metadata; map LayerID via layerIDs to
%   M23/M56/DS/VS. Keep MUA and good units (unitType 1 and 2).

    if ~isfield(opts, 'subjectName') || isempty(opts.subjectName)
        error('opts.subjectName must be set before loading semicircle spike times');
    end

    dataFile = fullfile(paths.semicircleDataPath, opts.subjectName, [sessionName, '.mat']);
    if ~exist(dataFile, 'file')
        error('Semicircle data file not found: %s', dataFile);
    end
    dataS = load(dataFile);

    idChan = get_semicircle_idchan(dataS);
    layerIds = get_semicircle_layer_ids(dataS);
    areaByLayer = map_semicircle_layer_to_area(layerIds);

    % LayerID col 7; unitType col 4 (1=MUA, 2=Good); exclude LayerID==0
    layerIdCol = idChan(:, 7);
    unitTypeCol = idChan(:, 4);
    validLayerIds = cell2mat(keys(areaByLayer));
    useNeurons = find(layerIdCol ~= 0 & ismember(unitTypeCol, [1 2]) & ...
        ismember(layerIdCol, validLayerIds));
    neuronIDs = idChan(useNeurons, 1);
    neuronAreas = cell(numel(useNeurons), 1);
    for i = 1:numel(useNeurons)
        neuronAreas{i} = areaByLayer(layerIdCol(useNeurons(i)));
    end

    spikeTimes = dataS.CSV(:, 1);
    spikeClusters = dataS.CSV(:, 2);

    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        opts.collectStart = 0;
    end
    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = [];
    end
    sessionEnd = max(spikeTimes);
    if isfield(dataS, 'TaskMatrix') && ~isempty(dataS.TaskMatrix)
        sessionEnd = max(sessionEnd, max(dataS.TaskMatrix(:, 8)));
    end
    opts.collectEnd = clamp_collect_end_to_session(opts.collectEnd, sessionEnd, opts.collectStart);

    validSpikes = ismember(spikeClusters, neuronIDs) & ...
        spikeTimes >= opts.collectStart & ...
        spikeTimes <= opts.collectEnd;
    spikeTimes = spikeTimes(validSpikes);
    spikeClusters = spikeClusters(validSpikes);

    if opts.removeSome
        [spikeTimes, spikeClusters, neuronIDs, neuronAreas] = ...
            filter_by_firing_rate(spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts);
    end

    spikeData = struct();
    spikeData.spikeTimes = spikeTimes;
    spikeData.spikeClusters = spikeClusters;
    spikeData.neuronIDs = neuronIDs;
    spikeData.neuronAreas = neuronAreas;
    spikeData.idLabels = neuronIDs;
    spikeData.areaLabelsUnique = unique(neuronAreas);
    spikeData.totalTime = opts.collectEnd - opts.collectStart;
    spikeData.collectStart = opts.collectStart;
    spikeData.collectEnd = opts.collectEnd;
end

function idChan = get_semicircle_idchan(dataS)
% GET_SEMICIRCLE_IDCHAN - Resolve IdChan / idchan field name

    if isfield(dataS, 'IdChan')
        idChan = dataS.IdChan;
    elseif isfield(dataS, 'idchan')
        idChan = dataS.idchan;
    else
        error('Semicircle file missing IdChan / idchan');
    end
end

function layerIds = get_semicircle_layer_ids(dataS)
% GET_SEMICIRCLE_LAYER_IDS - Cell of area names indexed by LayerID

    if ~isfield(dataS, 'layerIDs') || isempty(dataS.layerIDs)
        error('Semicircle file missing layerIDs');
    end
    layerIds = dataS.layerIDs;
    if ~iscell(layerIds)
        error('layerIDs must be a cell array of area name strings');
    end
end

function areaByLayer = map_semicircle_layer_to_area(layerIds)
% MAP_SEMICIRCLE_LAYER_TO_AREA - Map LayerID index -> pipeline area label
%
% Variables:
%   layerIds - Cell of strings from data file (e.g. {'M1 L23','M1 L5','DMS','VS'})
%
% Goal:
%   Build containers.Map from LayerID (1..N) to M23/M56/DS/VS used elsewhere.

    areaByLayer = containers.Map('KeyType', 'double', 'ValueType', 'char');
    for iLayer = 1:numel(layerIds)
        rawName = lower(strtrim(char(layerIds{iLayer})));
        if contains(rawName, 'l23') || contains(rawName, 'm23')
            areaName = 'M23';
        elseif contains(rawName, 'l5') || contains(rawName, 'm56')
            areaName = 'M56';
        elseif contains(rawName, 'dms') || strcmp(rawName, 'ds')
            areaName = 'DS';
        elseif contains(rawName, 'vs')
            areaName = 'VS';
        else
            warning('load_spike_times_semicircle:UnknownLayer', ...
                'Unrecognized layerIDs{%d}=''%s''; skipping.', iLayer, layerIds{iLayer});
            continue;
        end
        areaByLayer(iLayer) = areaName;
    end
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
    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = [];
    end

    % Extract spike data using same approach as neural_matrix_schall_fef.m
    % Get spike unit array from SessionData
    if ~isfield(dataS, 'SessionData') || ~isfield(dataS.SessionData, 'spikeUnitArray')
        error('SessionData.spikeUnitArray not found in Schall data file');
    end
    
    spikeUnitArray = dataS.SessionData.spikeUnitArray;
    nUnits = length(spikeUnitArray);
    
    % Collect all spike times and cluster IDs (full session first, then clamp collectEnd)
    allSpikeTimes = [];
    allSpikeClusters = [];
    
    % Loop through each unit and extract spike times (matching neural_matrix_schall_fef.m)
    for i = 1:nUnits
        % Get spike times for this unit
        iSpikeTimeCell = dataS.(spikeUnitArray{i});
        
        % Convert spike times to session time (matching neural_matrix_schall_fef.m line 101)
        iSpikeTime = convert_to_session_time(iSpikeTimeCell, dataS.trialOnset) / 1000;  % Convert to seconds
        
        % Append to arrays
        allSpikeTimes = [allSpikeTimes; iSpikeTime(:)];
        allSpikeClusters = [allSpikeClusters; repmat(i, length(iSpikeTime), 1)];
    end

    if isempty(allSpikeTimes)
        sessionEnd = opts.collectStart;
    else
        sessionEnd = max(allSpikeTimes);
    end
    if isfield(dataS, 'trialOnset') && isfield(dataS, 'trialDuration') ...
        && ~isempty(dataS.trialOnset) && ~isempty(dataS.trialDuration)
        sessionEnd = max(sessionEnd, ceil(max(dataS.trialOnset(end) + dataS.trialDuration(end)) / 1000));
    end
    opts.collectEnd = clamp_collect_end_to_session(opts.collectEnd, sessionEnd, opts.collectStart);

    validSpikes = allSpikeTimes >= opts.collectStart & allSpikeTimes <= opts.collectEnd;
    allSpikeTimes = allSpikeTimes(validSpikes);
    allSpikeClusters = allSpikeClusters(validSpikes);
    
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

    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        timeStart = 0;
    else
        timeStart = opts.collectStart;
    end
    timeEnd = opts.collectEnd;

    keepNeurons = neuron_firing_rate_filter_spikes(opts, neuronIDs, spikeTimes, ...
        spikeClusters, timeStart, timeEnd);

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
