function dataStruct = load_schall_data(dataStruct, dataSource, paths, sessionName, opts, lfpCleanParams, bands)
% LOAD_SCHALL_DATA Load Schall choice countermanding data for sliding window analysis
%
% Variables:
%   dataStruct - Data structure to populate
%   dataSource - 'spikes' or 'lfp'
%   paths - Paths structure from get_paths
%   sessionName - Session name without .mat extension (e.g., 'joule/jp121n02')
%   opts - Options structure
%   lfpCleanParams - LFP cleaning parameters (if dataSource == 'lfp')
%   bands - Frequency bands (if dataSource == 'lfp')
%
% Goal:
%   Load Schall choice countermanding spike or LFP data and populate dataStruct.
%
% Returns:
%   dataStruct - Updated data structure

    
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
    
    [~, dataBaseName, ~] = fileparts(sessionName);
    dataStruct.dataBaseName = dataBaseName;
    dataStruct.saveDir = fullfile(paths.dropPath, 'schall/results', dataBaseName);
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end
    
    dataS = load(schallDataFile);
    dataStruct.dataS = dataS;

        % Set collectEnd if not set (matching neural_matrix_schall_fef.m logic)
    if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
        opts.collectEnd = ceil(max(dataS.trialOnset(end) + dataS.trialDuration(end))/1000);
    end

    % Process response onsets
    responseOnset = convert_to_session_time(dataS.responseOnset, dataS.trialOnset);
    responseOnset = responseOnset / 1000;
    responseOnset(responseOnset < opts.collectStart) = [];
    responseOnset(responseOnset > opts.collectEnd) = [];
    dataStruct.responseOnset = responseOnset;
    
    if strcmp(dataSource, 'spikes')
        % Check if we should use spike times approach (new) or dataMat (old)
        % Default to spike times if not specified
        if ~isfield(opts, 'useSpikeTimes') || isempty(opts.useSpikeTimes)
            opts.useSpikeTimes = true;  % Default to new approach
        end
        
        if opts.useSpikeTimes
            % Load spike times using new approach
            spikeData = load_spike_times('schall', paths, sessionName, opts);
            
            % % Update opts with values set by load_spike_times_schall (e.g., collectEnd)
            % opts.collectEnd = spikeData.collectEnd;
            % opts.collectStart = spikeData.collectStart;
            % 
            % Extract area information
            dataStruct.areas = {'FEF'};
            idFEF = [];
            
            % Find neuron indices for FEF area
            for i = 1:length(spikeData.neuronIDs)
                areaName = spikeData.neuronAreas{i};
                if strcmp(areaName, 'FEF')
                    idFEF = [idFEF, i];
                end
            end
            
            dataStruct.idMatIdx = {idFEF};
            dataStruct.idLabel = {spikeData.neuronIDs(idFEF)};
            
            % Store spike times for on-demand binning
            dataStruct.spikeTimes = spikeData.spikeTimes;
            dataStruct.spikeClusters = spikeData.spikeClusters;
            dataStruct.spikeData = spikeData;  % Store full structure for reference
            dataStruct.dataMat = [];  % Not used in new approach
            
            fprintf('%d FEF\n', length(idFEF));
        else
            % Load using old approach (dataMat)
            [dataMat, idLabels, areaLabels] = neural_matrix_schall_fef(dataS, opts);
            dataStruct.areas = {'FEF'};
            idFEF = 1:length(idLabels);
            dataStruct.idMatIdx = {idFEF};
            dataStruct.idLabel = {idLabels(idFEF)};
            dataStruct.dataMat = dataMat;
            dataStruct.spikeData = [];  % Can be loaded later if needed
            dataStruct.spikeTimes = [];
            dataStruct.spikeClusters = [];
            
            fprintf('%d FEF\n', length(idFEF));
        end
        
    elseif strcmp(dataSource, 'lfp')
        % Load schall LFP data
        if ~isfield(opts, 'fsLfp')
            opts.fsLfp = 1000;  % Default fallback
        end
        
        % Note: LFP loading for schall may need session-specific handling
        % This is a placeholder - actual implementation may vary
        error('LFP loading for schall data needs to be implemented based on your specific file structure');
    end
    
    dataStruct.opts = opts;
    dataStruct.areasToTest = 1;
end
