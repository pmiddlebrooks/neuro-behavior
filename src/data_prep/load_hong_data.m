function dataStruct = load_hong_data(dataStruct, dataSource, paths, opts, lfpCleanParams, bands)
% LOAD_HONG_DATA Load Hong whisker/lick data for sliding window analysis
%
% Variables:
%   dataStruct - Data structure to populate
%   dataSource - 'spikes' or 'lfp'
%   paths - Paths structure from get_paths
%   opts - Options structure
%   lfpCleanParams - LFP cleaning parameters (if dataSource == 'lfp')
%   bands - Frequency bands (if dataSource == 'lfp')
%
% Goal:
%   Load Hong whisker/lick spike data and populate dataStruct.
%
% Returns:
%   dataStruct - Updated data structure

    % Check if we should use spike times approach (new) or dataMat (old)
    % Default to spike times if not specified
    if ~isfield(opts, 'useSpikeTimes') || isempty(opts.useSpikeTimes)
        opts.useSpikeTimes = true;  % Default to new approach
    end
    
    if opts.useSpikeTimes
        % Load spike times using new approach
        % Note: sessionName may be empty for hong data, but load_spike_times_hong doesn't use it
        spikeData = load_spike_times('hong', paths, '', opts);
        
        % Update opts with values set by load_spike_times_hong (e.g., collectEnd)
        opts.collectEnd = spikeData.collectEnd;
        opts.collectStart = spikeData.collectStart;
        dataStruct.opts = opts;
        
        % Extract area information
        dataStruct.areas = {'S1', 'SC'};
        idS1 = [];
        idSC = [];
        
        % Find neuron indices for each area
        for i = 1:length(spikeData.neuronIDs)
            areaName = spikeData.neuronAreas{i};
            switch areaName
                case 'S1'
                    idS1 = [idS1, i];
                case 'SC'
                    idSC = [idSC, i];
            end
        end
        
        dataStruct.idMatIdx = {idS1, idSC};
        dataStruct.idLabel = {spikeData.neuronIDs(idS1), spikeData.neuronIDs(idSC)};
        
        % Store spike times for on-demand binning
        dataStruct.spikeTimes = spikeData.spikeTimes;
        dataStruct.spikeClusters = spikeData.spikeClusters;
        dataStruct.spikeData = spikeData;  % Store full structure for reference
        dataStruct.dataMat = [];  % Not used in new approach
        
        % Load behavior table for compatibility
        load(fullfile(paths.dropPath, 'hong/data', 'behaviorTable.mat'));
        dataStruct.T = T;  % Store behavior table
        
        fprintf('%d S1\n', length(idS1));
        fprintf('%d SC\n', length(idSC));
    else
        % Load using old approach (dataMat)
        load(fullfile(paths.dropPath, 'hong/data', 'spikeData.mat'));
        load(fullfile(paths.dropPath, 'hong/data', 'T_allUnits2.mat'));
        load(fullfile(paths.dropPath, 'hong/data', 'behaviorTable.mat'));
        
        % Set collectEnd
        opts.collectEnd = min(T.startTime_oe(end)+max(diff(T.startTime_oe)), max(sp.st));
        dataStruct.opts = opts;
        
        data.sp = sp;
        data.ci = T_allUnits;
        
        [dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_hong(data, opts);
        
        dataStruct.areas = {'S1', 'SC'};
        idS1 = find(strcmp(areaLabels, 'S1'));
        idSC = find(strcmp(areaLabels, 'SC'));
        dataStruct.idMatIdx = {idS1, idSC};
        dataStruct.idLabel = {idLabels(idS1), idLabels(idSC)};
        dataStruct.dataMat = dataMat;
        dataStruct.spikeData = [];  % Can be loaded later if needed
        dataStruct.spikeTimes = [];
        dataStruct.spikeClusters = [];
        dataStruct.T = T;  % Store behavior table
        
        fprintf('%d S1\n', length(idS1));
        fprintf('%d SC\n', length(idSC));
    end
    
    dataStruct.saveDir = fullfile(paths.dropPath, 'hong/results');
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end
    
    dataStruct.areasToTest = 1:2;
    
    % Initialize empty variables for compatibility
    dataStruct.dataR = [];
    dataStruct.startBlock2 = [];
    dataStruct.reachStart = [];
    dataStruct.reachClass = [];
    dataStruct.sessionName = '';
    
    if strcmp(dataSource, 'lfp')
        error('LFP loading for hong data is not yet implemented');
    end
end
