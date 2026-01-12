function dataStruct = load_spontaneous_data(dataStruct, dataSource, paths, opts, sessionName, lfpCleanParams, bands)
% load_spontaneous_data - Load spontaneous data for sliding window analysis
%
% This function incorporates the functionality of get_standard_data.m
% directly, removing the dependency on workspace variables.
%
% Input:
%   dataStruct - Data structure to populate
%   dataSource - 'spikes' or 'lfp'
%   paths - Paths structure from get_paths
%   opts - Options structure
%   sessionName - Session name (e.g., 'ag/ag112321/recording1')
%   lfpCleanParams - LFP cleaning parameters (if dataSource == 'lfp')
%   bands - Frequency bands (if dataSource == 'lfp')
%
% Output:
%   dataStruct - Updated data structure with all loaded data

    % Set default collectEnd if not set
    if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
        opts.collectEnd = 10 * 60;  % Default 10 minutes
    end
    
    % Set up paths for spontaneous data (incorporating get_standard_data logic)
    % Determine subdirectory based on first two characters of sessionName
    % Extract first path component or first two characters of sessionName
    pathParts = strsplit(sessionName, filesep);
    if ~isempty(pathParts{1}) && length(pathParts{1}) >= 2
        % Use first two characters of first path component
        subDir = pathParts{1}(1:2);
    elseif length(sessionName) >= 2
        % Fallback: use first two characters of sessionName itself
        subDir = sessionName(1:2);
    else
        % Use sessionName as-is if too short
        subDir = '';
    end
    
    % Build data path with subdirectory
    if ~isempty(subDir)
        opts.dataPath = fullfile(paths.freeDataPath, subDir);
    else
        opts.dataPath = paths.freeDataPath;
    end
    opts.sessionName = sessionName;
    
    if strcmp(dataSource, 'spikes')
        % Load spontaneous spike data
        % First load behavior data (needed for neural_matrix)
        dataBhv = load_data(opts, 'behavior');
        
        % Create bhvID vector using behavior sampling frequency (opts.fsBhv)
        % This creates bhvID at behavior sampling rate, not neural data frame rate
        if ~isfield(opts, 'fsBhv')
            error('behavior needs a sampling frequency, opts.fsBhv, in the opts struct')
        end
        bhvBinSize = 1 / opts.fsBhv;  % Behavior bin size in seconds
        nBhvBins = ceil(opts.collectEnd / bhvBinSize);
        dataBhv.StartFrame = 1 + round(dataBhv.StartTime / bhvBinSize);
        bhvID = zeros(nBhvBins, 1);
        for i = 1:size(dataBhv, 1) - 1
            iInd = dataBhv.StartFrame(i) : dataBhv.StartFrame(i+1) - 1;
            bhvID(iInd) = dataBhv.ID(i);
        end
        if ~isempty(dataBhv)
            iInd = dataBhv.StartFrame(end):nBhvBins;
            if ~isempty(iInd)
                bhvID(iInd) = dataBhv.ID(end);
            end
        end
        
        % Store bhvID in dataStruct for behavior proportion calculations
        dataStruct.bhvID = bhvID;
        dataStruct.dataBhv = dataBhv;  % Also store full behavior data for reference
        dataStruct.fsBhv = opts.fsBhv;  % Store behavior sampling frequency for window calculations
        
        % Check if we should use spike times approach (new) or dataMat (old)
        % Default to spike times if not specified
        if ~isfield(opts, 'useSpikeTimes') || isempty(opts.useSpikeTimes)
            opts.useSpikeTimes = true;  % Default to new approach
        end
        
        if opts.useSpikeTimes
            % Load spike times using new approach
            spikeData = load_spike_times('spontaneous', paths, sessionName, opts);
            
            % Extract area information
            dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
            idM23 = [];
            idM56 = [];
            idDS = [];
            idVS = [];
            
            % Find neuron indices for each area
            for i = 1:length(spikeData.neuronIDs)
                areaName = spikeData.neuronAreas{i};
                
                switch areaName
                    case 'M23'
                        idM23 = [idM23, i];
                    case 'M56'
                        idM56 = [idM56, i];
                    case 'DS'
                        idDS = [idDS, i];
                    case 'VS'
                        idVS = [idVS, i];
                end
            end
            
            dataStruct.idMatIdx = {idM23, idM56, idDS, idVS};
            dataStruct.idLabel = {spikeData.neuronIDs(idM23), spikeData.neuronIDs(idM56), ...
                                 spikeData.neuronIDs(idDS), spikeData.neuronIDs(idVS)};
            
            % Store spike times for on-demand binning
            dataStruct.spikeTimes = spikeData.spikeTimes;
            dataStruct.spikeClusters = spikeData.spikeClusters;
            dataStruct.spikeData = spikeData;  % Store full structure for reference
            dataStruct.dataMat = [];  % Not used in new approach
            dataStruct.areaLabels = spikeData.neuronAreas;
            
            fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        else
            % Load using old approach (dataMat)
            % Load spike data
            spikeDataRaw = load_data(opts, 'spikes');
            spikeDataRaw.bhvDur = dataBhv.Dur;
            
            % Find the neuron clusters (ids) in each brain region
            useMulti = 1;
            if ~useMulti
                allGood = strcmp(spikeDataRaw.ci.group, 'good');
            else
                allGood = strcmp(spikeDataRaw.ci.group, 'good') | (strcmp(spikeDataRaw.ci.group, 'mua'));
                warning('Warning in load_spontaneous_data: you are loading muas with the good spiking units.');
            end
            
            goodM23 = allGood & strcmp(spikeDataRaw.ci.area, 'M23');
            goodM56 = allGood & strcmp(spikeDataRaw.ci.area, 'M56');
            goodDS = allGood & strcmp(spikeDataRaw.ci.area, 'DS');
            goodVS = allGood & strcmp(spikeDataRaw.ci.area, 'VS');
            
            % Which neurons to use in the neural matrix
            opts.useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);
            
            % Create neural matrix
            [dataMat, idLabels, areaLabels, removedNeurons] = neural_matrix(spikeDataRaw, opts);
            
            % Extract area indices
            idM23 = find(strcmp(areaLabels, 'M23'));
            idM56 = find(strcmp(areaLabels, 'M56'));
            idDS = find(strcmp(areaLabels, 'DS'));
            idVS = find(strcmp(areaLabels, 'VS'));
            
            % Store in dataStruct
            dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
            dataStruct.idMatIdx = {idM23, idM56, idDS, idVS};
            dataStruct.idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};
            dataStruct.dataMat = dataMat;
            dataStruct.spikeData = [];  % Initialize empty, can be loaded later if needed
            dataStruct.spikeTimes = [];
            dataStruct.spikeClusters = [];
            dataStruct.areaLabels = areaLabels;
            dataStruct.removedNeurons = removedNeurons;
            
            fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        end
        
    elseif strcmp(dataSource, 'lfp')
        % Load spontaneous LFP data
        if ~isfield(opts, 'fsLfp')
            opts.fsLfp = 1250;
        end
        
        % Load LFP data
        lfpData = load_data(opts, 'lfp');
        lfpData = fliplr(lfpData); % flip data so first column (channel) is brain surface
        
        % Average channels to create lfpPerArea (incorporating get_standard_data logic)
        % Channels [3 5] -> M23, [9 11] -> M56, [19 23] -> DS, [30 34] -> VS
        lfpPerArea = [mean(lfpData(:,[3 5]), 2) mean(lfpData(:,[9 11]), 2) ...
                      mean(lfpData(:,[19 23]), 2) mean(lfpData(:,[30 34]), 2)];
        clear lfpData;
        
        % Lowpass filter LFP at 300 Hz
        lfpPerArea = lowpass(lfpPerArea, 300, opts.fsLfp);
        
        % Clean LFP artifacts
        lfpPerArea = clean_lfp_artifacts(lfpPerArea, opts.fsLfp, ...
            'spikeThresh', lfpCleanParams.spikeThresh, ...
            'spikeWinSize', lfpCleanParams.spikeWinSize, ...
            'notchFreqs', lfpCleanParams.notchFreqs, ...
            'lowpassFreq', lfpCleanParams.lowpassFreq, ...
            'useHampel', lfpCleanParams.useHampel, ...
            'hampelK', lfpCleanParams.hampelK, ...
            'hampelNsigma', lfpCleanParams.hampelNsigma, ...
            'detrendOrder', lfpCleanParams.detrendOrder, ...
            'visualize', false);
        
        dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
        dataStruct.lfpPerArea = lfpPerArea;
        dataStruct.bands = bands;
        
        % Compute binned envelopes
        dataStruct = compute_lfp_binned_envelopes(dataStruct, opts, lfpCleanParams, bands);
    end
    
    % Create save directory based on subjectID (similar to reach_task/results/{dataBaseName})
    % Extract subjectID from sessionName
    pathParts = strsplit(sessionName, filesep);
    if length(pathParts) > 1
        % Session name has path separator (e.g., 'ag112321/recording1e')
        subjectID = fullfile(pathParts{1}, pathParts{2});
    else
        % Session name is just the subjectID (e.g., 'ey042822')
        subjectID = sessionName;
    end
    
    % Use dropPath/spontaneous/results/{subjectID}/ similar to reach_task/results/{dataBaseName}
    dataStruct.saveDir = fullfile(paths.dropPath, 'spontaneous/results', subjectID);
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end
    
    % Store session name
    dataStruct.sessionName = sessionName;
    
    % Initialize reach-specific variables as empty (not used for spontaneous)
    dataStruct.dataR = [];
    dataStruct.startBlock2 = [];
    dataStruct.reachStart = [];
    dataStruct.reachClass = [];
end
