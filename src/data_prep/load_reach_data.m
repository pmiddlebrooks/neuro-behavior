function dataStruct = load_reach_data(dataStruct, dataSource, paths, sessionName, opts, lfpCleanParams, bands)
% LOAD_REACH_DATA Load reach task data for sliding window analysis
%
% Variables:
%   dataStruct - Data structure to populate
%   dataSource - 'spikes' or 'lfp'
%   paths - Paths structure from get_paths
%   sessionName - Session name without .mat extension (e.g., 'Y15_27-Aug-2025 14_02_21_NeuroBeh')
%   opts - Options structure
%   lfpCleanParams - LFP cleaning parameters (if dataSource == 'lfp')
%   bands - Frequency bands (if dataSource == 'lfp')
%
% Goal:
%   Load reach task spike or LFP data and populate dataStruct.
%
% Returns:
%   dataStruct - Updated data structure

    % Load reach data file (sessionName no longer includes .mat extension)
    reachDataFile = fullfile(paths.reachDataPath, [sessionName, '.mat']);
    
    [~, dataBaseName, ~] = fileparts(sessionName);
    dataStruct.dataBaseName = dataBaseName;
    dataStruct.saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end
    
    dataR = load(reachDataFile);
    dataStruct.dataR = dataR;
    dataStruct.reachClass = dataR.Block(:,3);
    dataStruct.reachStart = dataR.R(:,1) / 1000;  % Convert from ms to seconds
    dataStruct.startBlock2 = dataStruct.reachStart(find(ismember(dataStruct.reachClass, [3 4]), 1));
    
    % Set collectEnd
    if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
        opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);
    end
    dataStruct.opts = opts;
    
    if strcmp(dataSource, 'spikes')
        % Check if we should use spike times approach (new) or dataMat (old)
        % Default to spike times if not specified
        if ~isfield(opts, 'useSpikeTimes') || isempty(opts.useSpikeTimes)
            opts.useSpikeTimes = true;  % Default to new approach
        end
        
        if opts.useSpikeTimes
            % Load spike times using new approach
            spikeData = load_spike_times('reach', paths, sessionName, opts);
            
            % Extract area information
            dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
            idM23 = [];
            idM56 = [];
            idDS = [];
            idVS = [];
            
            % Find neuron indices for each area
            for i = 1:length(spikeData.neuronIDs)
                areaName = spikeData.neuronAreas{i};
                neuronID = spikeData.neuronIDs(i);
                
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
            
            fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        else
            % Load using old approach (dataMat)
            [dataMat, idLabels, areaLabels] = reach_neural_matrix(dataR, opts);
            dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
            idM23 = find(strcmp(areaLabels, 'M23'));
            idM56 = find(strcmp(areaLabels, 'M56'));
            idDS = find(strcmp(areaLabels, 'DS'));
            idVS = find(strcmp(areaLabels, 'VS'));
            dataStruct.idMatIdx = {idM23, idM56, idDS, idVS};
            dataStruct.idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};
            dataStruct.dataMat = dataMat;
            dataStruct.spikeData = [];  % Can be loaded later if needed
            dataStruct.spikeTimes = [];
            dataStruct.spikeClusters = [];
            
            fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        end
        
    elseif strcmp(dataSource, 'lfp')
        % Load reach LFP data
        % sessionName no longer includes .mat extension
        lfpBaseName = strrep(sessionName, '_NeuroBeh', '');
        
        reachPath = paths.reachDataPath;
        lfpFile = fullfile(reachPath, [lfpBaseName, '_g0_t0.imec0.lf.bin']);
        metaFile = fullfile(reachPath, [lfpBaseName, '_g0_t0.imec0.lf.meta']);
        
        if ~exist(lfpFile, 'file') || ~exist(metaFile, 'file')
            error('LFP files not found. Expected: %s and %s', lfpFile, metaFile);
        end
        
        % Read metadata
        if ~exist('ReadMeta', 'file')
            error('ReadMeta function not found. Please ensure SpikeGLX toolbox is in path.');
        end
        m = ReadMeta(metaFile);
        opts.fsLfp = round(str2double(m.imSampRate));
        nChan = str2double(m.nSavedChans);
        nSamps = str2double(m.fileSizeBytes)/2/nChan;
        
        % Read LFP data
        mmf = memmapfile(lfpFile, 'Format', {'int16', [nChan nSamps], 'x'});
        lfpPerArea = double(flipud([...
            mmf.Data.x(360,:);...
            mmf.Data.x(280,:);...
            mmf.Data.x(170,:);...
            mmf.Data.x(40,:)...
            ])');
        
        % Clean LFP artifacts
        lfpPerArea = clean_lfp_artifacts(lfpPerArea, opts.fsLfp, ...
            'spikeThresh', lfpCleanParams.spikeThresh, ...
            'spikeWinSize', lfpCleanParams.spikeWinSize, ...
            'notchFreqs', lfpCleanParams.notchFreqs, ...
            'lowpassFreq', lfpCleanParams.lowpassFreq, ...
            'useHampel', lfpCleanParams.useHampel, ...
            'hampelK', lfpCleanParams.hampelK, ...
            'hampelNsigma', lfpCleanParams.hampelNsigma, ...
            'detrendOrder', lfpCleanParams.detrendOrder);
        
        dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
        dataStruct.lfpPerArea = lfpPerArea;
        dataStruct.bands = bands;
        dataStruct.opts = opts;
        
        % Compute binned envelopes
        dataStruct = compute_lfp_binned_envelopes(dataStruct, opts, lfpCleanParams, bands);
    end
    
    dataStruct.areasToTest = 1:4;
end
