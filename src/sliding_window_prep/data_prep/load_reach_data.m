function dataStruct = load_reach_data(dataStruct, dataSource, paths, sessionName, opts, lfpCleanParams, bands)
% LOAD_REACH_DATA Load reach task data for sliding window analysis
%
% Variables:
%   dataStruct - Data structure to populate
%   dataSource - 'spikes' or 'lfp'
%   paths - Paths structure from get_paths
%   sessionName - Session name (e.g., 'Y15_27-Aug-2025 14_02_21_NeuroBeh.mat')
%   opts - Options structure
%   lfpCleanParams - LFP cleaning parameters (if dataSource == 'lfp')
%   bands - Frequency bands (if dataSource == 'lfp')
%
% Goal:
%   Load reach task spike or LFP data and populate dataStruct.
%
% Returns:
%   dataStruct - Updated data structure

    % Load reach data file
    reachDataFile = fullfile(paths.reachDataPath, sessionName);
    
    [~, dataBaseName, ~] = fileparts(reachDataFile);
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
        opts.collectEnd = 8*60;  % Override to 8 minutes
    end
    dataStruct.opts = opts;
    
    if strcmp(dataSource, 'spikes')
        % Load reach spike data
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
        
        fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        
    elseif strcmp(dataSource, 'lfp')
        % Load reach LFP data
        [~, lfpBaseName, ~] = fileparts(sessionName);
        lfpBaseName = strrep(lfpBaseName, '_NeuroBeh', '');
        
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

