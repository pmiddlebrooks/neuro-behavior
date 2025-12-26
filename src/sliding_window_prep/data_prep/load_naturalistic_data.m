function dataStruct = load_naturalistic_data(dataStruct, dataSource, paths, opts, lfpCleanParams, bands)
% LOAD_NATURALISTIC_DATA Load naturalistic data for sliding window analysis
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
%   Load naturalistic spike or LFP data and populate dataStruct with
%   appropriate fields.
%
% Returns:
%   dataStruct - Updated data structure

    % Set default collectEnd if not set
    if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
        opts.collectEnd = 10 * 60;  % Default 10 minutes
    end
    
    if strcmp(dataSource, 'spikes')
        % Load naturalistic spike data
        getDataType = 'spikes';
        get_standard_data
        
        % Capture workspace variables set by get_standard_data
        dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
        dataStruct.idMatIdx = {idM23, idM56, idDS, idVS};
        dataStruct.idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};
        dataStruct.dataMat = dataMat;  % dataMat set by get_standard_data
        dataStruct.spikeData = [];  % Initialize empty, can be loaded later if needed
        
    elseif strcmp(dataSource, 'lfp')
        % Load naturalistic LFP data
        getDataType = 'lfp';
        if ~isfield(opts, 'fsLfp')
            opts.fsLfp = 1250;
        end
        get_standard_data
        
        % lfpPerArea is set by get_standard_data in workspace
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
    
    % Create save directory
    dataStruct.saveDir = fullfile(paths.dropPath, 'criticality/results');
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end
    
    % Initialize reach-specific variables as empty
    dataStruct.dataR = [];
    dataStruct.startBlock2 = [];
    dataStruct.reachStart = [];
    dataStruct.reachClass = [];
    dataStruct.sessionName = '';
    dataStruct.areasToTest = 1:4;
end

