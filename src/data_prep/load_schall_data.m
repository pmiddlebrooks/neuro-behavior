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

    % Set default collectEnd if not set
    if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
        opts.collectEnd = 7*60;  % Default 7 minutes
    end
    
    % Load schall data file (sessionName no longer includes .mat extension)
    schallDataFile = fullfile(paths.schallDataPath, [sessionName, '.mat']);
    
    [~, dataBaseName, ~] = fileparts(sessionName);
    dataStruct.dataBaseName = dataBaseName;
    dataStruct.saveDir = fullfile(paths.dropPath, 'schall/results', dataBaseName);
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end
    
    dataS = load(schallDataFile);
    dataStruct.dataS = dataS;
    
    % Process response onsets
    responseOnset = convert_to_session_time(dataS.responseOnset, dataS.trialOnset);
    responseOnset = responseOnset / 1000;
    responseOnset(responseOnset < opts.collectStart) = [];
    responseOnset(responseOnset > opts.collectEnd) = [];
    dataStruct.responseOnset = responseOnset;
    
    if strcmp(dataSource, 'spikes')
        % Load schall spike data
        [dataMat, idLabels, areaLabels] = neural_matrix_schall_fef(dataS, opts);
        dataStruct.areas = {'FEF'};
        idFEF = 1:length(idLabels);
        dataStruct.idMatIdx = {idFEF};
        dataStruct.idLabel = {idLabels(idFEF)};
        dataStruct.dataMat = dataMat;
        dataStruct.spikeData = [];
        
        fprintf('%d FEF\n', length(idFEF));
        
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
