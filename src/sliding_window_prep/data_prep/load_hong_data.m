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

    % Load Hong data files
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
    dataStruct.spikeData = [];
    dataStruct.T = T;  % Store behavior table
    
    fprintf('%d S1\n', length(idS1));
    fprintf('%d SC\n', length(idSC));
    
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

