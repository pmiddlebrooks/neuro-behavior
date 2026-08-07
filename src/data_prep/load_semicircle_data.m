function dataStruct = load_semicircle_data(dataStruct, dataSource, paths, opts, subjectName, sessionName, lfpCleanParams, bands)
% LOAD_SEMICIRCLE_DATA - Load semicircle reward task session data
%
% Variables:
%   dataStruct     - Data structure to populate
%   dataSource     - 'spikes' or 'lfp'
%   paths          - Paths structure from get_paths
%   opts           - Options structure
%   subjectName    - Subject folder under semicircle_reward_task/data (e.g. 'AS1')
%   sessionName    - Session .mat basename (e.g. 'AS1_0618_WellLearned')
%   lfpCleanParams - LFP cleaning parameters (if dataSource == 'lfp')
%   bands          - Frequency bands (if dataSource == 'lfp')
%
% Goal:
%   Load semicircle spike (or LFP) data into dataStruct for session analyses.
%   Area labels map layerIDs to pipeline names: M1 L23->M23, M1 L5->M56,
%   DMS->DS, VS->VS.
%
% Returns:
%   dataStruct - Updated data structure

    if isempty(subjectName) || isempty(sessionName)
        error('subjectName and sessionName must be provided for semicircle data');
    end

    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = [];
    end

    opts.subjectName = subjectName;
    opts.sessionName = sessionName;

    dataFile = fullfile(paths.semicircleDataPath, subjectName, [sessionName, '.mat']);
    if ~exist(dataFile, 'file')
        error('Semicircle data file not found: %s', dataFile);
    end

    dataS = load(dataFile);
    dataStruct.dataS = dataS;
    dataStruct.saveDir = fullfile(paths.semicircleResultsPath, subjectName, sessionName);
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end
    dataStruct.dataBaseName = sessionName;

    % Trial / task events (seconds); see data README TaskMatrix columns
    if isfield(dataS, 'TaskMatrix') && ~isempty(dataS.TaskMatrix)
        dataStruct.taskMatrix = dataS.TaskMatrix;
        dataStruct.trialStart = dataS.TaskMatrix(:, 2);
        dataStruct.trialOutcome = dataS.TaskMatrix(:, 3);  % -1 failed, 0 unrewarded, 1 rewarded
        dataStruct.choicePort = dataS.TaskMatrix(:, 4);
        dataStruct.choicePokeTime = dataS.TaskMatrix(:, 6);
        dataStruct.trialEnd = dataS.TaskMatrix(:, 8);
        dataStruct.leaveHomeFirst = dataS.TaskMatrix(:, 9);
        dataStruct.enterHomeStart = dataS.TaskMatrix(:, 10);
        dataStruct.leaveHomeLast = dataS.TaskMatrix(:, 11);
    else
        dataStruct.taskMatrix = [];
        dataStruct.trialStart = [];
        dataStruct.trialOutcome = [];
        dataStruct.choicePort = [];
        dataStruct.choicePokeTime = [];
        dataStruct.trialEnd = [];
        dataStruct.leaveHomeFirst = [];
        dataStruct.enterHomeStart = [];
        dataStruct.leaveHomeLast = [];
    end

    if strcmp(dataSource, 'spikes')
        if ~isfield(opts, 'useSpikeTimes') || isempty(opts.useSpikeTimes)
            opts.useSpikeTimes = true;
        end
        if ~opts.useSpikeTimes
            error('useSpikeTimes=false is not supported for semicircle data; set opts.useSpikeTimes=true');
        end

        spikeData = load_spike_times('semicircle', paths, sessionName, opts);
        opts.collectEnd = spikeData.collectEnd;
        opts.collectStart = spikeData.collectStart;
        dataStruct.opts = opts;

        dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
        idM23 = [];
        idM56 = [];
        idDS = [];
        idVS = [];

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
        dataStruct.spikeTimes = spikeData.spikeTimes;
        dataStruct.spikeClusters = spikeData.spikeClusters;
        dataStruct.spikeData = spikeData;
        dataStruct.dataMat = [];
        dataStruct.areaLabels = spikeData.neuronAreas;

        fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));

        dataStruct.bhvID = [];
        dataStruct.dataBhv = [];
        dataStruct.bhvTimeOrigin = [];
        if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
            dataStruct.bhvTimeOrigin = opts.collectStart;
        end
        dataStruct.fsBhv = [];

    elseif strcmp(dataSource, 'lfp')
        error('LFP loading is not yet implemented for semicircle reward task data');
    else
        error('Unsupported dataSource for semicircle: %s', dataSource);
    end

    dataStruct.areasToTest = 1:4;
end
