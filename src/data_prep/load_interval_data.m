function dataStruct = load_interval_data(dataStruct, dataSource, paths, opts, subjectName, sessionName, lfpCleanParams, bands)
% LOAD_INTERVAL_DATA - Load interval timing task session data
%
% Variables:
%   dataStruct - Data structure to populate
%   dataSource - 'spikes' or 'lfp'
%   paths - Paths structure from get_paths
%   opts - Options structure
%   subjectName - Subject folder under interval_timing_task/data (e.g. 'ey9166')
%   sessionName - Session folder under subject (e.g. 'ey9166_2026_04_03')
%   lfpCleanParams - LFP cleaning parameters (if dataSource == 'lfp')
%   bands - Frequency bands (if dataSource == 'lfp')
%
% Goal: Load interval task neural data from paths.intervalDataPath/subjectName/sessionName

    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = 10 * 60;
    end

    opts.dataPath = fullfile(paths.intervalDataPath, subjectName);
    opts.sessionName = sessionName;
    opts.subjectName = subjectName;

    if strcmp(dataSource, 'spikes')
        if ~isfield(opts, 'useSpikeTimes') || isempty(opts.useSpikeTimes)
            opts.useSpikeTimes = true;
        end

        if opts.useSpikeTimes
            spikeData = load_spike_times('interval', paths, sessionName, opts);

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
        else
            error('useSpikeTimes=false is not supported for interval data; set opts.useSpikeTimes=true');
        end

        dataStruct.bhvID = [];
        dataStruct.dataBhv = [];
        dataStruct.fsBhv = [];

    elseif strcmp(dataSource, 'lfp')
        if ~isfield(opts, 'fsLfp')
            opts.fsLfp = 1250;
        end

        lfpData = load_data(opts, 'lfp');
        lfpData = fliplr(lfpData);

        lfpPerArea = [mean(lfpData(:,[3 5]), 2) mean(lfpData(:,[9 11]), 2) ...
            mean(lfpData(:,[19 23]), 2) mean(lfpData(:,[30 34]), 2)];
        clear lfpData;

        lfpPerArea = lowpass(lfpPerArea, 300, opts.fsLfp);
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
        dataStruct = compute_lfp_binned_envelopes(dataStruct, opts, lfpCleanParams, bands);
    end

    dataStruct.saveDir = fullfile(paths.intervalResultsPath, subjectName, sessionName);
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end

    dataStruct.sessionName = sessionName;
    dataStruct.subjectName = subjectName;

    dataStruct.dataR = [];
    dataStruct.startBlock2 = [];
    dataStruct.reachStart = [];
    dataStruct.reachClass = [];
end
