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
% Goal: Load interval task neural data from paths.intervalDataPath/subjectName/sessionName.
%   Spike quality uses ci.group (good / mua) when that column is populated;
%   otherwise units with ci.rf_label == 'real' are kept.

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
            spikeData = load_interval_spike_times(paths, sessionName, opts);
            opts.collectEnd = spikeData.collectEnd;
            opts.collectStart = spikeData.collectStart;

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
            dataStruct.opts = opts;

            fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        else
            error('useSpikeTimes=false is not supported for interval data; set opts.useSpikeTimes=true');
        end

        dataStruct.bhvID = [];
        dataStruct.dataBhv = [];
        dataStruct.bhvTimeOrigin = [];
        if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
            dataStruct.bhvTimeOrigin = opts.collectStart;
        end
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

function spikeData = load_interval_spike_times(paths, sessionName, opts)
% LOAD_INTERVAL_SPIKE_TIMES - Load interval-task spikes from cluster_info / npy
%
% Variables:
%   paths       - Paths structure from get_paths
%   sessionName - Session folder under the subject
%   opts        - Options; requires subjectName. Optional: fsSpike, collectStart,
%                 collectEnd, removeSome, useMulti (true: good+mua; false: good only)
%
% Goal:
%   Load cluster_info.tsv, spike_times.npy, and spike_clusters.npy for an
%   interval session. Keep units from ci.group when that column is populated
%   (good / mua per opts.useMulti); otherwise keep ci.rf_label == 'real'.

    if ~isfield(opts, 'subjectName') || isempty(opts.subjectName)
        error('opts.subjectName must be set before loading interval spike times');
    end
    if ~isfield(opts, 'fsSpike') || isempty(opts.fsSpike)
        opts.fsSpike = 30000;
    end

    sessionFolder = fullfile(paths.intervalDataPath, opts.subjectName, sessionName);
    ci = load_interval_cluster_info(sessionFolder, sessionName);

    spikeTimesPath = fullfile(sessionFolder, 'spike_times.npy');
    if ~exist(spikeTimesPath, 'file')
        error('spike_times.npy not found in %s', sessionFolder);
    end
    spikeTimes = double(readNPY(spikeTimesPath)) / opts.fsSpike;

    spikeClustersPath = fullfile(sessionFolder, 'spike_clusters.npy');
    if ~exist(spikeClustersPath, 'file')
        error('spike_clusters.npy not found in %s', sessionFolder);
    end
    spikeClusters = readNPY(spikeClustersPath);

    allGood = cluster_quality_mask(ci, opts);

    goodM23 = allGood & strcmp(ci.area, 'M23');
    goodM56 = allGood & strcmp(ci.area, 'M56');
    goodDS = allGood & strcmp(ci.area, 'DS');
    goodVS = allGood & strcmp(ci.area, 'VS');
    useNeurons = find(goodM23 | goodM56 | goodDS | goodVS);

    if ismember('id', ci.Properties.VariableNames)
        neuronIDs = ci.id(useNeurons);
    else
        neuronIDs = ci.cluster_id(useNeurons);
    end
    neuronAreas = ci.area(useNeurons);

    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        opts.collectStart = 0;
    end
    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = [];
    end
    opts.collectEnd = clamp_collect_end_to_session(opts.collectEnd, max(spikeTimes), opts.collectStart);

    validSpikes = ismember(spikeClusters, neuronIDs) & ...
        spikeTimes >= opts.collectStart & ...
        spikeTimes <= opts.collectEnd;
    spikeTimes = spikeTimes(validSpikes);
    spikeClusters = spikeClusters(validSpikes);

    if isfield(opts, 'removeSome') && opts.removeSome
        [spikeTimes, spikeClusters, neuronIDs, neuronAreas] = ...
            filter_interval_by_firing_rate(spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts);
    end

    spikeData = struct();
    spikeData.spikeTimes = spikeTimes;
    spikeData.spikeClusters = spikeClusters;
    spikeData.neuronIDs = neuronIDs;
    spikeData.neuronAreas = cell(size(neuronAreas));
    for i = 1:length(neuronAreas)
        spikeData.neuronAreas{i} = char(neuronAreas(i));
    end
    spikeData.idLabels = neuronIDs;
    spikeData.areaLabelsUnique = unique(neuronAreas);
    spikeData.totalTime = opts.collectEnd - opts.collectStart;
    spikeData.collectStart = opts.collectStart;
    spikeData.collectEnd = opts.collectEnd;
end

function ci = load_interval_cluster_info(sessionFolder, sessionName)
% LOAD_INTERVAL_CLUSTER_INFO - Read cluster_info.tsv and assign brain areas
%
% Variables:
%   sessionFolder - Path to the interval session directory
%   sessionName   - Session folder name (for logging)
%
% Goal:
%   Return a cluster_info table with depth 0 = superficial (M23), 3840 = deep
%   (VS), and an area column from get_brain_area_depth_ranges.

    clusterInfoPath = fullfile(sessionFolder, 'cluster_info.tsv');
    if ~exist(clusterInfoPath, 'file')
        error('cluster_info.tsv not found in %s', sessionFolder);
    end
    ci = readtable(clusterInfoPath, 'FileType', 'text', 'Delimiter', '\t');

    ci = sortrows(ci, 'depth');
    ci.depth = 3840 - ci.depth;
    ci = flipud(ci);

    [m23, m56, cc, ds, vs, depthSource] = get_brain_area_depth_ranges(sessionFolder);
    if strcmp(depthSource, 'session')
        fprintf('Using brain_area_depths.mat for %s\n', sessionName);
    end

    area = cell(size(ci, 1), 1);
    area(ci.depth >= m23(1) & ci.depth <= m23(2)) = {'M23'};
    area(ci.depth >= m56(1) & ci.depth <= m56(2)) = {'M56'};
    area(ci.depth >= cc(1) & ci.depth <= cc(2)) = {'CC'};
    area(ci.depth >= ds(1) & ci.depth <= ds(2)) = {'DS'};
    area(ci.depth >= vs(1) & ci.depth <= vs(2)) = {'VS'};
    ci.area = area;
end

function [spikeTimes, spikeClusters, neuronIDs, neuronAreas] = ...
    filter_interval_by_firing_rate(spikeTimes, spikeClusters, neuronIDs, neuronAreas, opts)
% FILTER_INTERVAL_BY_FIRING_RATE - Drop interval units outside rate bounds
%
% Variables:
%   spikeTimes    - Spike times (seconds)
%   spikeClusters - Cluster id per spike
%   neuronIDs     - Unit ids to filter
%   neuronAreas   - Area label per unit
%   opts          - Options with collectStart, collectEnd, rate bounds
%
% Goal:
%   Keep neurons that pass neuron_firing_rate_filter_spikes and restrict
%   spikeTimes / spikeClusters to those units.

    if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
        timeStart = 0;
    else
        timeStart = opts.collectStart;
    end
    timeEnd = opts.collectEnd;

    keepNeurons = neuron_firing_rate_filter_spikes(opts, neuronIDs, spikeTimes, ...
        spikeClusters, timeStart, timeEnd);

    fprintf('\nKeeping %d of %d neurons after firing rate filtering\n', ...
        sum(keepNeurons), length(keepNeurons));

    neuronIDs = neuronIDs(keepNeurons);
    neuronAreas = neuronAreas(keepNeurons);

    validSpikes = ismember(spikeClusters, neuronIDs);
    spikeTimes = spikeTimes(validSpikes);
    spikeClusters = spikeClusters(validSpikes);
end
