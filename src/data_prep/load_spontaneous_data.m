function dataStruct = load_spontaneous_data(dataStruct, dataSource, paths, opts, subjectName, sessionName, lfpCleanParams, bands)
% LOAD_SPONTANEOUS_DATA - Load spontaneous session data
%
% Variables:
%   dataStruct - Data structure to populate
%   dataSource - 'spikes' or 'lfp'
%   paths - Paths structure from get_paths
%   opts - Options structure
%   subjectName - Subject folder under spontaneous/data (e.g. 'ag', 'ey')
%   sessionName - Session folder under subject (e.g. 'ag112321_1')
%   lfpCleanParams - LFP cleaning parameters (if dataSource == 'lfp')
%   bands - Frequency bands (if dataSource == 'lfp')
%
% Goal: Load from paths.spontaneousDataPath/subjectName/sessionName
%
% Notes:
%   opts.collectEnd = [] means analyze the full session (resolved from spikes).
%   Behavior CSVs are not loaded here. Call load_spontaneous_behavior_labels
%   after this function when bhvID is needed.

    % collectStart / collectEnd are seconds (same as neuro_behavior_options)
    if ~isfield(opts, 'collectEnd')
        opts.collectEnd = 10 * 60;  % seconds
    end

    opts.dataPath = fullfile(paths.spontaneousDataPath, subjectName);
    opts.sessionName = sessionName;
    opts.subjectName = subjectName;
    
    if strcmp(dataSource, 'spikes')
        % Check if we should use spike times approach (new) or dataMat (old)
        % Default to spike times if not specified
        if ~isfield(opts, 'useSpikeTimes') || isempty(opts.useSpikeTimes)
            opts.useSpikeTimes = true;  % Default to new approach
        end
        
        if opts.useSpikeTimes
            % Resolve collectEnd from spikes ([] = full session)
            spikeData = load_spike_times('spontaneous', paths, sessionName, opts);
            opts.collectEnd = spikeData.collectEnd;
            opts.collectStart = spikeData.collectStart;
            
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
            dataStruct.opts = opts;
            
            fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        else
            % Load using old approach (dataMat)
            spikeDataRaw = load_data(opts, 'spikes');
            % opts.collectEnd may still be [] here; load_data resolves it from spikes
            if isempty(opts.collectEnd) && isfield(spikeDataRaw, 'spikeTimes') ...
                    && ~isempty(spikeDataRaw.spikeTimes)
                opts.collectEnd = max(spikeDataRaw.spikeTimes);
            end

            % ci is already good / mua / real from load_data (cluster_quality_mask)
            inAreas = strcmp(spikeDataRaw.ci.area, 'M23') | strcmp(spikeDataRaw.ci.area, 'M56') | ...
                strcmp(spikeDataRaw.ci.area, 'DS') | strcmp(spikeDataRaw.ci.area, 'VS');
            opts.useNeurons = find(inAreas);

            [dataMat, idLabels, areaLabels, removedNeurons] = neural_matrix(spikeDataRaw, opts);

            idM23 = find(strcmp(areaLabels, 'M23'));
            idM56 = find(strcmp(areaLabels, 'M56'));
            idDS = find(strcmp(areaLabels, 'DS'));
            idVS = find(strcmp(areaLabels, 'VS'));

            dataStruct.areas = {'M23', 'M56', 'DS', 'VS'};
            dataStruct.idMatIdx = {idM23, idM56, idDS, idVS};
            dataStruct.idLabel = {idLabels(idM23), idLabels(idM56), idLabels(idDS), idLabels(idVS)};
            dataStruct.dataMat = dataMat;
            dataStruct.spikeData = [];
            dataStruct.spikeTimes = [];
            dataStruct.spikeClusters = [];
            dataStruct.areaLabels = areaLabels;
            dataStruct.removedNeurons = removedNeurons;

            fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS));
        end

        dataStruct.opts = opts;
        dataStruct = attach_empty_behavior_fields(dataStruct, opts);

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
    
    dataStruct.saveDir = fullfile(paths.spontaneousResultsPath, subjectName, sessionName);
    if ~exist(dataStruct.saveDir, 'dir')
        mkdir(dataStruct.saveDir);
    end

    dataStruct.sessionName = sessionName;
    dataStruct.subjectName = subjectName;
    
    % Initialize reach-specific variables as empty (not used for spontaneous)
    dataStruct.dataR = [];
    dataStruct.startBlock2 = [];
    dataStruct.reachStart = [];
    dataStruct.reachClass = [];
end

function dataStruct = attach_empty_behavior_fields(dataStruct, opts)
% ATTACH_EMPTY_BEHAVIOR_FIELDS - Placeholder bhv fields (labels loaded separately)

dataStruct.bhvID = [];
dataStruct.bhvTimeOrigin = [];
dataStruct.dataBhv = [];
if isfield(opts, 'fsBhv') && ~isempty(opts.fsBhv)
    dataStruct.fsBhv = opts.fsBhv;
else
    dataStruct.fsBhv = [];
end
end
