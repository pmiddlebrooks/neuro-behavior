function results = hmm_mazz_analysis(dataStruct, config)
% HMM_MAZZ_ANALYSIS Run Mazzucato-style HMM analysis for spontaneous/reach data.
%
% Variables:
%   dataStruct - Structure with precomputed data and metadata:
%       .sessionType     - 'spontaneous' or 'reach' (preferred)
%                          (legacy: .natOrReach = 'Nat' or 'Reach')
%       .paths           - Output of get_paths()
%       .opts            - Options struct from neuro_behavior_options(), with
%                          .HmmParam optionally pre-populated
%       .areas           - Cell array of brain area names (e.g., {'M23','M56','DS','VS'})
%       .idList          - Cell array of neuron id vectors per area (same order as areas)
%       .spikeData       - [time, neuronId, areaIdx] spike table
%       .trialDur        - Trial duration (seconds) for metadata
%       .sessionName     - (Reach) Session name string for saving/metadata
%       .reachDataFile   - (Reach, optional) Full path to NeuroBeh.mat file
%
%   config - Structure controlling analysis behavior:
%       .modelSelectionMethod - Model selection method string (default: 'XVAL')
%       .minNumNeurons        - Minimum neuron count per area (default: 15)
%       .areasToTest          - Optional vector of area indices to analyze
%       .saveData             - Save results/summary to disk (default: true)
%       .useParallel          - Use parallel pool for hmm.funHMM (default: true)
%
% Goal:
%   Perform HMM fitting per brain area, build continuous state and metastate
%   sequences, and save a results struct compatible with hmm_load_saved_model
%   and existing peri-event analysis scripts.

if nargin < 2 || isempty(config)
    config = struct();
end

% Backward-compatible handling of Nat/Reach vs sessionType
if isfield(dataStruct, 'sessionType') && ~isempty(dataStruct.sessionType)
    sessionType = dataStruct.sessionType;
elseif isfield(dataStruct, 'natOrReach') && ~isempty(dataStruct.natOrReach)
    switch dataStruct.natOrReach
        case 'Nat'
            sessionType = 'spontaneous';
        case 'Reach'
            sessionType = 'reach';
        otherwise
            error('Unrecognized natOrReach value: %s', dataStruct.natOrReach);
    end
else
    error('dataStruct must contain either sessionType or natOrReach.');
end

paths = dataStruct.paths;
opts = dataStruct.opts;
areas = dataStruct.areas;
idList = dataStruct.idList;
spikeData = dataStruct.spikeData;
trialDur = dataStruct.trialDur;

if ~isfield(config, 'modelSelectionMethod') || isempty(config.modelSelectionMethod)
    config.modelSelectionMethod = 'XVAL';
end
if ~isfield(config, 'minNumNeurons') || isempty(config.minNumNeurons)
    config.minNumNeurons = 15;
end
if ~isfield(config, 'saveData') || isempty(config.saveData)
    config.saveData = true;
end
if ~isfield(config, 'useParallel') || isempty(config.useParallel)
    config.useParallel = true;
end

modelSel = config.modelSelectionMethod;
minNumNeurons = config.minNumNeurons;

% HMM parameter defaults (allow override via opts.HmmParam or config.HmmParam)
if isfield(config, 'HmmParam') && ~isempty(config.HmmParam)
    hmmParam = config.HmmParam;
elseif isfield(opts, 'HmmParam') && ~isempty(opts.HmmParam)
    hmmParam = opts.HmmParam;
else
    hmmParam = struct();
    hmmParam.AdjustT = 0.0;
    hmmParam.BinSize = 0.01;
    hmmParam.MinDur = 0.04;
    hmmParam.MinP = 0.8;
    hmmParam.NumSteps = 8;
    hmmParam.NumRuns = 33;
    hmmParam.singleSeqXval.K = 5;
end

opts.HmmParam = hmmParam;

% Determine areasToTest based on neuron counts and optional mask
numAreas = numel(areas);
if isfield(config, 'areasToTest') && ~isempty(config.areasToTest)
    areasToTest = config.areasToTest(:)';
else
    areaCounts = zeros(1, numAreas);
    for areaIdx = 1:numAreas
        areaCounts(areaIdx) = numel(idList{areaIdx});
    end
    areaMask = areaCounts >= minNumNeurons;
    areasToTest = find(areaMask);
end

fprintf('HMM_MAZZ_ANALYSIS: %s data\n', sessionType);
fprintf('Model selection: %s\n', modelSel);
fprintf('Areas: %s\n', strjoin(areas, ', '));
fprintf('Min neurons per area: %d\n', minNumNeurons);
fprintf('Areas to test (indices): %s\n', mat2str(areasToTest));

allResults = struct();
allResults.areas = areas;
allResults.binSize = zeros(1, numAreas);
allResults.numStates = zeros(1, numAreas);
allResults.hmm_results = cell(1, numAreas);

% Optional parallel pool
poolStartedHere = false;
if config.useParallel
    if isempty(gcp('nocreate'))
        poolCluster = parcluster('local');
        numWorkers = min(poolCluster.NumWorkers, 4);
        parpool('local', numWorkers);
        poolStartedHere = true;
    end
end

for areaIdx = areasToTest
    areaName = areas{areaIdx};
    neuronIds = idList{areaIdx};

    fprintf('\n=== Processing Brain Area: %s (Index %d) ===\n', areaName, areaIdx);

    numUnits = numel(neuronIds);
    if numUnits < minNumNeurons
        fprintf('Skipping area %s: only %d neurons (min %d)\n', ...
            areaName, numUnits, minNumNeurons);
        continue;
    end

    % Build spikes struct directly from spikeData
    spikesStruct = struct('spk', cell(1, numUnits));

    % Determine trial window in seconds using requested collection duration
    if isfield(opts, 'collectEnd') && ~isempty(opts.collectEnd)
        trialStartSec = 0;
        trialEndSec = opts.collectEnd;
    else
        areaMask = ismember(spikeData(:, 2), neuronIds) & (spikeData(:, 3) == areaIdx);
        if any(areaMask)
            trialStartSec = 0;
            trialEndSec = max(spikeData(areaMask, 1));
        else
            fprintf('No spikes found for area %s, skipping.\n', areaName);
            continue;
        end
    end
    trialDurationSec = max(trialEndSec - trialStartSec, 0);
    winTrain = [trialStartSec, trialEndSec];

    for neuronIdx = 1:numUnits
        neuronId = neuronIds(neuronIdx);
        neuronMask = spikeData(:, 2) == neuronId & spikeData(:, 3) == areaIdx;
        spikeTimes = spikeData(neuronMask, 1);
        spikeTimes = spikeTimes(spikeTimes >= trialStartSec & spikeTimes <= trialEndSec);
        spikeTimes = spikeTimes - trialStartSec;
        spikesStruct(1, neuronIdx).spk = spikeTimes(:)';
    end

    fprintf('\nRunning HMM on %s %s for area %s\n\n', ...
        sessionType, dataStruct.sessionName, areaName);

    dataIn = struct();
    dataIn.spikes = spikesStruct;
    dataIn.win = winTrain;
    dataIn.METHOD = modelSel;
    dataIn.HmmParam = opts.HmmParam;

    res = hmm.funHMM(dataIn);

    fprintf('HMM analysis completed successfully for %s.\n', areaName);
    fprintf('Number of states: %d\n', res.HmmParam.VarStates(res.BestStateInd));
    fprintf('Log-likelihood: %.2f\n', res.hmm_bestfit.LLtrain);

    % Transform trial results back to continuous time series
    continuousPStates = [];
    continuousSequence = [];
    totalTimeBins = [];
    if ~isempty(res) && isfield(res, 'hmm_results') && isfield(res, 'hmm_postfit')
        fprintf('Transforming trial results back to continuous time series...\n');

        numTrials = numel(res.hmm_results);
        numTimePerTrial = size(res.hmm_results(1).pStates, 2);
        numStates = size(res.hmm_results(1).pStates, 1);
        numNeurons = size(res.hmm_results(1).rates, 2);
        totalTimeBins = numTrials * numTimePerTrial;

        continuousPStates = zeros(totalTimeBins, numStates);
        continuousRates = zeros(totalTimeBins, numStates, numNeurons); %#ok<NASGU>
        continuousSequence = zeros(totalTimeBins, 1);

        for trialIdx = 1:numTrials
            timeIdx = ((trialIdx - 1) * numTimePerTrial + 1):(trialIdx * numTimePerTrial);
            continuousPStates(timeIdx, :) = res.hmm_results(trialIdx).pStates';
        end

        for binIdx = 1:totalTimeBins
            stateProbs = continuousPStates(binIdx, :);
            [maxProb, maxState] = max(stateProbs);
            if maxProb >= res.HmmParam.MinP
                continuousSequence(binIdx) = maxState;
            end
        end

        fprintf('Successfully transformed results to continuous format\n');
        fprintf('Proportion of undefined states: %.2f\n', ...
            sum(continuousSequence == 0) / numel(continuousSequence));
    end

    % Metastate detection
    communities = [];
    continuousMetastates = [];
    numMetastates = 0;
    if ~isempty(res) && isfield(res, 'hmm_bestfit') && isfield(res.hmm_bestfit, 'tpm')
        fprintf('Detecting metastates from transition probability matrix...\n');
        tpm = res.hmm_bestfit.tpm;
        numStates = size(tpm, 1);
        fprintf('Transition probability matrix size: %dx%d\n', numStates, numStates);

        try
            communities = detect_metastates_vidaurre(tpm, true);
            continuousMetastates = zeros(size(continuousSequence));
            for binIdx = 1:numel(continuousSequence)
                stateLabel = continuousSequence(binIdx);
                if stateLabel > 0 && stateLabel <= numStates
                    continuousMetastates(binIdx) = communities(stateLabel);
                else
                    continuousMetastates(binIdx) = 0;
                end
            end

            uniqueMetastates = unique(communities);
            numMetastates = numel(uniqueMetastates);
            fprintf('\tNumber of metastates: %d (from %d states)\n', numMetastates, numStates);
            for metastateIdx = 1:numMetastates
                metastateLabel = uniqueMetastates(metastateIdx);
                statesInMetastate = find(communities == metastateLabel);
                fprintf('  Metastate %d: states [%s]\n', ...
                    metastateLabel, num2str(statesInMetastate));
            end

            fprintf('Successfully created continuous metastates sequence\n');
        catch metastateError
            fprintf('Error in metastate detection for %s: %s\n', ...
                areaName, metastateError.message);
            communities = [];
            continuousMetastates = [];
            numMetastates = 0;
            clear metastateError;
        end
    else
        fprintf('No transition probability matrix available for metastate detection in %s\n', areaName);
    end

    % Build per-area result struct
    hmmRes = struct();
    hmmRes.metadata = struct();
    hmmRes.metadata.data_type = sessionType;
    hmmRes.metadata.brain_area = areaName;
    hmmRes.metadata.analysis_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    hmmRes.metadata.analysis_status = 'SUCCESS';
    hmmRes.metadata.model_selection_method = modelSel;

    hmmRes.data_params = struct();
    hmmRes.data_params.bin_size = opts.frameSize;
    hmmRes.data_params.collect_start = opts.collectStart;
    hmmRes.data_params.collect_duration = opts.collectEnd;
    hmmRes.data_params.min_act_time = opts.minActTime;
    hmmRes.data_params.min_firing_rate = opts.minFiringRate;

    hmmRes.trial_params = struct();
    hmmRes.trial_params.trial_windows = winTrain;
    hmmRes.trial_params.trial_duration = trialDur;

    hmmRes.hmm_data = res.hmm_data;
    hmmRes.hmm_results = res.hmm_results;
    hmmRes.hmm_postfit = res.hmm_postfit;
    hmmRes.hmm_multispikes = res.hmm_multispikes;
    hmmRes.HmmParam = res.HmmParam;
    hmmRes.colors = res.colors;
    hmmRes.LLtot = res.LLtot;

    hmmRes.best_model = struct();
    hmmRes.best_model.num_states = res.HmmParam.VarStates(res.BestStateInd);
    hmmRes.best_model.transition_matrix = res.hmm_bestfit.tpm;
    hmmRes.best_model.emission_matrix = res.hmm_bestfit.epm;
    hmmRes.best_model.log_likelihood = res.hmm_bestfit.LLtrain;
    hmmRes.best_model.best_state_index = res.BestStateInd;

    hmmRes.continuous_results = struct();
    hmmRes.continuous_results.pStates = continuousPStates;
    hmmRes.continuous_results.sequence = continuousSequence;
    if ~isempty(totalTimeBins)
        hmmRes.continuous_results.totalTime = totalTimeBins * res.HmmParam.BinSize;
    else
        hmmRes.continuous_results.totalTime = 0;
    end

    hmmRes.metastate_results = struct();
    hmmRes.metastate_results.communities = communities;
    hmmRes.metastate_results.continuous_metastates = continuousMetastates;
    hmmRes.metastate_results.num_metastates = numMetastates;
    if exist('tpm', 'var')
        hmmRes.metastate_results.transition_matrix = tpm;
    else
        hmmRes.metastate_results.transition_matrix = [];
    end

    allResults.binSize(areaIdx) = res.HmmParam.BinSize;
    allResults.numStates(areaIdx) = res.HmmParam.VarStates(res.BestStateInd);
    allResults.hmm_results{areaIdx} = hmmRes;

    slackMessage = sprintf('HMM_MAZZ: Completed analysis for brain area: %s\n', areaName);
    disp(slackMessage);
    if exist('slack_code_done', 'file')
        slack_code_done(slackMessage);
    end
end

% Build top-level results struct used by hmm_load_saved_model
results = struct();
results.areas = allResults.areas;
results.binSize = allResults.binSize;
results.numStates = allResults.numStates;
results.hmm_results = allResults.hmm_results;

% Save results and summary (paths and naming must match hmm_mazz.m)
if config.saveData
    binSizeSave = opts.HmmParam.BinSize;
    minDurSave = opts.HmmParam.MinDur;

    if strcmpi(sessionType, 'reach')
        if isfield(dataStruct, 'reachDataFile') && ~isempty(dataStruct.reachDataFile)
            [~, dataBaseName, ~] = fileparts(dataStruct.reachDataFile);
        elseif isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
            [~, dataBaseName, ~] = fileparts(dataStruct.sessionName);
        else
            dataBaseName = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
        end

        saveDirReach = fullfile(paths.reachResultsPath, dataBaseName);
        if ~exist(saveDirReach, 'dir')
            mkdir(saveDirReach);
        end

        filenameReach = sprintf('hmm_mazz_reach_bin%.3f_minDur%.3f.mat', ...
            binSizeSave, minDurSave);
        filePathReach = fullfile(saveDirReach, filenameReach);
        fprintf('Saving Reach HMM results to:\n%s\n', filePathReach);
        save(filePathReach, 'results', '-v7.3');

        summaryFilename = sprintf('HMM_summary_%s.txt', dataBaseName);
        summaryFilepath = fullfile(saveDirReach, summaryFilename);
        write_hmm_summary(summaryFilepath, sessionType, modelSel, ...
            areas, areasToTest, allResults);
    else
        % Spontaneous data: mirror reach-style per-session structure
        % Use paths.spontaneousResultsPath / <sessionFolder>
        if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
            % Remove any file extension from session name
            [~, dataBaseName, ~] = fileparts(dataStruct.sessionName);
        else
            dataBaseName = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
        end

        saveDirSpont = fullfile(paths.spontaneousResultsPath, dataBaseName);
        if ~exist(saveDirSpont, 'dir')
            mkdir(saveDirSpont);
        end

        filenameNat = sprintf('hmm_mazz_spontaneous_bin%.3f_minDur%.3f.mat', ...
            binSizeSave, minDurSave);
        filePathNat = fullfile(saveDirSpont, filenameNat);
        fprintf('Saving Spontaneous HMM results to:\n%s\n', filePathNat);
        save(filePathNat, 'results', '-v7.3');

        summaryFilename = sprintf('HMM_summary_%s.txt', dataBaseName);
        summaryFilepath = fullfile(saveDirSpont, summaryFilename);
        write_hmm_summary(summaryFilepath, sessionType, modelSel, ...
            areas, areasToTest, allResults);
    end
end

if poolStartedHere
    poolObj = gcp('nocreate');
    if ~isempty(poolObj)
        delete(poolObj);
    end
end

end

function write_hmm_summary(summaryFilepath, sessionType, modelSel, ...
    areas, areasToTest, allResults)
% WRITE_HMM_SUMMARY Write a simple text summary of HMM results per area.

fid = fopen(summaryFilepath, 'w');
if fid == -1
    fprintf('Could not open summary file for writing: %s\n', summaryFilepath);
    return;
end

fprintf(fid, 'HMM Analysis Summary\n');
fprintf(fid, '===================\n\n');
fprintf(fid, 'Analysis Date: %s\n', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
fprintf(fid, 'Data Type: %s\n', sessionType);
fprintf(fid, 'Model Selection Method: %s\n\n', modelSel);

fprintf(fid, 'Areas Analyzed:\n');
for areaIdx = areasToTest
    if ~isempty(allResults.hmm_results{areaIdx})
        fprintf(fid, '  %s: %d states, bin size %.6f s\n', ...
            areas{areaIdx}, allResults.numStates(areaIdx), allResults.binSize(areaIdx));
    else
        fprintf(fid, '  %s: Analysis failed or skipped\n', areas{areaIdx});
    end
end

fclose(fid);
fprintf('Summary saved to:\n%s\n', summaryFilepath);

end

