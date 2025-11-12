%% HMM Analysis for Naturalistic Data Only
% Mirrors hmm_mazz_reach.m but for naturalistic data
% Loads naturalistic data using get_standard_data and segments into fixed-duration trials
% before HMM fitting. Aggregates results across brain areas and saves a single results file.

paths = get_paths;

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.minFiringRate = .05;
opts.minFiringRate = .2;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;

monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :);
monitorTwo = monitorPositions(size(monitorPositions, 1), :);

% HMM parameter overrides (can be edited later)
HmmParam = struct();
HmmParam.AdjustT = 0.0;
HmmParam.BinSize = 0.002;
HmmParam.MinDur = 0.05;
HmmParam.MinP = 0.8;
HmmParam.NumSteps = 10;
HmmParam.NumRuns = 50;
HmmParam.singleSeqXval.K = 5; % Cross-validation
opts.HmmParam = HmmParam;

% Which data to analyze
natOrReach = 'Nat';
areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 1:4;

% Naturalistic data parameters
opts.collectStart = 0 * 60 * 60;
opts.collectEnd = 45 * 60;
opts.collectEnd = 5 * 60;

% Load naturalistic data using helper from hmm_mazz.m
getDataType = 'spikes';
get_standard_data

% At this point we expect variables: dataMat, idLabels, areaLabels, dataBhv

% Build area index lists from labels
idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));

idList = {idM23, idM56, idDS, idVS};

%% Event-based trial segmentation using behavioral starts from dataBhv
% Define behavior and segmentation parameters
bhvStartID = 15;          % behavior ID that marks trial starts
minBhvDur = 0.5;        % seconds (minimum behavior duration)
preBhvTime = -.2;       % seconds before behavior start to include (can be negative for after behavior start)
minTrialDur = 5.0;      % seconds (minimum separation/length between trial starts)


% Identify behavior onsets matching bhvStartID with sufficient duration
isStart = (dataBhv.ID == bhvStartID) & (dataBhv.Dur >= minBhvDur);
candidateStarts = dataBhv.StartTime(isStart); % seconds

% Enforce minimum spacing between trials (seconds)
candidateStarts = sort(candidateStarts(:));
if ~isempty(candidateStarts)
    keepIdx = true(size(candidateStarts));
    lastKept = -inf;
    for i = 1:numel(candidateStarts)
        if candidateStarts(i) - lastKept < minTrialDur
            keepIdx(i) = false;
        else
            lastKept = candidateStarts(i);
        end
    end
    trialStartTimes = candidateStarts(keepIdx) + preBhvTime; % shift by preBhvTime
else
    trialStartTimes = [];
end

% Define trial end times: frame before next trial starts (1 ms prior);
% last trial ends at last data sample
binSizeLoaded = opts.frameSize; % seconds per bin (assume 1 ms)
numTimeBins = size(dataMat, 1);
totalTime = numTimeBins * binSizeLoaded;

if ~isempty(trialStartTimes)
    nextStartTimes = [trialStartTimes(2:end); totalTime];
    trialEndTimes = nextStartTimes - 0.001; % 1 ms before next start
else
    trialEndTimes = [];
end

% Prepare results structure
results = struct();
results.data_type = 'Nat';
results.areas = areas;
results.hmm_results = cell(1, length(areas));
results.binSize = zeros(1, length(areas));
results.minDur = zeros(1, length(areas));
results.numStates = zeros(1, length(areas));
results.logLikelihood = zeros(1, length(areas));
results.numNeurons = zeros(1, length(areas));
results.numTrials = zeros(1, length(areas));
results.analysisStatus = cell(1, length(areas));

% Output directory and file
hmmdir = fullfile(paths.dropPath, 'metastability');
if ~exist(hmmdir, 'dir')
    mkdir(hmmdir);
end
% filename = sprintf('hmm_mazz_nat_bin%.3f_minDur%.3f_bhv%d_offset%.3f.mat', opts.HmmParam.BinSize, opts.HmmParam.MinDur, bhvStartID, preBhvTime);
filename = sprintf('hmm_mazz_nat_bin%.3f_minDur%.3f.mat', opts.HmmParam.BinSize, opts.HmmParam.MinDur);
filepath = fullfile(hmmdir, filename);

% Loop through areas
for areaIdx = areasToTest
    tic
    idAreaName = areas{areaIdx};
    idArea = idList{areaIdx};
    dataMatMain = dataMat(:, idArea);

    fprintf('\n=== Processing Brain Area (Nat): %s (Index %d) ===\n', idAreaName, areaIdx);

    % Segment continuous data into behavior-based trials (seconds)
    binSizeLoaded = opts.frameSize;
    gnunits = size(dataMatMain, 2);
    totalTime = size(dataMatMain, 1) * binSizeLoaded;

    % Filter trials that are within data bounds and meet minTrialDur (by seconds)
    validTrialMask = (trialStartTimes >= 0) & (trialEndTimes > trialStartTimes) & ((trialEndTimes - trialStartTimes) >= minTrialDur) & (trialEndTimes <= totalTime);
    trialStartTimesValid = trialStartTimes(validTrialMask);
    trialEndTimesValid = trialEndTimes(validTrialMask);
    ntrials = numel(trialStartTimesValid);
    fprintf('Total time: %.1f seconds\n', totalTime);
    fprintf('Creating %d behavior-based trials (min dur %.1f s)\n', ntrials, minTrialDur);

    % spikes = struct('spk', cell(ntrials, gnunits));
    % win_train = zeros(ntrials, 2);
    % 
    % for i_trial = 1:ntrials
    %     trialStart_s = trialStartTimesValid(i_trial);
    %     trialEnd_s = trialEndTimesValid(i_trial);
    %     start_bin = max(1, round(trialStart_s / binSizeLoaded) + 1);
    %     end_bin = min(size(dataMatMain, 1), round(trialEnd_s / binSizeLoaded));
    %     trialDuration_s = trialEnd_s - trialStart_s;
    %     if end_bin <= start_bin
    %         continue;
    %     end
    %     win_train(i_trial, :) = [0, trialDuration_s];
    %     for i_neuron = 1:gnunits
    %         trial_data = dataMatMain(start_bin:end_bin, i_neuron);
    %         spike_bins = find(trial_data > 0);
    %         spike_times = (spike_bins - 1) * binSizeLoaded;
    %         spikes(i_trial, i_neuron).spk = spike_times;
    %     end
    % end

    spikes = struct('spk', cell(1, gnunits));
        trialStart_s = 0;
        trialEnd_s = size(dataMatMain, 1) * opts.frameSize;
        start_bin = max(1, round(trialStart_s / binSizeLoaded) + 1);
        end_bin = min(size(dataMatMain, 1), round(trialEnd_s / binSizeLoaded));
        trialDuration_s = trialEnd_s - trialStart_s;
        if end_bin <= start_bin
            continue;
        end
        win_train = [0, trialDuration_s];
        for i_neuron = 1:gnunits
            trial_data = dataMatMain(start_bin:end_bin, i_neuron);
            spike_bins = find(trial_data > 0);
            spike_times = (spike_bins - 1) * binSizeLoaded;
            spikes(1, i_neuron).spk = spike_times;
        end

    % HMM analysis
    fprintf('\n\nRunning HMM on Nat data for area %s\n\n', idAreaName)
    disp('In fun_HMM_modelSel.m, select hmmParam values!!!')

    MODELSEL = 'XVAL';
    DATAIN = struct('spikes', spikes, 'win', win_train, 'METHOD', MODELSEL, 'HmmParam', opts.HmmParam);

     % try
        myCluster = parcluster('local');
        NumWorkers = min(myCluster.NumWorkers, 6);
        parpool('local', NumWorkers);

        res = hmm.funHMM(DATAIN);

        fprintf('HMM analysis completed successfully!\n');
        fprintf('\t\t\tNumber of states for %s: %d\n', idAreaName, res.HmmParam.VarStates(res.BestStateInd));
        fprintf('Log-likelihood: %.2f\n', res.hmm_bestfit.LLtrain);
    % catch ME
    %     fprintf('Error in HMM analysis: %s\n', ME.message);
    %     pid = gcp; delete(pid)
    %     results.hmm_results{areaIdx} = [];
    %     results.binSize(areaIdx) = NaN;
    %     results.minDur(areaIdx) = NaN;
    %     results.numStates(areaIdx) = NaN;
    %     results.logLikelihood(areaIdx) = NaN;
    %     results.numNeurons(areaIdx) = gnunits;
    %     results.numTrials(areaIdx) = ntrials;
    %     results.analysisStatus{areaIdx} = sprintf('ERROR: %s', ME.message);
    %     continue
    % end

    pid = gcp; delete(pid)

    % ----------------------------
    % TRANSFORM TRIAL RESULTS BACK TO CONTINUOUS TIME SERIES
    %----------------------------
%
    if ~isempty(res) && isfield(res, 'hmm_results') && isfield(res, 'hmm_postfit')
        fprintf('Transforming trial results back to continuous time series...\n');

        % Get dimensions
        numTrials = length(res.hmm_results);
        numStates = size(res.hmm_results(1).pStates, 1);
        numNeurons = size(res.hmm_results(1).rates, 2);
        
        % Calculate total time bins for variable-length trials
        totalTimeBins = 0;
        trialTimeBins = zeros(numTrials, 1);
        for iTrial = 1:numTrials
            trialTimeBins(iTrial) = size(res.hmm_results(iTrial).pStates, 2);
            totalTimeBins = totalTimeBins + trialTimeBins(iTrial);
        end

        % Initialize continuous arrays
        continuous_pStates = zeros(totalTimeBins, numStates);
        continuous_rates = zeros(totalTimeBins, numStates, numNeurons);
        continuous_sequence = zeros(totalTimeBins, 1);

        % Concatenate results from each trial
        currentBin = 1;
        for iTrial = 1:numTrials
            % Get number of time bins for this specific trial
            numTimePerTrial = trialTimeBins(iTrial);
            
            % Time indices for this trial based on current position
            timeIdx = currentBin:(currentBin + numTimePerTrial - 1);

            % Copy posterior state probabilities
            continuous_pStates(timeIdx, :) = res.hmm_results(iTrial).pStates';
            
            % Update current bin position for next trial
            currentBin = currentBin + numTimePerTrial;
        end

        % Compute continuous_sequence from continuous_pStates using MinP threshold
        % For each time bin, find the state with highest probability
        for iBin = 1:totalTimeBins
            stateProbs = continuous_pStates(iBin,:);
            [maxProb, maxState] = max(stateProbs);

            % Assign state if probability exceeds MinP threshold
            if maxProb >= res.HmmParam.MinP
                continuous_sequence(iBin) = maxState;
            end
            % Otherwise, keep as NaN (no confident state assignment)
        end


        fprintf('Successfully transformed results to continuous format\n');
        fprintf('Total time bins: %d (%.1f seconds)\n', totalTimeBins, totalTimeBins * res.HmmParam.BinSize);
    end

    % ----------------------------
    % METASTATE DETECTION
    %----------------------------
    
    if ~isempty(res) && isfield(res, 'hmm_bestfit') && isfield(res.hmm_bestfit, 'tpm')
        fprintf('Detecting metastates from transition probability matrix...\n');
        
        % Get transition probability matrix
        tpm = res.hmm_bestfit.tpm;
        numStates = size(tpm, 1);
        
        fprintf('Transition probability matrix size: %dx%d\n', numStates, numStates);
        
        % Detect metastate communities using Vidaurre method
        try
            communities = detect_metastates_vidaurre(tpm, true); % verbose output
            
            % Create continuous metastates sequence
            continuous_metastates = zeros(size(continuous_sequence));
            
            % Map each state to its metastate
            for iBin = 1:length(continuous_sequence)
                state = continuous_sequence(iBin);
                if ~isnan(state) && state > 0 && state <= numStates
                    continuous_metastates(iBin) = communities(state);
                else
                    continuous_metastates(iBin) = 0; % Keep undefined states as 0
                end
            end
            
            % Get unique metastates and their composition
            uniqueMetastates = unique(communities);
            numMetastates = length(uniqueMetastates);
            
            fprintf('\t\t\tNumber of metastates: %d (from %d states)\n', numMetastates, numStates);
            for m = 1:numMetastates
                metastateLabel = uniqueMetastates(m);
                statesInMetastate = find(communities == metastateLabel);
                fprintf('  Metastate %d: states [%s]\n', metastateLabel, num2str(statesInMetastate));
            end
            
            fprintf('Successfully created continuous metastates sequence\n');
            
        catch ME
            fprintf('Error in metastate detection: %s\n', ME.message);
            fprintf('Skipping metastate analysis for area %s\n', idAreaName);
            
            % Set empty metastate results
            communities = [];
            continuous_metastates = [];
            numMetastates = 0;
        end
    else
        fprintf('No transition probability matrix available for metastate detection\n');
        communities = [];
        continuous_metastates = [];
        numMetastates = 0;
    end

    % Package per-area results
    hmm_res = struct();
    hmm_res.metadata = struct();
    hmm_res.metadata.data_type = 'Nat';
    hmm_res.metadata.brain_area = idAreaName;
    hmm_res.metadata.analysis_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    hmm_res.metadata.analysis_status = 'SUCCESS';
    hmm_res.metadata.model_selection_method = MODELSEL;

    hmm_res.data_params = struct();
    hmm_res.data_params.num_neurons = gnunits;
    hmm_res.data_params.num_trials = ntrials;
    hmm_res.data_params.bin_size = opts.frameSize;
    hmm_res.data_params.collect_start = opts.collectStart;
    hmm_res.data_params.collect_duration = opts.collectEnd;
    hmm_res.data_params.min_act_time = opts.minActTime;
    hmm_res.data_params.min_firing_rate = opts.minFiringRate;

    hmm_res.trial_params = struct();
    hmm_res.trial_params.trial_windows = win_train;
    hmm_res.trial_params.trial_start_times = trialStartTimesValid;
    hmm_res.trial_params.trial_end_times = trialEndTimesValid;
    hmm_res.trial_params.min_trial_duration = minTrialDur;
    hmm_res.trial_params.bhv_start_id = bhvStartID;
    hmm_res.trial_params.min_bhv_duration = minBhvDur;
    hmm_res.trial_params.pre_bhv_time = preBhvTime;

    hmm_res.hmm_data = res.hmm_data;
    hmm_res.hmm_results = res.hmm_results;
    hmm_res.hmm_postfit = res.hmm_postfit;
    hmm_res.hmm_multispikes = res.hmm_multispikes;
    hmm_res.HmmParam = res.HmmParam;
    hmm_res.colors = res.colors;
    hmm_res.LLtot = res.LLtot;

    hmm_res.best_model = struct();
    hmm_res.best_model.num_states = res.HmmParam.VarStates(res.BestStateInd);
    hmm_res.best_model.transition_matrix = res.hmm_bestfit.tpm;
    hmm_res.best_model.emission_matrix = res.hmm_bestfit.epm;
    hmm_res.best_model.log_likelihood = res.hmm_bestfit.LLtrain;
    hmm_res.best_model.best_state_index = res.BestStateInd;

    hmm_res.continuous_results = struct();
    hmm_res.continuous_results.pStates = continuous_pStates;
    hmm_res.continuous_results.sequence = continuous_sequence;
    hmm_res.continuous_results.totalTime = totalTimeBins * res.HmmParam.BinSize;
    
    % Store metastate results
    hmm_res.metastate_results = struct();
    hmm_res.metastate_results.communities = communities;
    hmm_res.metastate_results.continuous_metastates = continuous_metastates;
    hmm_res.metastate_results.num_metastates = numMetastates;
    if exist('tpm', 'var')
        hmm_res.metastate_results.transition_matrix = tpm;
    else
        hmm_res.metastate_results.transition_matrix = [];
    end

    % Store in results
    results.hmm_results{areaIdx} = hmm_res;
    results.binSize(areaIdx) = res.HmmParam.BinSize;
    results.minDur(areaIdx) = res.HmmParam.MinDur;
    results.numStates(areaIdx) = res.HmmParam.VarStates(res.BestStateInd);
    results.logLikelihood(areaIdx) = res.hmm_bestfit.LLtrain;
    results.numNeurons(areaIdx) = gnunits;
    results.numTrials(areaIdx) = ntrials;
    results.analysisStatus{areaIdx} = 'SUCCESS';

    fprintf('Completed Nat analysis for area: %s\t%.1f\n', idAreaName, toc/60);
end

% Save aggregated results (single file without timestamp)
fprintf('Saving all HMM Nat results to: %s\n', filepath);
save(filepath, 'results', '-v7.3');

% Save a summary text file
% summary_filename = sprintf('HMM_summary_Nat_bin%.3f_minDur%.3f_bhv%d_offset%.3f.txt', opts.HmmParam.BinSize, opts.HmmParam.MinDur, bhvStartID, preBhvTime);
summary_filename = sprintf('HMM_summary_Nat_bin%.3f_minDur%.3f.txt', opts.HmmParam.BinSize, opts.HmmParam.MinDur);
summary_filepath = fullfile(hmmdir, summary_filename);
fid = fopen(summary_filepath, 'w');
if fid ~= -1
    fprintf(fid, 'HMM Analysis Summary (Naturalistic)\n');
    fprintf(fid, '===================================\n\n');
    fprintf(fid, 'Analysis Date: %s\n', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
    fprintf(fid, 'Data Type: Nat\n');
    fprintf(fid, '\n');
    fprintf(fid, 'Brain Area Results:\n');
    for areaIdx = areasToTest
        idAreaName = areas{areaIdx};
        fprintf(fid, '  %s:\n', idAreaName);
        if strcmp(results.analysisStatus{areaIdx}, 'SUCCESS')
            fprintf(fid, '    Status: SUCCESS\n');
            fprintf(fid, '    Number of neurons: %d\n', results.numNeurons(areaIdx));
            fprintf(fid, '    Number of trials: %d\n', results.numTrials(areaIdx));
            fprintf(fid, '    Number of states: %d\n', results.numStates(areaIdx));
            fprintf(fid, '    Log-likelihood: %.2f\n', results.logLikelihood(areaIdx));
            fprintf(fid, '    Bin size: %.6f seconds\n', results.binSize(areaIdx));
            fprintf(fid, '    Min duration: %.6f seconds\n', results.minDur(areaIdx));
        else
            fprintf(fid, '    Status: %s\n', results.analysisStatus{areaIdx});
        end
        fprintf(fid, '\n');
    end
    fprintf(fid, 'Files saved:\n');
    fprintf(fid, '  Results: %s\n', filename);
    fprintf(fid, '  Summary: %s\n', summary_filename);
    fclose(fid);
    fprintf('Summary saved to: %s\n', summary_filepath);
end

fprintf('\n=== Naturalistic HMM analysis completed ===\n');


