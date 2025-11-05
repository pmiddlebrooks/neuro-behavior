%% HMM Analysis for Reach Data Only
% Based on hmm_mazz.m, adapted specifically for reach data
% https://github.com/mazzulab/contamineuro_2019_spiking_net/tree/master
%
% This script performs HMM analysis on reach neural data and automatically saves:
% 1. Complete HMM results in .mat format with session-specific naming
% 2. Summary text file with key parameters and results
% 3. All relevant metadata including brain area, trial parameters
% 4. Analysis parameters, trial windows, and HMM model details
%
% Files are saved to: paths.dropPath/reach_data/[sessionName]/
% Naming convention: HMM_results_[BrainArea]_bin[BinSize]_minDur[MinDur].mat
%                   HMM_summary_[BrainArea]_bin[BinSize]_minDur[MinDur].txt

paths = get_paths;

opts = neuro_behavior_options;
opts.minActTime = .16;
opts.minFiringRate = .05;
opts.minFiringRate = .1;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;

monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one


HmmParam=struct();
HmmParam.AdjustT=0.; % interval to skip at trial start to avoid canonical choise of 1st state in matlab
% HmmParam.BinSize=0.002;%0.005; % time step of Markov chain
HmmParam.BinSize=0.002;%0.005; % time step of Markov chain
HmmParam.MinDur=0.05;   % .05 min duration of an admissible state (s) in HMM DECODING
HmmParam.MinP=0.8;      % pstate>MinP for an admissible state in HMM ADMISSIBLE STATES
HmmParam.NumSteps=10;%    %10 number of fits at fixed parameters to avoid non-convexity
HmmParam.singleSeqXval.K = 5; % Cross-validation
HmmParam.NumRuns=50;%     % 50% % number of times we iterate hmmtrain over all trials

opts.HmmParam = HmmParam;




%% Discover all reach data files and process each


reachDir = fullfile(paths.dropPath, 'reach_data');
matFiles = dir(fullfile(reachDir, '*.mat'));
reachDataFiles = cell(1, numel(matFiles));
for i = 1:numel(matFiles)
    reachDataFiles{i} = fullfile(reachDir, matFiles(i).name);
end

reachDataFiles = cell(1);
% reachDataFiles{1} = fullfile(paths.dropPath, 'reach_task/data/Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');
reachDataFiles{1} = fullfile(paths.dropPath, 'reach_task/data/AB2_01-May-2023 15_34_59_NeuroBeh.mat');
% reachDataFiles{1} = fullfile(paths.dropPath, 'reach_task/data/AB2_11-May-2023 17_31_00_NeuroBeh.mat');
% reachDataFiles{2} = fullfile(paths.dropPath, 'reach_task/data/AB2_28-Apr-2023 17_50_02_NeuroBeh.mat');
% reachDataFiles{3} = fullfile(paths.dropPath, 'reach_task/data/AB2_30-May-2023 12_49_52_NeuroBeh.mat');
% reachDataFiles{4} = fullfile(paths.dropPath, 'reach_task/data/AB6_02-Apr-2025 14_18_54_NeuroBeh.mat');
% reachDataFiles{5} = fullfile(paths.dropPath, 'reach_task/data/AB6_03-Apr-2025 13_34_09_NeuroBeh.mat');
% reachDataFiles{6} = fullfile(paths.dropPath, 'reach_task/data/AB6_27-Mar-2025 14_04_12_NeuroBeh.mat');
% reachDataFiles{7} = fullfile(paths.dropPath, 'reach_task/data/AB6_29-Mar-2025 15_21_05_NeuroBeh.mat');
% reachDataFiles{1} = fullfile(paths.dropPath, 'reach_task/data/Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');

reachDataFiles{1} = fullfile(paths.dropPath, 'reach_task/data/reach_test.mat');

areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 1:4; % Indices of areas to test


% Process each reach data file
for fileIdx = 1:numel(reachDataFiles)
    reachDataFile = reachDataFiles{fileIdx};
    fprintf('\n=== Processing Reach Data File: %s ===\n', reachDataFile);
    
    % try
        run_hmm_analysis_for_reach_file(reachDataFile, areas, areasToTest, opts, paths);
    % catch ME
    %     fprintf('Error processing %s: %s\n', reachDataFile, ME.message);
    %     continue;
    % end
end
fprintf('\n=== All reach data files processed ===\n');

function run_hmm_analysis_for_reach_file(reachDataFile, areas, areasToTest, opts, paths)

% Set "trial duration" for reach data (required for the hmm fitting in
% toolbox
trialDur = 30; % "trial" duration in seconds

% Load reach data
dataR = load(reachDataFile);

% Extract session name from filename
[~, sessionName, ~] = fileparts(reachDataFile);

% Create session-specific save directory
saveDir = fullfile(paths.reachResultsPath, sessionName);
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end
% Save all results in a single file
% Create filename with session name
filename = sprintf('hmm_mazz_reach_bin%.3f_minDur%.3f.mat', opts.HmmParam.BinSize, opts.HmmParam.MinDur);
filepath = fullfile(saveDir, filename);


% Set collection parameters based on reach data
opts.collectStart = 0;
opts.collectEnd = round(min(dataR.R(end,1) + 5000, max(dataR.CSV(:,1)*1000)) / 1000);

% Convert reach data to neural matrix format
[dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);

% Find brain area indices
idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));
fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))

idList = {idM23, idM56, idDS, idVS};

% % Find all reach starts
% block1Err = 1;
% block1Corr = 2;
% block2Err = 3;
% block2Corr = 4;
% 
% % First and second block trials
% trialIdx = ismember(dataR.Block(:, 3), 1:4);
% trialIdx1 = ismember(dataR.Block(:, 3), 1:2);
% trialIdx2 = ismember(dataR.Block(:, 3), 3:4);
% 
% % Find when blocks begin and end
% block1Last = find(trialIdx1, 1, 'last');
% block2First = find(trialIdx2, 1);
% block1End = dataR.R(block1Last, 2);
% 
% block2StartFrame = round(dataR.R(block2First, 1) / 1000 / opts.frameSize);
% rStarts = round(dataR.R(trialIdx,1)/opts.frameSize/1000); % in frames
% rStops = round(dataR.R(trialIdx,2)/opts.frameSize/1000); % in frames
% firstFrame = round(opts.collectStart / opts.frameSize);
% lastFrame = round((opts.collectEnd) / opts.frameSize);
% validTrials = rStarts > firstFrame & rStops < lastFrame;
% 
% rStarts = rStarts(validTrials);
% rStops = rStops(validTrials);
% rAcc = dataR.BlockWithErrors(validTrials, 4);


%% Initialize results structure for all areas
results = struct();
results.areas = areas;
results.hmm_results = cell(1, length(areas));
results.binSize = zeros(1, length(areas));
results.minDur = zeros(1, length(areas));
results.numStates = zeros(1, length(areas));
results.logLikelihood = zeros(1, length(areas));
results.numNeurons = zeros(1, length(areas));
results.numTrials = zeros(1, length(areas));
results.analysisStatus = cell(1, length(areas));

%% Loop through each brain area specified in areasToTest
for areaIdx = areasToTest
    tic
    idAreaName = areas{areaIdx};
    idArea = idList{areaIdx};
    dataMatMain = dataMat(:, idArea); % This should be your data
    
    fprintf('\n=== Processing Brain Area: %s (Index %d) ===\n', idAreaName, areaIdx);

    %----------------------------
    % CONVERT DATA TO SPIKE TIMES FORMAT WITH MULTIPLE TRIALS
    %----------------------------

    % Parameters
    binSizeLoaded = opts.frameSize; % seconds
    gnunits = size(dataMatMain, 2); % number of neurons

    % Reach-based trial segmentation: Each trial begins 3 seconds before reach start,
    % ends frame before 3 seconds of following reach
    fprintf('Using REACH-BASED approach: Segmenting data around reach events\n');

    % Extract reach start times (in milliseconds)
    reachStartTimes_ms = dataR.R(:,1); % reach start times in milliseconds
    reachEndTimes_ms = dataR.R(:,2);   % reach end times in milliseconds
    
    % Convert to seconds and sort
    reachStartTimes_s = reachStartTimes_ms / 1000; % convert to seconds
    reachEndTimes_s = reachEndTimes_ms / 1000;     % convert to seconds
    
    % Sort reach times to ensure proper ordering
    [reachStartTimes_s, sortIdx] = sort(reachStartTimes_s);
    reachEndTimes_s = reachEndTimes_s(sortIdx);
    
    % Define trial parameters
    preReachTime = 3.0; % seconds before reach start
    preReachTime = 0.0; % seconds before reach start
    
    % Calculate trial windows based on reach events
    trialWindows = [];
    trialStartTimes = [];
    trialEndTimes = [];
    
    % Process each reach to define trial windows
    for i_reach = 1:length(reachStartTimes_s)
        % Trial starts 3 seconds before this reach
        trialStart = reachStartTimes_s(i_reach) - preReachTime;
        
        % Trial ends frame before 3 seconds of next reach (or 5 seconds after last reach)
        if i_reach < length(reachStartTimes_s)
            % End frame before 3 seconds of next reach
            trialEnd = reachStartTimes_s(i_reach + 1) - preReachTime;
        else
            % For last reach, end 5 seconds after the reach
            trialEnd = reachStartTimes_s(i_reach) + 5.0;
        end
        
        % Only include trials with positive duration
        if trialEnd > trialStart
            trialWindows = [trialWindows; trialStart, trialEnd];
            trialStartTimes = [trialStartTimes; trialStart];
            trialEndTimes = [trialEndTimes; trialEnd];
        end
    end
    
    ntrials = size(trialWindows, 1);
    fprintf('Total time: %.1f seconds\n', opts.collectEnd);
    fprintf('Found %d reaches, created %d trials\n', length(reachStartTimes_s), ntrials);
    fprintf('Trial duration range: %.1f - %.1f seconds\n', min(trialEndTimes - trialStartTimes), max(trialEndTimes - trialStartTimes));

    % % Convert time bin matrix to spike times format with variable-length trials
    % spikes = struct('spk', cell(ntrials, gnunits));
    % win_train = zeros(ntrials, 2); % trial windows [start, end] for each trial
    % 
    % % For each trial
    % for i_trial = 1:ntrials
    %     % Get trial start and end times
    %     trialStart_s = trialStartTimes(i_trial);
    %     trialEnd_s = trialEndTimes(i_trial);
    %     trialDuration_s = trialEnd_s - trialStart_s;
    % 
    %     % Convert to bin indices (1-indexed)
    %     start_bin = round(trialStart_s / binSizeLoaded) + 1;
    %     end_bin = round(trialEnd_s / binSizeLoaded);
    % 
    %     % Ensure we don't exceed data bounds
    %     start_bin = max(1, start_bin);
    %     end_bin = min(end_bin, size(dataMatMain, 1));
    % 
    %     % Skip trials with invalid bounds
    %     if start_bin >= end_bin
    %         fprintf('Warning: Skipping trial %d due to invalid bounds\n', i_trial);
    %         continue;
    %     end
    % 
    %     % Trial window in seconds (relative to trial start)
    %     win_train(i_trial, :) = [0, trialDuration_s];
    % 
    %     % For each neuron in this trial
    %     for i_neuron = 1:gnunits
    %         % Get data for this trial
    %         trial_data = dataMatMain(start_bin:end_bin, i_neuron);
    % 
    %         % Find time bins where this neuron spiked (value > 0)
    %         spike_bins = find(trial_data > 0);
    % 
    %         % Convert bin indices to time in seconds (relative to trial start)
    %         spike_times = (spike_bins - 1) * binSizeLoaded; % -1 because bins are 1-indexed
    % 
    %         % Store spike times for this neuron in this trial
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

    %----------------------------
    % HMM ANALYSIS
    %----------------------------
    fprintf('\n\nRunning HMM on Reach data for area %s\n\n', idAreaName)
    disp('In fun_HMM_modelSel.m, select hhmParam values!!!')

    % Model selection method
    MODELSEL = 'XVAL'; % 'BIC'; % 'AIC';

    % Prepare data structure for HMM analysis
    DATAIN = struct('spikes', spikes, 'win', win_train, 'METHOD', MODELSEL, 'HmmParam', opts.HmmParam);

    % Run HMM analysis
    % try
        myCluster = parcluster('local');
        NumWorkers=min(myCluster.NumWorkers, 5);
        parpool('local', NumWorkers);

        res = hmm.funHMM(DATAIN);

        % OUTPUT OF HMM FIT
        % 'res' is a structure array with fields: 'hmm_data','hmm_bestfit','hmm_results',
        %                                        'hmm_postfit','hmm_multispikes','HmmParam',
        %                                        'win_train','colors','BestStateInd','LLtot'
        % Important fields:
        %     .hmm_bestfit.tpm: K x K transition probability matrix, where K is the number of hidden states
        %     .hmm_bestfit.epm: K x (nunits+1) emission probability matrix, where K is the number of hidden states, the (n+1)-th column represents the probability of silence - you can safely drop it
        %     .hmm_bestfit.LLtrain: -2*loglikelihood of the data
        %
        %     .hmm_results(i_trial).pStates: array of dim [K,time] with posterior probabilities of each state in trial i_trial
        %     .hmm_results(i_trial).rates: array of dim [K,nunits] with local estimate of emissions (i.e., firing rates in each state) conditioned on observations in trial i_trial
        %     .hmm_results(i_trial).Logpseq: -2*loglikelihood from local observations in trial i_trial
        %
        %     .hmm_postfit(i_trial).sequence: array of dimension [4,nseq] where columns represent detected states (intervals with prob(state)>0.8), in the order they appear in trial
        %         i_trial, and rows represent state [onset,offset,duration,label].
        %
        %      HmmParam.VarStates: number of states K chosen with model selection
        %      HmmParam.BinSize: bin size (seconds)
        %      HmmParam.MinDur: shortest duration of detected states; shorter states are excluded from hmm_postfit(i_trial).sequence
        %      HmmParam.MinP: state detection probability
        %      HmmParam.NumSteps: number of independent EM runs used to non-convexity issues
        %      HmmParam.NumRuns: maximum number of iterations each EM runs for

        fprintf('HMM analysis completed successfully!\n');
        fprintf('\t\t\tNumber of states for %s: %d\n', idAreaName, res.HmmParam.VarStates(res.BestStateInd));
        fprintf('Log-likelihood: %.2f\n', res.hmm_bestfit.LLtrain);

        fprintf('HMM completed successfully: Need to save it!\n');

    % catch ME
    %     fprintf('Error in HMM analysis: %s\n', ME.message);
    %     fprintf('You may need to implement the HMM fitting functions separately.\n');
    %     pid = gcp;
    %     delete(pid)
    % 
    %     % Store error status
    %     results.hmm_results{areaIdx} = [];
    %     results.binSize(areaIdx) = NaN;
    %     results.minDur(areaIdx) = NaN;
    %     results.numStates(areaIdx) = NaN;
    %     results.logLikelihood(areaIdx) = NaN;
    %     results.numNeurons(areaIdx) = gnunits;
    %     results.numTrials(areaIdx) = ntrials;
    %     results.analysisStatus{areaIdx} = sprintf('ERROR: %s', ME.message);
    % 
    %     continue % Skip to next area if HMM fails
    % end

    pid = gcp;
    delete(pid)

    % ----------------------------
    % TRANSFORM TRIAL RESULTS BACK TO CONTINUOUS TIME SERIES
    %----------------------------

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

    % ----------------------------
    % STORE RESULTS FOR THIS AREA
    %----------------------------

    % Create comprehensive results structure for this area
    hmm_res = struct();

    % Analysis metadata
    hmm_res.metadata = struct();
    hmm_res.metadata.data_type = 'Reach';
    hmm_res.metadata.brain_area = idAreaName; % Which brain area was analyzed
    hmm_res.metadata.session_name = sessionName;
    hmm_res.metadata.analysis_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    hmm_res.metadata.analysis_status = 'SUCCESS';
    hmm_res.metadata.model_selection_method = MODELSEL;

    % Data parameters
    hmm_res.data_params = struct();
    hmm_res.data_params.num_neurons = gnunits;
    hmm_res.data_params.num_trials = ntrials;
    hmm_res.data_params.bin_size = opts.frameSize;
    hmm_res.data_params.collect_start = opts.collectStart;
    hmm_res.data_params.collect_duration = opts.collectEnd;
    hmm_res.data_params.min_act_time = opts.minActTime;
    hmm_res.data_params.min_firing_rate = opts.minFiringRate;

    % Trial parameters
    hmm_res.trial_params = struct();
    hmm_res.trial_params.trial_windows = win_train; % [start, end] for each trial
    hmm_res.trial_params.trial_start_times = trialStartTimes; % absolute start times
    hmm_res.trial_params.trial_end_times = trialEndTimes; % absolute end times
    hmm_res.trial_params.reach_start_times = reachStartTimes_s; % reach start times in seconds
    hmm_res.trial_params.reach_end_times = reachEndTimes_s; % reach end times in seconds
    hmm_res.trial_params.pre_reach_time = preReachTime; % seconds before reach
    hmm_res.trial_params.trial_duration_range = [min(trialEndTimes - trialStartTimes), max(trialEndTimes - trialStartTimes)];

    % HMM results
    hmm_res.hmm_data = res.hmm_data;
    hmm_res.hmm_results = res.hmm_results;
    hmm_res.hmm_postfit = res.hmm_postfit;
    hmm_res.hmm_multispikes = res.hmm_multispikes;
    hmm_res.HmmParam = res.HmmParam;
    hmm_res.colors = res.colors;
    hmm_res.LLtot = res.LLtot;

    % Best model parameters
    hmm_res.best_model = struct();
    hmm_res.best_model.num_states = res.HmmParam.VarStates(res.BestStateInd);
    hmm_res.best_model.transition_matrix = res.hmm_bestfit.tpm;
    hmm_res.best_model.emission_matrix = res.hmm_bestfit.epm;
    hmm_res.best_model.log_likelihood = res.hmm_bestfit.LLtrain;
    hmm_res.best_model.best_state_index = res.BestStateInd;

    % Store continuous results in the hmm_res structure
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

    % Store results in the main results structure
    results.hmm_results{areaIdx} = hmm_res;
    results.binSize(areaIdx) = res.HmmParam.BinSize;
    results.minDur(areaIdx) = res.HmmParam.MinDur;
    results.numStates(areaIdx) = res.HmmParam.VarStates(res.BestStateInd);
    results.logLikelihood(areaIdx) = res.hmm_bestfit.LLtrain;
    results.numNeurons(areaIdx) = gnunits;
    results.numTrials(areaIdx) = ntrials;
    results.analysisStatus{areaIdx} = 'SUCCESS';

    fprintf('Completed analysis for brain area: %s\t%.1f\n', idAreaName, toc/60);
    
end % End of loop through brain areas





% Save the results
fprintf('Saving all HMM results to: %s\n', filepath);
save(filepath, 'results', '-v7.3');

% Also save a summary text file
summary_filename = sprintf('HMM_summary_%s.txt', sessionName);
summary_filepath = fullfile(saveDir, summary_filename);

% Create summary file
fid = fopen(summary_filepath, 'w');
if fid ~= -1
    fprintf(fid, 'HMM Analysis Summary\n');
    fprintf(fid, '===================\n\n');
    fprintf(fid, 'Analysis Date: %s\n', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
    fprintf(fid, 'Data Type: Reach\n');
    fprintf(fid, 'Session Name: %s\n', sessionName);
    fprintf(fid, 'Model Selection Method: %s\n', MODELSEL);
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

fprintf('\n=== All brain area analyses completed for session: %s ===\n', sessionName);

end
