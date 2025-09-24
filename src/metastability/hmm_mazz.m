%% HMM Analysis for Naturalistic and Reach Data
% Based on demo3_HMM_Full.m, adapted for naturalistic data format
% https://github.com/mazzulab/contamineuro_2019_spiking_net/tree/master
%
% This script performs HMM analysis on neural data and automatically saves:
% 1. Complete HMM results in .mat format with timestamp
% 2. Summary text file with key parameters and results
% 3. All relevant metadata including data type (Nat/Reach), brain area, trial parameters
% 4. Analysis parameters, trial windows, and HMM model details
%
% Files are saved to: paths.dropPath/hmm/
% Naming convention: HMM_results_[DataType]_[BrainArea]_[Timestamp].mat
%                   HMM_summary_[DataType]_[BrainArea]_[Timestamp].txt

paths = get_paths;
cd(fullfile(paths.homePath, 'toolboxes/contamineuro_2019_spiking_net/'));
%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.minFiringRate = .05;
opts.minFiringRate = .1;
opts.frameSize = .001;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 45 * 60; % seconds
opts.firingRateCheckTime = 5 * 60;

monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one




%%           ==========================         WHICH DATA DO YOU WANT TO ANALYZE?        =================================

natOrReach = 'Reach'; % 'Nat'  'Reach'

areas = {'M23', 'M56', 'DS', 'VS'};
areasToTest = 3:4; % Indices of areas to test

switch natOrReach
    case 'Nat'
        % Naturalistic data
        getDataType = 'spikes';
        get_standard_data
    case 'Reach'
        % Mark's reach data
        % dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));
        dataR = load(fullfile(paths.dropPath, 'reach_data/Copy_of_Y4_100623_Spiketimes_idchan_BEH.mat'));

        [dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);

        idM23 = find(strcmp(areaLabels, 'M23'));
        idM56 = find(strcmp(areaLabels, 'M56'));
        idDS = find(strcmp(areaLabels, 'DS'));
        idVS = find(strcmp(areaLabels, 'VS'));
    fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', length(idM23), length(idM56), length(idDS), length(idVS))
end

idList = {idM23, idM56, idDS, idVS};

%% Loop through each brain area specified in areasToTest
for areaIdx = areasToTest
    idAreaName = areas{areaIdx};
    idArea = idList{areaIdx};
    dataMatMain = dataMat(:, idArea); % This should be your data
    
    fprintf('\n=== Processing Brain Area: %s (Index %d) ===\n', idAreaName, areaIdx);


    %  Nat or Reach-specific variables

    switch natOrReach
        case 'Reach'
            % preTime = 4; % s before reach onset
            % postTime = 3;

            % Find all reach starts
            block1Err = 1;
            block1Corr = 2;
            block2Err = 3;
            block2Corr = 4;

            % First and second block trials
            trialIdx = ismember(dataR.block(:, 3), 1:4);
            trialIdx1 = ismember(dataR.block(:, 3), 1:2);
            trialIdx2 = ismember(dataR.block(:, 3), 3:4);

            % Find when blocks begin and end
            block1Last = find(trialIdx1, 1, 'last');
            block2First = find(trialIdx2, 1);
            block1End = dataR.R(block1Last, 2);

            block2StartFrame = round(dataR.R(block2First, 1) / 1000 / opts.frameSize);
            rStarts = round(dataR.R(trialIdx,1)/opts.frameSize/1000); % in frames
            rStops = round(dataR.R(trialIdx,2)/opts.frameSize/1000); % in frames
            firstFrame = round(opts.collectStart / opts.frameSize);
            lastFrame = round((opts.collectStart + opts.collectFor) / opts.frameSize);
            validTrials = rStarts > firstFrame & rStops < lastFrame;

            rStarts = rStarts(validTrials);
            rStops = rStops(validTrials);
            rAcc = dataR.BlockWithErrors(validTrials, 4);

            % Set trial duration for reach data (same as natural data for unified approach)
            trialDur = 20; % "trial" duration in seconds

        case 'Nat'
            % Parameters for natural trials
            trialDur = 20; % "trial" duration in seconds
    end

    %
    %----------------------------
    % CONVERT DATA TO SPIKE TIMES FORMAT WITH MULTIPLE TRIALS
    %----------------------------

    % Parameters
    binSizeLoaded = opts.frameSize; % seconds
    gnunits = size(dataMatMain, 2); % number of neurons

    % Unified approach: Segment continuous data by trialDur for both Nat and Reach
    fprintf('Using UNIFIED approach: Segmenting continuous data by trialDur\n');

    % Calculate number of complete trials
    totalTime = size(dataMatMain, 1) * binSizeLoaded; % total time in seconds
    ntrials = floor(totalTime / trialDur); % number of complete trials

    fprintf('Total time: %.1f seconds\n', totalTime);
    fprintf('Creating %d trials of %.1f seconds each\n', ntrials, trialDur);

    % Convert time bin matrix to spike times format with multiple trials
    spikes = struct('spk', cell(ntrials, gnunits));
    win_train = zeros(ntrials, 2); % trial windows [start, end] for each trial

    % For each trial (same approach for both Nat and Reach)
    for i_trial = 1:ntrials
        % Calculate trial start and end bins
        start_bin = (i_trial - 1) * trialDur / binSizeLoaded + 1;
        end_bin = i_trial * trialDur / binSizeLoaded;

        % Ensure we don't exceed data bounds
        end_bin = min(end_bin, size(dataMatMain, 1));

        % Trial window in seconds (relative to trial start)
        win_train(i_trial, :) = [0, trialDur];

        % For each neuron in this trial
        for i_neuron = 1:gnunits
            % Get data for this trial
            trial_data = dataMatMain(start_bin:end_bin, i_neuron);

            % Find time bins where this neuron spiked (value > 0)
            spike_bins = find(trial_data > 0);

            % Convert bin indices to time in seconds (relative to trial start)
            spike_times = (spike_bins - 1) * binSizeLoaded; % -1 because bins are 1-indexed

            % Store spike times for this neuron in this trial
            spikes(i_trial, i_neuron).spk = spike_times;
        end
    end

    %
    %----------------------------
    % HMM ANALYSIS
    %----------------------------
    fprintf('\n\nRunning HMM on %s for area %s\n\n', natOrReach, idAreaName)
    disp('In fun_HMM_modelSel.m, select hhmParam values!!!')

    % Model selection method
    MODELSEL = 'XVAL'; % 'BIC'; % 'AIC';

    % Setup directories
    hmmdir = fullfile(paths.dropPath, 'metastability');
    if ~exist(hmmdir, 'dir')
        mkdir(hmmdir);
    end

    % Prepare data structure for HMM analysis
    DATAIN = struct('spikes', spikes, 'win', win_train, 'METHOD', MODELSEL);

    % Run HMM analysis
    % Note: This assumes you have the hmm.funHMM function available
    % If not, you'll need to implement the HMM fitting separately
    try
        myCluster = parcluster('local');
        NumWorkers=min(myCluster.NumWorkers, 6);
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
        fprintf('Number of states: %d\n', res.HmmParam.VarStates(res.BestStateInd));
        fprintf('Log-likelihood: %.2f\n', res.hmm_bestfit.LLtrain);

        fprintf('HMM completed successfully: Need to save it!\n');


    catch ME
        fprintf('Error in HMM analysis: %s\n', ME.message);
        fprintf('You may need to implement the HMM fitting functions separately.\n');
        pid = gcp;
        delete(pid)
        continue % Skip to next area if HMM fails
    end


    pid = gcp;
    delete(pid)


    % ----------------------------
    % TRANSFORM TRIAL RESULTS BACK TO CONTINUOUS TIME SERIES
    %----------------------------

    if ~isempty(res) && isfield(res, 'hmm_results') && isfield(res, 'hmm_postfit')
        fprintf('Transforming trial results back to continuous time series...\n');

        % Get dimensions
        numTrials = length(res.hmm_results);
        numTimePerTrial = size(res.hmm_results(1).pStates, 2);
        numStates = size(res.hmm_results(1).pStates, 1);
        numNeurons = size(res.hmm_results(1).rates, 2);
        totalTimeBins = numTrials * numTimePerTrial;

        % Initialize continuous arrays
        continuous_pStates = zeros(totalTimeBins, numStates);
        continuous_rates = zeros(totalTimeBins, numStates, numNeurons);
        continuous_sequence = zeros(totalTimeBins, 1);

        % Concatenate results from each trial
        for iTrial = 1:numTrials
            % Time indices for this trial
            timeIdx = ((iTrial-1)*numTimePerTrial + 1):(iTrial*numTimePerTrial);

            % Copy posterior state probabilities
            continuous_pStates(timeIdx, :) = res.hmm_results(iTrial).pStates';

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
    % SAVE HMM RESULTS
    %----------------------------

    % Create comprehensive results structure
    hmm_res = struct();

    % Analysis metadata
    hmm_res.metadata = struct();
    hmm_res.metadata.data_type = natOrReach; % 'Nat' or 'Reach'
    hmm_res.metadata.brain_area = idAreaName; % Which brain area was analyzed
    hmm_res.metadata.analysis_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    hmm_res.metadata.analysis_status = 'SUCCESS';
    hmm_res.metadata.model_selection_method = MODELSEL;

    % Data parameters
    hmm_res.data_params = struct();
    hmm_res.data_params.num_neurons = gnunits;
    hmm_res.data_params.num_trials = ntrials;
    hmm_res.data_params.bin_size = opts.frameSize;
    hmm_res.data_params.collect_start = opts.collectStart;
    hmm_res.data_params.collect_duration = opts.collectFor;
    hmm_res.data_params.min_act_time = opts.minActTime;
    hmm_res.data_params.min_firing_rate = opts.minFiringRate;

    % Trial parameters
    hmm_res.trial_params = struct();
    hmm_res.trial_params.trial_windows = win_train; % [start, end] for each trial
    hmm_res.trial_params.trial_duration = trialDur;

    % Add reach-specific parameters if applicable
    if strcmp(natOrReach, 'Reach')
        hmm_res.trial_params.reach_starts = rStarts;
        hmm_res.trial_params.reach_stops = rStops;
        hmm_res.trial_params.reach_accuracy = rAcc;
        hmm_res.trial_params.valid_trials = validTrials;
    end

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

    % Create filename without timestamp
    filename = sprintf('HMM_results_%s_%s.mat', natOrReach, idAreaName);
    filepath = fullfile(hmmdir, filename);

    % Save the results
    fprintf('Saving HMM results to: %s\n', filepath);
    save(filepath, 'hmm_res', '-v7.3');

    % Also save a summary text file
    summary_filename = sprintf('HMM_summary_%s_%s.txt', natOrReach, idAreaName);
    summary_filepath = fullfile(hmmdir, summary_filename);

    % Create summary file
    fid = fopen(summary_filepath, 'w');
    if fid ~= -1
        fprintf(fid, 'HMM Analysis Summary\n');
        fprintf(fid, '===================\n\n');
        fprintf(fid, 'Analysis Date: %s\n', hmm_res.metadata.analysis_date);
        fprintf(fid, 'Data Type: %s\n', hmm_res.metadata.data_type);
        fprintf(fid, 'Brain Area: %s\n', hmm_res.metadata.brain_area);
        fprintf(fid, 'Model Selection Method: %s\n', hmm_res.metadata.model_selection_method);
        fprintf(fid, '\n');
        fprintf(fid, 'Data Parameters:\n');
        fprintf(fid, '  Number of neurons: %d\n', hmm_res.data_params.num_neurons);
        fprintf(fid, '  Number of trials: %d\n', hmm_res.data_params.num_trials);
        fprintf(fid, '  Frame size: %.6f seconds\n', opts.frameSize);
        fprintf(fid, '  Collection start: %.1f seconds\n', hmm_res.data_params.collect_start);
        fprintf(fid, '  Collection duration: %.1f seconds\n', hmm_res.data_params.collect_duration);
        fprintf(fid, '\n');
        fprintf(fid, 'Trial Parameters:\n');
        fprintf(fid, '  Trial duration: %.1f seconds\n', hmm_res.trial_params.trial_duration);
        % if strcmp(natOrReach, 'Reach')
        %     fprintf(fid, '  Pre-reach time: %.1f seconds\n', hmm_res.trial_params.pre_time);
        %     fprintf(fid, '  Post-reach time: %.1f seconds\n', hmm_res.trial_params.post_time);
        % end
        fprintf(fid, '\n');
        fprintf(fid, 'HMM Results:\n');
        fprintf(fid, '  Number of states: %d\n', hmm_res.best_model.num_states);
        fprintf(fid, '  Log-likelihood: %.2f\n', hmm_res.best_model.log_likelihood);
        fprintf(fid, '  Best state index: %d\n', hmm_res.best_model.best_state_index);
        fprintf(fid, '\n');
        fprintf(fid, 'Files saved:\n');
        fprintf(fid, '  Results: %s\n', filename);
        fprintf(fid, '  Summary: %s\n', summary_filename);
        fclose(fid);

        fprintf('Summary saved to: %s\n', summary_filepath);
    end

    fprintf('HMM analysis saved successfully!\n');

    %% Plot full sequence
    figure(8); clf;
    hold on;

    x = (1:length(continuous_sequence)) * res.HmmParam.BinSize;
    numStates = size(continuous_pStates, 2);

    % Get colors for states
  func = @sRGB_to_OKLab;
  cOpts.exc = [0,0,0;1,1,1];
  colors = maxdistcolor(numStates,func, cOpts);
    % colors = distinguishable_colors(numStates, {'w','k'}, func);

    % Plot rectangles for each state
    for state = 1:numStates
        stateIdx = continuous_sequence == state;
        if any(stateIdx)
            % Find contiguous segments
            d = diff([0; stateIdx; 0]);
            startIdx = find(d == 1);
            endIdx = find(d == -1) - 1;

            % Plot each segment as a rectangle
            for i = 1:length(startIdx)
                xStart = x(startIdx(i));
                xEnd = x(endIdx(i));
                patch([xStart xEnd xEnd xStart], [0 0 1 1], colors(state,:), ...
                    'FaceAlpha', 0.2, 'EdgeColor', 'none');
            end
        end
    end

    % Plot probability traces
    for state = 1:numStates
        plot(x, continuous_pStates(:,state), 'Color', colors(state,:), 'LineWidth', 2);
    end

    xlabel('Time (s)');
    ylabel('State Probability');
    ylim([0 1]);
    hold off;

    fprintf('Completed analysis for brain area: %s\n', idAreaName);
    
end % End of loop through brain areas

fprintf('\n=== All brain area analyses completed ===\n');










% % Helper function for distinguishable colors (if not available)
% function colors = distinguishable_colors(n)
% % Simple color generation if distinguishable_colors is not available
% if n <= 7
%     colors = lines(n);
% else
%     colors = hsv(n);
% end
% end
% 



