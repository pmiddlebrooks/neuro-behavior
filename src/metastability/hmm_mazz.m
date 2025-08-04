%% HMM Analysis for Naturalistic Data
% Based on demo3_HMM_Full.m, adapted for naturalistic data format

paths = get_paths;
cd(fullfile(paths.homePath, 'toolboxes/contamineuro_2019_spiking_net/'));
%%
opts = neuro_behavior_options;
opts.minActTime = .16;
opts.minFiringRate = .05;
opts.frameSize = .001;
opts.collectStart = 0 * 60 * 60; % seconds
opts.collectFor = 10 * 60; % seconds
opts.firingRateCheckTime = 5 * 60;

monitorPositions = get(0, 'MonitorPositions');
monitorOne = monitorPositions(1, :); % Just use single monitor if you don't have second one
monitorTwo = monitorPositions(size(monitorPositions, 1), :); % Just use single monitor if you don't have second one



%%           ==========================         WHICH DATA DO YOU WANT TO ANALYZE?        =================================
% Naturalistic data
getDataType = 'spikes';
get_standard_data

%% Mark's reach data
dataR = load(fullfile(paths.dropPath, 'reach_data/Y4_100623_Spiketimes_idchan_BEH.mat'));

[dataMatR, idLabels, areaLabels, rmvNeurons] = neural_matrix_mark_data(dataR, opts);

idM23 = find(strcmp(areaLabels, 'M23'));
idM56 = find(strcmp(areaLabels, 'M56'));
idDS = find(strcmp(areaLabels, 'DS'));
idVS = find(strcmp(areaLabels, 'VS'));

%%
preTime = 4; % s before reach onset
postTime = 3;

% Find all reach starts
block1Err = 1;
block1Corr = 2;
block2Err = 3;
block2Corr = 4;

% First and second block trials
trialIdx = ismember(dataR.block(:, 3), 1:4);
trialIdx1 = ismember(dataR.block(:, 3), 1:2);
trialIdx2 = ismember(dataR.block(:, 3), 3:4);

rStarts = round(dataR.R(trialIdx,1)/opts.frameSize/1000);

    for i = 1 : length(rStarts)    %
        reachWindow = rStarts(i) - floor(preTime/opts.frameSize) : rStarts(i) + floor(postTime/opts.frameSize) - 1;
    end

%%
areas = {'M23', 'M56', 'DS', 'VS'};
idList = {idM23, idM56, idDS, idVS};

% Load your data
% dataMatMain should be a time bin X neuron matrix with binSize = 0.001 seconds
idArea = idM56;
dataMatMain = dataMat(:, idArea); % This should be your data

%%
%----------------------------
% CONVERT DATA TO SPIKE TIMES FORMAT WITH MULTIPLE TRIALS
%----------------------------

% Parameters
binSize = opts.frameSize; % seconds
trialDur = 10; % trial duration in seconds
gnunits = size(dataMatMain, 2); % number of neurons

% Calculate number of complete trials
totalTime = size(dataMatMain, 1) * binSize; % total time in seconds
ntrials = floor(totalTime / trialDur); % number of complete trials

fprintf('Total time: %.1f seconds\n', totalTime);
fprintf('Creating %d trials of %.1f seconds each\n', ntrials, trialDur);

% Convert time bin matrix to spike times format with multiple trials
spikes = struct('spk', cell(ntrials, gnunits));
win_train = zeros(ntrials, 2); % trial windows [start, end] for each trial

% For each trial
for i_trial = 1:ntrials
    % Calculate trial start and end bins
    start_bin = (i_trial - 1) * trialDur / binSize + 1;
    end_bin = i_trial * trialDur / binSize;
    
    % Ensure we don't exceed data bounds
    end_bin = min(end_bin, size(dataMatMain, 1));
    
    % Trial window in seconds
    win_train(i_trial, :) = [(i_trial - 1) * trialDur, i_trial * trialDur];
    
    % For each neuron in this trial
    for i_neuron = 1:gnunits
        % Get data for this trial
        trial_data = dataMatMain(start_bin:end_bin, i_neuron);
        
        % Find time bins where this neuron spiked (value > 0)
        spike_bins = find(trial_data > 0);
        
        % Convert bin indices to time in seconds (relative to trial start)
        spike_times = (spike_bins - 1) * binSize; % -1 because bins are 1-indexed
        
        % Store spike times for this neuron in this trial
        spikes(i_trial, i_neuron).spk = spike_times;
    end
end

%%
%----------------------------
% HMM ANALYSIS
%----------------------------

% Model selection method
MODELSEL = 'XVAL'; % 'BIC'; % 'AIC';

% Setup directories
hmmdir = fullfile(paths.dropPath, 'hmm'); 
if ~exist(hmmdir, 'dir')
    mkdir(hmmdir); 
end
filesave = fullfile(hmmdir, 'HMM_naturalistic');

% Prepare data structure for HMM analysis
DATAIN = struct('spikes', spikes, 'win', win_train, 'METHOD', MODELSEL, 'filesave', filesave);

% Run HMM analysis
% Note: This assumes you have the hmm.funHMM function available
% If not, you'll need to implement the HMM fitting separately
try
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
    
    %----------------------------
    % PLOTS
    %----------------------------
    HmmParam = res.HmmParam;
    HmmParam.VarStates = res.HmmParam.VarStates(res.BestStateInd);
    hmm_bestfit = res.hmm_bestfit; 
    hmm_results = res.hmm_results; 
    hmm_postfit = res.hmm_postfit;
    
    % Create distinguishable colors for states
    colors = distinguishable_colors(max(HmmParam.VarStates, 4));
    
    % Plot transition probability matrix and emission probability matrix
    figure;
    subplot(1, 2, 1);
    imagesc(hmm_bestfit.tpm);
    colorbar;
    title('Transition Probability Matrix');
    xlabel('State'); ylabel('State');
    
    subplot(1, 2, 2);
    imagesc(hmm_bestfit.epm(:, 1:end-1)); % Exclude silence column
    colorbar;
    title('Emission Probability Matrix');
    xlabel('Neuron'); ylabel('State');
    
    % Plot state sequence over time
    figure;
    if isfield(hmm_results, 'pStates') && ~isempty(hmm_results)
        imagesc(hmm_results(1).pStates);
        colorbar;
        title('Posterior State Probabilities Over Time');
        xlabel('Time'); ylabel('State');
    end
    
    % Plot detected state sequences
    if isfield(hmm_postfit, 'sequence') && ~isempty(hmm_postfit)
        figure;
        for i_trial = 1:length(hmm_postfit)
            if ~isempty(hmm_postfit(i_trial).sequence)
                sequence = hmm_postfit(i_trial).sequence;
                for i_seq = 1:size(sequence, 2)
                    onset = sequence(1, i_seq);
                    offset = sequence(2, i_seq);
                    state = sequence(4, i_seq);
                    rectangle('Position', [onset, state-0.4, offset-onset, 0.8], ...
                             'FaceColor', colors(state, :), 'EdgeColor', 'none');
                end
            end
        end
        title('Detected State Sequences');
        xlabel('Time (s)'); ylabel('State');
        ylim([0.5, HmmParam.VarStates + 0.5]);
    end
    
catch ME
    fprintf('Error in HMM analysis: %s\n', ME.message);
    fprintf('You may need to implement the HMM fitting functions separately.\n');

    % % Alternative: Use your existing Gaussian HMM approach
    % fprintf('Falling back to Gaussian HMM approach...\n');
    % 
    % % Use PCA on the data
    % [coeff, score, ~, ~, explained, mu] = pca(dataMatMain);
    % expThresh = 70; % percent explained variance
    % nDim = find(cumsum(explained) > expThresh, 1);
    % 
    % % Fit Gaussian HMM
    % opts.stateRange = 3:15;
    % opts.numReps = 5;
    % opts.numFolds = 3;
    % opts.margLikMethod = 'laplace';
    % opts.numSamples = 100;
    % opts.selectBy = 'margLik';
    % opts.plotFlag = 1;
    % 
    % [bestModel, bestNumStates, stateSeq, allModels, allLogL, allBIC, allMargLik] = ...
    %     fit_gaussian_hmm(score(:, 1:nDim), opts);
    % 
    % % Plot results
    % posteriorProb = posterior(bestModel, score(:, 1:nDim));
    % 
    % figure;
    % imagesc(posteriorProb'); 
    % colorbar;
    % xlabel('Time'); 
    % ylabel('State');
    % title('Posterior Probabilities of States (Gaussian HMM)');
end

% Helper function for distinguishable colors (if not available)
function colors = distinguishable_colors(n)
    % Simple color generation if distinguishable_colors is not available
    if n <= 7
        colors = lines(n);
    else
        colors = hsv(n);
    end
end 