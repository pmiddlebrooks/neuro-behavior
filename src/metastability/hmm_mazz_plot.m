%% ----------------------------
% PLOTS
%----------------------------

%% USER CHOICE: Specify how to get HMM data
% Set this to 'workspace' to use existing 'res' variable, or 'load' to load a saved model
dataSource = 'load'; % 'workspace' or 'load'

if strcmp(dataSource, 'load')
    fprintf('Loading saved HMM model...\n');
    
    % Parameters for loading saved model - CHANGE THESE AS NEEDED
    natOrReach = 'Reach'; % 'Nat' or 'Reach'
    brainArea = 'M56';    % 'M23', 'M56', 'DS', 'VS'
    
    % Load the saved model
    [hmm_res] = hmm_load_saved_model(natOrReach, brainArea);
    

    
    fprintf('Loaded model: %s data from %s area\n', natOrReach, brainArea);
    
elseif strcmp(dataSource, 'workspace')
    fprintf('Using existing HMM results from workspace.\n');
    
    % Check if required variables exist
    if ~exist('res', 'var') || isempty(res)
        error('No ''res'' variable found in workspace. Set dataSource = ''load'' to load a saved model.');
    end
    
    % Check if we have the trial parameters from hmm_results_save
    if ~exist('trial_params', 'var') && exist('hmm_results_save', 'var')
        trial_params = hmm_results_save.trial_params;
    end
    
else
    error('Invalid dataSource. Must be ''workspace'' or ''load''');
end

% Extract HMM parameters
HmmParam = hmm_res.HmmParam;
if isfield(HmmParam, 'VarStates')
    if isfield(res, 'BestStateInd')
        HmmParam.VarStates = hmm_res.HmmParam.VarStates(hmm_res.BestStateInd);
    else
        HmmParam.VarStates = hmm_res.HmmParam.VarStates(1); % Use first if no best index
    end
end

hmm_bestfit = hmm_res.hmm_bestfit;
hmm_results = hmm_res.hmm_results;
hmm_postfit = hmm_res.hmm_postfit;

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

% Plot example state sequence over time
figure;
iTrial = 3;
if isfield(hmm_results, 'pStates') && ~isempty(hmm_results) && length(hmm_results) >= iTrial
    imagesc(hmm_results(iTrial).pStates);
    colorbar;
    title('Posterior State Probabilities Over Time');
    xlabel('Time'); ylabel('State');
else
    fprintf('Warning: Could not plot posterior probabilities - data not available\n');
end

%% Proportion of data in various states
% This section can be expanded later for additional analyses

%% Plot full sequence
figure(8); clf;
hold on;
sequence = hmm_results_save.continuous_results.sequence;
probabilities = hmm_results_save.continuous_results.pStates;

x = (1:length(sequence)) * HmmParam.BinSize;
numStates = size(sequence, 2);

% Get colors for states
colors = distinguishable_colors(numStates);

% Plot rectangles for each state
for state = 1:numStates
    stateIdx = sequence == state;
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
    plot(x, probabilities(:,state), 'Color', colors(state,:), 'LineWidth', 2);
end

xlabel('Time (s)');
ylabel('State Probability');
ylim([0 1]);
hold off;

%% Additional plots can be added here
fprintf('Basic HMM plots completed!\n');
fprintf('You can add more analysis and plotting code below.\n');

