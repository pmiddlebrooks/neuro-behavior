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
    [hmm_model, hmm_params, metadata] = hmm_load_saved_model(natOrReach, brainArea);
    
    % Create a res structure compatible with the plotting code
    res = struct();
    res.HmmParam = hmm_params.hmm_parameters;
    res.BestStateInd = hmm_model.best_state_index;
    res.hmm_bestfit.tpm = hmm_model.transition_matrix;
    res.hmm_bestfit.epm = hmm_model.emission_matrix;
    res.hmm_bestfit.LLtrain = hmm_model.log_likelihood;
    
    % Load the full results file to get state sequences and posterior probabilities
    try
        paths = get_paths;
        hmmdir = fullfile(paths.dropPath, 'hmm');
        
        % Find the file that was loaded
        files = dir(fullfile(hmmdir, '*.mat'));
        fileNames = {files.name};
        
        % Find the file that matches our criteria
        matchingFiles = {};
        for i = 1:length(fileNames)
            fileName = fileNames{i};
            if contains(fileName, 'HMM_results_') && ...
               contains(fileName, ['_' natOrReach '_']) && ...
               contains(fileName, ['_' brainArea '_']) && ...
               ~contains(fileName, 'FAILED')
                matchingFiles{end+1} = fileName;
            end
        end
        
        if ~isempty(matchingFiles)
            % Sort by date and get most recent
            filePaths = fullfile(hmmdir, matchingFiles);
            fileInfo = dir(filePaths{1});
            for i = 2:length(filePaths)
                fileInfo(i) = dir(filePaths{i});
            end
            [~, sortIdx] = sort([fileInfo.datenum], 'descend');
            mostRecentFile = matchingFiles{sortIdx(1)};
            
            % Load the full results
            fullResults = load(fullfile(hmmdir, mostRecentFile));
            if isfield(fullResults, 'hmm_results_save')
                hmm_results_save = fullResults.hmm_results_save;
                
                % Extract the needed fields
                if isfield(hmm_results_save, 'hmm_results')
                    res.hmm_results = hmm_results_save.hmm_results;
                end
                if isfield(hmm_results_save, 'state_sequences')
                    res.hmm_postfit = hmm_results_save.state_sequences;
                end
                if isfield(hmm_results_save, 'trial_params')
                    trial_params = hmm_results_save.trial_params;
                end
            end
        end
    catch ME
        fprintf('Warning: Could not load full results file: %s\n', ME.message);
        fprintf('Some plots may not work without full results.\n');
    end
    
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
HmmParam = res.HmmParam;
if isfield(HmmParam, 'VarStates')
    if isfield(res, 'BestStateInd')
        HmmParam.VarStates = res.HmmParam.VarStates(res.BestStateInd);
    else
        HmmParam.VarStates = res.HmmParam.VarStates(1); % Use first if no best index
    end
end

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

