%% ----------------------------
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

% Plot example state sequence over time
figure;
iTrial = 3;
if isfield(hmm_results, 'pStates') && ~isempty(hmm_results)
    imagesc(hmm_results(iTrial).pStates);
    colorbar;
    title('Posterior State Probabilities Over Time');
    xlabel('Time'); ylabel('State');
end


%% Proportion of data in various states



%% Plot detected state sequences
if isfield(hmm_postfit, 'sequence') && ~isempty(hmm_postfit)
    % figure(4); clf; hold on;
    figure(4);
    
    % Unified approach: Plot one trial window at a time for both Nat and Reach data
    fprintf('Plotting state sequences - one trial window at a time (unified approach)\n');
    
    for i_trial = 1:length(hmm_postfit)
         clf;
        ylim([-.5 HmmParam.VarStates + 0.5])
        
        if ~isempty(hmm_postfit(i_trial).sequence)
            % Get trial window information
            trial_start = hmm_results_save.trial_params.trial_windows(i_trial, 1);
            trial_end = hmm_results_save.trial_params.trial_windows(i_trial, 2);
            trial_duration = trial_end - trial_start;
            
            sequence = hmm_postfit(i_trial).sequence;
            for i_seq = 1:size(sequence, 2)
                onset = sequence(1, i_seq);
                offset = sequence(2, i_seq);
                state = sequence(4, i_seq);
                rectangle('Position', [onset, state-0.4, offset-onset, 0.8], ...
                    'FaceColor', colors(state, :), 'EdgeColor', 'none');
            end
        end
        
        % Create title for trial
        trial_num = i_trial;
        total_trials = length(hmm_postfit);
        trial_start_time = hmm_results_save.trial_params.trial_windows(i_trial, 1);
        trial_end_time = hmm_results_save.trial_params.trial_windows(i_trial, 2);
        
        % Determine data type for title
        if strcmp(hmm_results_save.metadata.data_type, 'Nat')
            data_label = 'Natural Data';
        else
            data_label = 'Reach Data';
        end
        
        title(sprintf('%s %s - Trial %d/%d (%.1f-%.1f s)', ...
            hmm_results_save.metadata.brain_area, data_label, trial_num, total_trials, trial_start_time, trial_end_time));
        xlabel('Time (s)'); ylabel('State');
    end
end

