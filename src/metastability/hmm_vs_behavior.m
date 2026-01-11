%% HMM vs Behavior Analysis for Spontaneous Data
% This script analyzes the relationship between HMM states and behavior
% using contingency matrices, optimal assignment, and consistency metrics
%
% DATA FORMAT EXPECTATIONS:
% - hmm_model.continuous_sequence: Vector where 0 = invalid (no confident state), >0 = state number
% - bhvID: Vector where -1 = invalid (undefined behavior), >=0 = behavior category ID
% - Both vectors should have the same length (totalTimeBins)

%% Load saved HMM results
% Choose data type and brain area
natOrReach = 'Nat'; % 'Nat' or 'Reach'
brainArea = 'M56'; % 'M23', 'M56', 'DS', 'VS'

% Load the HMM model
[hmm_results] = hmm_load_saved_model(natOrReach, brainArea);
totalTimeBins = length(hmm_results.continuous_results.sequence);

fprintf('Loaded HMM model for %s data in %s area\n', hmm_results.metadata.data_type, hmm_results.metadata.brain_area);
fprintf('Number of states: %d, Number of neurons: %d\n', hmm_results.best_model.num_states, hmm_results.data_params.num_neurons);

%% Load behavior data
switch natOrReach
    case 'Nat'
        opts = neuro_behavior_options;
        opts.frameSize = hmm_results.HmmParam.BinSize;
        opts.collectStart = 0 * 60 * 60; % seconds
        opts.collectEnd = 45 * 60; % seconds
        getDataType = 'behavior';
        get_standard_data
        [dataBhv, bhvID] = curate_behavior_labels(dataBhv, opts);
        bhvProbs = []; % Not available for spontaneous data
    case 'Reach'
        % Load saved HMM results from get_reach_bhv_labels.m
        fprintf('Loading reach behavior data from saved HMM results...\n');
        
        % Look for the most recent reach HMM results file
        hmmdir = fullfile(paths.dropPath, 'hmm');
        reachFiles = dir(fullfile(hmmdir, 'HMM_results_Reach_*.mat'));
        
        if isempty(reachFiles)
            error('No reach HMM results found in %s. Please run get_reach_bhv_labels.m first.', hmmdir);
        end
        
        % Get the most recent file
        [~, idx] = max([reachFiles.datenum]);
        latestReachFile = reachFiles(idx).name;
        reachFilePath = fullfile(hmmdir, latestReachFile);
        
        fprintf('Loading reach HMM results from: %s\n', latestReachFile);
        load(reachFilePath, 'hmm_results');
        
        % Extract behavior labels from HMM state estimates
        bhvID = hmm_results.best_model.state_estimates;
        
        % Extract behavior probabilities for potential future use
        bhvProbs = hmm_results.best_model.state_probabilities;
        
        fprintf('Reach behavior data loaded:\n');
        fprintf('  Number of time bins: %d\n', length(bhvID));
        fprintf('  Number of states: %d\n', hmm_results.best_model.num_states);
        fprintf('  State probability matrix size: %s\n', mat2str(size(bhvProbs)));
end

% ============================================================================

fprintf('Behavior data loaded: %d time bins\n', length(bhvID));
if strcmp(natOrReach, 'Reach')
    fprintf('Invalid behavior bins (bhvID = 0): %d (%.1f%%)\n', sum(bhvID == 0), sum(bhvID == 0)/totalTimeBins*100);
else
    fprintf('Invalid behavior bins (bhvID = -1): %d (%.1f%%)\n', sum(bhvID == -1), sum(bhvID == -1)/totalTimeBins*100);
end
fprintf('Invalid HMM bins (continuous_sequence = 0): %d (%.1f%%)\n', sum(hmm_results.continuous_results.sequence == 0), sum(hmm_results.continuous_results.sequence == 0)/totalTimeBins*100);

%% 0. Find Optimal Lag using Mutual Information
% This section finds the optimal lag (in bins and seconds) between HMM states and behavior using mutual information.
% The best lag (in seconds) will be displayed in the plot title.

% (Plotting and lag analysis code is below; the best lag will be shown in the plot title.)
fprintf('\n=== Step 0: Finding Optimal Lag using Mutual Information ===\n');

% Parameters for lag analysis
maxLagSec = 1.5; % Maximum lag to test (in seconds)
maxLagBin = round(maxLagSec / hmm_results.HmmParam.BinSize); % Maximum lag in time bins

fprintf('Testing lags from -%d to +%d time bins (%.1f to %.1f seconds)\n', ...
    maxLagBin, maxLagBin, -maxLagBin * hmm_results.HmmParam.BinSize, maxLagBin * hmm_results.HmmParam.BinSize);

% Initialize arrays for lag analysis
lags = -maxLagBin:maxLagBin;
mi_values = nan(length(lags), 1);
n_valid_pairs = zeros(length(lags), 1);

% Calculate mutual information at each lag
for i = 1:length(lags)
    lag = lags(i);

    if lag >= 0
        % Positive lag: HMM leads behavior
        hmm_start = 1;
        hmm_end = totalTimeBins - lag;
        bhv_start = 1 + lag;
        bhv_end = totalTimeBins;
    else
        % Negative lag: behavior leads HMM
        hmm_start = 1 - lag;
        hmm_end = totalTimeBins;
        bhv_start = 1;
        bhv_end = totalTimeBins + lag;
    end

    % Ensure valid indices
    if hmm_start < 1 || hmm_end > totalTimeBins || ...
            bhv_start < 1 || bhv_end > totalTimeBins || ...
            hmm_end < hmm_start || bhv_end < bhv_start
        mi_values(i) = NaN;
        n_valid_pairs(i) = 0;
        continue;
    end

    % Extract aligned segments
    hmm_segment = hmm_results.continuous_results.sequence(hmm_start:hmm_end);
    bhv_segment = bhvID(bhv_start:bhv_end);

    % Find valid time bins (exclude invalid values)
    if strcmp(natOrReach, 'Reach')
        valid_mask = (hmm_segment > 0) & (bhv_segment > 0); % For reach data, 0 = invalid
    else
        valid_mask = (hmm_segment > 0) & (bhv_segment >= 0); % For spontaneous data, -1 = invalid
    end

    n_valid_pairs(i) = sum(valid_mask);

    if n_valid_pairs(i) < 10
        % Need sufficient valid data to compute MI
        mi_values(i) = NaN;
        continue;
    end

    % Extract valid data
    hmm_valid = hmm_segment(valid_mask);
    bhv_valid = bhv_segment(valid_mask);

    % Calculate mutual information on valid data
    mi_values(i) = mutual_information(hmm_valid, bhv_valid);
end

% Find best lag
[best_mi, best_lag_idx] = max(mi_values);
best_lag = lags(best_lag_idx);

% Interpret the lag direction
if best_lag > 0
    lag_interpretation = sprintf('Neural activity (HMM) LEADS behavior by %d time bins (%.3f seconds)', ...
        best_lag, best_lag * hmm_results.HmmParam.BinSize);
elseif best_lag < 0
    lag_interpretation = sprintf('Behavior LEADS neural activity (HMM) by %d time bins (%.3f seconds)', ...
        abs(best_lag), abs(best_lag) * hmm_results.HmmParam.BinSize);
else
    lag_interpretation = 'Neural activity and behavior are synchronized (no lag)';
end

fprintf('\nBest lag: %d time bins = %.3f seconds (MI = %.4f, %d valid pairs)\n', ...
    best_lag, best_lag * hmm_results.HmmParam.BinSize, best_mi, n_valid_pairs(best_lag_idx));
fprintf('Lag interpretation: %s\n', lag_interpretation);

% Plot mutual information vs lag, include best lag in seconds in the title
figure('Position', [100, 100, 800, 400]);
plot(lags, mi_values, 'b-', 'LineWidth', 2);
hold on;
plot(best_lag, best_mi, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('Lag (time bins)');
ylabel('Mutual Information');
% Add best lag in seconds to the title
bestLagSeconds = best_lag * hmm_results.HmmParam.BinSize;
title(sprintf('Mutual Information vs Lag (Best lag: %d bins = %.3f s)', best_lag, bestLagSeconds));
grid on;
legend('MI values', 'Best lag', 'Location', 'best');

%% 1. Build Contingency Matrix (at optimal lag)
fprintf('\n=== Building Contingency Matrix ===\n');
fprintf('Using optimal lag: %d time bins (%.3f seconds)\n', best_lag, best_lag * hmm_results.HmmParam.BinSize);

% Get unique categories from both vectors
hmmStates = unique(hmm_results.continuous_results.sequence(hmm_results.continuous_results.sequence > 0)); % Exclude invalid HMM states (0)
if strcmp(natOrReach, 'Reach')
    bhvCategories = unique(bhvID(bhvID > 0)); % Exclude invalid behavior (0) for reach data
else
    bhvCategories = unique(bhvID(bhvID >= 0)); % Exclude invalid behavior (-1) for spontaneous data
end

nHmmStates = length(hmmStates);
nBhvCategories = length(bhvCategories);

fprintf('HMM states: %s\n', mat2str(hmmStates));
fprintf('Behavior categories: %s\n', mat2str(bhvCategories));

% Initialize contingency matrix
contingency_matrix = zeros(nHmmStates, nBhvCategories);

% Fill contingency matrix using optimal lag alignment
for i = 1:totalTimeBins
    % Get HMM state at current time
    hmm_state = hmm_results.continuous_results.sequence(i);

    % Get behavior at lagged time
    % Compute behavior time point by adding lag (positive or negative)
    bhv_time = i + best_lag;

    % Check if both time points are valid
    if bhv_time >= 1 && bhv_time <= totalTimeBins
        bhv_cat = bhvID(bhv_time);

        % Only count if both HMM state and behavior are valid
        if strcmp(natOrReach, 'Reach')
            if hmm_state > 0 && bhv_cat > 0 % For reach data, 0 = invalid
                hmm_idx = find(hmmStates == hmm_state);
                bhv_idx = find(bhvCategories == bhv_cat);

                if ~isempty(hmm_idx) && ~isempty(bhv_idx)
                    contingency_matrix(hmm_idx, bhv_idx) = contingency_matrix(hmm_idx, bhv_idx) + 1;
                end
            end
        else
            if hmm_state > 0 && bhv_cat >= 0 % For spontaneous data, -1 = invalid
                hmm_idx = find(hmmStates == hmm_state);
                bhv_idx = find(bhvCategories == bhv_cat);

                if ~isempty(hmm_idx) && ~isempty(bhv_idx)
                    contingency_matrix(hmm_idx, bhv_idx) = contingency_matrix(hmm_idx, bhv_idx) + 1;
                end
            end
        end
    end
end

% Display contingency matrix
fprintf('\nContingency Matrix (HMM states x Behavior categories):\n');
fprintf('Rows: HMM states, Columns: Behavior categories\n');
fprintf('Each cell shows count of co-occurrences\n\n');

% Create row and column labels
row_labels = cell(nHmmStates, 1);
col_labels = cell(1, nBhvCategories);

for i = 1:nHmmStates
    row_labels{i} = sprintf('HMM_%d', hmmStates(i));
end

for j = 1:nBhvCategories
    col_labels{j} = sprintf('BHV_%d', bhvCategories(j));
end

% Display with labels
fprintf('%8s', '');
for j = 1:nBhvCategories
    fprintf('%8s', col_labels{j});
end
fprintf('\n');

for i = 1:nHmmStates
    fprintf('%8s', row_labels{i});
    for j = 1:nBhvCategories
        fprintf('%8d', contingency_matrix(i, j));
    end
    fprintf('\n');
end

%% 2. Cluster-to-Label Matching using Hungarian Algorithm
fprintf('\n=== Cluster-to-Label Matching (Hungarian Algorithm) ===\n');

% Convert contingency matrix to cost matrix for Hungarian algorithm
% We want to maximize overlap, so use negative counts as costs
cost_matrix = -contingency_matrix;

% Apply Hungarian algorithm to find optimal assignment
% Use assignmentoptimal from MATLAB's Optimization Toolbox instead of munkres
% Try to use Hungarian algorithm from Optimization Toolbox if available
if exist('munkres', 'file') 
    [assignment, cost] = munkres(cost_matrix);
else
    % Fallback to a basic greedy assignment if neither function is available
    [assignment, cost] = greedy_assignment(cost_matrix);
end


fprintf('Optimal assignment found (cost = %.2f):\n', -cost);
fprintf('HMM State -> Behavior Category\n');

% Create mapping dictionary
hmm_to_bhv_mapping = containers.Map('KeyType', 'double', 'ValueType', 'double');
bhv_to_hmm_mapping = containers.Map('KeyType', 'double', 'ValueType', 'double');

for i = 1:length(assignment)
    if assignment(i) > 0
        % Only map states that exist in our arrays
        if i <= length(hmmStates) && assignment(i) <= length(bhvCategories)
            hmm_state = hmmStates(i);
            bhv_category = bhvCategories(assignment(i));
            hmm_to_bhv_mapping(hmm_state) = bhv_category;
            bhv_to_hmm_mapping(bhv_category) = hmm_state;
 
             fprintf('  HMM_%d -> BHV_%d\n', hmmStates(i), bhvCategories(assignment(i)));
       end

    end
end

%% 3. Compute Classification Metrics
% -----------------------------------------------------------
% This section computes classification metrics to evaluate
% how well the HMM state sequence predicts behavioral categories.
% Metrics include overall accuracy and per-category precision,
% recall, F1 score, and support (number of samples per category).
% The predicted behavior is derived from the HMM state sequence
% using the optimal mapping and lag determined earlier.
% -----------------------------------------------------------
fprintf('\n=== Classification Metrics (at optimal lag) ===\n');

% Create predicted behavior vector based on HMM states
predicted_bhv = nan(size(bhvID));

for i = 1:totalTimeBins
    hmm_state = hmm_results.continuous_results.sequence(i);

    if hmm_state > 0 && isKey(hmm_to_bhv_mapping, hmm_state)
        % Get the predicted behavior for this HMM state
        predicted_behavior = hmm_to_bhv_mapping(hmm_state);

        % Apply the lag to align with actual behavior
        % Apply lag to align prediction with behavior
        pred_time = i + best_lag;

        % Place prediction at the lagged time if it's within bounds
        if pred_time >= 1 && pred_time <= totalTimeBins
            predicted_bhv(pred_time) = predicted_behavior;
        end
    end
end

% Calculate accuracy (only for time points where we have both predictions and actual)
if strcmp(natOrReach, 'Reach')
    valid_idx = ~isnan(predicted_bhv) & (bhvID > 0); % For reach data, 0 = invalid
else
    valid_idx = ~isnan(predicted_bhv) & (bhvID >= 0); % For spontaneous data, -1 = invalid
end
if sum(valid_idx) > 0
    accuracy = sum(predicted_bhv(valid_idx) == bhvID(valid_idx)) / sum(valid_idx);
    fprintf('Overall Accuracy: %.3f (%.1f%%)\n', accuracy, accuracy * 100);
else
    fprintf('No valid predictions to compute accuracy\n');
end

% Compute per-category metrics
fprintf('\nPer-Category Metrics:\n');
fprintf('%-12s %-8s %-8s %-8s %-8s\n', 'Category', 'Precision', 'Recall', 'F1', 'Support');

for cat = bhvCategories'
    % Find indices for this category
    actual_pos = (bhvID == cat);
    pred_pos = (predicted_bhv == cat);

    % True positives, false positives, false negatives
    tp = sum(actual_pos & pred_pos & valid_idx);
    fp = sum(~actual_pos & pred_pos & valid_idx);
    fn = sum(actual_pos & ~pred_pos & valid_idx);

    % Calculate metrics
    precision = tp / (tp + fp + eps);
    recall = tp / (tp + fn + eps);
    f1 = 2 * (precision * recall) / (precision + recall + eps);
    support = sum(actual_pos & valid_idx);

    fprintf('%-12s %-8.3f %-8.3f %-8.3f %-8d\n', ...
        sprintf('BHV_%d', cat), precision, recall, f1, support);
end

%% 4. Consistency Rates Analysis (at optimal lag)
% This section computes how consistently each HMM state maps to a specific behavior category
% ("per-state purity"), and how much of the total time each state covers ("coverage").
% The analysis is performed using the optimal lag found earlier, aligning HMM states and behaviors.
fprintf('\n=== Consistency Rates Analysis (at optimal lag) ===\n');

% Per-state purity (how behaviorally consistent a state is)
fprintf('\nPer-State Purity (Behavioral Consistency):\n');
fprintf('%-12s %-15s %-15s %-15s\n', 'HMM State', 'Primary Behavior', 'Purity (%)', 'Coverage (%)');

for i = 1:nHmmStates
    hmm_state = hmmStates(i);

    % Find all time points for this HMM state
    state_mask = (hmm_results.continuous_results.sequence == hmm_state);
    state_time_points = sum(state_mask);

    if state_time_points > 0
        % Get behavior distribution for this state at lagged times
        bhv_distribution = [];
        for j = find(state_mask)'
            % Apply lag to get corresponding behavior time
            bhv_time = j + best_lag;
            bhv_time = j + best_lag;


            % Check if behavior time is valid and within bounds
            if bhv_time >= 1 && bhv_time <= totalTimeBins
                bhv_cat = bhvID(bhv_time);
                if strcmp(natOrReach, 'Reach')
                    if bhv_cat > 0  % Only include valid behavior categories for reach data
                        bhv_distribution = [bhv_distribution; bhv_cat];
                    end
                else
                    if bhv_cat >= 0  % Only include valid behavior categories for spontaneous data
                        bhv_distribution = [bhv_distribution; bhv_cat];
                    end
                end
            end
        end

        if ~isempty(bhv_distribution)
            % Find most common behavior (primary behavior)
            [bhv_counts, bhv_values] = histcounts(bhv_distribution, bhvCategories);
            [max_count, max_idx] = max(bhv_counts);
            primary_bhv = bhv_values(max_idx);

            % Calculate purity (fraction of time this state maps to its primary behavior)
            purity = max_count / length(bhv_distribution) * 100;

            % Calculate coverage (fraction of total time this state represents)
            coverage = state_time_points / totalTimeBins * 100;

            fprintf('%-12s %-15s %-15.1f %-15.1f\n', ...
                sprintf('HMM_%d', hmm_state), sprintf('BHV_%d', primary_bhv), purity, coverage);
        end
    end
end

% Per-behavior coverage (how many states it spreads across)
fprintf('\nPer-Behavior Coverage (State Distribution):\n');
fprintf('%-15s %-15s %-15s %-15s\n', 'Behavior', 'Primary HMM State', 'Coverage (%)', 'State Spread');

for j = 1:nBhvCategories
    bhv_category = bhvCategories(j);

    % Find all time points for this behavior
    bhv_mask = (bhvID == bhv_category);
    bhv_time_points = sum(bhv_mask);

    if bhv_time_points > 0
        % Get HMM state distribution for this behavior at lagged times
        hmm_distribution = [];
        for k = find(bhv_mask)'
            % Apply lag to get corresponding HMM time
            % No need for if/else since lag direction is always reversed
            hmm_time = k - best_lag;  % Reverse the lag direction
            hmm_time = k - best_lag;  % Reverse the lag direction


            % Check if HMM time is valid and within bounds
            if hmm_time >= 1 && hmm_time <= totalTimeBins
                hmm_state = hmm_results.continuous_results.sequence(hmm_time);
                if hmm_state > 0  % Only include valid HMM states
                    hmm_distribution = [hmm_distribution; hmm_state];
                end
            end
        end

        if ~isempty(hmm_distribution)
            % Find most common HMM state (primary state)
            [hmm_counts, hmm_values] = histcounts(hmm_distribution, hmmStates);
            [max_count, max_idx] = max(hmm_counts);
            primary_hmm = hmm_values(max_idx);

            % Calculate coverage (fraction of total time this behavior represents)
            coverage = bhv_time_points / totalTimeBins * 100;

            % Calculate state spread (how many different HMM states this behavior maps to)
            unique_states = unique(hmm_distribution);
            state_spread = length(unique_states);

            fprintf('%-15s %-15s %-15.1f %-15d\n', ...
                sprintf('BHV_%d', bhv_category), sprintf('HMM_%d', primary_hmm), coverage, state_spread);
        end
    end
end

%% 5. Visualization
fprintf('\n=== Creating Visualizations ===\n');

% Create figure with subplots
figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Contingency Matrix Heatmap
subplot(2, 3, 1);
imagesc(contingency_matrix);
colormap('hot');
colorbar;
title('Contingency Matrix Heatmap');
xlabel('Behavior Categories');
ylabel('HMM States');

% Add text labels
for i = 1:nHmmStates
    for j = 1:nBhvCategories
        text(j, i, num2str(contingency_matrix(i, j)), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    end
end

% Set tick labels
xticks(1:nBhvCategories);
xticklabels(col_labels);
yticks(1:nHmmStates);
yticklabels(row_labels);

% Subplot 2: HMM State Sequence
subplot(2, 3, 2);
plot(hmm_results.continuous_results.sequence, 'b-', 'LineWidth', 1);
title('HMM State Sequence Over Time');
xlabel('Time Bins');
ylabel('HMM State');
ylim([min(hmmStates)-0.5, max(hmmStates)+0.5]);

% Subplot 3: Behavior Sequence
subplot(2, 3, 3);
plot(bhvID, 'r-', 'LineWidth', 1);
title('Behavior Sequence Over Time');
xlabel('Time Bins');
ylabel('Behavior Category');
ylim([min(bhvCategories)-0.5, max(bhvCategories)+0.5]);

% Subplot 4: Predicted vs Actual Behavior
subplot(2, 3, 4);
plot(bhvID, 'r-', 'LineWidth', 2, 'DisplayName', 'Actual');
hold on;
plot(predicted_bhv, 'b--', 'LineWidth', 1, 'DisplayName', 'Predicted');
title('Predicted vs Actual Behavior');
xlabel('Time Bins');
ylabel('Behavior Category');
legend('Location', 'best');
ylim([min(bhvCategories)-0.5, max(bhvCategories)+0.5]);

% Subplot 5: State-Behavior Mapping
subplot(2, 3, 5);
mapping_matrix = zeros(nHmmStates, nBhvCategories);
for i = 1:nHmmStates
    hmm_state = hmmStates(i);
    if isKey(hmm_to_bhv_mapping, hmm_state)
        bhv_category = hmm_to_bhv_mapping(hmm_state);
        bhv_idx = find(bhvCategories == bhv_category);
        mapping_matrix(i, bhv_idx) = 1;
    end
end

imagesc(mapping_matrix);
colormap('gray');
title('Optimal State-Behavior Mapping');
xlabel('Behavior Categories');
ylabel('HMM States');

% Set tick labels
xticks(1:nBhvCategories);
xticklabels(col_labels);
yticks(1:nHmmStates);
yticklabels(row_labels);

% Subplot 6: Summary Statistics
subplot(2, 3, 6);
text(0.1, 0.9, sprintf('Total Time Bins: %d', totalTimeBins), 'FontSize', 10);
text(0.1, 0.8, sprintf('HMM States: %d', nHmmStates), 'FontSize', 10);
text(0.1, 0.7, sprintf('Behaviors: %d', nBhvCategories), 'FontSize', 10);
if exist('accuracy', 'var')
    text(0.1, 0.6, sprintf('Accuracy: %.1f%%', accuracy*100), 'FontSize', 10);
end
text(0.1, 0.5, sprintf('Mapping Cost: %.2f', -cost), 'FontSize', 10);
axis off;
title('Summary Statistics');

% Adjust layout
sgtitle(sprintf('HMM vs Behavior Analysis: %s %s', metadata.data_type, metadata.brain_area), 'FontSize', 14);

%% 6. Save Results
fprintf('\n=== Saving Analysis Results ===\n');

% Create results structure
analysis_results = struct();
analysis_results.contingency_matrix = contingency_matrix;
analysis_results.hmmStates = hmmStates;
analysis_results.bhvCategories = bhvCategories;
analysis_results.optimal_mapping = assignment;
analysis_results.mapping_cost = -cost;
analysis_results.hmm_to_bhv_mapping = hmm_to_bhv_mapping;
analysis_results.bhv_to_hmm_mapping = bhv_to_hmm_mapping;
analysis_results.accuracy = accuracy;
analysis_results.predicted_behavior = predicted_bhv;
analysis_results.analysis_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

% Add behavior probabilities if available (for reach data)
if exist('bhvProbs', 'var') && ~isempty(bhvProbs)
    analysis_results.behavior_probabilities = bhvProbs;
    fprintf('Behavior probabilities added to analysis results\n');
end

% Add lag analysis results
analysis_results.optimal_lag = best_lag;
analysis_results.optimal_lag_seconds = best_lag * hmm_results.HmmParam.BinSize;
analysis_results.best_mutual_information = best_mi;
analysis_results.lag_interpretation = lag_interpretation;
analysis_results.mutual_information_curve = struct('lags', lags, 'mi_values', mi_values, 'n_valid_pairs', n_valid_pairs);

% Save results
output_filename = sprintf('HMM_Behavior_Analysis_%s_%s_%s.mat', ...
    hmm_results.metadata.data_type, hmm_results.metadata.brain_area, datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
save(output_filename, 'analysis_results', 'hmm_results');

fprintf('Analysis results saved to: %s\n', output_filename);
fprintf('\nAnalysis complete!\n');










%% Helper Functions

function mi = mutual_information(x, y)
% Calculate mutual information between two categorical variables
% x, y: vectors of categorical values
% Returns: mutual information in bits

% Get unique values
unique_x = unique(x);
unique_y = unique(y);

% Calculate joint and marginal probabilities
n = length(x);
joint_prob = zeros(length(unique_x), length(unique_y));
marg_prob_x = zeros(length(unique_x), 1);
marg_prob_y = zeros(length(unique_y), 1);

for i = 1:length(unique_x)
    for j = 1:length(unique_y)
        joint_prob(i, j) = sum(x == unique_x(i) & y == unique_y(j)) / n;
    end
    marg_prob_x(i) = sum(x == unique_x(i)) / n;
end

for j = 1:length(unique_y)
    marg_prob_y(j) = sum(y == unique_y(j)) / n;
end

% Calculate mutual information
mi = 0;
for i = 1:length(unique_x)
    for j = 1:length(unique_y)
        if joint_prob(i, j) > 0 && marg_prob_x(i) > 0 && marg_prob_y(j) > 0
            mi = mi + joint_prob(i, j) * log2(joint_prob(i, j) / (marg_prob_x(i) * marg_prob_y(j)));
        end
    end
end
end





% Helper function for greedy assignment
function [assignment, total_cost] = greedy_assignment(cost_matrix)
    [n, m] = size(cost_matrix);
    assignment = zeros(n, 1);
    used = false(1, m);
    total_cost = 0;
    
    for i = 1:n
        [min_cost, best_j] = min(cost_matrix(i, ~used));
        unused_cols = find(~used);
        j = unused_cols(best_j);
        assignment(i) = j;
        used(j) = true;
        total_cost = total_cost + min_cost;
    end
end


