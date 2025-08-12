%% HMM vs Behavior Analysis for Naturalistic Data
% This script compares HMM state sequences with behavioral labels
% to find the best temporal alignment and measure agreement
% ASSUMES: bhvID is already loaded in the workspace
%
% UPDATED: Now uses continuous results directly from hmm_results_save.continuous_results
% instead of converting trial-based sequences to continuous time series

% Clear workspace and close figures


%% USER CHOICE: Specify how to get HMM data
% Set this to 'workspace' to use existing 'res' variable, or 'load' to load a saved model
dataSource = 'load'; % 'workspace' or 'load'

if strcmp(dataSource, 'load')
    fprintf('Loading saved HMM model...\n');
    
    % Parameters for loading saved model - CHANGE THESE AS NEEDED
    natOrReach = 'Nat'; % 'Nat' for naturalistic data
    brainArea = 'M56';  % 'M23', 'M56', 'DS', 'VS'
    
    % Load the saved model with continuous results
    [hmm_model, hmm_params, metadata, continuous_results] = hmm_load_saved_model(natOrReach, brainArea);
    
    % Check if continuous results are available
    if isempty(continuous_results.sequence)
        error('No continuous results found in the loaded data. Make sure the HMM analysis was completed successfully.');
    end
    
    fprintf('Loaded model: %s data from %s area\n', natOrReach, brainArea);
    fprintf('Continuous sequence length: %d time bins\n', length(continuous_results.sequence));
    
elseif strcmp(dataSource, 'workspace')
    fprintf('Using existing HMM results from workspace.\n');
    
    % Check if required variables exist
    if ~exist('res', 'var') || isempty(res)
        error('No ''res'' variable found in workspace. Set dataSource = ''load'' to load a saved model.');
    end
    
    % Check if continuous results exist in workspace
    if exist('hmm_res', 'var') && isfield(hmm_res, 'continuous_results')
        continuous_results = hmm_res.continuous_results;
        fprintf('Using continuous results from workspace\n');
    else
        error('No continuous results found in workspace. Set dataSource = ''load'' to load a saved model.');
    end
    
    % Get HMM parameters from workspace
    if exist('hmm_params', 'var')
        % Use existing hmm_params
    else
        % Create basic hmm_params structure
        hmm_params = struct();
        hmm_params.bin_size = 0.001; % Default bin size, adjust as needed
    end
    
else
    error('Invalid dataSource. Must be ''workspace'' or ''load''');
end

%% Check for behavioral data
% ASSUMES: bhvID is already loaded in the workspace
fprintf('Checking for behavioral data in workspace...\n');

if ~exist('bhvID', 'var')
    error('bhvID not found in workspace. Please load behavioral data first.');
end

fprintf('Found bhvID in workspace (length: %d)\n', length(bhvID));

%% Parameters for analysis
maxLag = 10 / hmm_params.bin_size; % Maximum lag to test (in time bins)
fprintf('Maximum lag for analysis: %d time bins (%.1f seconds)\n', maxLag, maxLag * hmm_params.bin_size);

%% Prepare data for analysis
fprintf('Preparing data for analysis...\n');

% Use continuous sequence directly (keeping all bins including 0s)
continuous_sequence = continuous_results.sequence;
totalTimeBins = length(continuous_sequence);

fprintf('Using continuous sequence of length %d time bins\n', totalTimeBins);

% Prepare behavioral labels - KEEP ALL BINS including -1 values
% This ensures vectors maintain the same length for analysis
bhvID_full = bhvID;
fprintf('Behavioral labels: %d total labels (including %d undefined = -1)\n', ...
        length(bhvID_full), sum(bhvID == -1));

% Ensure behavioral data has same length as sequence
if length(bhvID_full) ~= totalTimeBins
    fprintf('Warning: Behavioral data length (%d) differs from sequence length (%d)\n', ...
            length(bhvID_full), totalTimeBins);
    
    % Truncate to shorter length
    minLength = min(length(bhvID_full), totalTimeBins);
    continuous_sequence = continuous_sequence(1:minLength);
    bhvID_full = bhvID_full(1:minLength);
    totalTimeBins = minLength;
    
    fprintf('Truncated both to length %d\n', minLength);
end

% Use the full vectors for analysis (including invalid values)
% continuous_sequence: 0 = undefined state, >0 = state number
% bhvID_full: -1 = undefined behavior, >=0 = behavior ID
fprintf('Final data lengths: Sequence = %d, Behavior = %d\n', ...
        length(continuous_sequence), length(bhvID_full));
fprintf('Sequence: %d valid states, %d undefined (0)\n', ...
        sum(continuous_sequence > 0), sum(continuous_sequence == 0));
fprintf('Behavior: %d valid labels, %d undefined (-1)\n', ...
        sum(bhvID_full >= 0), sum(bhvID_full == -1));

%% 1. Find best lag using mutual information
fprintf('\n=== Step 1: Finding best lag using mutual information ===\n');
fprintf('Note: Invalid values (sequence=0, behavior=-1) are treated as separate categories\n');

% Calculate mutual information at different lags
lags = -maxLag:maxLag;
mi_values = zeros(length(lags), 1);

for i = 1:length(lags)
    lag = lags(i);
    
    if lag >= 0
        % Positive lag: sequence leads behavior
        seq_start = 1;
        seq_end = length(continuous_sequence) - lag;
        bhv_start = 1 + lag;
        bhv_end = length(bhvID_full);
    else
        % Negative lag: behavior leads sequence
        seq_start = 1 - lag;
        seq_end = length(continuous_sequence);
        bhv_start = 1;
        bhv_end = length(bhvID_full) + lag;
    end
    
    % Ensure valid indices
    if seq_start < 1 || seq_end > length(continuous_sequence) || ...
       bhv_start < 1 || bhv_end > length(bhvID_full) || ...
       seq_end < seq_start || bhv_end < bhv_start
        mi_values(i) = NaN;
        continue;
    end
    
    % Extract aligned segments
    seq_segment = continuous_sequence(seq_start:seq_end);
    bhv_segment = bhvID_full(bhv_start:bhv_end);
    
    % Calculate mutual information
    mi_values(i) = mutual_information(seq_segment, bhv_segment);
end

% Find best lag
[best_mi, best_lag_idx] = max(mi_values);
best_lag = lags(best_lag_idx);

fprintf('Best lag: %d time bins (MI = %.4f)\n', best_lag, best_mi);

%% 2. Calculate agreement metrics at best lag
fprintf('\n=== Step 2: Calculating agreement metrics at best lag ===\n');

% Align data at best lag
if best_lag >= 0
    % Positive lag: sequence leads behavior
    seq_start = 1;
    seq_end = length(continuous_sequence) - best_lag;
    bhv_start = 1 + best_lag;
    bhv_end = length(bhvID_full);
else
    % Negative lag: behavior leads sequence
    seq_start = 1 - best_lag;
    seq_end = length(continuous_sequence);
    bhv_start = 1;
    bhv_end = length(bhvID_full) + best_lag;
end

% Extract aligned segments
seq_aligned = continuous_sequence(seq_start:seq_end);
bhv_aligned = bhvID_full(bhv_start:bhv_end);

% Calculate percent agreement
agreement = sum(seq_aligned == bhv_aligned) / length(seq_aligned) * 100;

% Calculate Cohen's kappa
kappa = cohens_kappa(seq_aligned, bhv_aligned);

fprintf('Percent Agreement: %.2f%%\n', agreement);
fprintf('Cohen''s Kappa: %.4f\n', kappa);

%% 3. Print and plot summary results
fprintf('\n=== Step 3: Summary Results ===\n');

% Display data type and brain area if available
if exist('natOrReach', 'var')
    fprintf('Data Type: %s\n', natOrReach);
else
    fprintf('Data Type: Workspace data\n');
end

if exist('brainArea', 'var')
    fprintf('Brain Area: %s\n', brainArea);
else
    fprintf('Brain Area: Workspace data\n');
end

fprintf('Total Time Bins: %d\n', totalTimeBins);
fprintf('Valid Sequence Points: %d\n', length(continuous_sequence));
fprintf('Valid Behavior Points: %d\n', length(bhvID_full));
fprintf('Best Lag: %d time bins (%.3f seconds)\n', best_lag, best_lag * hmm_params.bin_size);
fprintf('Mutual Information at Best Lag: %.4f\n', best_mi);
fprintf('Percent Agreement: %.2f%%\n', agreement);
fprintf('Cohen''s Kappa: %.4f\n', kappa);

% Interpret kappa
if kappa < 0
    kappa_interpretation = 'Poor agreement';
elseif kappa < 0.2
    kappa_interpretation = 'Slight agreement';
elseif kappa < 0.4
    kappa_interpretation = 'Fair agreement';
elseif kappa < 0.6
    kappa_interpretation = 'Moderate agreement';
elseif kappa < 0.8
    kappa_interpretation = 'Substantial agreement';
else
    kappa_interpretation = 'Almost perfect agreement';
end
fprintf('Kappa Interpretation: %s\n', kappa_interpretation);

%% Create plots
fprintf('\n=== Creating plots ===\n');

% Plot 1: Mutual information vs lag
figure(1); clf;
plot(lags, mi_values, 'b-', 'LineWidth', 2);
hold on;
plot(best_lag, best_mi, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('Lag (time bins)');
ylabel('Mutual Information');
title('Mutual Information vs Lag');
grid on;
legend('MI values', 'Best lag', 'Location', 'best');

% Plot 2: Aligned sequences comparison
figure(2); clf;
subplot(2,1,1);
plot(seq_aligned, 'b-', 'LineWidth', 1.5);
title('HMM State Sequence (Aligned)');
ylabel('State');
xlabel('Time Bin');
grid on;

subplot(2,1,2);
plot(bhv_aligned, 'r-', 'LineWidth', 1.5);
title('Behavioral Labels (Aligned)');
ylabel('Behavior ID');
xlabel('Time Bin');
grid on;

% Plot 3: Agreement over time
figure(3); clf;
agreement_over_time = (seq_aligned == bhv_aligned) * 100;
plot(agreement_over_time, 'g-', 'LineWidth', 1.5);
title('Agreement Over Time (100% = Match, 0% = Mismatch)');
ylabel('Agreement (%)');
xlabel('Time Bin');
ylim([0 100]);
grid on;

% Plot 4: Confusion matrix
figure(4); clf;
% Get unique values
unique_states = unique(seq_aligned);
unique_behaviors = unique(bhv_aligned);

% Create confusion matrix
confusion_mat = zeros(length(unique_states), length(unique_behaviors));
for i = 1:length(unique_states)
    for j = 1:length(unique_behaviors)
        confusion_mat(i, j) = sum(seq_aligned == unique_states(i) & bhv_aligned == unique_behaviors(j));
    end
end

% Plot confusion matrix
imagesc(confusion_mat);
colorbar;
title('Confusion Matrix: HMM States vs Behavioral Labels');
xlabel('Behavior ID');
ylabel('HMM State');
xticks(1:length(unique_behaviors));
xticklabels(arrayfun(@num2str, unique_behaviors, 'UniformOutput', false));
yticks(1:length(unique_states));
yticklabels(arrayfun(@num2str, unique_states, 'UniformOutput', false));

fprintf('Plots created successfully!\n');

%% Helper Functions

function mi = mutual_information(x, y)
    % Calculate mutual information between two categorical variables
    % x, y: vectors of categorical values (including invalid values)
    % x: 0 = undefined state, >0 = state number
    % y: -1 = undefined behavior, >=0 = behavior ID
    
    % Get unique values (including invalid ones)
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

function kappa = cohens_kappa(x, y)
    % Calculate Cohen's kappa between two categorical variables
    % x, y: vectors of categorical values (including invalid values)
    % x: 0 = undefined state, >0 = state number
    % y: -1 = undefined behavior, >=0 = behavior ID
    
    % Get unique values (including invalid ones)
    unique_x = unique(x);
    unique_y = unique(y);
    
    % Create confusion matrix
    n = length(x);
    confusion_mat = zeros(length(unique_x), length(unique_y));
    
    for i = 1:length(unique_x)
        for j = 1:length(unique_y)
            confusion_mat(i, j) = sum(x == unique_x(i) & y == unique_y(j));
        end
    end
    
    % Calculate observed agreement (diagonal)
    observed_agreement = sum(diag(confusion_mat)) / n;
    
    % Calculate expected agreement
    row_sums = sum(confusion_mat, 2);
    col_sums = sum(confusion_mat, 1);
    expected_agreement = sum(row_sums .* col_sums) / (n^2);
    
    % Calculate kappa
    if expected_agreement == 1
        kappa = 1; % Perfect agreement
    else
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement);
    end
end 