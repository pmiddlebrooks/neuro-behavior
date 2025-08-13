%% HMM vs Behavior Analysis for Naturalistic Data
% This script compares HMM state sequences with behavioral labels
% to find the best temporal alignment and measure agreement
% ASSUMES: bhvID is already loaded in the workspace
%
% UPDATED: Now uses continuous results directly from hmm_results.continuous_results
% instead of converting trial-based sequences to continuous time series


%% USER CHOICE: Specify how to get HMM data
% Set this to 'workspace' to use existing 'res' variable, or 'load' to load a saved model
    fprintf('Loading saved HMM model...\n');

    % Parameters for loading saved model - CHANGE THESE AS NEEDED
    natOrReach = 'Nat'; % 'Nat' for naturalistic data
    brainArea = 'M56';  % 'M23', 'M56', 'DS', 'VS'

    % Load the saved model with continuous results
    [hmm_res] = hmm_load_saved_model(natOrReach, brainArea);

    % Check if continuous results are available
    if isempty(hmm_res.continuous_results.sequence)
        error('No continuous results found in the loaded data. Make sure the HMM analysis was completed successfully.');
    end

    fprintf('Loaded model: %s data from %s area\n', natOrReach, brainArea);
    fprintf('Continuous sequence length: %d time bins\n', length(hmm_res.continuous_results.sequence));


%% Get behavioral data at the same sampling size as the HMM
opts = neuro_behavior_options;
opts.frameSize = hmm_res.HmmParam.BinSize;
opts.collectStart =     hmm_res.data_params.collect_start;
opts.collectFor = hmm_res.data_params.collect_duration;
getDataType = 'behavior';
get_standard_data



%% Parameters for analysis
maxLag = 4 / hmm_res.HmmParam.BinSize; % Maximum lag to test (in time bins)
fprintf('Maximum lag for analysis: %d time bins (%.1f seconds)\n', maxLag, maxLag * hmm_res.HmmParam.BinSize);

%% Prepare data for analysis
fprintf('Preparing data for analysis...\n');

% Use continuous sequence directly (keeping all bins including 0s)
continuous_sequence = hmm_res.continuous_results.sequence;
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
fprintf(['Note: Invalid values (sequence=0, behavior=-1) are excluded from mutual information calculation at each lag.\n' ...
    'This means only time bins where both are valid are used, which may result in non-contiguous time bins being compared.\n' ...
    'This is standard for mutual information between two variables, but be aware that temporal gaps are ignored in the MI calculation.\n']);

% Calculate mutual information at different lags, EXCLUDING invalid values
lags = -maxLag:maxLag;
mi_values = nan(length(lags), 1);
n_valid_pairs = zeros(length(lags), 1);

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
        n_valid_pairs(i) = 0;
        continue;
    end

    % Extract aligned segments
    seq_segment = continuous_sequence(seq_start:seq_end);
    bhv_segment = bhvID_full(bhv_start:bhv_end);

    % Mask out invalid values (sequence==0 or behavior==-1)
    valid_mask = (seq_segment > 0) & (bhv_segment >= 0);

    n_valid_pairs(i) = sum(valid_mask);

    if n_valid_pairs(i) < 2
        % Not enough valid data to compute MI
        mi_values(i) = NaN;
        continue;
    end

    seq_valid = seq_segment(valid_mask);
    bhv_valid = bhv_segment(valid_mask);

    % Calculate mutual information on valid data only
    mi_values(i) = mutual_information(seq_valid, bhv_valid);
end

% Find best lag
[best_mi, best_lag_idx] = max(mi_values);
best_lag = lags(best_lag_idx);

% Interpret the lag direction
if best_lag > 0
    lag_interpretation = sprintf('Neural activity (HMM sequence) LEADS behavior by %d time bins (%.3f seconds)', ...
        best_lag, best_lag * hmm_res.HmmParam.BinSize);
elseif best_lag < 0
    lag_interpretation = sprintf('Behavior LEADS neural activity (HMM sequence) by %d time bins (%.3f seconds)', ...
        abs(best_lag), abs(best_lag) * hmm_res.HmmParam.BinSize);
else
    lag_interpretation = 'Neural activity and behavior are synchronized (no lag)';
end

fprintf('Best lag: %d time bins (MI = %.4f, %d valid pairs)\n', best_lag, best_mi, n_valid_pairs(best_lag_idx));
fprintf('Lag interpretation: %s\n', lag_interpretation);
fprintf('Note: MI is computed only on time bins where both sequence and behavior are valid at each lag.\n');
fprintf('\nLag direction explanation:\n');
fprintf('  Positive lag (+): HMM sequence comes FIRST, behavior follows\n');
fprintf('  Negative lag (-): Behavior comes FIRST, HMM sequence follows\n');
fprintf('  Zero lag (0): HMM sequence and behavior are synchronized\n');

%% 2. Calculate agreement metrics at best lag
fprintf('\n=== Step 2: Calculating agreement metrics at best lag ===\n');
fprintf('Note: Agreement metrics exclude invalid values (sequence=0, behavior=-1)\n');

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

% Mask out invalid values for agreement calculation
valid_mask = (seq_aligned > 0) & (bhv_aligned >= 0);
n_valid_agreement = sum(valid_mask);

fprintf('Total aligned time bins: %d\n', length(seq_aligned));
fprintf('Valid time bins for agreement: %d\n', n_valid_agreement);

if n_valid_agreement < 2
    error('Not enough valid data pairs for agreement calculation (need at least 2, got %d)', n_valid_agreement);
end

% Extract valid data for agreement calculation
seq_valid = seq_aligned(valid_mask);
bhv_valid = bhv_aligned(valid_mask);

% Calculate percent agreement on valid data only
agreement = sum(seq_valid == bhv_valid) / length(seq_valid) * 100;

% Calculate Cohen's kappa on valid data only
kappa = cohens_kappa(seq_valid, bhv_valid);

fprintf('Percent Agreement (valid data only): %.2f%%\n', agreement);
fprintf('Cohen''s Kappa (valid data only): %.4f\n', kappa);

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
fprintf('Valid Sequence Points: %d\n', sum(continuous_sequence > 0));
fprintf('Valid Behavior Points: %d\n', sum(bhvID_full >= 0));
fprintf('Best Lag: %d time bins (%.3f seconds)\n', best_lag, best_lag * hmm_res.HmmParam.BinSize);
fprintf('Lag Direction: %s\n', lag_interpretation);
fprintf('Mutual Information at Best Lag: %.4f (on %d valid pairs)\n', best_mi, n_valid_pairs(best_lag_idx));
fprintf('Agreement Calculation: %d valid time bins out of %d aligned\n', n_valid_agreement, length(seq_aligned));
fprintf('Percent Agreement (valid data only): %.2f%%\n', agreement);
fprintf('Cohen''s Kappa (valid data only): %.4f\n', kappa);

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

% Plot 4: Confusion matrix (valid data only)
figure(4); clf;
% Get unique values from valid data only
unique_states = unique(seq_valid);
unique_behaviors = unique(bhv_valid);

% Create confusion matrix using valid data
confusion_mat = zeros(length(unique_states), length(unique_behaviors));
for i = 1:length(unique_states)
    for j = 1:length(unique_behaviors)
        confusion_mat(i, j) = sum(seq_valid == unique_states(i) & bhv_valid == unique_behaviors(j));
    end
end

% Plot confusion matrix
imagesc(confusion_mat);
colorbar;
title('Confusion Matrix: HMM States vs Behavioral Labels (Valid Data Only)');
xlabel('Behavior ID');
ylabel('HMM State');
xticks(1:length(unique_behaviors));
xticklabels(arrayfun(@num2str, unique_behaviors, 'UniformOutput', false));
yticks(1:length(unique_states));
yticklabels(arrayfun(@num2str, unique_states, 'UniformOutput', false));

fprintf('Plots created successfully!\n');
fprintf('Note: Confusion matrix shows only valid data pairs (excludes sequence=0 and behavior=-1)\n');

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