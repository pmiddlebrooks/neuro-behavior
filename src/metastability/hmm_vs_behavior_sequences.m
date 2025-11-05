%% HMM vs Behavior Sequence Analysis - Motif Discovery Pipeline
% This script analyzes temporal sequences to discover prevalent motifs between 
% HMM states and behavioral categories, identifying bidirectional relationships
% and temporal patterns of length 1-4 time bins.
%
% DATA FORMAT EXPECTATIONS:
% - hmm_model.continuous_sequence: Vector where 0 = invalid (no confident state), >0 = state number
% - bhvID: Vector where -1 = invalid (undefined behavior), >=0 = behavior category ID
% - Both vectors should have the same length (totalTimeBins)
%
% OUTPUTS:
% - Motif frequency analysis for both directions (HMM->Behavior, Behavior->HMM)
% - Predictive power metrics for each motif type
% - Statistical significance testing against shuffled data
% - Visualization of discovered motifs and their temporal dynamics

%% Load saved HMM results
% Choose data type and brain area
natOrReach = 'Nat'; % 'Nat' or 'Reach'
brainArea = 'M56'; % 'M23', 'M56', 'DS', 'VS'

% Load the HMM model
[hmm_results] = hmm_load_saved_model(natOrReach, brainArea);

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
    case 'Reach'
        bhvData = get_reach_bhv_labels(fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NIBEH.mat'));
end

% For now, create placeholder - you'll replace this with actual loading
totalTimeBins = length(hmm_results.continuous_results.sequence);
% ============================================================================

fprintf('Behavior data loaded: %d time bins\n', length(bhvID));
fprintf('Invalid behavior bins (bhvID = -1): %d (%.1f%%)\n', sum(bhvID == -1), sum(bhvID == -1)/totalTimeBins*100);
fprintf('Invalid HMM bins (continuous_sequence = 0): %d (%.1f%%)\n', sum(hmm_results.continuous_results.sequence == 0), sum(hmm_results.continuous_results.sequence == 0)/totalTimeBins*100);

%% Analysis Parameters
maxMotifLength = 4; % Maximum motif length to analyze
nShuffles = 1000; % Number of shuffles for significance testing
maxLagSec = 2; % Maximum lag to test (in seconds)
maxLagBin = round(maxLagSec / hmm_results.HmmParam.BinSize); % Maximum lag in time bins

fprintf('\n=== Analysis Parameters ===\n');
fprintf('Maximum motif length: %d time bins\n', maxMotifLength);
fprintf('Number of shuffles for significance testing: %d\n', nShuffles);
fprintf('Maximum lag: %.3f seconds (%d time bins)\n', maxLagSec, maxLagBin);

%% 1. Temporal Alignment & Lag Analysis
fprintf('\n=== Step 1: Temporal Alignment & Lag Analysis ===\n');

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
    valid_mask = (hmm_segment > 0) & (bhv_segment >= 0);
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

% Plot mutual information vs lag
figure('Position', [100, 100, 800, 400]);
plot(lags, mi_values, 'b-', 'LineWidth', 2);
hold on;
plot(best_lag, best_mi, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('Lag (time bins)');
ylabel('Mutual Information');
bestLagSeconds = best_lag * hmm_results.HmmParam.BinSize;
title(sprintf('Mutual Information vs Lag (Best lag: %d bins = %.3f s)', best_lag, bestLagSeconds));
grid on;
legend('MI values', 'Best lag', 'Location', 'best');

%% 2. Core Motif Discovery Functions
fprintf('\n=== Step 2: Core Motif Discovery ===\n');

% Get unique categories from both vectors
hmmStates = unique(hmm_results.continuous_results.sequence(hmm_results.continuous_results.sequence > 0));
bhvCategories = unique(bhvID(bhvID >= 0));

nHmmStates = length(hmmStates);
nBhvCategories = length(bhvCategories);

fprintf('HMM states: %s\n', mat2str(hmmStates));
fprintf('Behavior categories: %s\n', mat2str(bhvCategories));

% Initialize motif storage
motif_results = struct();
motif_results.hmm_to_bhv = cell(maxMotifLength, 1);
motif_results.bhv_to_hmm = cell(maxMotifLength, 1);
motif_results.optimal_lag = best_lag;
motif_results.optimal_lag_seconds = best_lag * hmm_results.HmmParam.BinSize;

% Extract motifs for both directions
fprintf('\nExtracting HMM->Behavior motifs...\n');
for motifLen = 1:maxMotifLength
    fprintf('  Analyzing motif length %d...\n', motifLen);
    motif_results.hmm_to_bhv{motifLen} = extract_motifs(hmm_results.continuous_results.sequence, bhvID, motifLen, best_lag, 'hmm_to_bhv');
end

fprintf('\nExtracting Behavior->HMM motifs...\n');
for motifLen = 1:maxMotifLength
    fprintf('  Analyzing motif length %d...\n', motifLen);
    motif_results.bhv_to_hmm{motifLen} = extract_motifs(bhvID, hmm_results.continuous_results.sequence, motifLen, -best_lag, 'bhv_to_hmm');
end

%% 3. Motif Analysis & Scoring
fprintf('\n=== Step 3: Motif Analysis & Scoring ===\n');

% Analyze motifs for both directions
motif_results.hmm_to_bhv_analysis = analyze_motifs(motif_results.hmm_to_bhv, 'HMM->Behavior');
motif_results.bhv_to_hmm_analysis = analyze_motifs(motif_results.bhv_to_hmm, 'Behavior->HMM');

% Find top motifs by frequency and predictive power
motif_results.top_motifs = find_top_motifs(motif_results);

%% 4. Statistical Significance Testing
fprintf('\n=== Step 4: Statistical Significance Testing ===\n');

% Perform shuffling tests for significance
fprintf('Performing %d shuffles for significance testing...\n', nShuffles);
motif_results.significance = test_motif_significance(motif_results, nShuffles, hmm_results.continuous_results.sequence, bhvID, best_lag);

%% 5. Visualization
fprintf('\n=== Step 5: Creating Visualizations ===\n');

% Create comprehensive visualization
create_motif_visualizations(motif_results, hmm_results, bhvID);

%% 6. Save Results
fprintf('\n=== Step 6: Saving Analysis Results ===\n');

% Create comprehensive results structure
analysis_results = struct();
analysis_results.motif_results = motif_results;
analysis_results.hmm_results = hmm_results;
analysis_results.analysis_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
analysis_results.parameters = struct('maxMotifLength', maxMotifLength, 'nShuffles', nShuffles, 'maxLagSec', maxLagSec);

% Save results
output_filename = sprintf('HMM_Behavior_Sequences_%s_%s_%s.mat', ...
    hmm_results.metadata.data_type, hmm_results.metadata.brain_area, datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
save(output_filename, 'analysis_results');

fprintf('Analysis results saved to: %s\n', output_filename);
fprintf('\nMotif discovery analysis complete!\n');

%% Helper Functions

function motifs = extract_motifs(source_sequence, target_sequence, motif_length, lag, direction)
% EXTRACT_MOTIFS Extract temporal motifs and their corresponding target patterns
%
% INPUTS:
%   source_sequence - Source sequence (HMM states or behavior categories)
%   target_sequence - Target sequence to predict
%   motif_length - Length of motifs to extract
%   lag - Temporal lag between source and target
%   direction - 'hmm_to_bhv' or 'bhv_to_hmm'
%
% OUTPUTS:
%   motifs - Structure containing extracted motifs and statistics

% Initialize motif storage
motifs = struct();
motifs.length = motif_length;
motifs.direction = direction;
motifs.unique_motifs = {};
motifs.motif_counts = [];
motifs.target_patterns = {};
motifs.frequencies = [];
motifs.predictive_powers = [];

% Extract all possible motifs from source sequence
valid_indices = find(source_sequence > 0); % Exclude invalid states
n_valid = length(valid_indices);

if n_valid < motif_length
    fprintf('    Warning: Insufficient valid data for motif length %d\n', motif_length);
    return;
end

% Extract motifs and find corresponding target patterns
motif_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
motif_counts = containers.Map('KeyType', 'char', 'ValueType', 'uint32');

for i = 1:(n_valid - motif_length + 1)
    start_idx = valid_indices(i);
    end_idx = start_idx + motif_length - 1;
    
    % Extract motif
    motif = source_sequence(start_idx:end_idx);
    motif_key = mat2str(motif);
    
    % Find corresponding target pattern
    target_start = start_idx + lag;
    target_end = target_start + motif_length - 1;
    
    if target_start >= 1 && target_end <= length(target_sequence)
        target_pattern = target_sequence(target_start:target_end);
        
        % Only include if target pattern is valid
        if all(target_pattern >= 0)
            if isKey(motif_map, motif_key)
                motif_map(motif_key) = [motif_map(motif_key); target_pattern];
                motif_counts(motif_key) = motif_counts(motif_key) + 1;
            else
                motif_map(motif_key) = target_pattern;
                motif_counts(motif_key) = 1;
            end
        end
    end
end

% Convert to arrays
motif_keys = keys(motif_map);
n_motifs = length(motif_keys);

if n_motifs == 0
    fprintf('    No valid motifs found for length %d\n', motif_length);
    return;
end

motifs.unique_motifs = motif_keys;
motifs.motif_counts = zeros(n_motifs, 1);
motifs.target_patterns = cell(n_motifs, 1);
motifs.frequencies = zeros(n_motifs, 1);
motifs.predictive_powers = zeros(n_motifs, 1);

for i = 1:n_motifs
    motif_key = motif_keys{i};
    motifs.motif_counts(i) = motif_counts(motif_key);
    motifs.target_patterns{i} = motif_map(motif_key);
    
    % Calculate frequency
    motifs.frequencies(i) = motifs.motif_counts(i) / n_valid;
    
    % Calculate predictive power (how consistent the target patterns are)
    target_patterns = motifs.target_patterns{i};
    if size(target_patterns, 1) > 1
        % Calculate entropy of target pattern distribution
        unique_targets = unique(target_patterns, 'rows');
        target_entropy = 0;
        for j = 1:size(unique_targets, 1)
            pattern_count = sum(all(target_patterns == unique_targets(j, :), 2));
            pattern_prob = pattern_count / size(target_patterns, 1);
            if pattern_prob > 0
                target_entropy = target_entropy - pattern_prob * log2(pattern_prob);
            end
        end
        % Predictive power is inverse of entropy (lower entropy = higher predictive power)
        motifs.predictive_powers(i) = 1 / (1 + target_entropy);
    else
        motifs.predictive_powers(i) = 1; % Perfect prediction for single pattern
    end
end

fprintf('    Found %d unique motifs\n', n_motifs);
end

function analysis = analyze_motifs(motif_cell_array, direction_name)
% ANALYZE_MOTIFS Analyze extracted motifs and compute summary statistics
%
% INPUTS:
%   motif_cell_array - Cell array of motifs for different lengths
%   direction_name - String describing the analysis direction
%
% OUTPUTS:
%   analysis - Structure containing analysis results

analysis = struct();
analysis.direction = direction_name;
analysis.motif_lengths = [];
analysis.total_motifs = [];
analysis.avg_frequencies = [];
analysis.avg_predictive_powers = [];
analysis.top_motifs_by_frequency = {};
analysis.top_motifs_by_predictive_power = {};

n_lengths = length(motif_cell_array);

for i = 1:n_lengths
    motifs = motif_cell_array{i};
    
    if ~isempty(motifs) && isfield(motifs, 'unique_motifs')
        analysis.motif_lengths(i) = motifs.length;
        analysis.total_motifs(i) = length(motifs.unique_motifs);
        analysis.avg_frequencies(i) = mean(motifs.frequencies);
        analysis.avg_predictive_powers(i) = mean(motifs.predictive_powers);
        
        % Find top motifs by frequency
        [~, freq_idx] = sort(motifs.frequencies, 'descend');
        top_freq = motifs.unique_motifs(freq_idx(1:min(5, length(freq_idx))));
        analysis.top_motifs_by_frequency{i} = top_freq;
        
        % Find top motifs by predictive power
        [~, pred_idx] = sort(motifs.predictive_powers, 'descend');
        top_pred = motifs.unique_motifs(pred_idx(1:min(5, length(pred_idx))));
        analysis.top_motifs_by_predictive_power{i} = top_pred;
    else
        analysis.motif_lengths(i) = i;
        analysis.total_motifs(i) = 0;
        analysis.avg_frequencies(i) = 0;
        analysis.avg_predictive_powers(i) = 0;
        analysis.top_motifs_by_frequency{i} = {};
        analysis.top_motifs_by_predictive_power{i} = {};
    end
end

fprintf('  %s Analysis:\n', direction_name);
for i = 1:n_lengths
    if analysis.total_motifs(i) > 0
        fprintf('    Length %d: %d motifs, avg freq=%.4f, avg pred power=%.4f\n', ...
            analysis.motif_lengths(i), analysis.total_motifs(i), ...
            analysis.avg_frequencies(i), analysis.avg_predictive_powers(i));
    end
end
end

function top_motifs = find_top_motifs(motif_results)
% FIND_TOP_MOTIFS Find the most significant motifs across all lengths and directions
%
% INPUTS:
%   motif_results - Complete motif results structure
%
% OUTPUTS:
%   top_motifs - Structure containing top motifs by different criteria

top_motifs = struct();

% Combine all motifs from both directions
all_motifs = [];
all_frequencies = [];
all_predictive_powers = [];
all_directions = {};
all_lengths = [];

% Collect HMM->Behavior motifs
for i = 1:length(motif_results.hmm_to_bhv)
    motifs = motif_results.hmm_to_bhv{i};
    if ~isempty(motifs) && isfield(motifs, 'unique_motifs')
        for j = 1:length(motifs.unique_motifs)
            all_motifs{end+1} = motifs.unique_motifs{j};
            all_frequencies(end+1) = motifs.frequencies(j);
            all_predictive_powers(end+1) = motifs.predictive_powers(j);
            all_directions{end+1} = 'HMM->Behavior';
            all_lengths(end+1) = motifs.length;
        end
    end
end

% Collect Behavior->HMM motifs
for i = 1:length(motif_results.bhv_to_hmm)
    motifs = motif_results.bhv_to_hmm{i};
    if ~isempty(motifs) && isfield(motifs, 'unique_motifs')
        for j = 1:length(motifs.unique_motifs)
            all_motifs{end+1} = motifs.unique_motifs{j};
            all_frequencies(end+1) = motifs.frequencies(j);
            all_predictive_powers(end+1) = motifs.predictive_powers(j);
            all_directions{end+1} = 'Behavior->HMM';
            all_lengths(end+1) = motifs.length;
        end
    end
end

if isempty(all_motifs)
    fprintf('  No motifs found to analyze\n');
    return;
end

% Find top motifs by frequency
[~, freq_idx] = sort(all_frequencies, 'descend');
top_motifs.by_frequency = struct('motif', {}, 'frequency', {}, 'predictive_power', {}, 'direction', {}, 'length', {});
for i = 1:min(10, length(freq_idx))
    idx = freq_idx(i);
    top_motifs.by_frequency(i).motif = all_motifs{idx};
    top_motifs.by_frequency(i).frequency = all_frequencies(idx);
    top_motifs.by_frequency(i).predictive_power = all_predictive_powers(idx);
    top_motifs.by_frequency(i).direction = all_directions{idx};
    top_motifs.by_frequency(i).length = all_lengths(idx);
end

% Find top motifs by predictive power
[~, pred_idx] = sort(all_predictive_powers, 'descend');
top_motifs.by_predictive_power = struct('motif', {}, 'frequency', {}, 'predictive_power', {}, 'direction', {}, 'length', {});
for i = 1:min(10, length(pred_idx))
    idx = pred_idx(i);
    top_motifs.by_predictive_power(i).motif = all_motifs{idx};
    top_motifs.by_predictive_power(i).frequency = all_frequencies(idx);
    top_motifs.by_predictive_power(i).predictive_power = all_predictive_powers(idx);
    top_motifs.by_predictive_power(i).direction = all_directions{idx};
    top_motifs.by_predictive_power(i).length = all_lengths(idx);
end

% Find top motifs by combined score (frequency * predictive power)
combined_scores = all_frequencies .* all_predictive_powers;
[~, combined_idx] = sort(combined_scores, 'descend');
top_motifs.by_combined_score = struct('motif', {}, 'frequency', {}, 'predictive_power', {}, 'combined_score', {}, 'direction', {}, 'length', {});
for i = 1:min(10, length(combined_idx))
    idx = combined_idx(i);
    top_motifs.by_combined_score(i).motif = all_motifs{idx};
    top_motifs.by_combined_score(i).frequency = all_frequencies(idx);
    top_motifs.by_combined_score(i).predictive_power = all_predictive_powers(idx);
    top_motifs.by_combined_score(i).combined_score = combined_scores(idx);
    top_motifs.by_combined_score(i).direction = all_directions{idx};
    top_motifs.by_combined_score(i).length = all_lengths(idx);
end

fprintf('\nTop motifs by frequency:\n');
for i = 1:min(5, length(top_motifs.by_frequency))
    fprintf('  %d. %s (freq=%.4f, pred=%.4f, %s, length=%d)\n', ...
        i, top_motifs.by_frequency(i).motif, ...
        top_motifs.by_frequency(i).frequency, ...
        top_motifs.by_frequency(i).predictive_power, ...
        top_motifs.by_frequency(i).direction, ...
        top_motifs.by_frequency(i).length);
end

fprintf('\nTop motifs by predictive power:\n');
for i = 1:min(5, length(top_motifs.by_predictive_power))
    fprintf('  %d. %s (freq=%.4f, pred=%.4f, %s, length=%d)\n', ...
        i, top_motifs.by_predictive_power(i).motif, ...
        top_motifs.by_predictive_power(i).frequency, ...
        top_motifs.by_predictive_power(i).predictive_power, ...
        top_motifs.by_predictive_power(i).direction, ...
        top_motifs.by_predictive_power(i).length);
end
end

function significance = test_motif_significance(motif_results, n_shuffles, hmm_sequence, bhv_sequence, lag)
% TEST_MOTIF_SIGNIFICANCE Test statistical significance of motifs using shuffling
%
% INPUTS:
%   motif_results - Complete motif results structure
%   n_shuffles - Number of shuffles to perform
%   hmm_sequence - Original HMM sequence
%   bhv_sequence - Original behavior sequence
%   lag - Optimal lag found
%
% OUTPUTS:
%   significance - Structure containing significance test results

fprintf('  Testing significance for HMM->Behavior motifs...\n');
significance.hmm_to_bhv = test_direction_significance(motif_results.hmm_to_bhv, n_shuffles, hmm_sequence, bhv_sequence, lag, 'hmm_to_bhv');

fprintf('  Testing significance for Behavior->HMM motifs...\n');
significance.bhv_to_hmm = test_direction_significance(motif_results.bhv_to_hmm, n_shuffles, bhv_sequence, hmm_sequence, -lag, 'bhv_to_hmm');
end

function sig_results = test_direction_significance(motif_cell_array, n_shuffles, source_sequence, target_sequence, lag, direction)
% TEST_DIRECTION_SIGNIFICANCE Test significance for one direction
%
% INPUTS:
%   motif_cell_array - Cell array of motifs for different lengths
%   n_shuffles - Number of shuffles to perform
%   source_sequence - Source sequence
%   target_sequence - Target sequence
%   lag - Temporal lag
%   direction - Direction string
%
% OUTPUTS:
%   sig_results - Significance test results

sig_results = struct();
sig_results.direction = direction;
sig_results.motif_lengths = [];
sig_results.p_values = [];
sig_results.shuffled_frequencies = {};

n_lengths = length(motif_cell_array);

for i = 1:n_lengths
    motifs = motif_cell_array{i};
    
    if ~isempty(motifs) && isfield(motifs, 'unique_motifs')
        fprintf('    Testing length %d motifs...\n', i);
        
        sig_results.motif_lengths(i) = i;
        
        % Get observed frequencies
        observed_frequencies = motifs.frequencies;
        
        % Perform shuffling
        shuffled_freqs = zeros(length(observed_frequencies), n_shuffles);
        
        for shuffle = 1:n_shuffles
            % Shuffle the source sequence while preserving temporal structure
            shuffled_source = shuffle_sequence(source_sequence);
            
            % Extract motifs from shuffled sequence
            shuffled_motifs = extract_motifs(shuffled_source, target_sequence, i, lag, direction);
            
            if ~isempty(shuffled_motifs) && isfield(shuffled_motifs, 'frequencies')
                % Match shuffled motifs to observed motifs
                for j = 1:length(observed_frequencies)
                    if j <= length(shuffled_motifs.frequencies)
                        shuffled_freqs(j, shuffle) = shuffled_motifs.frequencies(j);
                    end
                end
            end
        end
        
        sig_results.shuffled_frequencies{i} = shuffled_freqs;
        
        % Calculate p-values
        p_values = zeros(length(observed_frequencies), 1);
        for j = 1:length(observed_frequencies)
            if j <= size(shuffled_freqs, 1)
                % Count how many shuffled frequencies are >= observed
                n_greater_equal = sum(shuffled_freqs(j, :) >= observed_frequencies(j));
                p_values(j) = (n_greater_equal + 1) / (n_shuffles + 1); % Add 1 for continuity correction
            else
                p_values(j) = 1; % No data for this motif
            end
        end
        
        sig_results.p_values{i} = p_values;
        
        % Report significant motifs
        significant_motifs = find(p_values < 0.05);
        if ~isempty(significant_motifs)
            fprintf('      %d significant motifs found (p < 0.05)\n', length(significant_motifs));
        else
            fprintf('      No significant motifs found\n');
        end
    else
        sig_results.motif_lengths(i) = i;
        sig_results.p_values{i} = [];
        sig_results.shuffled_frequencies{i} = [];
    end
end
end

function shuffled_sequence = shuffle_sequence(sequence)
% SHUFFLE_SEQUENCE Shuffle sequence while preserving temporal structure
%
% INPUTS:
%   sequence - Original sequence
%
% OUTPUTS:
%   shuffled_sequence - Shuffled sequence

% Simple random shuffle for now - could be enhanced with more sophisticated methods
valid_indices = find(sequence > 0);
shuffled_values = sequence(valid_indices);
shuffled_values = shuffled_values(randperm(length(shuffled_values)));

shuffled_sequence = sequence;
shuffled_sequence(valid_indices) = shuffled_values;
end

function create_motif_visualizations(motif_results, hmm_results, bhv_sequence)
% CREATE_MOTIF_VISUALIZATIONS Create comprehensive visualizations of discovered motifs
%
% INPUTS:
%   motif_results - Complete motif results structure
%   hmm_results - HMM analysis results
%   bhv_sequence - Behavior sequence

fprintf('  Creating motif visualizations...\n');

% Create main figure
figure('Position', [100, 100, 1400, 1000]);

% Subplot 1: Motif frequency by length
subplot(3, 4, 1);
plot_motif_frequency_by_length(motif_results);

% Subplot 2: Predictive power by length
subplot(3, 4, 2);
plot_predictive_power_by_length(motif_results);

% Subplot 3: Top motifs heatmap
subplot(3, 4, 3);
plot_top_motifs_heatmap(motif_results);

% Subplot 4: Direction comparison
subplot(3, 4, 4);
plot_direction_comparison(motif_results);

% Subplot 5: Motif length distribution
subplot(3, 4, 5);
plot_motif_length_distribution(motif_results);

% Subplot 6: Significance results
subplot(3, 4, 6);
plot_significance_results(motif_results);

% Subplot 7: Temporal motif occurrence
subplot(3, 4, 7);
plot_temporal_motif_occurrence(motif_results, hmm_results, bhv_sequence);

% Subplot 8: Motif transition network
subplot(3, 4, 8);
plot_motif_transition_network(motif_results);

% Subplot 9: Combined score analysis
subplot(3, 4, 9);
plot_combined_score_analysis(motif_results);

% Subplot 10: Motif stability analysis
subplot(3, 4, 10);
plot_motif_stability(motif_results);

% Subplot 11: Summary statistics
subplot(3, 4, 11);
plot_summary_statistics(motif_results);

% Subplot 12: Analysis parameters
subplot(3, 4, 12);
plot_analysis_parameters(motif_results);

% Adjust layout
sgtitle('HMM-Behavior Motif Discovery Analysis', 'FontSize', 16);
end

function plot_motif_frequency_by_length(motif_results)
% Plot motif frequency by length for both directions
hmm_lengths = [];
hmm_freqs = [];
bhv_lengths = [];
bhv_freqs = [];

for i = 1:length(motif_results.hmm_to_bhv)
    if ~isempty(motif_results.hmm_to_bhv{i}) && isfield(motif_results.hmm_to_bhv{i}, 'avg_frequencies')
        hmm_lengths(end+1) = i;
        hmm_freqs(end+1) = mean(motif_results.hmm_to_bhv{i}.frequencies);
    end
end

for i = 1:length(motif_results.bhv_to_hmm)
    if ~isempty(motif_results.bhv_to_hmm{i}) && isfield(motif_results.bhv_to_hmm{i}, 'avg_frequencies')
        bhv_lengths(end+1) = i;
        bhv_freqs(end+1) = mean(motif_results.bhv_to_hmm{i}.frequencies);
    end
end

plot(hmm_lengths, hmm_freqs, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
hold on;
plot(bhv_lengths, bhv_freqs, 'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('Motif Length');
ylabel('Average Frequency');
title('Motif Frequency by Length');
legend('HMM->Behavior', 'Behavior->HMM', 'Location', 'best');
grid on;
end

function plot_predictive_power_by_length(motif_results)
% Plot predictive power by length for both directions
hmm_lengths = [];
hmm_powers = [];
bhv_lengths = [];
bhv_powers = [];

for i = 1:length(motif_results.hmm_to_bhv)
    if ~isempty(motif_results.hmm_to_bhv{i}) && isfield(motif_results.hmm_to_bhv{i}, 'predictive_powers')
        hmm_lengths(end+1) = i;
        hmm_powers(end+1) = mean(motif_results.hmm_to_bhv{i}.predictive_powers);
    end
end

for i = 1:length(motif_results.bhv_to_hmm)
    if ~isempty(motif_results.bhv_to_hmm{i}) && isfield(motif_results.bhv_to_hmm{i}, 'predictive_powers')
        bhv_lengths(end+1) = i;
        bhv_powers(end+1) = mean(motif_results.bhv_to_hmm{i}.predictive_powers);
    end
end

plot(hmm_lengths, hmm_powers, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
hold on;
plot(bhv_lengths, bhv_powers, 'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'r');
xlabel('Motif Length');
ylabel('Average Predictive Power');
title('Predictive Power by Length');
legend('HMM->Behavior', 'Behavior->HMM', 'Location', 'best');
grid on;
end

function plot_top_motifs_heatmap(motif_results)
% Plot heatmap of top motifs
if isfield(motif_results, 'top_motifs') && isfield(motif_results.top_motifs, 'by_combined_score')
    top_motifs = motif_results.top_motifs.by_combined_score;
    
    if length(top_motifs) > 0
        % Create data matrix for heatmap
        n_top = min(10, length(top_motifs));
        data_matrix = zeros(n_top, 3);
        
        for i = 1:n_top
            data_matrix(i, 1) = top_motifs(i).frequency;
            data_matrix(i, 2) = top_motifs(i).predictive_power;
            data_matrix(i, 3) = top_motifs(i).length;
        end
        
        % Normalize data for visualization
        data_norm = data_matrix ./ max(data_matrix, [], 1);
        
        imagesc(data_norm);
        colormap('hot');
        colorbar;
        
        % Set labels
        xticks(1:3);
        xticklabels({'Freq', 'Pred', 'Len'});
        yticks(1:n_top);
        yticklabels(arrayfun(@(x) sprintf('Motif %d', x), 1:n_top, 'UniformOutput', false));
        
        title('Top Motifs Heatmap');
    else
        text(0.5, 0.5, 'No motifs found', 'HorizontalAlignment', 'center');
        axis off;
    end
else
    text(0.5, 0.5, 'No motif data', 'HorizontalAlignment', 'center');
    axis off;
end
end

function plot_direction_comparison(motif_results)
% Compare HMM->Behavior vs Behavior->HMM performance
hmm_performance = [];
bhv_performance = [];

for i = 1:length(motif_results.hmm_to_bhv)
    if ~isempty(motif_results.hmm_to_bhv{i}) && isfield(motif_results.hmm_to_bhv{i}, 'frequencies')
        hmm_performance(end+1) = mean(motif_results.hmm_to_bhv{i}.frequencies);
    end
end

for i = 1:length(motif_results.bhv_to_hmm)
    if ~isempty(motif_results.bhv_to_hmm{i}) && isfield(motif_results.bhv_to_hmm{i}, 'frequencies')
        bhv_performance(end+1) = bhv_performance + mean(motif_results.bhv_to_hmm{i}.frequencies);
    end
end

if ~isempty(hmm_performance) && ~isempty(bhv_performance)
    bar([mean(hmm_performance), mean(bhv_performance)]);
    xticklabels({'HMM->Behavior', 'Behavior->HMM'});
    ylabel('Average Performance');
    title('Direction Performance Comparison');
    grid on;
else
    text(0.5, 0.5, 'Insufficient data', 'HorizontalAlignment', 'center');
    axis off;
end
end

function plot_motif_length_distribution(motif_results)
% Plot distribution of motifs by length
lengths = 1:length(motif_results.hmm_to_bhv);
hmm_counts = zeros(size(lengths));
bhv_counts = zeros(size(lengths));

for i = 1:length(lengths)
    if ~isempty(motif_results.hmm_to_bhv{i}) && isfield(motif_results.hmm_to_bhv{i}, 'unique_motifs')
        hmm_counts(i) = length(motif_results.hmm_to_bhv{i}.unique_motifs);
    end
    if ~isempty(motif_results.bhv_to_hmm{i}) && isfield(motif_results.bhv_to_hmm{i}, 'unique_motifs')
        bhv_counts(i) = length(motif_results.bhv_to_hmm{i}.unique_motifs);
    end
end

bar(lengths, [hmm_counts', bhv_counts']);
xlabel('Motif Length');
ylabel('Number of Unique Motifs');
title('Motif Distribution by Length');
legend('HMM->Behavior', 'Behavior->HMM', 'Location', 'best');
grid on;
end

function plot_significance_results(motif_results)
% Plot significance test results
if isfield(motif_results, 'significance')
    sig = motif_results.significance;
    
    % Count significant motifs by length
    hmm_sig_counts = [];
    bhv_sig_counts = [];
    
    for i = 1:length(sig.hmm_to_bhv.p_values)
        if ~isempty(sig.hmm_to_bhv.p_values{i})
            hmm_sig_counts(end+1) = sum(sig.hmm_to_bhv.p_values{i} < 0.05);
        end
    end
    
    for i = 1:length(sig.bhv_to_hmm.p_values)
        if ~isempty(sig.bhv_to_hmm.p_values{i})
            bhv_sig_counts(end+1) = sum(sig.bhv_to_hmm.p_values{i} < 0.05);
        end
    end
    
    if ~isempty(hmm_sig_counts) && ~isempty(bhv_sig_counts)
        lengths = 1:length(hmm_sig_counts);
        bar(lengths, [hmm_sig_counts', bhv_sig_counts']);
        xlabel('Motif Length');
        ylabel('Significant Motifs (p < 0.05)');
        title('Significant Motifs by Length');
        legend('HMM->Behavior', 'Behavior->HMM', 'Location', 'best');
        grid on;
    else
        text(0.5, 0.5, 'No significance data', 'HorizontalAlignment', 'center');
        axis off;
    end
else
    text(0.5, 0.5, 'No significance data', 'HorizontalAlignment', 'center');
    axis off;
end
end

function plot_temporal_motif_occurrence(motif_results, hmm_results, bhv_sequence)
% Plot temporal occurrence of top motifs
if isfield(motif_results, 'top_motifs') && isfield(motif_results.top_motifs, 'by_frequency')
    top_motifs = motif_results.top_motifs.by_frequency;
    
    if length(top_motifs) > 0
        % Plot first few top motifs
        n_plot = min(3, length(top_motifs));
        
        for i = 1:n_plot
            % This is a simplified plot - could be enhanced with actual temporal data
            plot([i, i+1], [top_motifs(i).frequency, top_motifs(i).predictive_power], 'o-', 'LineWidth', 2);
            hold on;
        end
        
        xlabel('Motif Index');
        ylabel('Performance Metric');
        title('Top Motifs Performance');
        legend(arrayfun(@(x) sprintf('Motif %d', x), 1:n_plot, 'UniformOutput', false), 'Location', 'best');
        grid on;
    else
        text(0.5, 0.5, 'No motif data', 'HorizontalAlignment', 'center');
        axis off;
    end
else
    text(0.5, 0.5, 'No motif data', 'HorizontalAlignment', 'center');
    axis off;
end
end

function plot_motif_transition_network(motif_results)
% Plot motif transition network (simplified)
text(0.5, 0.5, 'Motif Transition Network\n(Implementation needed)', 'HorizontalAlignment', 'center');
axis off;
end

function plot_combined_score_analysis(motif_results)
% Plot combined score analysis
if isfield(motif_results, 'top_motifs') && isfield(motif_results.top_motifs, 'by_combined_score')
    top_motifs = motif_results.top_motifs.by_combined_score;
    
    if length(top_motifs) > 0
        n_plot = min(10, length(top_motifs));
        scores = zeros(n_plot, 1);
        labels = cell(n_plot, 1);
        
        for i = 1:n_plot
            scores(i) = top_motifs(i).combined_score;
            labels{i} = sprintf('M%d', i);
        end
        
        bar(scores);
        xticks(1:n_plot);
        xticklabels(labels);
        ylabel('Combined Score');
        title('Top Motifs by Combined Score');
        grid on;
    else
        text(0.5, 0.5, 'No motif data', 'HorizontalAlignment', 'center');
        axis off;
    end
else
    text(0.5, 0.5, 'No motif data', 'HorizontalAlignment', 'center');
    axis off;
end
end

function plot_motif_stability(motif_results)
% Plot motif stability analysis
text(0.5, 0.5, 'Motif Stability Analysis\n(Implementation needed)', 'HorizontalAlignment', 'center');
axis off;
end

function plot_summary_statistics(motif_results)
% Plot summary statistics
if isfield(motif_results, 'hmm_to_bhv_analysis') && isfield(motif_results, 'bhv_to_hmm_analysis')
    hmm_analysis = motif_results.hmm_to_bhv_analysis;
    bhv_analysis = motif_results.bhv_to_hmm_analysis;
    
    % Calculate summary stats
    total_hmm_motifs = sum(hmm_analysis.total_motifs);
    total_bhv_motifs = sum(bhv_analysis.total_motifs);
    avg_hmm_freq = mean(hmm_analysis.avg_frequencies);
    avg_bhv_freq = mean(bhv_analysis.avg_frequencies);
    
    text(0.1, 0.9, sprintf('Total HMM->Behavior Motifs: %d', total_hmm_motifs), 'FontSize', 10);
    text(0.1, 0.8, sprintf('Total Behavior->HMM Motifs: %d', total_bhv_motifs), 'FontSize', 10);
    text(0.1, 0.7, sprintf('Avg HMM->Behavior Freq: %.4f', avg_hmm_freq), 'FontSize', 10);
    text(0.1, 0.6, sprintf('Avg Behavior->HMM Freq: %.4f', avg_bhv_freq), 'FontSize', 10);
    text(0.1, 0.5, sprintf('Optimal Lag: %.3f s', motif_results.optimal_lag_seconds), 'FontSize', 10);
    
    axis off;
    title('Summary Statistics');
else
    text(0.5, 0.5, 'No analysis data', 'HorizontalAlignment', 'center');
    axis off;
end
end

function plot_analysis_parameters(motif_results)
% Plot analysis parameters
text(0.1, 0.9, 'Analysis Parameters:', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.8, sprintf('Max Motif Length: %d', 4), 'FontSize', 10);
text(0.1, 0.7, sprintf('Shuffles: %d', 1000), 'FontSize', 10);
text(0.1, 0.6, sprintf('Max Lag: %.3f s', 2.0), 'FontSize', 10);
text(0.1, 0.5, sprintf('Data Type: %s', 'Naturalistic'), 'FontSize', 10);
text(0.1, 0.4, sprintf('Brain Area: %s', 'M56'), 'FontSize', 10);

axis off;
title('Analysis Parameters');
end

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
