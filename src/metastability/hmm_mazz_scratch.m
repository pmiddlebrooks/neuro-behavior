%%
% HMM State Assignment Comparison: Reach vs Spontaneous
% 
% Loads HMM results for both reach and spontaneous data and compares
% the proportion of time assigned to states (state ~= 0) across brain areas
%
% Variables:
%   resultsReach - loaded reach HMM results
%   resultsNat - loaded spontaneous HMM results
%   areas - brain areas analyzed
%   propAssignedReach - proportion of assigned states for reach data per area
%   propAssignedNat - proportion of assigned states for spontaneous data per area

%% Setup
paths = get_paths;

% HMM parameters (must match the files you want to load)
binSize = 0.01;
minDur = 0.04;

% Brain areas
areas = {'M23', 'M56', 'DS', 'VS'};

%% Load Reach HMM Results
fprintf('\n=== Loading Reach HMM Results ===\n');
try
    resultsReach = hmm_load_saved_model('Reach', 'binSize', binSize, 'minDur', minDur);
    if isfield(resultsReach, 'hmm_results')
        % All areas loaded
        hmmResultsReach = resultsReach.hmm_results;
        areasReach = resultsReach.areas;
    else
        error('Reach results structure not in expected format');
    end
    fprintf('Successfully loaded reach HMM results\n');
catch ME
    error('Failed to load reach HMM results: %s', ME.message);
end

% Load Spontaneous HMM Results
fprintf('\n=== Loading Spontaneous HMM Results ===\n');
try
    resultsNat = hmm_load_saved_model('Nat', 'binSize', binSize, 'minDur', minDur);
    if isfield(resultsNat, 'hmm_results')
        % All areas loaded
        hmmResultsNat = resultsNat.hmm_results;
        areasNat = resultsNat.areas;
    else
        error('Spontaneous results structure not in expected format');
    end
    fprintf('Successfully loaded spontaneous HMM results\n');
catch ME
    error('Failed to load spontaneous HMM results: %s', ME.message);
end

%% Calculate Proportion of Assigned States for Each Area
fprintf('\n=== Calculating State Assignment Proportions ===\n');

% Initialize storage
propAssignedReach = nan(1, length(areas));
propAssignedNat = nan(1, length(areas));

% Map area names to indices
areaMap = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});

for a = 1:length(areas)
    areaName = areas{a};
    areaIdx = areaMap(areaName);
    
    fprintf('\nArea: %s\n', areaName);
    
    % Reach data
    if areaIdx <= length(hmmResultsReach) && ~isempty(hmmResultsReach{areaIdx})
        hmmResReach = hmmResultsReach{areaIdx};
        if isfield(hmmResReach, 'continuous_results') && ...
           isfield(hmmResReach.continuous_results, 'sequence')
            seqReach = hmmResReach.continuous_results.sequence;
            if ~isempty(seqReach)
                % Calculate proportion where state ~= 0 (assigned states)
                propAssignedReach(a) = sum(seqReach ~= 0) / length(seqReach);
                fprintf('  Reach: %.3f assigned (%.1f%%)\n', propAssignedReach(a), propAssignedReach(a)*100);
            else
                fprintf('  Reach: No sequence data\n');
            end
        else
            fprintf('  Reach: No continuous results\n');
        end
    else
        fprintf('  Reach: No HMM results\n');
    end
    
    % Spontaneous data
    if areaIdx <= length(hmmResultsNat) && ~isempty(hmmResultsNat{areaIdx})
        hmmResNat = hmmResultsNat{areaIdx};
        if isfield(hmmResNat, 'continuous_results') && ...
           isfield(hmmResNat.continuous_results, 'sequence')
            seqNat = hmmResNat.continuous_results.sequence;
            if ~isempty(seqNat)
                % Calculate proportion where state ~= 0 (assigned states)
                propAssignedNat(a) = sum(seqNat ~= 0) / length(seqNat);
                fprintf('  Spontaneous: %.3f assigned (%.1f%%)\n', propAssignedNat(a), propAssignedNat(a)*100);
            else
                fprintf('  Spontaneous: No sequence data\n');
            end
        else
            fprintf('  Spontaneous: No continuous results\n');
        end
    else
        fprintf('  Spontaneous: No HMM results\n');
    end
end

%% Create Bar Plot
fprintf('\n=== Creating Comparison Plot ===\n');

figure; clf;
hold on;

% Prepare data for grouped bar plot
barData = [propAssignedReach; propAssignedNat]'; % Areas x Conditions

% Create grouped bar plot
b = bar(barData, 'grouped');

% Set colors
b(1).FaceColor = [0.8500 0.3250 0.0980]; % Orange-red for Reach
b(2).FaceColor = [0 0.4470 0.7410];     % Blue for Spontaneous

% Customize axes
xticks(1:length(areas))
set(gca, 'XTickLabel', areas);
xlabel('Brain Area', 'FontSize', 12);
ylabel('Proportion of Assigned States', 'FontSize', 12);
title(sprintf('HMM State Assignment: Reach vs Spontaneous\n(binSize=%.3f, minDur=%.3f)', binSize, minDur), 'FontSize', 14);

% Add legend
legend({'Reach', 'Spontaneous'}, 'Location', 'best', 'FontSize', 11);

% Add grid
grid on;
set(gca, 'GridAlpha', 0.3);

% Set y-axis limits
ylim([0, 1]);

% Format y-axis as percentage
yticklabels = get(gca, 'YTickLabel');
if ~isempty(yticklabels)
    yticks = get(gca, 'YTick');
    yticklabels = cellstr(num2str(yticks' * 100, '%.0f%%'));
    set(gca, 'YTickLabel', yticklabels);
end

% Add value labels on bars
for a = 1:length(areas)
    % Reach bar
    if ~isnan(propAssignedReach(a))
        text(a - 0.15, propAssignedReach(a) + 0.02, sprintf('%.2f', propAssignedReach(a)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', [0.8500 0.3250 0.0980]);
    end
    % Spontaneous bar
    if ~isnan(propAssignedNat(a))
        text(a + 0.15, propAssignedNat(a) + 0.02, sprintf('%.2f', propAssignedNat(a)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', [0 0.4470 0.7410]);
    end
end

hold off;

% Adjust figure size
set(gcf, 'Position', [100, 100, 800, 600]);

% Save figure
hmmdir = fullfile(paths.dropPath, 'metastability');
if ~exist(hmmdir, 'dir')
    mkdir(hmmdir);
end
savePath = fullfile(hmmdir, sprintf('hmm_state_assignment_comparison_bin%.3f_minDur%.3f.png', binSize, minDur));
exportgraphics(gcf, savePath, 'Resolution', 300);
fprintf('Saved plot to: %s\n', savePath);

% Also save as .fig
figPath = fullfile(hmmdir, sprintf('hmm_state_assignment_comparison_bin%.3f_minDur%.3f.fig', binSize, minDur));
saveas(gcf, figPath);
fprintf('Saved figure to: %s\n', figPath);

%% Calculate State Transition Rates
fprintf('\n=== Calculating State Transition Rates ===\n');

% Initialize storage
transitionRateReach = nan(1, length(areas));
transitionRateNat = nan(1, length(areas));


for a = 1:length(areas)
    areaName = areas{a};
    areaIdx = areaMap(areaName);
    
    fprintf('\nArea: %s\n', areaName);
    
    % Reach data
    if areaIdx <= length(hmmResultsReach) && ~isempty(hmmResultsReach{areaIdx})
        hmmResReach = hmmResultsReach{areaIdx};
        if isfield(hmmResReach, 'continuous_results') && ...
           isfield(hmmResReach.continuous_results, 'sequence')
            seqReach = hmmResReach.continuous_results.sequence;
            if ~isempty(seqReach)
                % Get bin size for this area
                if isfield(hmmResReach, 'HmmParam') && isfield(hmmResReach.HmmParam, 'BinSize')
                    binSizeReach = hmmResReach.HmmParam.BinSize;
                else
                    binSizeReach = binSize; % Use default
                end
                
                % Collapse unassigned states between same states
                seqCollapsed = collapse_unassigned_between_same_state(seqReach);
                
                % Count transitions
                numTransitions = count_transitions(seqCollapsed);
                
                % Calculate total time in seconds
                totalTime = length(seqReach) * binSizeReach;
                
                % Calculate transition rate (transitions per second)
                if totalTime > 0
                    transitionRateReach(a) = numTransitions / totalTime;
                    fprintf('  Reach: %d transitions, %.2f s total time, %.3f transitions/s\n', ...
                        numTransitions, totalTime, transitionRateReach(a));
                else
                    fprintf('  Reach: No time data\n');
                end
            else
                fprintf('  Reach: No sequence data\n');
            end
        else
            fprintf('  Reach: No continuous results\n');
        end
    else
        fprintf('  Reach: No HMM results\n');
    end
    
    % Spontaneous data
    if areaIdx <= length(hmmResultsNat) && ~isempty(hmmResultsNat{areaIdx})
        hmmResNat = hmmResultsNat{areaIdx};
        if isfield(hmmResNat, 'continuous_results') && ...
           isfield(hmmResNat.continuous_results, 'sequence')
            seqNat = hmmResNat.continuous_results.sequence;
            if ~isempty(seqNat)
                % Get bin size for this area
                if isfield(hmmResNat, 'HmmParam') && isfield(hmmResNat.HmmParam, 'BinSize')
                    binSizeNat = hmmResNat.HmmParam.BinSize;
                else
                    binSizeNat = binSize; % Use default
                end
                
                % Collapse unassigned states between same states
                seqCollapsed = collapse_unassigned_between_same_state(seqNat);
                
                % Count transitions
                numTransitions = count_transitions(seqCollapsed);
                
                % Calculate total time in seconds
                totalTime = length(seqNat) * binSizeNat;
                
                % Calculate transition rate (transitions per second)
                if totalTime > 0
                    transitionRateNat(a) = numTransitions / totalTime;
                    fprintf('  Spontaneous: %d transitions, %.2f s total time, %.3f transitions/s\n', ...
                        numTransitions, totalTime, transitionRateNat(a));
                else
                    fprintf('  Spontaneous: No time data\n');
                end
            else
                fprintf('  Spontaneous: No sequence data\n');
            end
        else
            fprintf('  Spontaneous: No continuous results\n');
        end
    else
        fprintf('  Spontaneous: No HMM results\n');
    end
end

%% Create Transition Rate Bar Plot
fprintf('\n=== Creating Transition Rate Comparison Plot ===\n');

figure; clf;
hold on;

% Prepare data for grouped bar plot
barData = [transitionRateReach; transitionRateNat]'; % Areas x Conditions

% Create grouped bar plot
b = bar(barData, 'grouped');

% Set colors
b(1).FaceColor = [0.8500 0.3250 0.0980]; % Orange-red for Reach
b(2).FaceColor = [0 0.4470 0.7410];     % Blue for Spontaneous

% Customize axes
xticks(1:length(areas))
set(gca, 'XTickLabel', areas);
xlabel('Brain Area', 'FontSize', 12);
ylabel('State Transitions per Second', 'FontSize', 12);
title(sprintf('HMM State Transition Rate: Reach vs Spontaneous\n(binSize=%.3f, minDur=%.3f)', binSize, minDur), 'FontSize', 14);

% Add legend
legend({'Reach', 'Spontaneous'}, 'Location', 'best', 'FontSize', 11);

% Add grid
grid on;
set(gca, 'GridAlpha', 0.3);

% Add value labels on bars
for a = 1:length(areas)
    % Reach bar
    if ~isnan(transitionRateReach(a))
        text(a - 0.15, transitionRateReach(a) + max(max(transitionRateReach, transitionRateNat, 'omitnan')) * 0.02, ...
            sprintf('%.3f', transitionRateReach(a)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', [0.8500 0.3250 0.0980]);
    end
    % Spontaneous bar
    if ~isnan(transitionRateNat(a))
        text(a + 0.15, transitionRateNat(a) + max(max(transitionRateReach, transitionRateNat, 'omitnan')) * 0.02, ...
            sprintf('%.3f', transitionRateNat(a)), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, 'Color', [0 0.4470 0.7410]);
    end
end

hold off;

% Adjust figure size
set(gcf, 'Position', [100, 100, 800, 600]);

% Save figure
savePath = fullfile(hmmdir, sprintf('hmm_transition_rate_comparison_bin%.3f_minDur%.3f.png', binSize, minDur));
exportgraphics(gcf, savePath, 'Resolution', 300);
fprintf('Saved plot to: %s\n', savePath);

% Also save as .fig
figPath = fullfile(hmmdir, sprintf('hmm_transition_rate_comparison_bin%.3f_minDur%.3f.fig', binSize, minDur));
saveas(gcf, figPath);
fprintf('Saved figure to: %s\n', figPath);

%% Calculate and Plot Histograms of Unassigned State (state == 0) Durations
fprintf('\n=== Calculating Unassigned State Duration Distributions ===\n');

% Initialize storage for unassigned state durations
unassignedDurationsReach = cell(1, length(areas));
unassignedDurationsNat = cell(1, length(areas));

% Function to find stretches of state == 0 and calculate durations
find_unassigned_stretches = @(seq, binSize) ...
    arrayfun(@(start, stop) (stop - start + 1) * binSize, ...
    find(diff([1; seq == 0; 1]) == 1), ...
    find(diff([1; seq == 0; 1]) == -1) - 1);

for a = 1:length(areas)
    areaName = areas{a};
    areaIdx = areaMap(areaName);
    
    fprintf('\nArea: %s\n', areaName);
    
    % Reach data
    if areaIdx <= length(hmmResultsReach) && ~isempty(hmmResultsReach{areaIdx})
        hmmResReach = hmmResultsReach{areaIdx};
        if isfield(hmmResReach, 'continuous_results') && ...
           isfield(hmmResReach.continuous_results, 'sequence')
            seqReach = hmmResReach.continuous_results.sequence;
            if ~isempty(seqReach)
                % Get bin size for this area
                if isfield(hmmResReach, 'HmmParam') && isfield(hmmResReach.HmmParam, 'BinSize')
                    binSizeReach = hmmResReach.HmmParam.BinSize;
                else
                    binSizeReach = binSize; % Use default
                end
                
                % Find all stretches where state == 0
                isUnassigned = (seqReach == 0);
                if any(isUnassigned)
                    % Find transitions into and out of unassigned stretches
                    transitions = diff([0; isUnassigned; 0]);
                    starts = find(transitions == 1);
                    stops = find(transitions == -1) - 1;
                    
                    % Calculate durations in seconds
                    durations = (stops - starts + 1) * binSizeReach;
                    unassignedDurationsReach{a} = durations;
                    fprintf('  Reach: %d unassigned stretches, mean duration: %.3f s\n', ...
                        length(durations), mean(durations));
                else
                    unassignedDurationsReach{a} = [];
                    fprintf('  Reach: No unassigned stretches\n');
                end
            end
        end
    end
    
    % Spontaneous data
    if areaIdx <= length(hmmResultsNat) && ~isempty(hmmResultsNat{areaIdx})
        hmmResNat = hmmResultsNat{areaIdx};
        if isfield(hmmResNat, 'continuous_results') && ...
           isfield(hmmResNat.continuous_results, 'sequence')
            seqNat = hmmResNat.continuous_results.sequence;
            if ~isempty(seqNat)
                % Get bin size for this area
                if isfield(hmmResNat, 'HmmParam') && isfield(hmmResNat.HmmParam, 'BinSize')
                    binSizeNat = hmmResNat.HmmParam.BinSize;
                else
                    binSizeNat = binSize; % Use default
                end
                
                % Find all stretches where state == 0
                isUnassigned = (seqNat == 0);
                if any(isUnassigned)
                    % Find transitions into and out of unassigned stretches
                    transitions = diff([0; isUnassigned; 0]);
                    starts = find(transitions == 1);
                    stops = find(transitions == -1) - 1;
                    
                    % Calculate durations in seconds
                    durations = (stops - starts + 1) * binSizeNat;
                    unassignedDurationsNat{a} = durations;
                    fprintf('  Spontaneous: %d unassigned stretches, mean duration: %.3f s\n', ...
                        length(durations), mean(durations));
                else
                    unassignedDurationsNat{a} = [];
                    fprintf('  Spontaneous: No unassigned stretches\n');
                end
            end
        end
    end
end

%% Create Histogram Plots
fprintf('\n=== Creating Histogram Plots ===\n');

% Find maximum duration across all areas and both tasks for consistent xlim
maxDuration = 0;
for a = 1:length(areas)
    if ~isempty(unassignedDurationsReach{a})
        maxDuration = max(maxDuration, max(unassignedDurationsReach{a}));
    end
    if ~isempty(unassignedDurationsNat{a})
        maxDuration = max(maxDuration, max(unassignedDurationsNat{a}));
    end
end

% Set common xlim with some padding
if maxDuration > 0
    xlimCommon = [0, maxDuration * 1.05]; % 5% padding
    xlimCommon = [0, maxDuration * .3]; % 5% padding
else
    xlimCommon = [0, 1]; % Default if no data
end

fprintf('Using common x-axis limit: [%.3f, %.3f] seconds\n', xlimCommon(1), xlimCommon(2));

% Create combined figure: 2 rows (tasks) x 4 columns (areas)
figure(2); clf;
set(gcf, 'Position', [100, 100, 1600, 800]);

% Use tight_subplot: 2 rows (Reach, Spontaneous) x 4 columns (areas)
[ha, ~] = tight_subplot(2, 4, [0.08 0.06], [0.06 0.1], [0.06 0.04]);

% Plot Reach data (row 1: indices 1-4)
for a = 1:length(areas)
    areaName = areas{a};
    plotIdx = a; % Row 1, column a
    axes(ha(plotIdx));
    
    durations = unassignedDurationsReach{a};
    
    if ~isempty(durations) && any(~isnan(durations))
        % Create histogram
        h = histogram(durations, 'FaceColor', [0.8500 0.3250 0.0980], 'EdgeColor', 'none');
        
        % Add statistics
        meanDur = mean(durations);
        medianDur = median(durations);
        maxDur = max(durations);
        
        % Add vertical lines for mean and median
        hold on;
        xline(meanDur, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.3f s', meanDur));
        xline(medianDur, 'b--', 'LineWidth', 2, 'DisplayName', sprintf('Median: %.3f s', medianDur));
        hold off;
        
        % Formatting
        ylabel('Count', 'FontSize', 10);
        title(sprintf('%s - Reach\n(n=%d, max=%.3f s)', areaName, length(durations), maxDur), ...
            'FontSize', 11);
        grid on;
        set(gca, 'GridAlpha', 0.3);
        legend('Location', 'best', 'FontSize', 8);
        xlim(xlimCommon); % Set common xlim
    else
        % No data
        text(0.5, 0.5, 'No unassigned stretches', 'HorizontalAlignment', 'center', ...
            'FontSize', 11, 'Color', [0.5 0.5 0.5]);
        xlim(xlimCommon); % Set common xlim even for empty plots
        ylim([0, 1]);
        title(sprintf('%s - Reach\n(No Data)', areaName), 'FontSize', 11);
    end
end

% Plot Spontaneous data (row 2: indices 5-8)
for a = 1:length(areas)
    areaName = areas{a};
    plotIdx = 4 + a; % Row 2, column a
    axes(ha(plotIdx));
    
    durations = unassignedDurationsNat{a};
    
    if ~isempty(durations) && any(~isnan(durations))
        % Create histogram
        h = histogram(durations, 'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'none');
        
        % Add statistics
        meanDur = mean(durations);
        medianDur = median(durations);
        maxDur = max(durations);
        
        % Add vertical lines for mean and median
        hold on;
        xline(meanDur, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.3f s', meanDur));
        xline(medianDur, 'b--', 'LineWidth', 2, 'DisplayName', sprintf('Median: %.3f s', medianDur));
        hold off;
        
        % Formatting
        xlabel('Duration (seconds)', 'FontSize', 10);
        ylabel('Count', 'FontSize', 10);
        title(sprintf('%s - Spontaneous\n(n=%d, max=%.3f s)', areaName, length(durations), maxDur), ...
            'FontSize', 11);
        grid on;
        set(gca, 'GridAlpha', 0.3);
        legend('Location', 'best', 'FontSize', 8);
        xlim(xlimCommon); % Set common xlim
    else
        % No data
        text(0.5, 0.5, 'No unassigned stretches', 'HorizontalAlignment', 'center', ...
            'FontSize', 11, 'Color', [0.5 0.5 0.5]);
        xlim(xlimCommon); % Set common xlim even for empty plots
        ylim([0, 1]);
        title(sprintf('%s - Spontaneous\n(No Data)', areaName), 'FontSize', 11);
    end
end

% Add overall title
sgtitle(sprintf('Distribution of Unassigned State Durations\n(binSize=%.3f, minDur=%.3f)', ...
    binSize, minDur), 'FontSize', 13, 'FontWeight', 'bold');

% Save combined histogram figure
savePath = fullfile(hmmdir, sprintf('hmm_unassigned_durations_combined_bin%.3f_minDur%.3f.png', binSize, minDur));
exportgraphics(gcf, savePath, 'Resolution', 300);
fprintf('Saved combined histogram to: %s\n', savePath);

figPath = fullfile(hmmdir, sprintf('hmm_unassigned_durations_combined_bin%.3f_minDur%.3f.fig', binSize, minDur));
saveas(gcf, figPath);
fprintf('Saved combined histogram figure to: %s\n', figPath);

fprintf('\n=== Analysis Complete ===\n');









% Function to collapse unassigned states that are between the same state
% If state A -> unassigned -> back to state A, treat as continuous state A
function seqCollapsed = collapse_unassigned_between_same_state(seq)
    seqCollapsed = seq;
    if isempty(seq) || length(seq) < 3
        return;
    end
    
    % Find stretches of unassigned (state == 0)
    isUnassigned = (seq == 0);
    if ~any(isUnassigned)
        return; % No unassigned states to collapse
    end
    
    % Find transitions into and out of unassigned stretches
    transitions = diff([0; isUnassigned; 0]);
    unassignedStarts = find(transitions == 1);
    unassignedStops = find(transitions == -1) - 1;
    
    % Process each unassigned stretch
    for i = 1:length(unassignedStarts)
        startIdx = unassignedStarts(i);
        stopIdx = unassignedStops(i);
        
        % Get the state before and after the unassigned stretch
        stateBefore = 0;
        stateAfter = 0;
        
        if startIdx > 1
            stateBefore = seq(startIdx - 1);
        end
        if stopIdx < length(seq)
            stateAfter = seq(stopIdx + 1);
        end
        
        % If both states are the same and non-zero, collapse the unassigned stretch
        if stateBefore == stateAfter && stateBefore ~= 0
            seqCollapsed(startIdx:stopIdx) = stateBefore;
        end
    end
end

% Function to count transitions in a sequence (excluding transitions to/from same state)
function numTransitions = count_transitions(seq)
    if isempty(seq) || length(seq) < 2
        numTransitions = 0;
        return;
    end
    
    % Find where state changes (excluding transitions to/from same state)
    stateChanges = diff(seq) ~= 0;
    numTransitions = sum(stateChanges);
end

