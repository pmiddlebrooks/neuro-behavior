%%
% Peri-Trial HMM State Analysis for Naturalistic Data
% Loads results from hmm_mazz_nat.m and analyzes HMM state sequences
% around trial onset times for each brain area
%
% Variables:
%   results - loaded HMM analysis results
%   dataBhv - naturalistic behavioral data
%   areas - brain areas to analyze
%   trialStartTimes - trial start times for each area
%   stateWindows - HMM state sequences in windows around each trial
%   stateImages - imagesc plots of state sequences for each area

%% Load existing results if requested
paths = get_paths;
binSize = .002;
minDur = .05;
bhvStartID = 15;
preBhvTime = -.2;

areasToTest = 1:4;

% Define window parameters
windowDurationSec = 16; % 16-second window around each trial

% Determine save directory and filename based on parameters
hmmdir = fullfile(paths.dropPath, 'metastability');
filename = sprintf('hmm_mazz_nat_bin%.3f_minDur%.3f_bhv%d_offset%.3f.mat', binSize, minDur, bhvStartID, preBhvTime);
resultsPath = fullfile(hmmdir, filename);

% Extract first 10 characters of filename for titles and file names
filePrefix = 'Nat';

% Load HMM analysis results
fprintf('Loading HMM analysis results from: %s\n', resultsPath);
if ~exist(resultsPath, 'file')
    error('Results file not found: %s\nMake sure hmm_mazz_nat.m has been run for this dataset.', resultsPath);
end
results = load(resultsPath);
results = results.results;

% Extract areas and parameters
areas = results.areas;
binSizes = results.binSize;
numStates = results.numStates;
hmmResults = results.hmm_results;

% Load naturalistic behavioral data
fprintf('Loading naturalistic behavioral data...\n');
getDataType = 'spikes';
[dataSpikes, dataBhv, dataKin] = get_standard_data(getDataType);

% Extract trial parameters from results
trialStartTimes = results.trial_start_times;
trialEndTimes = results.trial_end_times;

% ==============================================     Peri-Trial Analysis     ==============================================

% Initialize storage for peri-trial state sequences
stateWindows = cell(1, length(areas));

% Initialize storage for peri-trial metastate sequences
metastateWindows = cell(1, length(areas));

% Flags indicating availability of data to analyze/plot per area
hasHmmArea = false(1, length(areas));

fprintf('\n=== Peri-Trial HMM State Analysis ===\n');

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});

    % Get HMM results for this area
    hmmRes = hmmResults{a};
    
    % Check if HMM analysis was successful for this area
    if isempty(hmmRes) || ~strcmp(hmmRes.metadata.analysis_status, 'SUCCESS')
        fprintf('Skipping area %s due to failed HMM analysis.\n', areas{a});
        continue
    end
    
    hasHmmArea(a) = true;
    
    % Get continuous state sequence and parameters
    continuousSequence = hmmRes.continuous_results.sequence;
    binSize = hmmRes.HmmParam.BinSize;
    totalTimeBins = length(continuousSequence);
    
    % Get continuous metastate sequence if available
    if isfield(hmmRes, 'metastate_results') && ~isempty(hmmRes.metastate_results.continuous_metastates)
        continuousMetastates = hmmRes.metastate_results.continuous_metastates;
        hasMetastates = true;
    else
        continuousMetastates = [];
        hasMetastates = false;
    end
    
    % Calculate window duration in bins
    windowDurationBins = ceil(windowDurationSec / binSize);
    halfWindow = floor(windowDurationBins / 2);
    
    % Initialize arrays for this area
    numTrials = length(trialStartTimes);
    
    % Initialize storage for all trial windows
    stateWindows{a} = nan(numTrials, windowDurationBins + 1);
    
    % Initialize storage for metastate windows
    if hasMetastates
        metastateWindows{a} = nan(numTrials, windowDurationBins + 1);
    end
    
    % Extract around each trial onset
    validTrials = 0;
    for t = 1:numTrials
        trialTime = trialStartTimes(t); % Already in seconds
        trialBin = round(trialTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = trialBin - halfWindow;
        winEnd = trialBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindows{a}(t, :) = continuousSequence(winStart:winEnd);
            if hasMetastates
                metastateWindows{a}(t, :) = continuousMetastates(winStart:winEnd);
            end
            validTrials = validTrials + 1;
        else
            stateWindows{a}(t, :) = nan(1, windowDurationBins + 1);
            if hasMetastates
                metastateWindows{a}(t, :) = nan(1, windowDurationBins + 1);
            end
        end
    end
    
    fprintf('Area %s: %d valid trials\n', areas{a}, validTrials);
end

% ==============================================     Plotting Results     ==============================================

% Create peri-trial plots for each area: single row per area
figure(400); clf;
% Prefer plotting on second screen if available
monitorPositions = get(0, 'MonitorPositions');
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    set(gcf, 'Position', monitorTwo);
else
    set(gcf, 'Position', monitorPositions(1, :));
end

% Use tight_subplot for layout: single row with areas as columns
numCols = length(areasToTest);
ha = tight_subplot(1, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

% Create time axis for peri-trial window (centered on trial onset)
timeAxisPeriTrial = (-halfWindow:halfWindow) * binSizes(1); % Use first area's bin size for time axis

% Plot each area (column)
for areaIdx = 1:length(areasToTest)
    a = areasToTest(areaIdx);
    
    axes(ha(areaIdx));
    hold on;
    
    % Get state data for this area
    stateData = stateWindows{a};
    
    % Check if we have data for this area
    if ~hasHmmArea(a) || isempty(stateData) || all(isnan(stateData(:)))
        % No data - show blank plot
        xlim([timeAxisPeriTrial(1), timeAxisPeriTrial(end)]);
        ylim([0.5, 1.5]);
        title(sprintf('%s\n(No Data)', areas{a}), 'FontSize', 10);
        continue;
    end
    
    % Remove rows that are all NaN and ensure trials are in order
    validRows = ~all(isnan(stateData), 2);
    if ~any(validRows)
        xlim([timeAxisPeriTrial(1), timeAxisPeriTrial(end)]);
        ylim([0.5, 1.5]);
        title(sprintf('%s\n(No Valid Trials)', areas{a}), 'FontSize', 10);
        continue;
    end
    
    % Keep trials in original order (top to bottom)
    stateDataValid = stateData(validRows, :);
    
    % Create imagesc plot
    imagesc(timeAxisPeriTrial, 1:size(stateDataValid, 1), stateDataValid);
    
    % Set colormap to distinguish states with state 0 as white
    if numStates(a) > 0
        % Create custom colormap with white for state 0
        cmap = lines(numStates(a));
        % Ensure state 0 (undefined) is white
        cmap(1, :) = [1 1 1]; % White for state 0
        colormap(cmap);
    else
        colormap('lines');
    end
    
    % Add colorbar
    c = colorbar;
    c.Label.String = 'HMM State';
    c.Label.FontSize = 8;
    
    % Add vertical line at trial onset
    plot([0 0], ylim, 'k--', 'LineWidth', 2);
    
    % Formatting
    xlabel('Time relative to trial onset (s)', 'FontSize', 8);
    ylabel('Trial', 'FontSize', 8);
    title(sprintf('%s\n(%d trials)', areas{a}, sum(validRows)), 'FontSize', 10);
    
    % Set axis limits
    xlim([timeAxisPeriTrial(1), timeAxisPeriTrial(end)]);
    ylim([0.5, size(stateDataValid, 1) + 0.5]);
    
    % Set tick labels
    xTicks = ceil(timeAxisPeriTrial(1)):floor(timeAxisPeriTrial(end));
    if isempty(xTicks)
        xTicks = linspace(timeAxisPeriTrial(1), timeAxisPeriTrial(end), 5);
    end
    xticks(xTicks);
    xticklabels(string(xTicks));
    
    % Set y-axis ticks to show trial numbers
    if size(stateDataValid, 1) <= 10
        yticks(1:size(stateDataValid, 1));
    else
        yticks(1:5:size(stateDataValid, 1));
    end
    
    grid on;
    set(gca, 'GridAlpha', 0.3);
end

% Add overall title
sgtitle(sprintf('%s - Peri-Trial HMM State Sequences (Window: %gs)', filePrefix, windowDurationSec), 'FontSize', 16);

% Save combined figure
filename = fullfile(hmmdir, sprintf('%s_peri_trial_hmm_states_win%gs.png', filePrefix, windowDurationSec));
exportgraphics(gcf, filename, 'Resolution', 300);
fprintf('Saved peri-trial HMM state plot to: %s\n', filename);

% ==============================================     Metastate Plotting     ==============================================

% Check if any area has metastate data
hasAnyMetastates = false;
for a = areasToTest
    if hasHmmArea(a) && ~isempty(results.hmm_results{a}) && isfield(results.hmm_results{a}, 'metastate_results') && ~isempty(results.hmm_results{a}.metastate_results.continuous_metastates)
        hasAnyMetastates = true;
        break;
    end
end

if hasAnyMetastates
    fprintf('\n=== Creating Peri-Trial Metastate Plot ===\n');
    
    % Create peri-trial metastate plots for each area: single row per area
    figure(401); clf;
    % Prefer plotting on second screen if available
    monitorPositions = get(0, 'MonitorPositions');
    monitorTwo = monitorPositions(size(monitorPositions, 1), :);
    if size(monitorPositions, 1) >= 2
        set(gcf, 'Position', monitorTwo);
    else
        set(gcf, 'Position', monitorPositions(1, :));
    end

    % Use tight_subplot for layout: single row with areas as columns
    numCols = length(areasToTest);
    ha = tight_subplot(1, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

    % Plot each area (column)
    for areaIdx = 1:length(areasToTest)
        a = areasToTest(areaIdx);
        
        axes(ha(areaIdx));
        hold on;
        
        % Get metastate data for this area
        metastateData = metastateWindows{a};
        
        % Check if we have metastate data for this area
        if ~hasHmmArea(a) || isempty(metastateData) || all(isnan(metastateData(:)))
            % No data - show blank plot
            xlim([timeAxisPeriTrial(1), timeAxisPeriTrial(end)]);
            ylim([0.5, 1.5]);
            title(sprintf('%s\n(No Metastate Data)', areas{a}), 'FontSize', 10);
            continue;
        end
        
        % Remove rows that are all NaN and ensure trials are in order
        validRows = ~all(isnan(metastateData), 2);
        if ~any(validRows)
            xlim([timeAxisPeriTrial(1), timeAxisPeriTrial(end)]);
            ylim([0.5, 1.5]);
            title(sprintf('%s\n(No Valid Trials)', areas{a}), 'FontSize', 10);
            continue;
        end
        
        % Keep trials in original order (top to bottom)
        metastateDataValid = metastateData(validRows, :);
        
        % Create imagesc plot
        imagesc(timeAxisPeriTrial, 1:size(metastateDataValid, 1), metastateDataValid);
        
        % Set colormap to distinguish metastates with metastate 0 as white
        if ~isempty(results.hmm_results{a}) && isfield(results.hmm_results{a}, 'metastate_results') && ~isempty(results.hmm_results{a}.metastate_results.communities)
            numMetastates = results.hmm_results{a}.metastate_results.num_metastates;
            if numMetastates > 0
                % Create custom colormap with white for metastate 0
                cmap = lines(numMetastates + 1); % +1 for metastate 0
                % Ensure metastate 0 (undefined) is white
                cmap(1, :) = [1 1 1]; % White for metastate 0
                colormap(cmap);
            else
                colormap('lines');
            end
        else
            colormap('lines');
        end
        
        % Add colorbar
        c = colorbar;
        c.Label.String = 'Metastate';
        c.Label.FontSize = 8;
        
        % Add vertical line at trial onset
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
        
        % Formatting
        xlabel('Time relative to trial onset (s)', 'FontSize', 8);
        ylabel('Trial', 'FontSize', 8);
        title(sprintf('%s\n(%d trials)', areas{a}, sum(validRows)), 'FontSize', 10);
        
        % Set axis limits
        xlim([timeAxisPeriTrial(1), timeAxisPeriTrial(end)]);
        ylim([0.5, size(metastateDataValid, 1) + 0.5]);
        
        % Set tick labels
        xTicks = ceil(timeAxisPeriTrial(1)):floor(timeAxisPeriTrial(end));
        if isempty(xTicks)
            xTicks = linspace(timeAxisPeriTrial(1), timeAxisPeriTrial(end), 5);
        end
        xticks(xTicks);
        xticklabels(string(xTicks));
        
        % Set y-axis ticks to show trial numbers
        if size(metastateDataValid, 1) <= 10
            yticks(1:size(metastateDataValid, 1));
        else
            yticks(1:5:size(metastateDataValid, 1));
        end
        
        grid on;
        set(gca, 'GridAlpha', 0.3);
    end

    % Add overall title
    sgtitle(sprintf('%s - Peri-Trial Metastate Sequences (Window: %gs)', filePrefix, windowDurationSec), 'FontSize', 16);

    % Save combined metastate figure
    filename = fullfile(hmmdir, sprintf('%s_peri_trial_metastates_win%gs.png', filePrefix, windowDurationSec));
    exportgraphics(gcf, filename, 'Resolution', 300);
    fprintf('Saved peri-trial metastate plot to: %s\n', filename);
else
    fprintf('\nNo metastate data available for plotting\n');
end

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Peri-Trial HMM State Analysis Summary ===\n');
for a = areasToTest
    if hasHmmArea(a)
        fprintf('\nArea %s:\n', areas{a});
        fprintf('  Total trials analyzed: %d\n', length(trialStartTimes));
        fprintf('  Valid trials: %d\n', sum(~all(isnan(stateWindows{a}), 2)));
        fprintf('  Number of HMM states: %d\n', numStates(a));
        fprintf('  Bin size: %.6f seconds\n', binSizes(a));
    else
        fprintf('\nArea %s: HMM analysis failed\n', areas{a});
    end
end

fprintf('\nPeri-trial HMM state analysis complete!\n');
