%%
% Peri-Reach HMM State Analysis
% Loads results from hmm_mazz_reach.m and analyzes HMM state sequences
% around reach onset times for each brain area
%
% Variables:
%   results - loaded HMM analysis results
%   dataR - reach behavioral data
%   areas - brain areas to analyze
%   reachStartFrame - reach start times in frame units for each area
%   stateWindows - HMM state sequences in windows around each reach
%   stateImages - imagesc plots of state sequences for each condition

%% Load existing results if requested
paths = get_paths;

% User-specified reach data file (should match the one used in hmm_mazz_reach.m)
reachDataFile = fullfile(paths.dropPath, 'reach_data/Y4_06-Oct-2023 14_14_53_NeuroBeh.mat');

areasToTest = 1:4;

% Toggle plotting error reaches
plotErrors = true;

% Define window parameters
windowDurationSec = 40; % 40-second window around each reach

% Extract session name from filename
[~, sessionName, ~] = fileparts(reachDataFile);

% Determine save directory based on loaded data file name (same as hmm_mazz_reach.m)
saveDir = fullfile(paths.dropPath, 'reach_data', sessionName);
resultsPath = fullfile(saveDir, sprintf('HMM_results_%s.mat', sessionName));

% Extract first 10 characters of filename for titles and file names
filePrefix = sessionName(1:min(10, length(sessionName)));

% Load HMM analysis results
fprintf('Loading HMM analysis results from: %s\n', resultsPath);
if ~exist(resultsPath, 'file')
    error('Results file not found: %s\nMake sure hmm_mazz_reach.m has been run for this dataset.', resultsPath);
end
results = load(resultsPath);
results = results.results;

% Extract areas and parameters
areas = results.areas;
binSizes = results.binSize;
numStates = results.numStates;
hmmResults = results.hmm_results;

% Load reach behavioral data
fprintf('Loading reach behavioral data from: %s\n', reachDataFile);
dataR = load(reachDataFile);

reachStart = dataR.R(:,1); % In seconds
reachStop = dataR.R(:,2);
reachAmp = dataR.R(:,3); % Amplitude of each reach (distance from 0)

% Use Block(:,3) for reach classification
reachClass = dataR.Block(:,3);

% Define reach conditions
reachStartCorr1 = reachStart(ismember(reachClass, 2)); % Block 1 correct
reachStartCorr2 = reachStart(ismember(reachClass, 4)); % Block 2 correct
reachStartErr1 = reachStart(ismember(reachClass, 1));  % Block 1 error
reachStartErr2 = reachStart(ismember(reachClass, 3));  % Block 2 error

% ==============================================     Peri-Reach Analysis     ==============================================

% Initialize storage for peri-reach state sequences (separate correct/error by block)
stateWindowsCorr1 = cell(1, length(areas));
stateWindowsCorr2 = cell(1, length(areas));
stateWindowsErr1 = cell(1, length(areas));
stateWindowsErr2 = cell(1, length(areas));

% Flags indicating availability of data to analyze/plot per area
hasHmmArea = false(1, length(areas));

fprintf('\n=== Peri-Reach HMM State Analysis ===\n');

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
    
    % Calculate window duration in bins
    windowDurationBins = ceil(windowDurationSec / binSize);
    halfWindow = floor(windowDurationBins / 2);
    
    % Initialize arrays for this area
    numReachesCorr1 = length(reachStartCorr1);
    numReachesCorr2 = length(reachStartCorr2);
    numReachesErr1 = length(reachStartErr1);
    numReachesErr2 = length(reachStartErr2);
    
    % Initialize storage for all reach windows (by block)
    stateWindowsCorr1{a} = nan(numReachesCorr1, windowDurationBins + 1);
    stateWindowsCorr2{a} = nan(numReachesCorr2, windowDurationBins + 1);
    stateWindowsErr1{a} = nan(numReachesErr1, windowDurationBins + 1);
    stateWindowsErr2{a} = nan(numReachesErr2, windowDurationBins + 1);
    
    % Extract around each correct reach (Block 1)
    validCorr1 = 0;
    for r = 1:numReachesCorr1
        reachTime = reachStartCorr1(r)/1000; % Convert to seconds
        reachBin = round(reachTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = reachBin - halfWindow;
        winEnd = reachBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindowsCorr1{a}(r, :) = continuousSequence(winStart:winEnd);
            validCorr1 = validCorr1 + 1;
        else
            stateWindowsCorr1{a}(r, :) = nan(1, windowDurationBins + 1);
        end
    end
    
    % Extract around each correct reach (Block 2)
    validCorr2 = 0;
    for r = 1:numReachesCorr2
        reachTime = reachStartCorr2(r)/1000; % Convert to seconds
        reachBin = round(reachTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = reachBin - halfWindow;
        winEnd = reachBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindowsCorr2{a}(r, :) = continuousSequence(winStart:winEnd);
            validCorr2 = validCorr2 + 1;
        else
            stateWindowsCorr2{a}(r, :) = nan(1, windowDurationBins + 1);
        end
    end
    
    % Extract around each error reach (Block 1)
    validErr1 = 0;
    for r = 1:numReachesErr1
        reachTime = reachStartErr1(r)/1000; % Convert to seconds
        reachBin = round(reachTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = reachBin - halfWindow;
        winEnd = reachBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindowsErr1{a}(r, :) = continuousSequence(winStart:winEnd);
            validErr1 = validErr1 + 1;
        else
            stateWindowsErr1{a}(r, :) = nan(1, windowDurationBins + 1);
        end
    end
    
    % Extract around each error reach (Block 2)
    validErr2 = 0;
    for r = 1:numReachesErr2
        reachTime = reachStartErr2(r)/1000; % Convert to seconds
        reachBin = round(reachTime / binSize) + 1; % Convert to bin index (1-indexed)
        
        winStart = reachBin - halfWindow;
        winEnd = reachBin + halfWindow;
        
        if winStart >= 1 && winEnd <= totalTimeBins
            stateWindowsErr2{a}(r, :) = continuousSequence(winStart:winEnd);
            validErr2 = validErr2 + 1;
        else
            stateWindowsErr2{a}(r, :) = nan(1, windowDurationBins + 1);
        end
    end
    
    fprintf('Area %s: Corr B1=%d, Corr B2=%d, Err B1=%d, Err B2=%d valid reaches\n', areas{a}, validCorr1, validCorr2, validErr1, validErr2);
end

% ==============================================     Plotting Results     ==============================================

% Create peri-reach plots for each area: conditions x areas
figure(400); clf;
% Prefer plotting on second screen if available
monitorPositions = get(0, 'MonitorPositions');
monitorTwo = monitorPositions(size(monitorPositions, 1), :);
if size(monitorPositions, 1) >= 2
    set(gcf, 'Position', monitorTwo);
else
    set(gcf, 'Position', monitorPositions(1, :));
end

% Determine number of conditions to plot
numConditions = 2; % Correct Block 1, Correct Block 2
if plotErrors
    numConditions = numConditions + 2; % Add Error Block 1, Error Block 2
end

% Use tight_subplot for layout: conditions (rows) x areas (columns)
numCols = length(areasToTest);
ha = tight_subplot(numConditions, numCols, [0.08 0.04], [0.08 0.1], [0.06 0.04]);

% Create time axis for peri-reach window (centered on reach onset)
timeAxisPeriReach = (-halfWindow:halfWindow) * binSizes(1); % Use first area's bin size for time axis

% Define colors for different conditions
colors = {'k', [0 0 .6], [.6 0 0], [0 0.6 0]};

% Plot each condition (row) and area (column)
plotIdx = 0;
for condIdx = 1:numConditions
    for areaIdx = 1:length(areasToTest)
        a = areasToTest(areaIdx);
        plotIdx = plotIdx + 1;
        
        axes(ha(plotIdx));
        hold on;
        
        % Determine which condition to plot
        if condIdx == 1
            % Correct Block 1
            stateData = stateWindowsCorr1{a};
            condName = 'Correct Block 1';
            colorIdx = 1;
        elseif condIdx == 2
            % Correct Block 2
            stateData = stateWindowsCorr2{a};
            condName = 'Correct Block 2';
            colorIdx = 2;
        elseif condIdx == 3 && plotErrors
            % Error Block 1
            stateData = stateWindowsErr1{a};
            condName = 'Error Block 1';
            colorIdx = 3;
        elseif condIdx == 4 && plotErrors
            % Error Block 2
            stateData = stateWindowsErr2{a};
            condName = 'Error Block 2';
            colorIdx = 4;
        else
            continue; % Skip if not plotting errors
        end
        
        % Check if we have data for this area and condition
        if ~hasHmmArea(a) || isempty(stateData) || all(isnan(stateData(:)))
            % No data - show blank plot
            xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
            ylim([0.5, 1.5]);
            title(sprintf('%s - %s\n(No Data)', areas{a}, condName), 'FontSize', 10);
            continue;
        end
        
        % Remove rows that are all NaN
        validRows = ~all(isnan(stateData), 2);
        if ~any(validRows)
            xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
            ylim([0.5, 1.5]);
            title(sprintf('%s - %s\n(No Valid Reaches)', areas{a}, condName), 'FontSize', 10);
            continue;
        end
        
        stateDataValid = stateData(validRows, :);
        
        % Create imagesc plot
        imagesc(timeAxisPeriReach, 1:size(stateDataValid, 1), stateDataValid);
        
        % Set colormap to distinguish states
        if numStates(a) > 0
            cmap = lines(numStates(a));
            colormap(cmap);
        else
            colormap('lines');
        end
        
        % Add colorbar
        c = colorbar;
        c.Label.String = 'HMM State';
        c.Label.FontSize = 8;
        
        % Add vertical line at reach onset
        plot([0 0], ylim, 'k--', 'LineWidth', 2);
        
        % Formatting
        xlabel('Time relative to reach onset (s)', 'FontSize', 8);
        ylabel('Reach Trial', 'FontSize', 8);
        title(sprintf('%s - %s\n(%d trials)', areas{a}, condName, sum(validRows)), 'FontSize', 10);
        
        % Set axis limits
        xlim([timeAxisPeriReach(1), timeAxisPeriReach(end)]);
        ylim([0.5, size(stateDataValid, 1) + 0.5]);
        
        % Set tick labels
        xTicks = ceil(timeAxisPeriReach(1)):floor(timeAxisPeriReach(end));
        if isempty(xTicks)
            xTicks = linspace(timeAxisPeriReach(1), timeAxisPeriReach(end), 5);
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
end

% Add overall title
sgtitle(sprintf('%s - Peri-Reach HMM State Sequences (Window: %gs)', filePrefix, windowDurationSec), 'FontSize', 16);

% Save combined figure (in same data-specific folder)
filename = fullfile(saveDir, sprintf('%s_peri_reach_hmm_states_win%gs.png', filePrefix, windowDurationSec));
exportgraphics(gcf, filename, 'Resolution', 300);
fprintf('Saved peri-reach HMM state plot to: %s\n', filename);

%% ==============================================     Summary Statistics     ==============================================

fprintf('\n=== Peri-Reach HMM State Analysis Summary ===\n');
for a = areasToTest
    if hasHmmArea(a)
        fprintf('\nArea %s:\n', areas{a});
        fprintf('  Total reaches analyzed: %d\n', length(reachStart));
        fprintf('  Valid reaches Block 1 Correct: %d\n', sum(~all(isnan(stateWindowsCorr1{a}), 2)));
        fprintf('  Valid reaches Block 2 Correct: %d\n', sum(~all(isnan(stateWindowsCorr2{a}), 2)));
        if plotErrors
            fprintf('  Valid reaches Block 1 Error: %d\n', sum(~all(isnan(stateWindowsErr1{a}), 2)));
            fprintf('  Valid reaches Block 2 Error: %d\n', sum(~all(isnan(stateWindowsErr2{a}), 2)));
        end
        fprintf('  Number of HMM states: %d\n', numStates(a));
        fprintf('  Bin size: %.6f seconds\n', binSizes(a));
    else
        fprintf('\nArea %s: HMM analysis failed\n', areas{a});
    end
end

fprintf('\nPeri-reach HMM state analysis complete!\n');
