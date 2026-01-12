%%
% Criticality AR Analysis Across Sessions
% Compares d2 values from a single large window between spontaneous and reach sessions
%
% Variables:
%   nMin - Window size in minutes (default: 20)
%   windowStartTime - Start time for the window in seconds (default: 0)
%   useNormalized - Whether to use normalized d2 values (default: true)
%
% Goal:
%   Load data for each session, analyze a single large window, and compare
%   raw and normalized d2 values between spontaneous and reach sessions.

%% Configuration
paths = get_paths;

nMin = 20;  % Window size in minutes
windowStartTime = 0;  % Start time for the window in seconds (0 = start of recording)
useNormalized = true;  % Use normalized d2 if available


%% Get session lists
fprintf('\n=== Criticality AR Analysis Across Sessions ===\n');
fprintf('Window size: %d minutes\n', nMin);
fprintf('Window start time: %.1f seconds\n', windowStartTime);

spontaneousSessions = spontaneous_session_list();
reachSessions = reach_session_list();

fprintf('Spontaneous sessions: %d\n', length(spontaneousSessions));
fprintf('Reach sessions: %d\n', length(reachSessions));

%% Configure analysis options
opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = windowStartTime;
opts.collectEnd = windowStartTime + nMin * 60;  % nMin minutes
opts.minFiringRate = .1;
opts.maxFiringRate = 100;

% Analysis config - use single large window
config = struct();
config.binSize = .03;
config.slidingWindowSize = nMin * 60;  % Window size in seconds
config.stepSize = nMin * 60 + 1;  % Larger than window size so only one window
config.minSpikesPerBin = 2.5;
config.minBinsPerWindow = 1000;
config.analyzeD2 = true;
config.analyzeMrBr = false;
config.pcaFlag = 1;
config.pcaFirstFlag = 1;
config.nDim = 4;
config.enablePermutations = true;
config.nShuffles = 20;
config.analyzeModulation = false;
config.makePlots = false;  % Don't create individual plots
config.saveData = false;  % Don't save individual session results
config.useOptimalBinWindowFunction = false;
config.pOrder = 10;
config.critType = 2;
config.normalizeD2 = true;
config.maxSpikesPerBin = 50;
config.nMinNeurons = 15;
config.includeM2356 = true;

%% Initialize results storage
spontaneousData = struct();
spontaneousData.d2Raw = {};  % Cell array: {areaIdx}{sessionIdx} = d2 value
spontaneousData.d2Normalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized d2 value
spontaneousData.sessionNames = {};
spontaneousData.areas = [];

reachData = struct();
reachData.d2Raw = {};  % Cell array: {areaIdx}{sessionIdx} = d2 value
reachData.d2Normalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized d2 value
reachData.sessionNames = {};
reachData.areas = [];

%% Process spontaneous sessions
fprintf('\n=== Processing Spontaneous Sessions ===\n');
for s = 1:length(spontaneousSessions)
    sessionName = spontaneousSessions{s};
    fprintf('\nProcessing session %d/%d: %s\n', s, length(spontaneousSessions), sessionName);
    
    try
        % Load data
        dataStruct = load_sliding_window_data('spontaneous', 'spikes', ...
            'sessionName', sessionName, 'opts', opts);
        
        % Run analysis
        results = criticality_ar_analysis(dataStruct, config);
        
        % Extract areas if not yet set
        if isempty(spontaneousData.areas) && isfield(results, 'areas')
            spontaneousData.areas = results.areas;
            numAreas = length(spontaneousData.areas);
            spontaneousData.d2Raw = cell(1, numAreas);
            spontaneousData.d2Normalized = cell(1, numAreas);
            for a = 1:numAreas
                spontaneousData.d2Raw{a} = [];
                spontaneousData.d2Normalized{a} = [];
            end
        end
        
        % Extract d2 values per area (should be single value per area)
        if isfield(results, 'd2') && isfield(results, 'd2Normalized')
            for a = 1:length(results.areas)
                if a <= length(results.d2) && ~isempty(results.d2{a})
                    % Get the first (and only) value from the window
                    d2Values = results.d2{a}(~isnan(results.d2{a}));
                    if ~isempty(d2Values)
                        spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, d2Values(1)];
                    else
                        spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                    end
                else
                    spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                end
                
                if a <= length(results.d2Normalized) && ~isempty(results.d2Normalized{a})
                    % Get the first (and only) value from the window
                    d2NormValues = results.d2Normalized{a}(~isnan(results.d2Normalized{a}));
                    if ~isempty(d2NormValues)
                        spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, d2NormValues(1)];
                    else
                        spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
                    end
                else
                    spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
                end
            end
        else
            warning('d2 or d2Normalized not found in results for session %s', sessionName);
            % Add NaN for all areas
            for a = 1:length(spontaneousData.areas)
                spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
            end
        end
        
        spontaneousData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session completed successfully\n');
        
    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(spontaneousData.areas)
            for a = 1:length(spontaneousData.areas)
                spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
            end
        end
        spontaneousData.sessionNames{end+1} = sessionName;
    end
end

%% Process reach sessions
fprintf('\n=== Processing Reach Sessions ===\n');
% For reach sessions, set window end time (collectEnd empty means use all data, but we want a specific window)
% Reset opts for reach sessions
opts.collectStart = windowStartTime;
opts.collectEnd = windowStartTime + nMin * 60;  % nMin minutes from start time

for s = 1:length(reachSessions)
    sessionName = reachSessions{s};
    fprintf('\nProcessing session %d/%d: %s\n', s, length(reachSessions), sessionName);
    
    try
        % Load data
        dataStruct = load_sliding_window_data('reach', 'spikes', ...
            'sessionName', sessionName, 'opts', opts);
        
        % Run analysis
        results = criticality_ar_analysis(dataStruct, config);
        
        % Extract areas if not yet set
        if isempty(reachData.areas) && isfield(results, 'areas')
            reachData.areas = results.areas;
            numAreas = length(reachData.areas);
            reachData.d2Raw = cell(1, numAreas);
            reachData.d2Normalized = cell(1, numAreas);
            for a = 1:numAreas
                reachData.d2Raw{a} = [];
                reachData.d2Normalized{a} = [];
            end
        end
        
        % Extract d2 values per area (should be single value per area)
        if isfield(results, 'd2') && isfield(results, 'd2Normalized')
            for a = 1:length(results.areas)
                if a <= length(results.d2) && ~isempty(results.d2{a})
                    % Get the first (and only) value from the window
                    d2Values = results.d2{a}(~isnan(results.d2{a}));
                    if ~isempty(d2Values)
                        reachData.d2Raw{a} = [reachData.d2Raw{a}, d2Values(1)];
                    else
                        reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                    end
                else
                    reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                end
                
                if a <= length(results.d2Normalized) && ~isempty(results.d2Normalized{a})
                    % Get the first (and only) value from the window
                    d2NormValues = results.d2Normalized{a}(~isnan(results.d2Normalized{a}));
                    if ~isempty(d2NormValues)
                        reachData.d2Normalized{a} = [reachData.d2Normalized{a}, d2NormValues(1)];
                    else
                        reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
                    end
                else
                    reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
                end
            end
        else
            warning('d2 or d2Normalized not found in results for session %s', sessionName);
            % Add NaN for all areas
            for a = 1:length(reachData.areas)
                reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
            end
        end
        
        reachData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session completed successfully\n');
        
    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(reachData.areas)
            for a = 1:length(reachData.areas)
                reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
            end
        end
        reachData.sessionNames{end+1} = sessionName;
    end
end

%% Determine common areas
if isempty(spontaneousData.areas) || isempty(reachData.areas)
    error('No data loaded. Check that sessions were processed successfully.');
end

% Find common areas
commonAreas = intersect(spontaneousData.areas, reachData.areas);
if isempty(commonAreas)
    error('No common areas found between spontaneous and reach sessions.');
end

fprintf('\n=== Common Areas Found ===\n');
for a = 1:length(commonAreas)
    fprintf('  %s\n', commonAreas{a});
end

%% Create plots
fprintf('\n=== Creating Plots ===\n');

% Determine which areas to plot (use common areas)
areasToPlot = commonAreas;
numAreasToPlot = length(areasToPlot);

% Determine which metric to plot
if useNormalized
    metricName = 'd2Normalized';
    metricLabel = 'd2 Normalized';
else
    metricName = 'd2Raw';
    metricLabel = 'd2 Raw';
end

% Create figure with two subplots: raw and normalized
figure(3000); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [100, 100, 1400, 300 * numAreasToPlot]);

% Create subplots for raw d2
for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Get d2 values for this area
    natValues = spontaneousData.d2Raw{natAreaIdx};
    reachValues = reachData.d2Raw{reachAreaIdx};
    
    % Remove NaN values
    natValues = natValues(~isnan(natValues));
    reachValues = reachValues(~isnan(reachValues));
    
    if isempty(natValues) && isempty(reachValues)
        continue;
    end
    
    % Create subplot
    subplot(numAreasToPlot, 2, (a-1)*2 + 1);
    hold on;
    
    % Create bar plot
    numNat = length(natValues);
    numReach = length(reachValues);
    
    % Plot individual bars for each session
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    
    if numNat > 0
        bar(xNat, natValues, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachValues, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    
    % Add group labels
    xlim([0.5, numNat + numReach + 0.5]);
    if numNat > 0 && numReach > 0
        xticks([mean(xNat), mean(xReach)]);
        xticklabels({'Spontaneous', 'Reach'});
    elseif numNat > 0
        xticks(mean(xNat));
        xticklabels({'Spontaneous'});
    elseif numReach > 0
        xticks(mean(xReach));
        xticklabels({'Reach'});
    end
    
    ylabel('d2 Raw');
    title(sprintf('%s - d2 Raw', areaName));
    grid on;
    
    % Add mean lines
    if ~isempty(natValues)
        yline(mean(natValues), 'b--', 'LineWidth', 2);
    end
    if ~isempty(reachValues)
        yline(mean(reachValues), 'r--', 'LineWidth', 2);
    end
    
    hold off;
    
    % Create subplot for normalized d2
    subplot(numAreasToPlot, 2, (a-1)*2 + 2);
    hold on;
    
    % Get normalized d2 values for this area
    natValuesNorm = spontaneousData.d2Normalized{natAreaIdx};
    reachValuesNorm = reachData.d2Normalized{reachAreaIdx};
    
    % Remove NaN values
    natValuesNorm = natValuesNorm(~isnan(natValuesNorm));
    reachValuesNorm = reachValuesNorm(~isnan(reachValuesNorm));
    
    numNat = length(natValuesNorm);
    numReach = length(reachValuesNorm);
    
    % Plot individual bars for each session
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    
    if numNat > 0
        bar(xNat, natValuesNorm, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachValuesNorm, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    
    % Add group labels
    xlim([0.5, numNat + numReach + 0.5]);
    if numNat > 0 && numReach > 0
        xticks([mean(xNat), mean(xReach)]);
        xticklabels({'Spontaneous', 'Reach'});
    elseif numNat > 0
        xticks(mean(xNat));
        xticklabels({'Spontaneous'});
    elseif numReach > 0
        xticks(mean(xReach));
        xticklabels({'Reach'});
    end
    
    ylabel('d2 Normalized');
    title(sprintf('%s - d2 Normalized', areaName));
    grid on;
    
    % Add mean lines
    if ~isempty(natValuesNorm)
        yline(mean(natValuesNorm), 'b--', 'LineWidth', 2);
    end
    if ~isempty(reachValuesNorm)
        yline(mean(reachValuesNorm), 'r--', 'LineWidth', 2);
    end
    
    hold off;
end

% Add overall title
sgtitle(sprintf('Criticality AR Comparison: Spontaneous vs Reach (Window: %d min)', nMin), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

plotFilename = sprintf('criticality_ar_across_sessions_%dmin.png', nMin);
plotPath = fullfile(saveDir, plotFilename);

exportgraphics(gcf, plotPath, 'Resolution', 300);
fprintf('\nSaved plot to: %s\n', plotPath);

fprintf('\n=== Analysis Complete ===\n');
