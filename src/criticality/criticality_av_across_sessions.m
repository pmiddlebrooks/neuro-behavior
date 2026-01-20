%%
% Criticality Avalanche Analysis Across Sessions
% Compares avalanche metrics (dcc, kappa) from pre-analyzed results between spontaneous and reach sessions
%
% Variables:
%   timeRange - Time range in seconds [startTime, endTime] (default: [0, 20*60])
%   filenameSuffix - Optional suffix for results files (e.g., '_pca', default: '')
%   areasToPlot - Cell array of area names to plot (default: [] = all common areas)
%                 Example: {'M1', 'PMd'} or [] for all areas
%
% Goal:
%   Load pre-analyzed results for each session, filter by time range, and compare
%   avalanche metrics (dcc and kappa) between spontaneous and reach sessions.

%% Configuration
paths = get_paths;

timeRange = [0, 20*60];  % Time range in seconds [startTime, endTime]
filenameSuffix = '';  % Optional suffix for results files (e.g., '_pca')
areasToPlot = {};  % Cell array of area names to plot (empty = all common areas)
                   % Example: {'M1', 'PMd'} or [] for all areas

% Calculate window parameters from timeRange
windowStartTime = timeRange(1);
windowEndTime = timeRange(2);
windowDuration = windowEndTime - windowStartTime;  % Window duration in seconds
nMin = windowDuration / 60;  % Window size in minutes (for display purposes)

%% Get session lists
fprintf('\n=== Criticality Avalanche Analysis Across Sessions ===\n');
fprintf('Time range: [%.1f, %.1f] seconds (%.1f minutes)\n', ...
    windowStartTime, windowEndTime, nMin);

spontaneousSessions = spontaneous_session_list();
reachSessions = reach_session_list();

fprintf('Spontaneous sessions: %d\n', length(spontaneousSessions));
fprintf('Reach sessions: %d\n', length(reachSessions));

%% Initialize results storage
spontaneousData = struct();
spontaneousData.dcc = {};  % Cell array: {areaIdx}{sessionIdx} = dcc value
spontaneousData.dccNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized dcc value
spontaneousData.kappa = {};  % Cell array: {areaIdx}{sessionIdx} = kappa value
spontaneousData.kappaNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized kappa value
spontaneousData.sessionNames = {};
spontaneousData.areas = [];

reachData = struct();
reachData.dcc = {};  % Cell array: {areaIdx}{sessionIdx} = dcc value
reachData.dccNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized dcc value
reachData.kappa = {};  % Cell array: {areaIdx}{sessionIdx} = kappa value
reachData.kappaNormalized = {};  % Cell array: {areaIdx}{sessionIdx} = normalized kappa value
reachData.sessionNames = {};
reachData.areas = [];

%% Process spontaneous sessions
fprintf('\n=== Loading Spontaneous Session Results ===\n');
for s = 1:length(spontaneousSessions)
    sessionName = spontaneousSessions{s};
    fprintf('\nLoading session %d/%d: %s\n', s, length(spontaneousSessions), sessionName);
    
    try
        % Find results file
        % Extract subjectID from sessionName (similar to how load_spontaneous_data does it)
        pathParts = strsplit(sessionName, filesep);
        if length(pathParts) > 1
            subjectID = fullfile(pathParts{1}, pathParts{2});
        else
            subjectID = sessionName;
        end
        
        % Use dropPath/spontaneous/results/{subjectID}/
        saveDir = fullfile(paths.dropPath, 'spontaneous/results', subjectID);
        
        % Find results file
        resultsPath = find_results_file('criticality_av', 'spontaneous', sessionName, saveDir, filenameSuffix, '');
        
        % Load results
        if isempty(resultsPath) || ~exist(resultsPath, 'file')
            warning('Results file not found for session: %s\nSkipping session.', sessionName);
            continue;
        end
        
        loaded = load(resultsPath);
        if ~isfield(loaded, 'results')
            warning('Results structure not found in file: %s\nSkipping session.', resultsPath);
            continue;
        end
        results = loaded.results;
        
        % Extract areas if not yet set
        if isempty(spontaneousData.areas) && isfield(results, 'areas')
            spontaneousData.areas = results.areas;
            numAreas = length(spontaneousData.areas);
            spontaneousData.dcc = cell(1, numAreas);
            spontaneousData.dccNormalized = cell(1, numAreas);
            spontaneousData.kappa = cell(1, numAreas);
            spontaneousData.kappaNormalized = cell(1, numAreas);
            for a = 1:numAreas
                spontaneousData.dcc{a} = [];
                spontaneousData.dccNormalized{a} = [];
                spontaneousData.kappa{a} = [];
                spontaneousData.kappaNormalized{a} = [];
            end
        end
        
        % Filter results by time range and extract metric values per area
        if isfield(results, 'dcc') && isfield(results, 'kappa') && isfield(results, 'startS')
            for a = 1:length(results.areas)
                % Find windows within time range
                if ~isempty(results.startS{a})
                    timeMask = results.startS{a} >= windowStartTime & results.startS{a} <= windowEndTime;
                    
                    % Extract dcc raw
                    if a <= length(results.dcc) && ~isempty(results.dcc{a})
                        dccValues = results.dcc{a}(timeMask);
                        dccValues = dccValues(~isnan(dccValues));
                        if ~isempty(dccValues)
                            spontaneousData.dcc{a} = [spontaneousData.dcc{a}, mean(dccValues)];
                        else
                            spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                        end
                    else
                        spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                    end
                    
                    % Extract dcc normalized
                    if a <= length(results.dccNormalized) && ~isempty(results.dccNormalized{a})
                        dccNormValues = results.dccNormalized{a}(timeMask);
                        dccNormValues = dccNormValues(~isnan(dccNormValues));
                        if ~isempty(dccNormValues)
                            spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, mean(dccNormValues)];
                        else
                            spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                        end
                    else
                        spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                    end
                    
                    % Extract kappa raw
                    if a <= length(results.kappa) && ~isempty(results.kappa{a})
                        kappaValues = results.kappa{a}(timeMask);
                        kappaValues = kappaValues(~isnan(kappaValues));
                        if ~isempty(kappaValues)
                            spontaneousData.kappa{a} = [spontaneousData.kappa{a}, mean(kappaValues)];
                        else
                            spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                        end
                    else
                        spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                    end
                    
                    % Extract kappa normalized
                    if a <= length(results.kappaNormalized) && ~isempty(results.kappaNormalized{a})
                        kappaNormValues = results.kappaNormalized{a}(timeMask);
                        kappaNormValues = kappaNormValues(~isnan(kappaNormValues));
                        if ~isempty(kappaNormValues)
                            spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, mean(kappaNormValues)];
                        else
                            spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
                        end
                    else
                        spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
                    end
                else
                    spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                    spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                    spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                    spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
                end
            end
        else
            warning('dcc, kappa, or startS not found in results for session %s', sessionName);
            % Add NaN for all areas
            for a = 1:length(spontaneousData.areas)
                spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
            end
        end
        
        spontaneousData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session loaded successfully\n');
        
    catch ME
        fprintf('  ✗ Error loading session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(spontaneousData.areas)
            for a = 1:length(spontaneousData.areas)
                spontaneousData.dcc{a} = [spontaneousData.dcc{a}, nan];
                spontaneousData.dccNormalized{a} = [spontaneousData.dccNormalized{a}, nan];
                spontaneousData.kappa{a} = [spontaneousData.kappa{a}, nan];
                spontaneousData.kappaNormalized{a} = [spontaneousData.kappaNormalized{a}, nan];
            end
        end
        spontaneousData.sessionNames{end+1} = sessionName;
    end
end

%% Process reach sessions
fprintf('\n=== Loading Reach Session Results ===\n');
for s = 1:length(reachSessions)
    sessionName = reachSessions{s};
    fprintf('\nLoading session %d/%d: %s\n', s, length(reachSessions), sessionName);
    
    try
        % Find results file
        [~, dataBaseName, ~] = fileparts(sessionName);
        saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
        
        % Find results file
        resultsPath = find_results_file('criticality_av', 'reach', sessionName, saveDir, filenameSuffix, '');
        
        % Load results
        if isempty(resultsPath) || ~exist(resultsPath, 'file')
            warning('Results file not found for session: %s\nSkipping session.', sessionName);
            continue;
        end
        
        loaded = load(resultsPath);
        if ~isfield(loaded, 'results')
            warning('Results structure not found in file: %s\nSkipping session.', resultsPath);
            continue;
        end
        results = loaded.results;
        
        % Extract areas if not yet set
        if isempty(reachData.areas) && isfield(results, 'areas')
            reachData.areas = results.areas;
            numAreas = length(reachData.areas);
            reachData.dcc = cell(1, numAreas);
            reachData.dccNormalized = cell(1, numAreas);
            reachData.kappa = cell(1, numAreas);
            reachData.kappaNormalized = cell(1, numAreas);
            for a = 1:numAreas
                reachData.dcc{a} = [];
                reachData.dccNormalized{a} = [];
                reachData.kappa{a} = [];
                reachData.kappaNormalized{a} = [];
            end
        end
        
        % Filter results by time range and extract metric values per area
        if isfield(results, 'dcc') && isfield(results, 'kappa') && isfield(results, 'startS')
            for a = 1:length(results.areas)
                % Find windows within time range
                if ~isempty(results.startS{a})
                    timeMask = results.startS{a} >= windowStartTime & results.startS{a} <= windowEndTime;
                    
                    % Extract dcc raw
                    if a <= length(results.dcc) && ~isempty(results.dcc{a})
                        dccValues = results.dcc{a}(timeMask);
                        dccValues = dccValues(~isnan(dccValues));
                        if ~isempty(dccValues)
                            reachData.dcc{a} = [reachData.dcc{a}, mean(dccValues)];
                        else
                            reachData.dcc{a} = [reachData.dcc{a}, nan];
                        end
                    else
                        reachData.dcc{a} = [reachData.dcc{a}, nan];
                    end
                    
                    % Extract dcc normalized
                    if a <= length(results.dccNormalized) && ~isempty(results.dccNormalized{a})
                        dccNormValues = results.dccNormalized{a}(timeMask);
                        dccNormValues = dccNormValues(~isnan(dccNormValues));
                        if ~isempty(dccNormValues)
                            reachData.dccNormalized{a} = [reachData.dccNormalized{a}, mean(dccNormValues)];
                        else
                            reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                        end
                    else
                        reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                    end
                    
                    % Extract kappa raw
                    if a <= length(results.kappa) && ~isempty(results.kappa{a})
                        kappaValues = results.kappa{a}(timeMask);
                        kappaValues = kappaValues(~isnan(kappaValues));
                        if ~isempty(kappaValues)
                            reachData.kappa{a} = [reachData.kappa{a}, mean(kappaValues)];
                        else
                            reachData.kappa{a} = [reachData.kappa{a}, nan];
                        end
                    else
                        reachData.kappa{a} = [reachData.kappa{a}, nan];
                    end
                    
                    % Extract kappa normalized
                    if a <= length(results.kappaNormalized) && ~isempty(results.kappaNormalized{a})
                        kappaNormValues = results.kappaNormalized{a}(timeMask);
                        kappaNormValues = kappaNormValues(~isnan(kappaNormValues));
                        if ~isempty(kappaNormValues)
                            reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, mean(kappaNormValues)];
                        else
                            reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
                        end
                    else
                        reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
                    end
                else
                    reachData.dcc{a} = [reachData.dcc{a}, nan];
                    reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                    reachData.kappa{a} = [reachData.kappa{a}, nan];
                    reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
                end
            end
        else
            warning('dcc, kappa, or startS not found in results for session %s', sessionName);
            % Add NaN for all areas
            for a = 1:length(reachData.areas)
                reachData.dcc{a} = [reachData.dcc{a}, nan];
                reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                reachData.kappa{a} = [reachData.kappa{a}, nan];
                reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
            end
        end
        
        reachData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session loaded successfully\n');
        
    catch ME
        fprintf('  ✗ Error loading session: %s\n', ME.message);
        % Add NaN for all areas
        if ~isempty(reachData.areas)
            for a = 1:length(reachData.areas)
                reachData.dcc{a} = [reachData.dcc{a}, nan];
                reachData.dccNormalized{a} = [reachData.dccNormalized{a}, nan];
                reachData.kappa{a} = [reachData.kappa{a}, nan];
                reachData.kappaNormalized{a} = [reachData.kappaNormalized{a}, nan];
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

% Determine which areas to plot
if isempty(areasToPlot)
    % Use all common areas if not specified
    areasToPlot = commonAreas;
else
    % Filter to only include areas that exist in common areas
    areasToPlot = intersect(areasToPlot, commonAreas);
    if isempty(areasToPlot)
        error('None of the specified areas to plot are found in common areas.');
    end
    fprintf('Plotting specified areas: %s\n', strjoin(areasToPlot, ', '));
end
numAreasToPlot = length(areasToPlot);

% Create figure with four columns: dcc raw, dcc normalized, kappa raw, kappa normalized
figure(3001); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [100, 100, 2000, 300 * numAreasToPlot]);

% Create subplots for dcc and kappa (raw and normalized)
for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Get dcc raw values for this area
    natValuesDcc = spontaneousData.dcc{natAreaIdx};
    reachValuesDcc = reachData.dcc{reachAreaIdx};
    
    % Remove NaN values
    natValuesDcc = natValuesDcc(~isnan(natValuesDcc));
    reachValuesDcc = reachValuesDcc(~isnan(reachValuesDcc));
    
    % Create subplot for dcc raw
    subplot(numAreasToPlot, 4, (a-1)*4 + 1);
    hold on;
    
    % Create bar plot
    numNat = length(natValuesDcc);
    numReach = length(reachValuesDcc);
    
    % Plot individual bars for each session
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    
    if numNat > 0
        bar(xNat, natValuesDcc, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachValuesDcc, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
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
    
    ylabel('dcc Raw');
    title(sprintf('%s - dcc Raw', areaName));
    grid on;
    
    % Add mean lines
    if ~isempty(natValuesDcc)
        yline(mean(natValuesDcc), 'b--', 'LineWidth', 2);
    end
    if ~isempty(reachValuesDcc)
        yline(mean(reachValuesDcc), 'r--', 'LineWidth', 2);
    end
    
    hold off;
    
    % Create subplot for dcc normalized
    subplot(numAreasToPlot, 4, (a-1)*4 + 2);
    hold on;
    
    % Get dcc normalized values for this area
    natValuesDccNorm = spontaneousData.dccNormalized{natAreaIdx};
    reachValuesDccNorm = reachData.dccNormalized{reachAreaIdx};
    
    % Remove NaN values
    natValuesDccNorm = natValuesDccNorm(~isnan(natValuesDccNorm));
    reachValuesDccNorm = reachValuesDccNorm(~isnan(reachValuesDccNorm));
    
    numNat = length(natValuesDccNorm);
    numReach = length(reachValuesDccNorm);
    
    % Plot individual bars for each session
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    
    if numNat > 0
        bar(xNat, natValuesDccNorm, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachValuesDccNorm, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
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
    
    ylabel('dcc Normalized');
    title(sprintf('%s - dcc Normalized', areaName));
    grid on;
    
    % Add mean lines
    if ~isempty(natValuesDccNorm)
        yline(mean(natValuesDccNorm), 'b--', 'LineWidth', 2);
    end
    if ~isempty(reachValuesDccNorm)
        yline(mean(reachValuesDccNorm), 'r--', 'LineWidth', 2);
    end
    
    hold off;
    
    % Create subplot for kappa raw
    subplot(numAreasToPlot, 4, (a-1)*4 + 3);
    hold on;
    
    % Get kappa raw values for this area
    natValuesKappa = spontaneousData.kappa{natAreaIdx};
    reachValuesKappa = reachData.kappa{reachAreaIdx};
    
    % Remove NaN values
    natValuesKappa = natValuesKappa(~isnan(natValuesKappa));
    reachValuesKappa = reachValuesKappa(~isnan(reachValuesKappa));
    
    numNat = length(natValuesKappa);
    numReach = length(reachValuesKappa);
    
    % Plot individual bars for each session
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    
    if numNat > 0
        bar(xNat, natValuesKappa, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachValuesKappa, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
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
    
    ylabel('kappa Raw');
    title(sprintf('%s - kappa Raw', areaName));
    grid on;
    
    % Add mean lines
    if ~isempty(natValuesKappa)
        yline(mean(natValuesKappa), 'b--', 'LineWidth', 2);
    end
    if ~isempty(reachValuesKappa)
        yline(mean(reachValuesKappa), 'r--', 'LineWidth', 2);
    end
    
    hold off;
    
    % Create subplot for kappa normalized
    subplot(numAreasToPlot, 4, (a-1)*4 + 4);
    hold on;
    
    % Get kappa normalized values for this area
    natValuesKappaNorm = spontaneousData.kappaNormalized{natAreaIdx};
    reachValuesKappaNorm = reachData.kappaNormalized{reachAreaIdx};
    
    % Remove NaN values
    natValuesKappaNorm = natValuesKappaNorm(~isnan(natValuesKappaNorm));
    reachValuesKappaNorm = reachValuesKappaNorm(~isnan(reachValuesKappaNorm));
    
    numNat = length(natValuesKappaNorm);
    numReach = length(reachValuesKappaNorm);
    
    % Plot individual bars for each session
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    
    if numNat > 0
        bar(xNat, natValuesKappaNorm, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachValuesKappaNorm, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
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
    
    ylabel('kappa Normalized');
    title(sprintf('%s - kappa Normalized', areaName));
    grid on;
    
    % Add mean lines
    if ~isempty(natValuesKappaNorm)
        yline(mean(natValuesKappaNorm), 'b--', 'LineWidth', 2);
    end
    if ~isempty(reachValuesKappaNorm)
        yline(mean(reachValuesKappaNorm), 'r--', 'LineWidth', 2);
    end
    
    hold off;
end

% Add overall title
sgtitle(sprintf('Criticality Avalanche Comparison: Spontaneous vs Reach (Time Range: %.1f-%.1f s)', ...
    windowStartTime, windowEndTime), 'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

plotFilenamePng = sprintf('criticality_av_across_sessions_%.0f-%.0fs.png', windowStartTime, windowEndTime);
plotPathPng = fullfile(saveDir, plotFilenamePng);

exportgraphics(gcf, plotPathPng, 'Resolution', 300);
fprintf('\nSaved PNG plot to: %s\n', plotPathPng);

plotFilenameEps = sprintf('criticality_av_across_sessions_%.0f-%.0fs.eps', windowStartTime, windowEndTime);
plotPathEps = fullfile(saveDir, plotFilenameEps);

exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
fprintf('Saved EPS plot to: %s\n', plotPathEps);

fprintf('\n=== Analysis Complete ===\n');

%% Helper function to find results file
function resultsPath = find_results_file(analysisType, sessionType, sessionName, saveDir, filenameSuffix, dataSource)
    % Build search pattern based on analysis type
    switch analysisType
        case 'criticality_ar'
            % Format: criticality_sliding_window_ar{filenameSuffix}_{sessionName}.mat
            if ~isempty(sessionName)
                pattern = sprintf('criticality_sliding_window_ar%s_%s.mat', filenameSuffix, sessionName);
            else
                pattern = sprintf('criticality_sliding_window_ar%s_*.mat', filenameSuffix);
            end
            
        case 'criticality_av'
            % Format: criticality_sliding_window_av{filenameSuffix}_{sessionName}.mat
            if ~isempty(sessionName)
                pattern = sprintf('criticality_sliding_window_av%s_%s.mat', filenameSuffix, sessionName);
            else
                pattern = sprintf('criticality_sliding_window_av%s_*.mat', filenameSuffix);
            end
            
        otherwise
            error('Unknown analysis type: %s', analysisType);
    end
    
    % Search for matching files
    if ~exist(saveDir, 'dir')
        resultsPath = '';
        return;
    end
    
    % For sessionName with path separators, replace them with underscores in pattern
    if contains(sessionName, filesep)
        sessionNameForPattern = strrep(sessionName, filesep, '_');
        pattern = strrep(pattern, sessionName, sessionNameForPattern);
    end
    
    files = dir(fullfile(saveDir, pattern));
    
    % Also search in subdirectories if they exist
    if exist(saveDir, 'dir')
        subDirs = dir(saveDir);
        subDirs = subDirs([subDirs.isdir] & ~strncmp({subDirs.name}, '.', 1));
        for d = 1:length(subDirs)
            subDirPath = fullfile(saveDir, subDirs(d).name);
            subFiles = dir(fullfile(subDirPath, pattern));
            if ~isempty(subFiles)
                files = [files; subFiles];
            end
        end
    end
    
    if isempty(files)
        resultsPath = '';
        return;
    end
    
    % Use the first matching file (or most recent if multiple)
    if length(files) > 1
        [~, idx] = sort([files.datenum], 'descend');
        files = files(idx);
        fprintf('  Found %d matching files, using most recent: %s\n', length(files), files(1).name);
    end
    
    if files(1).isdir
        resultsPath = '';
        return;
    end
    
    resultsPath = fullfile(files(1).folder, files(1).name);
end
