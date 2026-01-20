%%
% Criticality AR Analysis Across Sessions
% Compares d2 values from pre-analyzed results between spontaneous and reach sessions
%
% Variables:
%   timeRange - Time range in seconds [startTime, endTime] (default: [0, 20*60])
%   useNormalized - Whether to use normalized d2 values (default: true)
%   filenameSuffix - Optional suffix for results files (e.g., '_pca', default: '')
%   areasToPlot - Cell array of area names to plot (default: [] = all common areas)
%                 Example: {'M1', 'PMd'} or [] for all areas
%
% Goal:
%   Load pre-analyzed results for each session, filter by time range, and compare
%   raw and normalized d2 values between spontaneous and reach sessions.

%% Configuration
paths = get_paths;

timeRange = [0, 20*60];  % Time range in seconds [startTime, endTime]
useNormalized = true;  % Use normalized d2 if available
filenameSuffix = '';  % Optional suffix for results files (e.g., '_pca')
areasToPlot = {};  % Cell array of area names to plot (empty = all common areas)
                   % Example: {'M1', 'PMd'} or [] for all areas

% Calculate window parameters from timeRange
windowStartTime = timeRange(1);
windowEndTime = timeRange(2);
windowDuration = windowEndTime - windowStartTime;  % Window duration in seconds
nMin = windowDuration / 60;  % Window size in minutes (for display purposes)

%% Get session lists
fprintf('\n=== Criticality AR Analysis Across Sessions ===\n');
fprintf('Time range: [%.1f, %.1f] seconds (%.1f minutes)\n', ...
    windowStartTime, windowEndTime, nMin);

spontaneousSessions = spontaneous_session_list();
reachSessions = reach_session_list();

fprintf('Spontaneous sessions: %d\n', length(spontaneousSessions));
fprintf('Reach sessions: %d\n', length(reachSessions));

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
        resultsPath = find_results_file('criticality_ar', 'spontaneous', sessionName, saveDir, filenameSuffix, '');
        
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
            spontaneousData.d2Raw = cell(1, numAreas);
            spontaneousData.d2Normalized = cell(1, numAreas);
            for a = 1:numAreas
                spontaneousData.d2Raw{a} = [];
                spontaneousData.d2Normalized{a} = [];
            end
        end
        
        % Filter results by time range and extract d2 values per area
        if isfield(results, 'd2') && isfield(results, 'd2Normalized') && isfield(results, 'startS')
            for a = 1:length(results.areas)
                % Find windows within time range
                if ~isempty(results.startS{a})
                    timeMask = results.startS{a} >= windowStartTime & results.startS{a} <= windowEndTime;
                    
                    % Extract d2 values within time range
                    if a <= length(results.d2) && ~isempty(results.d2{a})
                        d2Values = results.d2{a}(timeMask);
                        d2Values = d2Values(~isnan(d2Values));
                        if ~isempty(d2Values)
                            % Use mean of values in time range (or first if single window)
                            spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, mean(d2Values)];
                        else
                            spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                        end
                    else
                        spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                    end
                    
                    % Extract normalized d2 values within time range
                    if a <= length(results.d2Normalized) && ~isempty(results.d2Normalized{a})
                        d2NormValues = results.d2Normalized{a}(timeMask);
                        d2NormValues = d2NormValues(~isnan(d2NormValues));
                        if ~isempty(d2NormValues)
                            % Use mean of values in time range (or first if single window)
                            spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, mean(d2NormValues)];
                        else
                            spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
                        end
                    else
                        spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
                    end
                else
                    spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                    spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
                end
            end
        else
            warning('d2, d2Normalized, or startS not found in results for session %s', sessionName);
            % Add NaN for all areas
            for a = 1:length(spontaneousData.areas)
                spontaneousData.d2Raw{a} = [spontaneousData.d2Raw{a}, nan];
                spontaneousData.d2Normalized{a} = [spontaneousData.d2Normalized{a}, nan];
            end
        end
        
        spontaneousData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session loaded successfully\n');
        
    catch ME
        fprintf('  ✗ Error loading session: %s\n', ME.message);
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
fprintf('\n=== Loading Reach Session Results ===\n');
for s = 1:length(reachSessions)
    sessionName = reachSessions{s};
    fprintf('\nLoading session %d/%d: %s\n', s, length(reachSessions), sessionName);
    
    try
        % Find results file
        [~, dataBaseName, ~] = fileparts(sessionName);
        saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
        
        % Find results file
        resultsPath = find_results_file('criticality_ar', 'reach', sessionName, saveDir, filenameSuffix, '');
        
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
            reachData.d2Raw = cell(1, numAreas);
            reachData.d2Normalized = cell(1, numAreas);
            for a = 1:numAreas
                reachData.d2Raw{a} = [];
                reachData.d2Normalized{a} = [];
            end
        end
        
        % Filter results by time range and extract d2 values per area
        if isfield(results, 'd2') && isfield(results, 'd2Normalized') && isfield(results, 'startS')
            for a = 1:length(results.areas)
                % Find windows within time range
                if ~isempty(results.startS{a})
                    timeMask = results.startS{a} >= windowStartTime & results.startS{a} <= windowEndTime;
                    
                    % Extract d2 values within time range
                    if a <= length(results.d2) && ~isempty(results.d2{a})
                        d2Values = results.d2{a}(timeMask);
                        d2Values = d2Values(~isnan(d2Values));
                        if ~isempty(d2Values)
                            % Use mean of values in time range (or first if single window)
                            reachData.d2Raw{a} = [reachData.d2Raw{a}, mean(d2Values)];
                        else
                            reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                        end
                    else
                        reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                    end
                    
                    % Extract normalized d2 values within time range
                    if a <= length(results.d2Normalized) && ~isempty(results.d2Normalized{a})
                        d2NormValues = results.d2Normalized{a}(timeMask);
                        d2NormValues = d2NormValues(~isnan(d2NormValues));
                        if ~isempty(d2NormValues)
                            % Use mean of values in time range (or first if single window)
                            reachData.d2Normalized{a} = [reachData.d2Normalized{a}, mean(d2NormValues)];
                        else
                            reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
                        end
                    else
                        reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
                    end
                else
                    reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                    reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
                end
            end
        else
            warning('d2, d2Normalized, or startS not found in results for session %s', sessionName);
            % Add NaN for all areas
            for a = 1:length(reachData.areas)
                reachData.d2Raw{a} = [reachData.d2Raw{a}, nan];
                reachData.d2Normalized{a} = [reachData.d2Normalized{a}, nan];
            end
        end
        
        reachData.sessionNames{end+1} = sessionName;
        fprintf('  ✓ Session loaded successfully\n');
        
    catch ME
        fprintf('  ✗ Error loading session: %s\n', ME.message);
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
areasToPlot = {'M56', 'DS'};
natColor = [0 191 255] ./ 255;
reachColor = [255 215 0] ./ 255;
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

% Determine which metric to plot
if useNormalized
    metricName = 'd2Normalized';
    metricLabel = 'd2 Normalized';
else
    metricName = 'd2Raw';
    metricLabel = 'd2 Raw';
end

% First, collect all values to determine global y-axis limits
allD2Raw = [];
allD2Normalized = [];

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Collect d2 raw values
    natValues = spontaneousData.d2Raw{natAreaIdx};
    reachValues = reachData.d2Raw{reachAreaIdx};
    natValues = natValues(~isnan(natValues));
    reachValues = reachValues(~isnan(reachValues));
    allD2Raw = [allD2Raw, natValues, reachValues];
    
    % Collect d2 normalized values
    natValuesNorm = spontaneousData.d2Normalized{natAreaIdx};
    reachValuesNorm = reachData.d2Normalized{reachAreaIdx};
    natValuesNorm = natValuesNorm(~isnan(natValuesNorm));
    reachValuesNorm = reachValuesNorm(~isnan(reachValuesNorm));
    allD2Normalized = [allD2Normalized, natValuesNorm, reachValuesNorm];
end

% Calculate global y-axis limits with some padding
if ~isempty(allD2Raw)
    ylimD2Raw = [0, min(.25, max(allD2Raw) * 1.05)];
    if ylimD2Raw(1) == ylimD2Raw(2)
        ylimD2Raw = ylimD2Raw(1) + [-0.1, 0.1];
    end
else
    ylimD2Raw = [0, 1];
end

if ~isempty(allD2Normalized)
    ylimD2Normalized = [0, max(allD2Normalized) * 1.05];
    if ylimD2Normalized(1) == ylimD2Normalized(2)
        ylimD2Normalized = ylimD2Normalized(1) + [-0.1, 0.1];
    end
else
    ylimD2Normalized = [0, 1];
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
        bar(xNat, natValues, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachValues, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    
    % Set consistent y-axis limits across all areas
    ylim(ylimD2Raw);
    
    % Add mean lines
    if ~isempty(natValues)
        yline(mean(natValues), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachValues)
        yline(mean(reachValues), '--', 'color', reachColor, 'LineWidth', 2);
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
        bar(xNat, natValuesNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachValuesNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
    
    % Set consistent y-axis limits across all areas
    ylim(ylimD2Normalized);
    
    % Add mean lines
    if ~isempty(natValuesNorm)
        yline(mean(natValuesNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachValuesNorm)
        yline(mean(reachValuesNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    
    hold off;
end

% Add overall title
sgtitle(sprintf('Criticality AR Comparison: Spontaneous vs Reach (Time Range: %.1f-%.1f s)', ...
    windowStartTime, windowEndTime), 'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

plotFilenamePng = sprintf('criticality_ar_across_sessions_%.0f-%.0fs.png', windowStartTime, windowEndTime);
plotPathPng = fullfile(saveDir, plotFilenamePng);

exportgraphics(gcf, plotPathPng, 'Resolution', 300);
fprintf('\nSaved PNG plot to: %s\n', plotPathPng);

plotFilenameEps = sprintf('criticality_ar_across_sessions_%.0f-%.0fs.eps', windowStartTime, windowEndTime);
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
