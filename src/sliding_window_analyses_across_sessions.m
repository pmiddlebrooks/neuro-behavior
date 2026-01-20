%%
% Sliding Window Analyses Across Sessions
% Compares metrics between spontaneous (spontaneous) and reach sessions
%
% Variables:
%   analysisType - Type of analysis: 'criticality_ar', 'complexity', 'rqa', 
%                  'criticality_av', 'criticality_lfp' (default: 'criticality_ar')
%   metricName - Name of metric to plot (e.g., 'd2', 'lzComplexity', 'recurrenceRate')
%                 If empty, uses default for analysis type
%   useNormalized - Whether to use normalized metric if available (default: true)
%   filenameSuffix - Optional suffix for results files (e.g., '_pca')
%
% Note: The script automatically finds results files by matching session name
%       and analysis type, regardless of window size.
%
% Goal:
%   Load saved results from spontaneous and reach sessions, calculate median
%   metric values per session per area, and create bar plots comparing the two groups.

%% Configuration
analysisType = 'rqa';  % Options: 'criticality_ar', 'complexity', 'rqa', 'criticality_av', 'criticality_lfp'
metricName = '';  % If empty, uses default for analysis type
useNormalized = true;  % Use normalized metric if available
filenameSuffix = '';  % Optional suffix (e.g., '_pca')
timeRange = [0, 1200];  % Time range in seconds to analyze [startTime, endTime]. Use [] to analyze all data.

%% Add paths
% Get paths structure
paths = get_paths;

basePath = fullfile(paths.homePath, 'neuro-behavior/src/');  % src
addpath(fullfile(basePath, 'sliding_window_prep', 'utils'));
addpath(fullfile(basePath, 'spontaneous'));
addpath(fullfile(basePath, 'reach_task'));


%% Get session lists
spontaneousSessions = spontaneous_session_list();
reachSessions = reach_session_list();

fprintf('\n=== Sliding Window Analysis Comparison ===\n');
fprintf('Analysis type: %s\n', analysisType);
fprintf('Spontaneous sessions: %d\n', length(spontaneousSessions));
fprintf('Reach sessions: %d\n', length(reachSessions));
if ~isempty(timeRange) && length(timeRange) == 2
    fprintf('Time range: [%.1f, %.1f] s\n', timeRange(1), timeRange(2));
else
    fprintf('Time range: All data\n');
end

%% Determine default metric name if not provided
if isempty(metricName)
    switch analysisType
        case 'criticality_ar'
            if useNormalized
                metricName = 'd2Normalized';
            else
                metricName = 'd2';
            end
        case 'complexity'
            if useNormalized
                metricName = 'lzComplexityNormalized';
            else
                metricName = 'lzComplexity';
            end
        case 'rqa'
            if useNormalized
                metricName = 'recurrenceRateNormalized';
            else
                metricName = 'recurrenceRate';
            end
        case 'criticality_av'
            metricName = 'dcc';  % Default to dcc for avalanche analysis
        case 'criticality_lfp'
            if useNormalized
                metricName = 'd2Normalized';
            else
                metricName = 'd2';
            end
        otherwise
            error('Unknown analysis type: %s', analysisType);
    end
end

fprintf('Metric: %s\n', metricName);

%% Load results for spontaneous sessions
fprintf('\n=== Loading Spontaneous Session Results ===\n');
spontaneousData = struct();
spontaneousData.medians = {};  % Cell array: {areaIdx}{sessionIdx} = median value
spontaneousData.sessionNames = {};
spontaneousData.areas = [];

for s = 1:length(spontaneousSessions)
    sessionName = spontaneousSessions{s};
    fprintf('Loading session %d/%d: %s\n', s, length(spontaneousSessions), sessionName);
    
    % Find results file by pattern matching
    % Extract subjectID from sessionName (similar to how load_spontaneous_data does it)
    pathParts = strsplit(sessionName, filesep);
    if length(pathParts) > 1
        % Session name has path separator (e.g., 'ag112321/recording1e')
        % Use full path for subjectID to create subdirectory
        subjectID = fullfile(pathParts{1}, pathParts{2});
    else
        % Session name is just the subjectID (e.g., 'ey042822')
        subjectID = sessionName;
    end
    
    % Use dropPath/spontaneous/results/{subjectID}/ similar to reach_task/results/{dataBaseName}
    saveDir = fullfile(paths.dropPath, 'spontaneous/results', subjectID);
    
    % Determine dataSource for pattern matching
    if strcmp(analysisType, 'complexity') || strcmp(analysisType, 'rqa')
        dataSource = 'spikes';  % Default, could be made configurable
    else
        dataSource = '';
    end
    
    % For RQA, filenameSuffix should include PCA dimensions (e.g., '_pca4')
    % If not provided, try common values
    actualFilenameSuffix = filenameSuffix;
    if strcmp(analysisType, 'rqa') && isempty(filenameSuffix)
        % Try common PCA dimensions
        actualFilenameSuffix = '_pca4';  % Default, could be made configurable
    end
    
    resultsPath = find_results_file(analysisType, 'spontaneous', sessionName, saveDir, actualFilenameSuffix, dataSource);
    
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
        spontaneousData.medians = cell(1, numAreas);
        for a = 1:numAreas
            spontaneousData.medians{a} = [];
        end
    end
    
    % Filter results by time range if specified
    if ~isempty(timeRange) && length(timeRange) == 2 && isfield(results, 'startS')
        timeStart = timeRange(1);
        timeEnd = timeRange(2);
        numAreas = length(results.areas);
        for a = 1:numAreas
            if ~isempty(results.startS{a})
                % Find indices within time range
                timeMask = results.startS{a} >= timeStart & results.startS{a} <= timeEnd;
                
                % Filter startS
                results.startS{a} = results.startS{a}(timeMask);
                
                % Filter metric data if it exists
                if isfield(results, metricName) && iscell(results.(metricName)) && ...
                        a <= length(results.(metricName)) && ~isempty(results.(metricName){a})
                    results.(metricName){a} = results.(metricName){a}(timeMask);
                end
            end
        end
    end
    
    % Extract metric values per area
    if isfield(results, metricName)
        metricData = results.(metricName);
        if iscell(metricData)
            % Cell array format (one cell per area)
            for a = 1:length(metricData)
                if a <= length(spontaneousData.medians) && ~isempty(metricData{a})
                    values = metricData{a}(:);
                    values = values(~isnan(values));
                    if ~isempty(values)
                        medianVal = median(values);
                        spontaneousData.medians{a} = [spontaneousData.medians{a}, medianVal];
                    else
                        spontaneousData.medians{a} = [spontaneousData.medians{a}, nan];
                    end
                end
            end
        else
            % Single array format (shouldn't happen for these analyses, but handle it)
            warning('Metric %s is not in cell array format for session %s', metricName, sessionName);
        end
    else
        warning('Metric %s not found in results for session %s', metricName, sessionName);
        % Add NaN for all areas
        for a = 1:length(spontaneousData.medians)
            spontaneousData.medians{a} = [spontaneousData.medians{a}, nan];
        end
    end
    
    spontaneousData.sessionNames{end+1} = sessionName;
end

%% Load results for reach sessions
fprintf('\n=== Loading Reach Session Results ===\n');
reachData = struct();
reachData.medians = {};  % Cell array: {areaIdx}{sessionIdx} = median value
reachData.sessionNames = {};
reachData.areas = [];

for s = 1:length(reachSessions)
    sessionName = reachSessions{s};
    fprintf('Loading session %d/%d: %s\n', s, length(reachSessions), sessionName);
    
    % Find results file by pattern matching
    [~, dataBaseName, ~] = fileparts(sessionName);
    saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    
    % Determine dataSource for pattern matching
    if strcmp(analysisType, 'complexity') || strcmp(analysisType, 'rqa')
        dataSource = 'spikes';  % Default, could be made configurable
    else
        dataSource = '';
    end
    
    % For RQA, filenameSuffix should include PCA dimensions (e.g., '_pca4')
    % If not provided, try common values
    actualFilenameSuffix = filenameSuffix;
    if strcmp(analysisType, 'rqa') && isempty(filenameSuffix)
        % Try common PCA dimensions
        actualFilenameSuffix = '_pca4';  % Default, could be made configurable
    end
    
    resultsPath = find_results_file(analysisType, 'reach', sessionName, saveDir, actualFilenameSuffix, dataSource);
    
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
        reachData.medians = cell(1, numAreas);
        for a = 1:numAreas
            reachData.medians{a} = [];
        end
    end
    
    % Filter results by time range if specified
    if ~isempty(timeRange) && length(timeRange) == 2 && isfield(results, 'startS')
        timeStart = timeRange(1);
        timeEnd = timeRange(2);
        numAreas = length(results.areas);
        for a = 1:numAreas
            if ~isempty(results.startS{a})
                % Find indices within time range
                timeMask = results.startS{a} >= timeStart & results.startS{a} <= timeEnd;
                
                % Filter startS
                results.startS{a} = results.startS{a}(timeMask);
                
                % Filter metric data if it exists
                if isfield(results, metricName) && iscell(results.(metricName)) && ...
                        a <= length(results.(metricName)) && ~isempty(results.(metricName){a})
                    results.(metricName){a} = results.(metricName){a}(timeMask);
                end
            end
        end
    end
    
    % Extract metric values per area
    if isfield(results, metricName)
        metricData = results.(metricName);
        if iscell(metricData)
            % Cell array format (one cell per area)
            for a = 1:length(metricData)
                if a <= length(reachData.medians) && ~isempty(metricData{a})
                    values = metricData{a}(:);
                    values = values(~isnan(values));
                    if ~isempty(values)
                        medianVal = median(values);
                        reachData.medians{a} = [reachData.medians{a}, medianVal];
                    else
                        reachData.medians{a} = [reachData.medians{a}, nan];
                    end
                end
            end
        else
            % Single array format (shouldn't happen for these analyses, but handle it)
            warning('Metric %s is not in cell array format for session %s', metricName, sessionName);
        end
    else
        warning('Metric %s not found in results for session %s', metricName, sessionName);
        % Add NaN for all areas
        for a = 1:length(reachData.medians)
            reachData.medians{a} = [reachData.medians{a}, nan];
        end
    end
    
    reachData.sessionNames{end+1} = sessionName;
end

%% Determine common areas
if isempty(spontaneousData.areas) || isempty(reachData.areas)
    error('No data loaded. Check that results files exist and contain the expected structure.');
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

%% Create bar plots for each area
fprintf('\n=== Creating Bar Plots ===\n');

% Determine which areas to plot (use common areas)
areasToPlot = commonAreas;
numAreasToPlot = length(areasToPlot);

% Create figure
figure(2000); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [100, 100, 1200, 300 * numAreasToPlot]);

% Create subplots
for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Get median values for this area
    natMedians = spontaneousData.medians{natAreaIdx};
    reachMedians = reachData.medians{reachAreaIdx};
    
    % Remove NaN values
    natMedians = natMedians(~isnan(natMedians));
    reachMedians = reachMedians(~isnan(reachMedians));
    
    if isempty(natMedians) && isempty(reachMedians)
        continue;
    end
    
    % Create subplot
    subplot(numAreasToPlot, 1, a);
    hold on;
    
    % Prepare data for bar plot
    % Group 1: Spontaneous sessions
    % Group 2: Reach sessions
    allValues = [natMedians, reachMedians];
    groupLabels = [ones(1, length(natMedians)), 2*ones(1, length(reachMedians))];
    
    % Create bar plot
    numNat = length(natMedians);
    numReach = length(reachMedians);
    
    % Plot individual bars for each session
    % Spontaneous sessions: x positions 1 to numNat
    % Reach sessions: x positions (numNat + 1) to (numNat + numReach)
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    
    % Plot bars with spacing between groups
    if numNat > 0
        bar(xNat, natMedians, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachMedians, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
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
    
    ylabel(sprintf('%s (median)', metricName));
    title(sprintf('%s - %s', areaName, metricName));
    grid on;
    
    % Add mean lines
    if ~isempty(natMedians)
        yline(mean(natMedians), 'b--', 'LineWidth', 2, 'DisplayName', sprintf('Nat mean: %.3f', mean(natMedians)));
    end
    if ~isempty(reachMedians)
        yline(mean(reachMedians), 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Reach mean: %.3f', mean(reachMedians)));
    end
    
    hold off;
end

% Add overall title
sgtitle(sprintf('%s Comparison: Spontaneous vs Reach', ...
    analysisType), 'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% Add time range to filename if specified
timeRangeStr = '';
if ~isempty(timeRange) && length(timeRange) == 2
    timeRangeStr = sprintf('_t%.0f-%.0f', timeRange(1), timeRange(2));
end

plotFilename = sprintf('%s_%s_nat_vs_reach%s%s.png', ...
    analysisType, metricName, filenameSuffix, timeRangeStr);
plotPath = fullfile(saveDir, plotFilename);

exportgraphics(gcf, plotPath, 'Resolution', 300);
fprintf('\nSaved plot to: %s\n', plotPath);

fprintf('\n=== Analysis Complete ===\n');

%% Helper function to find results file by pattern
function resultsPath = find_results_file(analysisType, sessionType, sessionName, saveDir, filenameSuffix, dataSource)
    % Build search pattern based on analysis type
    % Updated to match current naming convention from create_results_path
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
            
        case 'criticality_lfp'
            % Format: criticality_sliding_lfp{filenameSuffix}_{sessionName}.mat
            if ~isempty(sessionName)
                pattern = sprintf('criticality_sliding_lfp%s_%s.mat', filenameSuffix, sessionName);
            else
                pattern = sprintf('criticality_sliding_lfp%s_*.mat', filenameSuffix);
            end
            
        case 'complexity'
            % Format: lzc_sliding_window{filenameSuffix}_{sessionName}.mat (for spikes)
            %         lzc_sliding_window_{dataSource}{filenameSuffix}_{sessionName}.mat (for lfp)
            if strcmp(dataSource, 'lfp')
                if ~isempty(sessionName)
                    pattern = sprintf('lzc_sliding_window_%s%s_%s.mat', dataSource, filenameSuffix, sessionName);
                else
                    pattern = sprintf('lzc_sliding_window_%s%s_*.mat', dataSource, filenameSuffix);
                end
            else
                % For spikes, dataSource is not in filename
                if ~isempty(sessionName)
                    pattern = sprintf('lzc_sliding_window%s_%s.mat', filenameSuffix, sessionName);
                else
                    pattern = sprintf('lzc_sliding_window%s_*.mat', filenameSuffix);
                end
            end
            
        case 'rqa'
            % Format: rqa_sliding_window{filenameSuffix}_{sessionName}.mat (for spikes)
            %         rqa_sliding_window_{dataSource}{filenameSuffix}_{sessionName}.mat (for lfp)
            % Note: filenameSuffix includes PCA dimensions (e.g., '_pca4')
            if strcmp(dataSource, 'lfp')
                if ~isempty(sessionName)
                    pattern = sprintf('rqa_sliding_window_%s%s_%s.mat', dataSource, filenameSuffix, sessionName);
                else
                    pattern = sprintf('rqa_sliding_window_%s%s_*.mat', dataSource, filenameSuffix);
                end
            else
                % For spikes, dataSource is not in filename
                if ~isempty(sessionName)
                    pattern = sprintf('rqa_sliding_window%s_%s.mat', filenameSuffix, sessionName);
                else
                    pattern = sprintf('rqa_sliding_window%s_*.mat', filenameSuffix);
                end
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
    % (since filenames can't contain path separators)
    if contains(sessionName, filesep)
        sessionNameForPattern = strrep(sessionName, filesep, '_');
        % Update pattern to use sessionNameForPattern
        pattern = strrep(pattern, sessionName, sessionNameForPattern);
    end
    
    files = dir(fullfile(saveDir, pattern));
    
    % Also search in subdirectories if they exist (for backward compatibility)
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

