%%
% Sliding Window Analyses Across Sessions
% Compares metrics between spontaneous (open_field) and reach sessions
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
analysisType = 'criticality_ar';  % Options: 'criticality_ar', 'complexity', 'rqa', 'criticality_av', 'criticality_lfp'
metricName = '';  % If empty, uses default for analysis type
useNormalized = true;  % Use normalized metric if available
filenameSuffix = '';  % Optional suffix (e.g., '_pca')

%% Add paths
% Get paths structure
paths = get_paths;

basePath = fullfile(paths.homePath, 'neuro-behavior/src/');  % src
addpath(fullfile(basePath, 'sliding_window_prep', 'utils'));
addpath(fullfile(basePath, 'open_field'));
addpath(fullfile(basePath, 'reach_task'));


%% Get session lists
naturalisticSessions = open_field_session_list();
reachSessions = reach_session_list();

fprintf('\n=== Sliding Window Analysis Comparison ===\n');
fprintf('Analysis type: %s\n', analysisType);
fprintf('Spontaneous sessions: %d\n', length(naturalisticSessions));
fprintf('Reach sessions: %d\n', length(reachSessions));

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
naturalisticData = struct();
naturalisticData.medians = {};  % Cell array: {areaIdx}{sessionIdx} = median value
naturalisticData.sessionNames = {};
naturalisticData.areas = [];

for s = 1:length(naturalisticSessions)
    sessionName = naturalisticSessions{s};
    fprintf('Loading session %d/%d: %s\n', s, length(naturalisticSessions), sessionName);
    
    % Find results file by pattern matching
    % Extract subjectID from sessionName (similar to how load_naturalistic_data does it)
    pathParts = strsplit(sessionName, filesep);
    if length(pathParts) > 1
        % Session name has path separator (e.g., 'ag112321/recording1e')
        % Use full path for subjectID to create subdirectory
        subjectID = fullfile(pathParts{1}, pathParts{2});
    else
        % Session name is just the subjectID (e.g., 'ey042822')
        subjectID = sessionName;
    end
    
    % Use dropPath/open_field/results/{subjectID}/ similar to reach_task/results/{dataBaseName}
    saveDir = fullfile(paths.dropPath, 'open_field/results', subjectID);
    
    % Determine dataSource for pattern matching
    if strcmp(analysisType, 'complexity') || strcmp(analysisType, 'rqa')
        dataSource = 'spikes';  % Default, could be made configurable
    else
        dataSource = '';
    end
    
    resultsPath = find_results_file(analysisType, 'spontaneous', sessionName, saveDir, filenameSuffix, dataSource);
    
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
    if isempty(naturalisticData.areas) && isfield(results, 'areas')
        naturalisticData.areas = results.areas;
        numAreas = length(naturalisticData.areas);
        naturalisticData.medians = cell(1, numAreas);
        for a = 1:numAreas
            naturalisticData.medians{a} = [];
        end
    end
    
    % Extract metric values per area
    if isfield(results, metricName)
        metricData = results.(metricName);
        if iscell(metricData)
            % Cell array format (one cell per area)
            for a = 1:length(metricData)
                if a <= length(naturalisticData.medians) && ~isempty(metricData{a})
                    values = metricData{a}(:);
                    values = values(~isnan(values));
                    if ~isempty(values)
                        medianVal = median(values);
                        naturalisticData.medians{a} = [naturalisticData.medians{a}, medianVal];
                    else
                        naturalisticData.medians{a} = [naturalisticData.medians{a}, nan];
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
        for a = 1:length(naturalisticData.medians)
            naturalisticData.medians{a} = [naturalisticData.medians{a}, nan];
        end
    end
    
    naturalisticData.sessionNames{end+1} = sessionName;
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
    
    resultsPath = find_results_file(analysisType, 'reach', sessionName, saveDir, filenameSuffix, dataSource);
    
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
if isempty(naturalisticData.areas) || isempty(reachData.areas)
    error('No data loaded. Check that results files exist and contain the expected structure.');
end

% Find common areas
commonAreas = intersect(naturalisticData.areas, reachData.areas);
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
    natAreaIdx = find(strcmp(naturalisticData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    % Get median values for this area
    natMedians = naturalisticData.medians{natAreaIdx};
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

plotFilename = sprintf('%s_%s_nat_vs_reach%s.png', ...
    analysisType, metricName, filenameSuffix);
plotPath = fullfile(saveDir, plotFilename);

exportgraphics(gcf, plotPath, 'Resolution', 300);
fprintf('\nSaved plot to: %s\n', plotPath);

fprintf('\n=== Analysis Complete ===\n');

%% Helper function to find results file by pattern
function resultsPath = find_results_file(analysisType, sessionType, sessionName, saveDir, filenameSuffix, dataSource)
    % Build search pattern based on analysis type
    switch analysisType
        case 'criticality_ar'
            if strcmp(sessionType, 'reach')
                pattern = sprintf('criticality_sliding_window_ar%s_win*_%s.mat', filenameSuffix, sessionName);
            else
                % For spontaneous, session name might be in subdirectory or filename
                pattern = sprintf('criticality_sliding_window_ar%s_win*.mat', filenameSuffix);
            end
            
        case 'criticality_av'
            if strcmp(sessionType, 'reach')
                pattern = sprintf('criticality_sliding_window_av%s_win*_%s.mat', filenameSuffix, sessionName);
            else
                pattern = sprintf('criticality_sliding_window_av%s_win*.mat', filenameSuffix);
            end
            
        case 'criticality_lfp'
            if strcmp(sessionType, 'reach')
                pattern = sprintf('criticality_sliding_lfp_win*_%s.mat', sessionName);
            else
                pattern = sprintf('criticality_sliding_lfp_win*.mat');
            end
            
        case 'complexity'
            if ~isempty(dataSource)
                if strcmp(sessionType, 'reach')
                    pattern = sprintf('complexity_sliding_window_%s_win*_%s.mat', dataSource, sessionName);
                else
                    % For spontaneous, session name might be in filename
                    pattern = sprintf('complexity_sliding_window_%s_win*.mat', dataSource);
                end
            else
                if strcmp(sessionType, 'reach')
                    pattern = sprintf('complexity_sliding_window_win*_%s.mat', sessionName);
                else
                    pattern = sprintf('complexity_sliding_window_win*.mat');
                end
            end
            
        case 'rqa'
            if ~isempty(dataSource)
                if strcmp(sessionType, 'reach')
                    pattern = sprintf('rqa_sliding_window_%s_win*_%s.mat', dataSource, sessionName);
                else
                    pattern = sprintf('rqa_sliding_window_%s_win*.mat', dataSource);
                end
            else
                if strcmp(sessionType, 'reach')
                    pattern = sprintf('rqa_sliding_window_win*_%s.mat', sessionName);
                else
                    pattern = sprintf('rqa_sliding_window_win*.mat');
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
    
    files = dir(fullfile(saveDir, pattern));
    
    % For spontaneous sessions, also check subdirectories if sessionName has path separators
    if strcmp(sessionType, 'spontaneous') && contains(sessionName, filesep)
        pathParts = strsplit(sessionName, filesep);
        if length(pathParts) > 1
            subDir = fullfile(saveDir, pathParts{1});
            if exist(subDir, 'dir')
                subFiles = dir(fullfile(subDir, pattern));
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

