%%
% Sliding Window Analyses Across Sessions
% Compares metrics between spontaneous (spontaneous) and reach sessions
%
% Variables:
%   analysisType - Type of analysis: 'criticality_ar', 'lzc', 'rqa', 
%                  'criticality_av', 'criticality_lfp', 'participation_ratio' (default: 'criticality_ar')
%   metricName - Name of metric to plot (e.g., 'd2', 'lzComplexity', 'recurrenceRate')
%                 If empty, uses default for analysis type
%   useNormalized - Whether to use normalized metric if available (default: true)
%   prNormalizationType - For participation_ratio only: 'none' (raw PR),
%                        'shuffle' (shuffle-normalized), 'neurons' (PR/nNeurons).
%                        Default: 'shuffle'. Ignored for other analysis types.
%   filenameSuffix - Optional suffix for results files (e.g., '_pca')
%
% Note: The script automatically finds results files by matching session name
%       and analysis type, regardless of window size.
%
% Goal:
%   Load saved results from spontaneous and reach sessions, calculate median
%   metric values per session per area, and create bar plots comparing the two groups.

%% Configuration
analysisType = 'rqa';  % Options: 'criticality_ar', 'lzc', 'rqa', 'criticality_av', 'criticality_lfp', 'participation_ratio'
metricName = '';  % If empty, uses default for analysis type
useNormalized = true;  % Use normalized metric if available (for non-PR analyses)
prNormalizationType = 'neurons';  % For participation_ratio: 'none', 'shuffle', 'neurons'
filenameSuffix = '';  % Optional suffix (e.g., '_pca')
timeRange = [0, 1200];  % Time range in seconds to analyze [startTime, endTime]. Use [] to analyze all data.

natColor = [0 191 255] ./ 255;   % Color for spontaneous sessions
reachColor = [255 215 0] ./ 255; % Color for reach sessions

% Add paths
% Get paths structure
paths = get_paths;

basePath = fullfile(paths.homePath, 'neuro-behavior/src/');  % src
addpath(fullfile(basePath, 'sliding_window_prep', 'utils'));
addpath(fullfile(basePath, 'spontaneous'));
addpath(fullfile(basePath, 'reach_task'));


% Get session lists
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

% Determine default metric name if not provided
% For RQA, we plot multiple metrics (determinism, laminarity, trapping time)
% so metricName is not used for RQA
if strcmp(analysisType, 'rqa')
    metricName = '';  % Not used for RQA (we plot multiple metrics)
    fprintf('RQA analysis: Will plot determinism, laminarity, and trapping time\n');
elseif isempty(metricName)
    switch analysisType
        case 'criticality_ar'
            if useNormalized
                metricName = 'd2Normalized';
            else
                metricName = 'd2';
            end
        case 'lzc'
            if useNormalized
                metricName = 'lzComplexityNormalized';
            else
                metricName = 'lzComplexity';
            end
        case 'criticality_av'
            metricName = 'dcc';  % Default to dcc for avalanche analysis
        case 'criticality_lfp'
            if useNormalized
                metricName = 'd2Normalized';
            else
                metricName = 'd2';
            end
        case 'participation_ratio'
            switch prNormalizationType
                case 'shuffle'
                    metricName = 'participationRatioNormalized';
                case {'none', 'neurons'}
                    metricName = 'participationRatio';
                otherwise
                    error('Unknown prNormalizationType: %s. Use ''none'', ''shuffle'', or ''neurons''.', prNormalizationType);
            end
        otherwise
            error('Unknown analysis type: %s', analysisType);
    end
    fprintf('Metric: %s\n', metricName);
    if strcmp(analysisType, 'participation_ratio')
        fprintf('PR normalization: %s\n', prNormalizationType);
    end
end

% Display name for plot labels (PR uses human-readable names)
if strcmp(analysisType, 'participation_ratio')
    switch prNormalizationType
        case 'none'
            metricDisplayName = 'PR (raw)';
        case 'shuffle'
            metricDisplayName = 'PR (shuffle-norm)';
        case 'neurons'
            metricDisplayName = 'PR/nNeurons';
        otherwise
            metricDisplayName = metricName;
    end
else
    metricDisplayName = metricName;
end

% Suffix for labels when using normalized values (PR display names already indicate normalization)
isNormalizedForPlot = (strcmp(analysisType, 'rqa') && useNormalized) || ...
    (ismember(analysisType, {'criticality_ar', 'lzc', 'criticality_lfp'}) && useNormalized);
if isNormalizedForPlot
    normalizedLabelSuffix = ' (normalized)';
else
    normalizedLabelSuffix = '';
end

% Load results for spontaneous sessions
fprintf('\n=== Loading Spontaneous Session Results ===\n');
spontaneousData = struct();
spontaneousData.medians = {};  % Cell array: {areaIdx}{sessionIdx} = median value
if strcmp(analysisType, 'rqa')
    % For RQA, store three metrics separately
    spontaneousData.detMedians = {};  % Determinism medians
    spontaneousData.lamMedians = {};  % Laminarity medians
    spontaneousData.ttMedians = {};  % Trapping time medians
end
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
    if strcmp(analysisType, 'lzc') || strcmp(analysisType, 'rqa') || strcmp(analysisType, 'participation_ratio')
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
        if strcmp(analysisType, 'rqa')
            spontaneousData.detMedians = cell(1, numAreas);
            spontaneousData.lamMedians = cell(1, numAreas);
            spontaneousData.ttMedians = cell(1, numAreas);
        end
        for a = 1:numAreas
            spontaneousData.medians{a} = [];
            if strcmp(analysisType, 'rqa')
                spontaneousData.detMedians{a} = [];
                spontaneousData.lamMedians{a} = [];
                spontaneousData.ttMedians{a} = [];
            end
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
                if strcmp(analysisType, 'rqa')
                    % Filter RQA metrics
                    metricNames = {'determinismNormalized', 'laminarityNormalized', 'trappingTimeNormalized'};
                    if useNormalized
                        metricNames = {'determinismNormalized', 'laminarityNormalized', 'trappingTimeNormalized'};
                    else
                        metricNames = {'determinism', 'laminarity', 'trappingTime'};
                    end
                    for m = 1:length(metricNames)
                        if isfield(results, metricNames{m}) && iscell(results.(metricNames{m})) && ...
                                a <= length(results.(metricNames{m})) && ~isempty(results.(metricNames{m}){a})
                            results.(metricNames{m}){a} = results.(metricNames{m}){a}(timeMask);
                        end
                    end
                elseif ~isempty(metricName) && isfield(results, metricName) && iscell(results.(metricName)) && ...
                        a <= length(results.(metricName)) && ~isempty(results.(metricName){a})
                    results.(metricName){a} = results.(metricName){a}(timeMask);
                end
                % Filter participationRatioOverNeurons when using PR/nNeurons
                if strcmp(analysisType, 'participation_ratio') && strcmp(prNormalizationType, 'neurons') && ...
                        isfield(results, 'participationRatioOverNeurons') && a <= length(results.participationRatioOverNeurons) && ...
                        ~isempty(results.participationRatioOverNeurons{a})
                    results.participationRatioOverNeurons{a} = results.participationRatioOverNeurons{a}(timeMask);
                end
            end
        end
    end
    
    % Extract metric values per area
    if strcmp(analysisType, 'rqa')
        % For RQA, extract three metrics
        if useNormalized
            metricNames = {'determinismNormalized', 'laminarityNormalized', 'trappingTimeNormalized'};
            dataFields = {'detMedians', 'lamMedians', 'ttMedians'};
        else
            metricNames = {'determinism', 'laminarity', 'trappingTime'};
            dataFields = {'detMedians', 'lamMedians', 'ttMedians'};
        end
        
        for m = 1:length(metricNames)
            metricNameRQA = metricNames{m};
            dataField = dataFields{m};
            if isfield(results, metricNameRQA)
                metricData = results.(metricNameRQA);
                if iscell(metricData)
                    for a = 1:length(metricData)
                        if a <= length(spontaneousData.(dataField)) && ~isempty(metricData{a})
                            values = metricData{a}(:);
                            values = values(~isnan(values));
                            if ~isempty(values)
                                medianVal = median(values);
                                spontaneousData.(dataField){a} = [spontaneousData.(dataField){a}, medianVal];
                            else
                                spontaneousData.(dataField){a} = [spontaneousData.(dataField){a}, nan];
                            end
                        end
                    end
                end
            else
                warning('Metric %s not found in results for session %s', metricNameRQA, sessionName);
                % Add NaN for all areas
                for a = 1:length(spontaneousData.(dataField))
                    spontaneousData.(dataField){a} = [spontaneousData.(dataField){a}, nan];
                end
            end
        end
    elseif strcmp(analysisType, 'participation_ratio') && strcmp(prNormalizationType, 'neurons')
        % PR normalized by neuron count: use saved participationRatioOverNeurons or compute from PR/idMatIdx
        useSaved = isfield(results, 'participationRatioOverNeurons') && ...
            any(cellfun(@(x) ~isempty(x) && any(~isnan(x)), results.participationRatioOverNeurons));
        if useSaved
            metricData = results.participationRatioOverNeurons;
            for a = 1:length(metricData)
                if a <= length(spontaneousData.medians)
                    if ~isempty(metricData{a})
                        values = metricData{a}(:);
                        values = values(~isnan(values));
                        if ~isempty(values)
                            medianVal = median(values);
                            spontaneousData.medians{a} = [spontaneousData.medians{a}, medianVal];
                        else
                            spontaneousData.medians{a} = [spontaneousData.medians{a}, nan];
                        end
                    else
                        spontaneousData.medians{a} = [spontaneousData.medians{a}, nan];
                    end
                end
            end
        elseif isfield(results, 'participationRatio') && isfield(results, 'idMatIdx')
            for a = 1:length(results.participationRatio)
                if a <= length(spontaneousData.medians) && ~isempty(results.participationRatio{a})
                    nNeurons = length(results.idMatIdx{a});
                    if nNeurons > 0
                        values = results.participationRatio{a}(:) / nNeurons;
                        values = values(~isnan(values));
                        if ~isempty(values)
                            medianVal = median(values);
                            spontaneousData.medians{a} = [spontaneousData.medians{a}, medianVal];
                        else
                            spontaneousData.medians{a} = [spontaneousData.medians{a}, nan];
                        end
                    else
                        spontaneousData.medians{a} = [spontaneousData.medians{a}, nan];
                    end
                end
            end
        else
            warning('participationRatioOverNeurons or (participationRatio and idMatIdx) not found for session %s', sessionName);
            for a = 1:length(spontaneousData.medians)
                spontaneousData.medians{a} = [spontaneousData.medians{a}, nan];
            end
        end
    elseif ~isempty(metricName) && isfield(results, metricName)
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
    elseif ~isempty(metricName)
        warning('Metric %s not found in results for session %s', metricName, sessionName);
        % Add NaN for all areas
        for a = 1:length(spontaneousData.medians)
            spontaneousData.medians{a} = [spontaneousData.medians{a}, nan];
        end
    end
    
    spontaneousData.sessionNames{end+1} = sessionName;
end

% Load results for reach sessions
fprintf('\n=== Loading Reach Session Results ===\n');
reachData = struct();
reachData.medians = {};  % Cell array: {areaIdx}{sessionIdx} = median value
if strcmp(analysisType, 'rqa')
    % For RQA, store three metrics separately
    reachData.detMedians = {};  % Determinism medians
    reachData.lamMedians = {};  % Laminarity medians
    reachData.ttMedians = {};  % Trapping time medians
end
reachData.sessionNames = {};
reachData.areas = [];

for s = 1:length(reachSessions)
    sessionName = reachSessions{s};
    fprintf('Loading session %d/%d: %s\n', s, length(reachSessions), sessionName);
    
    % Find results file by pattern matching
    [~, dataBaseName, ~] = fileparts(sessionName);
    saveDir = fullfile(paths.dropPath, 'reach_task/results', dataBaseName);
    
    % Determine dataSource for pattern matching
    if strcmp(analysisType, 'lzc') || strcmp(analysisType, 'rqa') || strcmp(analysisType, 'participation_ratio')
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
        if strcmp(analysisType, 'rqa')
            reachData.detMedians = cell(1, numAreas);
            reachData.lamMedians = cell(1, numAreas);
            reachData.ttMedians = cell(1, numAreas);
        end
        for a = 1:numAreas
            reachData.medians{a} = [];
            if strcmp(analysisType, 'rqa')
                reachData.detMedians{a} = [];
                reachData.lamMedians{a} = [];
                reachData.ttMedians{a} = [];
            end
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
                if strcmp(analysisType, 'rqa')
                    % Filter RQA metrics
                    if useNormalized
                        metricNames = {'determinismNormalized', 'laminarityNormalized', 'trappingTimeNormalized'};
                    else
                        metricNames = {'determinism', 'laminarity', 'trappingTime'};
                    end
                    for m = 1:length(metricNames)
                        if isfield(results, metricNames{m}) && iscell(results.(metricNames{m})) && ...
                                a <= length(results.(metricNames{m})) && ~isempty(results.(metricNames{m}){a})
                            results.(metricNames{m}){a} = results.(metricNames{m}){a}(timeMask);
                        end
                    end
                elseif ~isempty(metricName) && isfield(results, metricName) && iscell(results.(metricName)) && ...
                        a <= length(results.(metricName)) && ~isempty(results.(metricName){a})
                    results.(metricName){a} = results.(metricName){a}(timeMask);
                end
                % Filter participationRatioOverNeurons when using PR/nNeurons
                if strcmp(analysisType, 'participation_ratio') && strcmp(prNormalizationType, 'neurons') && ...
                        isfield(results, 'participationRatioOverNeurons') && a <= length(results.participationRatioOverNeurons) && ...
                        ~isempty(results.participationRatioOverNeurons{a})
                    results.participationRatioOverNeurons{a} = results.participationRatioOverNeurons{a}(timeMask);
                end
            end
        end
    end
    
    % Extract metric values per area
    if strcmp(analysisType, 'rqa')
        % For RQA, extract three metrics
        if useNormalized
            metricNames = {'determinismNormalized', 'laminarityNormalized', 'trappingTimeNormalized'};
            dataFields = {'detMedians', 'lamMedians', 'ttMedians'};
        else
            metricNames = {'determinism', 'laminarity', 'trappingTime'};
            dataFields = {'detMedians', 'lamMedians', 'ttMedians'};
        end
        
        for m = 1:length(metricNames)
            metricNameRQA = metricNames{m};
            dataField = dataFields{m};
            if isfield(results, metricNameRQA)
                metricData = results.(metricNameRQA);
                if iscell(metricData)
                    for a = 1:length(metricData)
                        if a <= length(reachData.(dataField)) && ~isempty(metricData{a})
                            values = metricData{a}(:);
                            values = values(~isnan(values));
                            if ~isempty(values)
                                medianVal = median(values);
                                reachData.(dataField){a} = [reachData.(dataField){a}, medianVal];
                            else
                                reachData.(dataField){a} = [reachData.(dataField){a}, nan];
                            end
                        end
                    end
                end
            else
                warning('Metric %s not found in results for session %s', metricNameRQA, sessionName);
                % Add NaN for all areas
                for a = 1:length(reachData.(dataField))
                    reachData.(dataField){a} = [reachData.(dataField){a}, nan];
                end
            end
        end
    elseif strcmp(analysisType, 'participation_ratio') && strcmp(prNormalizationType, 'neurons')
        % PR normalized by neuron count: use saved participationRatioOverNeurons or compute from PR/idMatIdx
        useSaved = isfield(results, 'participationRatioOverNeurons') && ...
            any(cellfun(@(x) ~isempty(x) && any(~isnan(x)), results.participationRatioOverNeurons));
        if useSaved
            metricData = results.participationRatioOverNeurons;
            for a = 1:length(metricData)
                if a <= length(reachData.medians)
                    if ~isempty(metricData{a})
                        values = metricData{a}(:);
                        values = values(~isnan(values));
                        if ~isempty(values)
                            medianVal = median(values);
                            reachData.medians{a} = [reachData.medians{a}, medianVal];
                        else
                            reachData.medians{a} = [reachData.medians{a}, nan];
                        end
                    else
                        reachData.medians{a} = [reachData.medians{a}, nan];
                    end
                end
            end
        elseif isfield(results, 'participationRatio') && isfield(results, 'idMatIdx')
            for a = 1:length(results.participationRatio)
                if a <= length(reachData.medians) && ~isempty(results.participationRatio{a})
                    nNeurons = length(results.idMatIdx{a});
                    if nNeurons > 0
                        values = results.participationRatio{a}(:) / nNeurons;
                        values = values(~isnan(values));
                        if ~isempty(values)
                            medianVal = median(values);
                            reachData.medians{a} = [reachData.medians{a}, medianVal];
                        else
                            reachData.medians{a} = [reachData.medians{a}, nan];
                        end
                    else
                        reachData.medians{a} = [reachData.medians{a}, nan];
                    end
                end
            end
        else
            warning('participationRatioOverNeurons or (participationRatio and idMatIdx) not found for session %s', sessionName);
            for a = 1:length(reachData.medians)
                reachData.medians{a} = [reachData.medians{a}, nan];
            end
        end
    elseif ~isempty(metricName) && isfield(results, metricName)
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
    elseif ~isempty(metricName)
        warning('Metric %s not found in results for session %s', metricName, sessionName);
        % Add NaN for all areas
        for a = 1:length(reachData.medians)
            reachData.medians{a} = [reachData.medians{a}, nan];
        end
    end
    
    reachData.sessionNames{end+1} = sessionName;
end

% Determine common areas
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

% Determine number of columns based on analysis type
if strcmp(analysisType, 'rqa')
    numCols = 3;  % Determinism, Laminarity, Trapping Time
else
    numCols = 1;  % Single metric
end

% Calculate global y-axis limits for each metric (common across areas)
bufferPercent = 0.05;  % 5% buffer on each side

if strcmp(analysisType, 'rqa')
    % For RQA, collect values for each of the 3 metrics
    metricDataFields = {'detMedians', 'lamMedians', 'ttMedians'};
    allMetricValues = cell(1, 3);  % One cell per metric
    
    for col = 1:3
        allValues = [];
        for a = 1:numAreasToPlot
            areaName = areasToPlot{a};
            natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
            reachAreaIdx = find(strcmp(reachData.areas, areaName));
            
            if ~isempty(natAreaIdx) && ~isempty(reachAreaIdx)
                natVals = spontaneousData.(metricDataFields{col}){natAreaIdx};
                reachVals = reachData.(metricDataFields{col}){reachAreaIdx};
                natVals = natVals(~isnan(natVals));
                reachVals = reachVals(~isnan(reachVals));
                allValues = [allValues, natVals, reachVals];
            end
        end
        allMetricValues{col} = allValues;
    end
    
    % Calculate y-axis limits for each metric
    ylims = cell(1, 3);
    for col = 1:3
        if ~isempty(allMetricValues{col})
            minVal = min(allMetricValues{col});
            maxVal = max(allMetricValues{col});
            rangeVal = maxVal - minVal;
            if rangeVal == 0
                rangeVal = 0.1;  % Default range if all values are the same
            end
            buffer = rangeVal * bufferPercent;
            ylims{col} = [minVal - buffer, maxVal + buffer];
            % Ensure lower limit doesn't go below 0 for normalized metrics
            if useNormalized
                ylims{col}(1) = max(0, ylims{col}(1));
            end
        else
            ylims{col} = [0, 2];  % Default if no data
        end
    end
else
    % For other analysis types, collect values for the single metric
    allValues = [];
    for a = 1:numAreasToPlot
        areaName = areasToPlot{a};
        natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
        reachAreaIdx = find(strcmp(reachData.areas, areaName));
        
        if ~isempty(natAreaIdx) && ~isempty(reachAreaIdx)
            natVals = spontaneousData.medians{natAreaIdx};
            reachVals = reachData.medians{reachAreaIdx};
            natVals = natVals(~isnan(natVals));
            reachVals = reachVals(~isnan(reachVals));
            allValues = [allValues, natVals, reachVals];
        end
    end
    
    % Calculate y-axis limits
    if ~isempty(allValues)
        minVal = min(allValues);
        maxVal = max(allValues);
        rangeVal = maxVal - minVal;
        if rangeVal == 0
            rangeVal = 0.1;  % Default range if all values are the same
        end
        buffer = rangeVal * bufferPercent;
        ylimSingle = [minVal - buffer, maxVal + buffer];
        % Ensure lower limit doesn't go below 0 for normalized metrics
        if useNormalized || (strcmp(analysisType, 'participation_ratio') && ...
                (strcmp(prNormalizationType, 'neurons') || strcmp(prNormalizationType, 'shuffle')))
            ylimSingle(1) = max(0, ylimSingle(1));
        end
    else
        ylimSingle = [0, 2];  % Default if no data
    end
end

% Create figure
figure(2000); clf;
set(gcf, 'Units', 'pixels');
if strcmp(analysisType, 'rqa')
    set(gcf, 'Position', [100, 100, 1800, 300 * numAreasToPlot]);
else
    set(gcf, 'Position', [100, 100, 1200, 300 * numAreasToPlot]);
end

% Create subplots
for a = 1:numAreasToPlot
    areaName = areasToPlot{a};
    
    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));
    
    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end
    
    if strcmp(analysisType, 'rqa')
        % For RQA, plot three metrics in separate columns
        metricDataFields = {'detMedians', 'lamMedians', 'ttMedians'};
        metricLabels = {'Determinism', 'Laminarity', 'Trapping Time'};
        metricShortLabels = {'DET', 'LAM', 'TT'};
        
        for col = 1:numCols
            % Get median values for this area and metric
            natMedians = spontaneousData.(metricDataFields{col}){natAreaIdx};
            reachMedians = reachData.(metricDataFields{col}){reachAreaIdx};
            
            % Remove NaN values
            natMedians = natMedians(~isnan(natMedians));
            reachMedians = reachMedians(~isnan(reachMedians));
            
            % Create subplot
            subplot(numAreasToPlot, numCols, (a-1)*numCols + col);
            hold on;
            
            % Create bar plot
            numNat = length(natMedians);
            numReach = length(reachMedians);
            
            if isempty(natMedians) && isempty(reachMedians)
                continue;
            end
            
            % Plot individual bars for each session
            xNat = 1:numNat;
            xReach = (numNat + 1):(numNat + numReach);
            
            % Plot bars with spacing between groups
            if numNat > 0
                bar(xNat, natMedians, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
            end
            if numReach > 0
                bar(xReach, reachMedians, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
            
            ylabel(sprintf('%s%s (median)', metricShortLabels{col}, normalizedLabelSuffix));
            if col == 1
                title(sprintf('%s - %s%s', areaName, metricLabels{col}, normalizedLabelSuffix));
            else
                title(sprintf('%s%s', metricLabels{col}, normalizedLabelSuffix));
            end
            grid on;
            
            % Apply global y-axis limits for this metric
            ylim(ylims{col});
            
            % Add mean lines using natColor and reachColor
            if ~isempty(natMedians)
                yline(mean(natMedians), '--', 'Color', natColor, 'LineWidth', 2, 'DisplayName', sprintf('Nat mean: %.3f', mean(natMedians)));
            end
            if ~isempty(reachMedians)
                yline(mean(reachMedians), '--', 'Color', reachColor, 'LineWidth', 2, 'DisplayName', sprintf('Reach mean: %.3f', mean(reachMedians)));
            end
            hold off;
        end
    else
        % For other analysis types, plot single metric
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
        
        % Create bar plot
        numNat = length(natMedians);
        numReach = length(reachMedians);
        
        % Plot individual bars for each session
        xNat = 1:numNat;
        xReach = (numNat + 1):(numNat + numReach);
        
        % Plot bars with spacing between groups
        if numNat > 0
            bar(xNat, natMedians, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
        end
        if numReach > 0
            bar(xReach, reachMedians, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
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
        
        ylabel(sprintf('%s%s (median)', metricDisplayName, normalizedLabelSuffix));
        title(sprintf('%s - %s%s', areaName, metricDisplayName, normalizedLabelSuffix));
        grid on;
        
        % Apply global y-axis limits
        ylim(ylimSingle);
        
        % Add mean lines
        if ~isempty(natMedians)
            yline(mean(natMedians), '--', 'LineWidth', 2, 'Color', natColor, 'DisplayName', sprintf('Nat mean: %.3f', mean(natMedians)));
        end
        if ~isempty(reachMedians)
            yline(mean(reachMedians), '--', 'LineWidth', 2, 'Color', reachColor, 'DisplayName', sprintf('Reach mean: %.3f', mean(reachMedians)));
        end
        
        hold off;
    end
end

% Add overall title
if strcmp(analysisType, 'participation_ratio')
    sgtitle(sprintf('%s (%s%s) Comparison: Spontaneous vs Reach', ...
        analysisType, metricDisplayName, normalizedLabelSuffix), 'FontSize', 14, 'FontWeight', 'bold', 'interpreter', 'none');
else
    sgtitle(sprintf('%s%s Comparison: Spontaneous vs Reach', ...
        analysisType, normalizedLabelSuffix), 'FontSize', 14, 'FontWeight', 'bold', 'interpreter', 'none');
end

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

% Generate filename
if strcmp(analysisType, 'rqa')
    plotFilenamePng = sprintf('%s_det_lam_tt_nat_vs_reach%s%s.png', ...
        analysisType, filenameSuffix, timeRangeStr);
    plotFilenameEps = sprintf('%s_det_lam_tt_nat_vs_reach%s%s.eps', ...
        analysisType, filenameSuffix, timeRangeStr);
elseif strcmp(analysisType, 'participation_ratio')
    plotFilenamePng = sprintf('%s_%s_nat_vs_reach%s%s.png', ...
        analysisType, prNormalizationType, filenameSuffix, timeRangeStr);
    plotFilenameEps = sprintf('%s_%s_nat_vs_reach%s%s.eps', ...
        analysisType, prNormalizationType, filenameSuffix, timeRangeStr);
else
    plotFilenamePng = sprintf('%s_%s_nat_vs_reach%s%s.png', ...
        analysisType, metricName, filenameSuffix, timeRangeStr);
    plotFilenameEps = sprintf('%s_%s_nat_vs_reach%s%s.eps', ...
        analysisType, metricName, filenameSuffix, timeRangeStr);
end
plotPathPng = fullfile(saveDir, plotFilenamePng);
plotPathEps = fullfile(saveDir, plotFilenameEps);

% Save PNG
exportgraphics(gcf, plotPathPng, 'Resolution', 300);
fprintf('\nSaved PNG plot to: %s\n', plotPathPng);

% Save EPS
exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
fprintf('Saved EPS plot to: %s\n', plotPathEps);

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
            
        case 'lzc'
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
            
        case 'participation_ratio'
            % Format: participation_ratio_sliding_window_{sessionName}.mat
            if ~isempty(sessionName)
                pattern = sprintf('participation_ratio_sliding_window_%s.mat', sessionName);
            else
                pattern = 'participation_ratio_sliding_window_*.mat';
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

