%%
% RQA_SINGLE_PER_SESSION (Single Large Window)
% Combines Recurrence Quantitative Analysis (RQA) metrics in a single large window
% Compares metrics between spontaneous and reach sessions.
%
% Variables:
%   timeRange - Time range in seconds [startTime, endTime] (default: [0, 20*60])
%   areasToPlot - Cell array of area names to plot (default: {} = all common areas)
%                 Example: {'M23', 'M56'} or {} for all areas
%   natColor - Color for spontaneous sessions (default: [0 191 255]/255)
%   reachColor - Color for reach sessions (default: [255 215 0]/255)
%
% Goal:
%   Load data for each session, analyze a single large window, and compare
%   RQA metrics (recurrence rate, determinism) between spontaneous and reach sessions.

%% Configuration
paths = get_paths;

timeRange = [0*60, 15*60];  % Time range in seconds [startTime, endTime]
areasToPlot = {};        % Cell array of area names to plot (empty = all common areas)
% Example: {'M23', 'M56'} or {} for all areas
natColor = [0 191 255] ./ 255;   % Color for spontaneous sessions
reachColor = [255 215 0] ./ 255; % Color for reach sessions

% Calculate window parameters from timeRange
windowStartTime = timeRange(1);
windowEndTime = timeRange(2);
windowDuration = windowEndTime - windowStartTime;  % Window duration in seconds
nMin = windowDuration / 60;  % Window size in minutes (for display purposes)

%% Get session lists
fprintf('\n=== RQA Analysis Across Sessions (Single Window) ===\n');
fprintf('Time range: [%.1f, %.1f] seconds (%.1f minutes)\n', ...
    windowStartTime, windowEndTime, nMin);

spontaneousSessions = spontaneous_session_list();
reachSessions = reach_session_list();

fprintf('Spontaneous sessions: %d\n', length(spontaneousSessions));
fprintf('Reach sessions: %d\n', length(reachSessions));

%% Configure analysis options
opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = windowStartTime;
opts.collectEnd = windowEndTime;  % Restrict data to timeRange
opts.minFiringRate = .1;
opts.maxFiringRate = 100;

% RQA Analysis config - use single large window
config = struct();
config.slidingWindowSize = windowDuration;     % Window size in seconds
config.stepSize = windowDuration + 1;          % Larger than window size so only one window
config.nShuffles = 3;                          % Number of shuffles for normalization
config.binSize = 0.03;                         % Bin size for spikes (seconds)
config.minTimeBins = 10000;                    % Minimum bins per window (used in defaults)
config.nPCADim = 4;                            % Number of PCA dimensions
config.recurrenceThreshold = 0.03;             % Target recurrence rate (2%)
config.distanceMetric = 'euclidean';           % Distance metric
config.nMinNeurons = 10;                       % Minimum neurons per area
config.makePlots = false;                      % Don't create per-session plots
config.useBernoulliControl = true;            % Compute Bernoulli normalized metric
config.saveRecurrencePlots = false;           % Don't store recurrence plots
config.includeM2356 = true;                   % Include combined M23+M56 area
config.useOptimalBinSize = false;             % We provide binSize explicitly
config.saveData = false;                      % Don't save individual session results
config.usePerWindowPCA = false;               % PCA on full session

%% Initialize results storage
spontaneousData = struct();
spontaneousData.rrRaw = {};              % Recurrence rate raw
spontaneousData.rrNormalized = {};       % Recurrence rate normalized (shuffled)
spontaneousData.detRaw = {};             % Determinism raw
spontaneousData.detNormalized = {};      % Determinism normalized (shuffled)
spontaneousData.lamRaw = {};             % Laminarity raw
spontaneousData.lamNormalized = {};      % Laminarity normalized (shuffled)
spontaneousData.ttRaw = {};              % Trapping time raw
spontaneousData.ttNormalized = {};       % Trapping time normalized (shuffled)
spontaneousData.sessionNames = {};
spontaneousData.areas = [];

reachData = struct();
reachData.rrRaw = {};
reachData.rrNormalized = {};
reachData.detRaw = {};
reachData.detNormalized = {};
reachData.lamRaw = {};
reachData.lamNormalized = {};
reachData.ttRaw = {};
reachData.ttNormalized = {};
reachData.sessionNames = {};
reachData.areas = [];

%% Set up parallel pool if running parallel in lzc_sliding_analysis
runParallel = 1;
% Check if parpool is already running, start one if not
if runParallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        % NumWorkers = min(4, length(dataStruct.areas));
        NumWorkers = 3;
        parpool('local', NumWorkers);
        fprintf('Started parallel pool with %d workers\n', NumWorkers);
    else
        fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
    end
end
% ===========================================================================================
%% Process spontaneous sessions
tic;
fprintf('\n=== Processing Spontaneous Sessions ===\n');
% Pre-allocate results cell array for parfor compatibility
spontaneousResults = cell(1, length(spontaneousSessions));

for s = 1:length(spontaneousSessions)
    sessionName = spontaneousSessions{s};
    fprintf('\nProcessing session %d/%d: %s\n', s, length(spontaneousSessions), sessionName);

    % Initialize session result structure
    sessionResult = struct();
    sessionResult.sessionName = sessionName;
    sessionResult.success = false;
    sessionResult.areas = {};
    sessionResult.rrRaw = {};
    sessionResult.rrNormalized = {};
    sessionResult.detRaw = {};
    sessionResult.detNormalized = {};
    % Laminarity and trapping time
    sessionResult.lamRaw = {};
    sessionResult.lamNormalized = {};
    sessionResult.ttRaw = {};
    sessionResult.ttNormalized = {};
    sessionResult.lamRaw = {};
    sessionResult.lamNormalized = {};
    sessionResult.ttRaw = {};
    sessionResult.ttNormalized = {};

    try
        % Load data
        dataStruct = load_sliding_window_data('spontaneous', 'spikes', ...
            'sessionName', sessionName, 'opts', opts);

        % Run RQA analysis
        fprintf('  Running RQA analysis...\n');
        results = rqa_sliding_analysis(dataStruct, config);

        % Extract areas from results
        if isfield(results, 'areas')
            sessionResult.areas = results.areas;
            numAreas = length(results.areas);

            % Initialize metric arrays
            sessionResult.rrRaw = cell(1, numAreas);
            sessionResult.rrNormalized = cell(1, numAreas);
            sessionResult.detRaw = cell(1, numAreas);
            sessionResult.detNormalized = cell(1, numAreas);
            sessionResult.lamRaw = cell(1, numAreas);
            sessionResult.lamNormalized = cell(1, numAreas);
            sessionResult.ttRaw = cell(1, numAreas);
            sessionResult.ttNormalized = cell(1, numAreas);
            sessionResult.lamRaw = cell(1, numAreas);
            sessionResult.lamNormalized = cell(1, numAreas);
            sessionResult.ttRaw = cell(1, numAreas);
            sessionResult.ttNormalized = cell(1, numAreas);

            % Extract metrics per area (single value per area)
            if isfield(results, 'recurrenceRate') && isfield(results, 'determinism') && ...
                    isfield(results, 'laminarity') && isfield(results, 'trappingTime')
                for a = 1:numAreas
                    % Recurrence rate raw
                    if a <= length(results.recurrenceRate) && ~isempty(results.recurrenceRate{a})
                        rrValues = results.recurrenceRate{a}(~isnan(results.recurrenceRate{a}));
                        if ~isempty(rrValues)
                            sessionResult.rrRaw{a} = rrValues(1);
                        else
                            sessionResult.rrRaw{a} = nan;
                        end
                    else
                        sessionResult.rrRaw{a} = nan;
                    end

                    % Recurrence rate normalized
                    if isfield(results, 'recurrenceRateNormalized') && ...
                            a <= length(results.recurrenceRateNormalized) && ...
                            ~isempty(results.recurrenceRateNormalized{a})
                        rrNormValues = results.recurrenceRateNormalized{a}(~isnan(results.recurrenceRateNormalized{a}));
                        if ~isempty(rrNormValues)
                            sessionResult.rrNormalized{a} = rrNormValues(1);
                        else
                            sessionResult.rrNormalized{a} = nan;
                        end
                    else
                        sessionResult.rrNormalized{a} = nan;
                    end

                    % Determinism raw
                    if a <= length(results.determinism) && ~isempty(results.determinism{a})
                        detValues = results.determinism{a}(~isnan(results.determinism{a}));
                        if ~isempty(detValues)
                            sessionResult.detRaw{a} = detValues(1);
                        else
                            sessionResult.detRaw{a} = nan;
                        end
                    else
                        sessionResult.detRaw{a} = nan;
                    end

                    % Determinism normalized
                    if isfield(results, 'determinismNormalized') && ...
                            a <= length(results.determinismNormalized) && ...
                            ~isempty(results.determinismNormalized{a})
                        detNormValues = results.determinismNormalized{a}(~isnan(results.determinismNormalized{a}));
                        if ~isempty(detNormValues)
                            sessionResult.detNormalized{a} = detNormValues(1);
                        else
                            sessionResult.detNormalized{a} = nan;
                        end
                    else
                        sessionResult.detNormalized{a} = nan;
                    end

                    % Laminarity raw
                    if a <= length(results.laminarity) && ~isempty(results.laminarity{a})
                        lamValues = results.laminarity{a}(~isnan(results.laminarity{a}));
                        if ~isempty(lamValues)
                            sessionResult.lamRaw{a} = lamValues(1);
                        else
                            sessionResult.lamRaw{a} = nan;
                        end
                    else
                        sessionResult.lamRaw{a} = nan;
                    end

                    % Laminarity normalized
                    if isfield(results, 'laminarityNormalized') && ...
                            a <= length(results.laminarityNormalized) && ...
                            ~isempty(results.laminarityNormalized{a})
                        lamNormValues = results.laminarityNormalized{a}(~isnan(results.laminarityNormalized{a}));
                        if ~isempty(lamNormValues)
                            sessionResult.lamNormalized{a} = lamNormValues(1);
                        else
                            sessionResult.lamNormalized{a} = nan;
                        end
                    else
                        sessionResult.lamNormalized{a} = nan;
                    end

                    % Trapping time raw
                    if a <= length(results.trappingTime) && ~isempty(results.trappingTime{a})
                        ttValues = results.trappingTime{a}(~isnan(results.trappingTime{a}));
                        if ~isempty(ttValues)
                            sessionResult.ttRaw{a} = ttValues(1);
                        else
                            sessionResult.ttRaw{a} = nan;
                        end
                    else
                        sessionResult.ttRaw{a} = nan;
                    end

                    % Trapping time normalized
                    if isfield(results, 'trappingTimeNormalized') && ...
                            a <= length(results.trappingTimeNormalized) && ...
                            ~isempty(results.trappingTimeNormalized{a})
                        ttNormValues = results.trappingTimeNormalized{a}(~isnan(results.trappingTimeNormalized{a}));
                        if ~isempty(ttNormValues)
                            sessionResult.ttNormalized{a} = ttNormValues(1);
                        else
                            sessionResult.ttNormalized{a} = nan;
                        end
                    else
                        sessionResult.ttNormalized{a} = nan;
                    end
                end
            else
                warning('RQA metrics not found in results for session %s', sessionName);
                for a = 1:numAreas
                    sessionResult.rrRaw{a} = nan;
                    sessionResult.rrNormalized{a} = nan;
                    sessionResult.detRaw{a} = nan;
                    sessionResult.detNormalized{a} = nan;
                    sessionResult.lamRaw{a} = nan;
                    sessionResult.lamNormalized{a} = nan;
                    sessionResult.ttRaw{a} = nan;
                    sessionResult.ttNormalized{a} = nan;
                end
            end

            sessionResult.success = true;
            fprintf('  ✓ Session completed successfully\n');
        else
            warning('Areas not found in results for session %s', sessionName);
        end

    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        sessionResult.success = false;
    end

    % Store result in cell array (parfor-compatible)
    spontaneousResults{s} = sessionResult;
end

% Aggregate results from all sessions
fprintf('\n=== Aggregating Spontaneous Session Results ===\n');
for s = 1:length(spontaneousResults)
    sessionResult = spontaneousResults{s};

    if isempty(sessionResult.areas)
        continue;
    end

    % Initialize areas on first valid session
    if isempty(spontaneousData.areas) && ~isempty(sessionResult.areas)
        spontaneousData.areas = sessionResult.areas;
        numAreas = length(spontaneousData.areas);
        spontaneousData.rrRaw = cell(1, numAreas);
        spontaneousData.rrNormalized = cell(1, numAreas);
        spontaneousData.detRaw = cell(1, numAreas);
        spontaneousData.detNormalized = cell(1, numAreas);
        spontaneousData.lamRaw = cell(1, numAreas);
        spontaneousData.lamNormalized = cell(1, numAreas);
        spontaneousData.ttRaw = cell(1, numAreas);
        spontaneousData.ttNormalized = cell(1, numAreas);
        for a = 1:numAreas
            spontaneousData.rrRaw{a} = [];
            spontaneousData.rrNormalized{a} = [];
            spontaneousData.detRaw{a} = [];
            spontaneousData.detNormalized{a} = [];
            spontaneousData.lamRaw{a} = [];
            spontaneousData.lamNormalized{a} = [];
            spontaneousData.ttRaw{a} = [];
            spontaneousData.ttNormalized{a} = [];
        end
    end

    % Append metrics for this session
    if ~isempty(spontaneousData.areas) && length(sessionResult.areas) == length(spontaneousData.areas)
        for a = 1:length(spontaneousData.areas)
            if a <= length(sessionResult.rrRaw) && ~isempty(sessionResult.rrRaw{a})
                spontaneousData.rrRaw{a} = [spontaneousData.rrRaw{a}, sessionResult.rrRaw{a}];
            else
                spontaneousData.rrRaw{a} = [spontaneousData.rrRaw{a}, nan];
            end

            if a <= length(sessionResult.rrNormalized) && ~isempty(sessionResult.rrNormalized{a})
                spontaneousData.rrNormalized{a} = [spontaneousData.rrNormalized{a}, sessionResult.rrNormalized{a}];
            else
                spontaneousData.rrNormalized{a} = [spontaneousData.rrNormalized{a}, nan];
            end

            if a <= length(sessionResult.detRaw) && ~isempty(sessionResult.detRaw{a})
                spontaneousData.detRaw{a} = [spontaneousData.detRaw{a}, sessionResult.detRaw{a}];
            else
                spontaneousData.detRaw{a} = [spontaneousData.detRaw{a}, nan];
            end

            if a <= length(sessionResult.detNormalized) && ~isempty(sessionResult.detNormalized{a})
                spontaneousData.detNormalized{a} = [spontaneousData.detNormalized{a}, sessionResult.detNormalized{a}];
            else
                spontaneousData.detNormalized{a} = [spontaneousData.detNormalized{a}, nan];
            end

            if a <= length(sessionResult.lamRaw) && ~isempty(sessionResult.lamRaw{a})
                spontaneousData.lamRaw{a} = [spontaneousData.lamRaw{a}, sessionResult.lamRaw{a}];
            else
                spontaneousData.lamRaw{a} = [spontaneousData.lamRaw{a}, nan];
            end

            if a <= length(sessionResult.lamNormalized) && ~isempty(sessionResult.lamNormalized{a})
                spontaneousData.lamNormalized{a} = [spontaneousData.lamNormalized{a}, sessionResult.lamNormalized{a}];
            else
                spontaneousData.lamNormalized{a} = [spontaneousData.lamNormalized{a}, nan];
            end

            if a <= length(sessionResult.ttRaw) && ~isempty(sessionResult.ttRaw{a})
                spontaneousData.ttRaw{a} = [spontaneousData.ttRaw{a}, sessionResult.ttRaw{a}];
            else
                spontaneousData.ttRaw{a} = [spontaneousData.ttRaw{a}, nan];
            end

            if a <= length(sessionResult.ttNormalized) && ~isempty(sessionResult.ttNormalized{a})
                spontaneousData.ttNormalized{a} = [spontaneousData.ttNormalized{a}, sessionResult.ttNormalized{a}];
            else
                spontaneousData.ttNormalized{a} = [spontaneousData.ttNormalized{a}, nan];
            end
        end
    end

    spontaneousData.sessionNames{end+1} = sessionResult.sessionName;
end

% ===========================================================================================
% Process reach sessions
fprintf('\n=== Processing Reach Sessions ===\n');
% Reset opts for reach sessions
opts.collectStart = windowStartTime;
opts.collectEnd = windowEndTime;  % End time from timeRange

% Pre-allocate results cell array for parfor compatibility
reachResults = cell(1, length(reachSessions));

for s = 1:length(reachSessions)
    sessionName = reachSessions{s};
    fprintf('\nProcessing session %d/%d: %s\n', s, length(reachSessions), sessionName);

    % Initialize session result structure
    sessionResult = struct();
    sessionResult.sessionName = sessionName;
    sessionResult.success = false;
    sessionResult.areas = {};
    sessionResult.rrRaw = {};
    sessionResult.rrNormalized = {};
    sessionResult.detRaw = {};
    sessionResult.detNormalized = {};
    % Laminarity and trapping time
    sessionResult.lamRaw = {};
    sessionResult.lamNormalized = {};
    sessionResult.ttRaw = {};
    sessionResult.ttNormalized = {};

    try
        % Load data
        dataStruct = load_sliding_window_data('reach', 'spikes', ...
            'sessionName', sessionName, 'opts', opts);

        % Run RQA analysis
        fprintf('  Running RQA analysis...\n');
        results = rqa_sliding_analysis(dataStruct, config);

        % Extract areas from results
        if isfield(results, 'areas')
            sessionResult.areas = results.areas;
            numAreas = length(results.areas);

            % Initialize metric arrays
            sessionResult.rrRaw = cell(1, numAreas);
            sessionResult.rrNormalized = cell(1, numAreas);
            sessionResult.detRaw = cell(1, numAreas);
            sessionResult.detNormalized = cell(1, numAreas);
            sessionResult.lamRaw = cell(1, numAreas);
            sessionResult.lamNormalized = cell(1, numAreas);
            sessionResult.ttRaw = cell(1, numAreas);
            sessionResult.ttNormalized = cell(1, numAreas);

            % Extract metrics per area (single value per area)
            if isfield(results, 'recurrenceRate') && isfield(results, 'determinism') && ...
                    isfield(results, 'laminarity') && isfield(results, 'trappingTime')
                for a = 1:numAreas
                    % Recurrence rate raw
                    if a <= length(results.recurrenceRate) && ~isempty(results.recurrenceRate{a})
                        rrValues = results.recurrenceRate{a}(~isnan(results.recurrenceRate{a}));
                        if ~isempty(rrValues)
                            sessionResult.rrRaw{a} = rrValues(1);
                        else
                            sessionResult.rrRaw{a} = nan;
                        end
                    else
                        sessionResult.rrRaw{a} = nan;
                    end

                    % Recurrence rate normalized
                    if isfield(results, 'recurrenceRateNormalized') && ...
                            a <= length(results.recurrenceRateNormalized) && ...
                            ~isempty(results.recurrenceRateNormalized{a})
                        rrNormValues = results.recurrenceRateNormalized{a}(~isnan(results.recurrenceRateNormalized{a}));
                        if ~isempty(rrNormValues)
                            sessionResult.rrNormalized{a} = rrNormValues(1);
                        else
                            sessionResult.rrNormalized{a} = nan;
                        end
                    else
                        sessionResult.rrNormalized{a} = nan;
                    end

                    % Determinism raw
                    if a <= length(results.determinism) && ~isempty(results.determinism{a})
                        detValues = results.determinism{a}(~isnan(results.determinism{a}));
                        if ~isempty(detValues)
                            sessionResult.detRaw{a} = detValues(1);
                        else
                            sessionResult.detRaw{a} = nan;
                        end
                    else
                        sessionResult.detRaw{a} = nan;
                    end

                    % Determinism normalized
                    if isfield(results, 'determinismNormalized') && ...
                            a <= length(results.determinismNormalized) && ...
                            ~isempty(results.determinismNormalized{a})
                        detNormValues = results.determinismNormalized{a}(~isnan(results.determinismNormalized{a}));
                        if ~isempty(detNormValues)
                            sessionResult.detNormalized{a} = detNormValues(1);
                        else
                            sessionResult.detNormalized{a} = nan;
                        end
                    else
                        sessionResult.detNormalized{a} = nan;
                    end

                    % Laminarity raw
                    if a <= length(results.laminarity) && ~isempty(results.laminarity{a})
                        lamValues = results.laminarity{a}(~isnan(results.laminarity{a}));
                        if ~isempty(lamValues)
                            sessionResult.lamRaw{a} = lamValues(1);
                        else
                            sessionResult.lamRaw{a} = nan;
                        end
                    else
                        sessionResult.lamRaw{a} = nan;
                    end

                    % Laminarity normalized
                    if isfield(results, 'laminarityNormalized') && ...
                            a <= length(results.laminarityNormalized) && ...
                            ~isempty(results.laminarityNormalized{a})
                        lamNormValues = results.laminarityNormalized{a}(~isnan(results.laminarityNormalized{a}));
                        if ~isempty(lamNormValues)
                            sessionResult.lamNormalized{a} = lamNormValues(1);
                        else
                            sessionResult.lamNormalized{a} = nan;
                        end
                    else
                        sessionResult.lamNormalized{a} = nan;
                    end

                    % Trapping time raw
                    if a <= length(results.trappingTime) && ~isempty(results.trappingTime{a})
                        ttValues = results.trappingTime{a}(~isnan(results.trappingTime{a}));
                        if ~isempty(ttValues)
                            sessionResult.ttRaw{a} = ttValues(1);
                        else
                            sessionResult.ttRaw{a} = nan;
                        end
                    else
                        sessionResult.ttRaw{a} = nan;
                    end

                    % Trapping time normalized
                    if isfield(results, 'trappingTimeNormalized') && ...
                            a <= length(results.trappingTimeNormalized) && ...
                            ~isempty(results.trappingTimeNormalized{a})
                        ttNormValues = results.trappingTimeNormalized{a}(~isnan(results.trappingTimeNormalized{a}));
                        if ~isempty(ttNormValues)
                            sessionResult.ttNormalized{a} = ttNormValues(1);
                        else
                            sessionResult.ttNormalized{a} = nan;
                        end
                    else
                        sessionResult.ttNormalized{a} = nan;
                    end
                end
            else
                warning('RQA metrics not found in results for session %s', sessionName);
                for a = 1:numAreas
                    sessionResult.rrRaw{a} = nan;
                    sessionResult.rrNormalized{a} = nan;
                    sessionResult.detRaw{a} = nan;
                    sessionResult.detNormalized{a} = nan;
                    sessionResult.lamRaw{a} = nan;
                    sessionResult.lamNormalized{a} = nan;
                    sessionResult.ttRaw{a} = nan;
                    sessionResult.ttNormalized{a} = nan;
                end
            end

            sessionResult.success = true;
            fprintf('  ✓ Session completed successfully\n');
        else
            warning('Areas not found in results for session %s', sessionName);
        end

    catch ME
        fprintf('  ✗ Error processing session: %s\n', ME.message);
        sessionResult.success = false;
    end

    % Store result in cell array (parfor-compatible)
    reachResults{s} = sessionResult;
end

% Aggregate results from all sessions
fprintf('\n=== Aggregating Reach Session Results ===\n');
for s = 1:length(reachResults)
    sessionResult = reachResults{s};

    if isempty(sessionResult.areas)
        continue;
    end

    % Initialize areas on first valid session
    if isempty(reachData.areas) && ~isempty(sessionResult.areas)
        reachData.areas = sessionResult.areas;
        numAreas = length(reachData.areas);
        reachData.rrRaw = cell(1, numAreas);
        reachData.rrNormalized = cell(1, numAreas);
        reachData.detRaw = cell(1, numAreas);
        reachData.detNormalized = cell(1, numAreas);
        reachData.lamRaw = cell(1, numAreas);
        reachData.lamNormalized = cell(1, numAreas);
        reachData.ttRaw = cell(1, numAreas);
        reachData.ttNormalized = cell(1, numAreas);
        for a = 1:numAreas
            reachData.rrRaw{a} = [];
            reachData.rrNormalized{a} = [];
            reachData.detRaw{a} = [];
            reachData.detNormalized{a} = [];
            reachData.lamRaw{a} = [];
            reachData.lamNormalized{a} = [];
            reachData.ttRaw{a} = [];
            reachData.ttNormalized{a} = [];
        end
    end

    % Append metrics for this session
    if ~isempty(reachData.areas) && length(sessionResult.areas) == length(reachData.areas)
        for a = 1:length(reachData.areas)
            if a <= length(sessionResult.rrRaw) && ~isempty(sessionResult.rrRaw{a})
                reachData.rrRaw{a} = [reachData.rrRaw{a}, sessionResult.rrRaw{a}];
            else
                reachData.rrRaw{a} = [reachData.rrRaw{a}, nan];
            end

            if a <= length(sessionResult.rrNormalized) && ~isempty(sessionResult.rrNormalized{a})
                reachData.rrNormalized{a} = [reachData.rrNormalized{a}, sessionResult.rrNormalized{a}];
            else
                reachData.rrNormalized{a} = [reachData.rrNormalized{a}, nan];
            end

            if a <= length(sessionResult.detRaw) && ~isempty(sessionResult.detRaw{a})
                reachData.detRaw{a} = [reachData.detRaw{a}, sessionResult.detRaw{a}];
            else
                reachData.detRaw{a} = [reachData.detRaw{a}, nan];
            end

            if a <= length(sessionResult.detNormalized) && ~isempty(sessionResult.detNormalized{a})
                reachData.detNormalized{a} = [reachData.detNormalized{a}, sessionResult.detNormalized{a}];
            else
                reachData.detNormalized{a} = [reachData.detNormalized{a}, nan];
            end

            if a <= length(sessionResult.lamRaw) && ~isempty(sessionResult.lamRaw{a})
                reachData.lamRaw{a} = [reachData.lamRaw{a}, sessionResult.lamRaw{a}];
            else
                reachData.lamRaw{a} = [reachData.lamRaw{a}, nan];
            end

            if a <= length(sessionResult.lamNormalized) && ~isempty(sessionResult.lamNormalized{a})
                reachData.lamNormalized{a} = [reachData.lamNormalized{a}, sessionResult.lamNormalized{a}];
            else
                reachData.lamNormalized{a} = [reachData.lamNormalized{a}, nan];
            end

            if a <= length(sessionResult.ttRaw) && ~isempty(sessionResult.ttRaw{a})
                reachData.ttRaw{a} = [reachData.ttRaw{a}, sessionResult.ttRaw{a}];
            else
                reachData.ttRaw{a} = [reachData.ttRaw{a}, nan];
            end

            if a <= length(sessionResult.ttNormalized) && ~isempty(sessionResult.ttNormalized{a})
                reachData.ttNormalized{a} = [reachData.ttNormalized{a}, sessionResult.ttNormalized{a}];
            else
                reachData.ttNormalized{a} = [reachData.ttNormalized{a}, nan];
            end
        end
    end

    reachData.sessionNames{end+1} = sessionResult.sessionName;
end

    fprintf('  All completed in %.1f hours\n', toc/60/60);

% Determine common areas
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

% Create plots
fprintf('\n=== Creating RQA Plots ===\n');

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

% Collect all values (normalized metrics only) to determine global y-axis limits
allDetNorm = [];
allLamNorm = [];
allTTNorm = [];

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};

    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));

    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end

    % Determinism
    natDetNorm = spontaneousData.detNormalized{natAreaIdx};
    reachDetNorm = reachData.detNormalized{reachAreaIdx};
    natDetNorm = natDetNorm(~isnan(natDetNorm));
    reachDetNorm = reachDetNorm(~isnan(reachDetNorm));
    allDetNorm = [allDetNorm, natDetNorm, reachDetNorm];

    % Laminarity
    natLamNorm = spontaneousData.lamNormalized{natAreaIdx};
    reachLamNorm = reachData.lamNormalized{reachAreaIdx};
    natLamNorm = natLamNorm(~isnan(natLamNorm));
    reachLamNorm = reachLamNorm(~isnan(reachLamNorm));
    allLamNorm = [allLamNorm, natLamNorm, reachLamNorm];

    % Trapping time
    natTTNorm = spontaneousData.ttNormalized{natAreaIdx};
    reachTTNorm = reachData.ttNormalized{reachAreaIdx};
    natTTNorm = natTTNorm(~isnan(natTTNorm));
    reachTTNorm = reachTTNorm(~isnan(reachTTNorm));
    allTTNorm = [allTTNorm, natTTNorm, reachTTNorm];
end

% Calculate global y-axis limits with some padding (normalized metrics)
if ~isempty(allDetNorm)
    ylimDetNorm = [0, max(allDetNorm) * 1.05];
    if ylimDetNorm(1) == ylimDetNorm(2)
        ylimDetNorm = ylimDetNorm(1) + [-0.1, 0.1];
    end
else
    ylimDetNorm = [0, 2];
end

if ~isempty(allLamNorm)
    ylimLamNorm = [0, max(allLamNorm) * 1.05];
    if ylimLamNorm(1) == ylimLamNorm(2)
        ylimLamNorm = ylimLamNorm(1) + [-0.1, 0.1];
    end
else
    ylimLamNorm = [0, 2];
end

if ~isempty(allTTNorm)
    ylimTTNorm = [0, max(allTTNorm) * 1.05];
    if ylimTTNorm(1) == ylimTTNorm(2)
        ylimTTNorm = ylimTTNorm(1) + [-0.1, 0.1];
    end
else
    ylimTTNorm = [0, 2];
end

% Create figure with 3 columns: DET norm, LAM norm, TT norm
figure(3200); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', [100, 100, 1800, 300 * numAreasToPlot]);

for a = 1:numAreasToPlot
    areaName = areasToPlot{a};

    % Find area index in spontaneous and reach data
    natAreaIdx = find(strcmp(spontaneousData.areas, areaName));
    reachAreaIdx = find(strcmp(reachData.areas, areaName));

    if isempty(natAreaIdx) || isempty(reachAreaIdx)
        continue;
    end

    % Determinism (normalized)
    natDetNorm = spontaneousData.detNormalized{natAreaIdx};
    reachDetNorm = reachData.detNormalized{reachAreaIdx};
    natDetNorm = natDetNorm(~isnan(natDetNorm));
    reachDetNorm = reachDetNorm(~isnan(reachDetNorm));

    % Laminarity (normalized)
    natLamNorm = spontaneousData.lamNormalized{natAreaIdx};
    reachLamNorm = reachData.lamNormalized{reachAreaIdx};
    natLamNorm = natLamNorm(~isnan(natLamNorm));
    reachLamNorm = reachLamNorm(~isnan(reachLamNorm));

    % Trapping time (normalized)
    natTTNorm = spontaneousData.ttNormalized{natAreaIdx};
    reachTTNorm = reachData.ttNormalized{reachAreaIdx};
    natTTNorm = natTTNorm(~isnan(natTTNorm));
    reachTTNorm = reachTTNorm(~isnan(reachTTNorm));

    % DET Normalized
    subplot(numAreasToPlot, 3, (a-1)*3 + 1);
    hold on;
    numNat = length(natDetNorm);
    numReach = length(reachDetNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natDetNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachDetNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
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
    ylabel('DET (norm)');
    title(sprintf('%s - DET (Norm)', areaName));
    grid on;
    ylim(ylimDetNorm);
    if ~isempty(natDetNorm)
        yline(mean(natDetNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachDetNorm)
        yline(mean(reachDetNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;

    % LAM Normalized
    subplot(numAreasToPlot, 3, (a-1)*3 + 2);
    hold on;
    numNat = length(natLamNorm);
    numReach = length(reachLamNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natLamNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachLamNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
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
    ylabel('LAM (norm)');
    title(sprintf('%s - LAM (Norm)', areaName));
    grid on;
    ylim(ylimLamNorm);
    if ~isempty(natLamNorm)
        yline(mean(natLamNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachLamNorm)
        yline(mean(reachLamNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;

    % TT Normalized
    subplot(numAreasToPlot, 3, (a-1)*3 + 3);
    hold on;
    numNat = length(natTTNorm);
    numReach = length(reachTTNorm);
    xNat = 1:numNat;
    xReach = (numNat + 1):(numNat + numReach);
    if numNat > 0
        bar(xNat, natTTNorm, 'FaceColor', natColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    if numReach > 0
        bar(xReach, reachTTNorm, 'FaceColor', reachColor, 'EdgeColor', 'k', 'LineWidth', 1);
    end
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
    ylabel('TT (norm)');
    title(sprintf('%s - TT (Norm)', areaName));
    grid on;
    ylim(ylimTTNorm);
    if ~isempty(natTTNorm)
        yline(mean(natTTNorm), '--', 'color', natColor, 'LineWidth', 2);
    end
    if ~isempty(reachTTNorm)
        yline(mean(reachTTNorm), '--', 'color', reachColor, 'LineWidth', 2);
    end
    hold off;
end

% Add overall title
sgtitle(sprintf('RQA: Spontaneous vs Reach (Time Range: %.1f-%.1f s) PCA Dim 1-%d', ...
    windowStartTime, windowEndTime, config.nPCADim), 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
saveDir = fullfile(paths.dropPath, 'sliding_window_comparisons');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

plotFilenamePng = sprintf('rqa_single_per_session_%.0f-%.0fs.png', windowStartTime, windowEndTime);
plotPathPng = fullfile(saveDir, plotFilenamePng);

exportgraphics(gcf, plotPathPng, 'Resolution', 300);
fprintf('\nSaved PNG plot to: %s\n', plotPathPng);

plotFilenameEps = sprintf('rqa_single_per_session_%.0f-%.0fs.eps', windowStartTime, windowEndTime);
plotPathEps = fullfile(saveDir, plotFilenameEps);

exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
fprintf('Saved EPS plot to: %s\n', plotPathEps);

fprintf('\n=== RQA Single-Window Analysis Complete ===\n');

