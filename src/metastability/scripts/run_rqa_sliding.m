%%
% RQA Sliding Window Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture
%
% This script maintains compatibility with the old workflow while using
% the new modular functions. You can still set variables in workspace
% and run this script, or use rqa_sliding_analysis() function directly.

% use parallel processing?
runParallel = 1;

% Set to 1 to load and plot existing results instead of running analysis
loadAndPlot = 0;
% When loadAndPlot = 1, you can optionally specify a time range to plot:
%   config.plotTimeRange = [startTime, endTime];  % in seconds
%   Example: config.plotTimeRange = [100, 500];  % Plot from 100s to 500s
%   If not specified or empty, all data will be plotted
if loadAndPlot
    config.nPCADim = 4;
    config.usePerWindowPCA = false;  % Set to true to perform PCA on each window (addresses representational drift)
    % Optionally specify which areas to plot (default: all areas)
    % Can be a vector of indices (e.g., [1, 2]) or cell array of area names (e.g., {'S1', 'SC'})
    % Example: config.areasToPlot = [1, 2];  % Plot only first two areas
    % Example: config.areasToPlot = {'S1', 'SC'};  % Plot areas by name
    % If not specified or empty, all areas will be plotted
    config.areasToPlot = 2:3;
end

paths = get_paths;

% Add paths (check if directories exist first to avoid warnings)
basePath = fileparts(mfilename('fullpath'));  % metastability/scripts
srcPath = fullfile(basePath, '..', '..');     % src

% Add sliding_window_prep paths
swDataPrepPath = fullfile(srcPath, 'sliding_window_prep', 'data_prep');
swUtilsPath = fullfile(srcPath, 'sliding_window_prep', 'utils');
analysesPath = fullfile(basePath, '..', 'analyses');

if exist(swDataPrepPath, 'dir')
    addpath(swDataPrepPath);
end
if exist(swUtilsPath, 'dir')
    addpath(swUtilsPath);
end
if exist(analysesPath, 'dir')
    addpath(analysesPath);
end

% Configure variables
opts = neuro_behavior_options;
opts.firingRateCheckTime = 3 * 60;
opts.collectStart = 0*60; %10*60;
opts.collectEnd = opts.collectStart + 45*60;
if strcmp(sessionType, 'reach') || strcmp(sessionType, 'hong')
    opts.collectEnd = [];
end
opts.minFiringRate = .15;
opts.maxFiringRate = 100;

% Load and plot existing results if requested
if loadAndPlot
    % Require sessionType and dataSource to be defined
    if ~exist('sessionType', 'var') || ~exist('dataSource', 'var')
        error('sessionType and dataSource must be defined to load and plot results');
    end

    % Require config variables needed to construct filename
    % These should match what's used in rqa_sliding_analysis.m
    if ~exist('config', 'var')
        config = struct();
    end
    if ~isfield(config, 'nPCADim')
        error('config.nPCADim must be defined to load RQA results (needed for filename)');
    end
    if ~isfield(config, 'usePerWindowPCA')
        config.usePerWindowPCA = false;  % Default if not specified
    end

    % Load dataStruct (needed for plotting)
    fprintf('Loading data using load_sliding_window_data...\n');
    dataStruct = load_sliding_window_data(sessionType, dataSource, ...
        'sessionName', sessionName, 'opts', opts);

    % Find results file - construct filenameSuffix matching rqa_sliding_analysis.m
    sessionNameForPath = '';
    if exist('sessionName', 'var') && ~isempty(sessionName)
        sessionNameForPath = sessionName;
    end

    % Construct filenameSuffix exactly as in rqa_sliding_analysis.m
    % Always include PCA dimensions in RQA filenames
    filenameSuffix = sprintf('_pca%d', config.nPCADim);
    if config.usePerWindowPCA
        filenameSuffix = [filenameSuffix, '_drift'];
    end

    resultsPath = create_results_path('rqa', sessionType, sessionNameForPath, ...
        dataStruct.saveDir, 'dataSource', dataSource, ...
        'filenameSuffix', filenameSuffix, 'createDir', false);

    if ~exist(resultsPath, 'file')
        error('Results file not found: %s', resultsPath);
    end

    fprintf('Loading results from: %s\n', resultsPath);
    load(resultsPath, 'results');

    % Reconstruct config from results.params (overwrite with saved values)
    if isfield(results.params, 'slidingWindowSize')
        config.slidingWindowSize = results.params.slidingWindowSize;
    end
    if isfield(results.params, 'stepSize')
        config.stepSize = results.params.stepSize;
    end
    if isfield(results.params, 'nShuffles')
        config.nShuffles = results.params.nShuffles;
    end
    if isfield(results.params, 'useBernoulliControl')
        config.useBernoulliControl = results.params.useBernoulliControl;
    end
    if isfield(results.params, 'nPCADim')
        config.nPCADim = results.params.nPCADim;
    end
    if isfield(results.params, 'recurrenceThreshold')
        config.recurrenceThreshold = results.params.recurrenceThreshold;
    end
    if isfield(results.params, 'distanceMetric')
        config.distanceMetric = results.params.distanceMetric;
    end
    if isfield(results.params, 'usePerWindowPCA')
        config.usePerWindowPCA = results.params.usePerWindowPCA;
    end

    % Add saveDir from dataStruct (needed for plotting)
    config.saveDir = dataStruct.saveDir;

    % Allow user to specify time range for plotting
    % Example: config.plotTimeRange = [100, 500];  % Plot from 100s to 500s
    if ~isfield(config, 'plotTimeRange') || isempty(config.plotTimeRange)
        config.plotTimeRange = [];  % Empty means plot all data
    end

    % Filter results by time range if specified
    if ~isempty(config.plotTimeRange) && length(config.plotTimeRange) == 2
        timeStart = config.plotTimeRange(1);
        timeEnd = config.plotTimeRange(2);
        fprintf('Filtering results to time range [%.1f, %.1f] s\n', timeStart, timeEnd);

        % Filter each area's data
        numAreas = length(results.areas);
        for a = 1:numAreas
            if ~isempty(results.startS{a})
                % Find indices within time range
                timeMask = results.startS{a} >= timeStart & results.startS{a} <= timeEnd;

                % Filter all time-series data for this area
                results.startS{a} = results.startS{a}(timeMask);

                if ~isempty(results.recurrenceRate{a})
                    results.recurrenceRate{a} = results.recurrenceRate{a}(timeMask);
                end

                if ~isempty(results.determinism{a})
                    results.determinism{a} = results.determinism{a}(timeMask);
                end

                if ~isempty(results.laminarity{a})
                    results.laminarity{a} = results.laminarity{a}(timeMask);
                end

                if ~isempty(results.trappingTime{a})
                    results.trappingTime{a} = results.trappingTime{a}(timeMask);
                end

                if ~isempty(results.recurrenceRateNormalized{a})
                    results.recurrenceRateNormalized{a} = results.recurrenceRateNormalized{a}(timeMask);
                end

                if ~isempty(results.determinismNormalized{a})
                    results.determinismNormalized{a} = results.determinismNormalized{a}(timeMask);
                end

                if ~isempty(results.laminarityNormalized{a})
                    results.laminarityNormalized{a} = results.laminarityNormalized{a}(timeMask);
                end

                if ~isempty(results.trappingTimeNormalized{a})
                    results.trappingTimeNormalized{a} = results.trappingTimeNormalized{a}(timeMask);
                end

                if ~isempty(results.recurrenceRateNormalizedBernoulli{a})
                    results.recurrenceRateNormalizedBernoulli{a} = results.recurrenceRateNormalizedBernoulli{a}(timeMask);
                end

                if ~isempty(results.determinismNormalizedBernoulli{a})
                    results.determinismNormalizedBernoulli{a} = results.determinismNormalizedBernoulli{a}(timeMask);
                end

                if ~isempty(results.laminarityNormalizedBernoulli{a})
                    results.laminarityNormalizedBernoulli{a} = results.laminarityNormalizedBernoulli{a}(timeMask);
                end

                if ~isempty(results.trappingTimeNormalizedBernoulli{a})
                    results.trappingTimeNormalizedBernoulli{a} = results.trappingTimeNormalizedBernoulli{a}(timeMask);
                end

                if isfield(results, 'behaviorProportion') && ~isempty(results.behaviorProportion{a})
                    results.behaviorProportion{a} = results.behaviorProportion{a}(timeMask);
                end

                % Filter recurrence plots if they exist (cell array of cell arrays)
                if isfield(results, 'recurrencePlots') && ~isempty(results.recurrencePlots{a})
                    results.recurrencePlots{a} = results.recurrencePlots{a}(timeMask);
                end
            end
        end
        fprintf('Filtered results: %d areas processed\n', numAreas);
    end

    % Setup plotting
    plotArgs = {};
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        plotArgs = [plotArgs, {'sessionName', dataStruct.sessionName}];
    end
    if isfield(dataStruct, 'dataBaseName') && ~isempty(dataStruct.dataBaseName)
        plotArgs = [plotArgs, {'dataBaseName', dataStruct.dataBaseName}];
    end
    plotConfig = setup_plotting(dataStruct.saveDir, plotArgs{:});

    % Plot results
    fprintf('Plotting results...\n');
    rqa_sliding_plot(results, plotConfig, config, dataStruct);

    fprintf('\n=== Plotting Complete ===\n');
    return;
end

% ============================================================================================================

% Normal analysis flow - check if data is already loaded (workspace variables)
% If not, try to load using new function
% Try to load data using new function
if exist('sessionType', 'var') && exist('dataSource', 'var')
    fprintf('Loading data using load_sliding_window_data...\n');
    dataStruct = load_sliding_window_data(sessionType, dataSource, ...
        'sessionName', sessionName, 'opts', opts);
else
    error('sessionType and dataSource must be defined, or data must be pre-loaded in workspace');
end

% Set up configuration
config = struct();
% Use optimal bin size calculation (default)
config.useOptimalBinSize = true;  % Set to true to calculate optimal bin size per area
config.minSpikesPerBin = 3.5;  % Minimum spikes per bin for optimal bin size calculation
config.minTimeBins = 10000;  % Minimum number of time bins per window (used to auto-calculate slidingWindowSize)
config.minWindowSize = 120;
% slidingWindowSize will be auto-calculated from binSize and minTimeBins in rqa_sliding_analysis.m
% config.slidingWindowSize = 10*60;  % Optional: specify directly (overrides auto-calculation)
config.stepSize = 30;
config.nShuffles = 4;
config.makePlots = true;
config.useBernoulliControl = false;  % Set to false to skip Bernoulli normalization (faster computation)
config.nPCADim = 5;
config.recurrenceThreshold = 0.03;  % Target recurrence rate (0.02 = 2%, range: 0.01-0.05)
config.distanceMetric = 'cosine';  % 'euclidean' or 'cosine' (cosine recommended for sparse spiking)
config.nMinNeurons = 15;  % Minimum number of neurons required (areas with fewer will be skipped)
config.saveRecurrencePlots = false;  % Set to true to compute and store recurrence plots (uses a lot of memory)
config.usePerWindowPCA = false;  % Set to true to perform PCA on each window (addresses representational drift)
config.saveData = true;  % Set to false to skip saving results
config.includeM2356 = true;  % Set to true to include combined M23+M56 area

if strcmp(sessionType, 'spontaneous')
    config.behaviorNumeratorIDs = 5:10;
    config.behaviorDenominatorIDs = [config.behaviorNumeratorIDs, 0:2, 15:17];
end

% Run analysis
if runParallel
    % Check if parpool is already running, start one if not
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        NumWorkers = min(4, length(dataStruct.areas));
        parpool('local', NumWorkers);
        fprintf('Started parallel pool with %d workers\n', NumWorkers);
    else
        fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
    end
end

results = rqa_sliding_analysis(dataStruct, config);


fprintf('\n=== Analysis Complete ===\n');

