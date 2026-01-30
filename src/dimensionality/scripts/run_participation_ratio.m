%%
% Participation Ratio Sliding Window Analysis - Wrapper Script
% Uses participation_ratio_analysis() and participation_ratio_plot().
% Set sessionType, sessionName (and optionally dataSource) before running.
% Set loadAndPlot = 1 to load saved results and plot only.

% Set to 1 to load and plot existing results instead of running analysis
loadAndPlot = 0;
% When loadAndPlot = 1, optionally set config.plotTimeRange = [startTime, endTime]
% to restrict the plotted time range (seconds).

% Paths
basePath = fileparts(mfilename('fullpath'));   % dimensionality/scripts
srcPath = fullfile(basePath, '..', '..');      % src

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

% Options for data loading
opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;
if exist('sessionType', 'var') && (strcmp(sessionType, 'reach') || strcmp(sessionType, 'hong'))
    opts.collectEnd = [];
end

% Load and plot existing results if requested
if loadAndPlot
    if ~exist('sessionType', 'var')
        error('sessionType must be defined to load and plot results');
    end

    fprintf('Loading data using load_sliding_window_data...\n');
    dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
        'sessionName', sessionName, 'opts', opts);

    sessionNameForPath = '';
    if exist('sessionName', 'var') && ~isempty(sessionName)
        sessionNameForPath = sessionName;
    end

    resultsPath = create_results_path('participation_ratio', sessionType, sessionNameForPath, ...
        dataStruct.saveDir, 'createDir', false);

    if ~exist(resultsPath, 'file')
        error('Results file not found: %s', resultsPath);
    end

    fprintf('Loading results from: %s\n', resultsPath);
    load(resultsPath, 'results');

    config = struct();
    if isfield(results.params, 'slidingWindowSize')
        config.slidingWindowSize = results.params.slidingWindowSize;
    end
    if isfield(results.params, 'stepSize')
        config.stepSize = results.params.stepSize;
    end
    config.saveDir = dataStruct.saveDir;

    if ~isfield(config, 'plotTimeRange')
        config.plotTimeRange = [];
    end
    if ~isempty(config.plotTimeRange) && length(config.plotTimeRange) == 2
        timeStart = config.plotTimeRange(1);
        timeEnd = config.plotTimeRange(2);
        fprintf('Filtering results to time range [%.1f, %.1f] s\n', timeStart, timeEnd);
        numAreas = length(results.areas);
        for a = 1:numAreas
            if ~isempty(results.startS{a})
                timeMask = results.startS{a} >= timeStart & results.startS{a} <= timeEnd;
                results.startS{a} = results.startS{a}(timeMask);
                if ~isempty(results.participationRatio{a})
                    results.participationRatio{a} = results.participationRatio{a}(timeMask);
                end
                if isfield(results, 'popActivityWindows') && ~isempty(results.popActivityWindows{a})
                    results.popActivityWindows{a} = results.popActivityWindows{a}(timeMask);
                end
                if isfield(results, 'popActivityFull') && ~isempty(results.popActivityFull{a})
                    results.popActivityFull{a} = results.popActivityFull{a}(timeMask);
                end
            end
        end
        fprintf('Filtered results by time range\n');
    end

    plotArgs = {};
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        plotArgs = [plotArgs, {'sessionName', dataStruct.sessionName}];
    end
    if isfield(dataStruct, 'dataBaseName') && ~isempty(dataStruct.dataBaseName)
        plotArgs = [plotArgs, {'dataBaseName', dataStruct.dataBaseName}];
    end
    plotConfig = setup_plotting(dataStruct.saveDir, plotArgs{:});
    fprintf('Plotting results...\n');
    participation_ratio_plot(results, plotConfig, config, dataStruct);
    fprintf('\n=== Plotting Complete ===\n');
    return;
end

% Normal analysis flow
if ~exist('sessionType', 'var') || ~exist('dataSource', 'var')
    error('sessionType and dataSource must be defined (e.g. sessionType=''spontaneous''; dataSource=''spikes''; sessionName=''...'';)');
end

fprintf('Loading data using load_sliding_window_data...\n');
dataStruct = load_sliding_window_data(sessionType, dataSource, ...
    'sessionName', sessionName, 'opts', opts);

config = struct();
config.slidingWindowSize = 10;
config.stepSize = 0.1;
config.minSpikesPerBin = 2.5;
config.minBinsPerWindow = 1000;
config.windowSizeNeuronMultiple = 10;  % time bins per window = this * nNeurons
config.useOptimalBinWindowFunction = true;
config.makePlots = true;
config.saveData = true;
config.nMinNeurons = 10;

results = participation_ratio_analysis(dataStruct, config);

fprintf('\n=== Analysis Complete ===\n');
