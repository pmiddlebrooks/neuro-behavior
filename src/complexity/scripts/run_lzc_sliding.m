%%
% LZC Sliding Window Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture
%
% This script maintains compatibility with the old workflow while using
% the new modular functions. You can still set variables in workspace
% and run this script, or use lzc_sliding_analysis() function directly.


% Want to parallelize the area-wise analysis?
runParallel = 0;

% Set to 1 to load and plot existing results instead of running analysis
loadAndPlot = 0;
% When loadAndPlot = 1, you can optionally specify a time range to plot:
%   config.plotTimeRange = [startTime, endTime];  % in seconds
%   Example: config.plotTimeRange = [100, 500];  % Plot from 100s to 500s
%   If not specified or empty, all data will be plotted


% Set to 1 if you want to explore what binSize to use
findBinSize = 0;

% Add paths (check if directories exist first to avoid warnings)
basePath = fileparts(mfilename('fullpath'));  % complexity/scripts
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
opts.frameSize = .001;
opts.firingRateCheckTime = 3 * 60;
opts.collectStart = 0*60; %10*60;
opts.collectEnd = opts.collectStart + 45*60;
if strcmp(sessionType, 'reach') || strcmp(sessionType, 'hong')
opts.collectEnd = [];
end
opts.minFiringRate = .15;
opts.maxFiringRate = 200;

% Load and plot existing results if requested
if loadAndPlot
    % Require sessionType and dataSource to be defined
    if ~exist('sessionType', 'var') || ~exist('dataSource', 'var')
        error('sessionType and dataSource must be defined to load and plot results');
    end
    
    % Load dataStruct (needed for plotting)
    fprintf('Loading data using load_sliding_window_data...\n');
    dataStruct = load_sliding_window_data(sessionType, dataSource, ...
        'sessionName', sessionName, 'opts', opts);
    
    % Find results file
    sessionNameForPath = '';
    if exist('sessionName', 'var') && ~isempty(sessionName)
        sessionNameForPath = sessionName;
    end
    
    resultsPath = create_results_path('lzc', sessionType, sessionNameForPath, ...
        dataStruct.saveDir, 'dataSource', dataSource, 'createDir', false);
    
    if ~exist(resultsPath, 'file')
        error('Results file not found: %s', resultsPath);
    end
    
    fprintf('Loading results from: %s\n', resultsPath);
    load(resultsPath, 'results');
    
    % Reconstruct config from results.params
    % Note: LZC doesn't use filenameSuffix, so no config variables needed for filename construction
    config = struct();
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
    if isfield(results.params, 'minDataPoints')
        config.minDataPoints = results.params.minDataPoints;
    end
    if strcmp(dataSource, 'lfp')
        if isfield(results.params, 'lfpLowpassFreq')
            config.lfpLowpassFreq = results.params.lfpLowpassFreq;
        end
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
                
                if ~isempty(results.lzComplexity{a})
                    results.lzComplexity{a} = results.lzComplexity{a}(timeMask);
                end
                
                if ~isempty(results.lzComplexityNormalized{a})
                    results.lzComplexityNormalized{a} = results.lzComplexityNormalized{a}(timeMask);
                end
                
                if ~isempty(results.lzComplexityNormalizedBernoulli{a})
                    results.lzComplexityNormalizedBernoulli{a} = results.lzComplexityNormalizedBernoulli{a}(timeMask);
                end
                
                if isfield(results, 'behaviorProportion') && ~isempty(results.behaviorProportion{a})
                    results.behaviorProportion{a} = results.behaviorProportion{a}(timeMask);
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
    lzc_sliding_plot(results, plotConfig, config, dataStruct);
    
    fprintf('\n=== Plotting Complete ===\n');
    return;
end

% Normal analysis flow - check if data is already loaded (workspace variables)
% If not, try to load using new function
% Try to load data using new function
if exist('sessionType', 'var') && exist('dataSource', 'var')
    fprintf('Loading data using load_sliding_window_data...\n');
    dataStruct = load_sliding_window_data(sessionType, dataSource, ...
        'sessionName', sessionName, 'opts', opts);

    if findBinSize
        if strcmp(dataSource, 'spikes')
            %%
            idRun = dataStruct.idMatIdx{2};
            [proportions, spikeCounts] = neural_pct_spike_count(dataStruct.dataMat(:,idRun), [.005 .01 .02], 4);
        elseif strcmp(dataSource, 'lfp')
        end
        disp("Code didn't run b/c you were determine what bin size to use")
        return
    end
else
    error('sessionType and dataSource must be defined, or data must be pre-loaded in workspace');
end

% Set up configuration
binSize = .02;  % Only used if useOptimalBinSize = false

config = struct();
config.useOptimalWindowSize = true;
config.useOptimalBinSize = true;  % Set to true to automatically calculate optimal bin size per area
config.slidingWindowSize = 60;
config.minSlidingWindowSize = 5;
config.maxSlidingWindowSize = 60;
config.stepSize = 15;
config.nShuffles = 3;
config.makePlots = true;
config.saveData = true;  % Set to false to skip saving results
config.nMinNeurons = 15;
config.minSpikesPerBin = 0.08;  % Minimum spikes per bin for optimal bin size calculation (we want about minSpikesPerPin proportion of bins with a spike)
config.minDataPoints = 2*10^5;
config.useBernoulliControl = false;  % Set to false to skip Bernoulli normalization (faster computation)
config.includeM2356 = true;  % Set to true to include combined M23+M56 area

if strcmp(dataSource, 'spikes')
    config.binSize = binSize;
elseif strcmp(dataSource, 'lfp')
    config.lfpLowpassFreq = 80;
end

if strcmp(sessionType, 'spontaneous')
    config.behaviorNumeratorIDs = 5:10;
    config.behaviorDenominatorIDs = [config.behaviorNumeratorIDs, 0:2, 15:17];
end

% Run analysis
% Check if parpool is already running, start one if not
if runParallel
    currentPool = gcp('nocreate');
if isempty(currentPool)
    NumWorkers = min(4, length(dataStruct.areas));
    parpool('local', NumWorkers);
    fprintf('Started parallel pool with %d workers\n', NumWorkers);
else
    fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
end
end

results = lzc_sliding_analysis(dataStruct, config);

fprintf('\n=== Analysis Complete ===\n');

