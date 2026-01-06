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
opts.collectEnd = opts.collectStart + 30*60;
opts.minFiringRate = .25;
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
config.nMinNeurons = 15;
config.minSpikesPerBin = 0.08;  % Minimum spikes per bin for optimal bin size calculation (we want about minSpikesPerPin proportion of bins with a spike)
config.minDataPoints = 2*10^5;
config.useBernoulliControl = false;  % Set to false to skip Bernoulli normalization (faster computation)

if strcmp(dataSource, 'spikes')
    config.binSize = binSize;
elseif strcmp(dataSource, 'lfp')
    config.lfpLowpassFreq = 80;
end

if strcmp(sessionType, 'naturalistic')
    config.behaviorNumeratorIDs = 5:10;
    config.behaviorDenominatorIDs = [config.behaviorNumeratorIDs, 0:2, 15:17];
end

% Run analysis
% Check if parpool is already running, start one if not
if runParallel
    currentPool = gcp('nocreate');
if isempty(currentPool)
    NumWorkers = length(dataStruct.areas);
    parpool('local', NumWorkers);
    fprintf('Started parallel pool with %d workers\n', NumWorkers);
else
    fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
end
end

results = lzc_sliding_analysis(dataStruct, config);

fprintf('\n=== Analysis Complete ===\n');

