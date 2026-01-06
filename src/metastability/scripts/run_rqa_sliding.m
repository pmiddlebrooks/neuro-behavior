%%
% RQA Sliding Window Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture
%
% This script maintains compatibility with the old workflow while using
% the new modular functions. You can still set variables in workspace
% and run this script, or use rqa_sliding_analysis() function directly.


runParallel = 1;

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
opts.frameSize = .001;
opts.firingRateCheckTime = 3 * 60;
opts.collectStart = 0*60; %10*60;
opts.collectEnd = opts.collectStart + 30*60;
opts.minFiringRate = .25;
opts.maxFiringRate = 100;

% Check if data is already loaded (workspace variables)
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
config.stepSize = .5*60;
config.nShuffles = 3;
config.makePlots = true;
config.useBernoulliControl = false;  % Set to false to skip Bernoulli normalization (faster computation)
config.nPCADim = 4;
config.recurrenceThreshold = 0.03;  % Target recurrence rate (0.02 = 2%, range: 0.01-0.05)
config.distanceMetric = 'cosine';  % 'euclidean' or 'cosine' (cosine recommended for sparse spiking)
config.nMinNeurons = 15;  % Minimum number of neurons required (areas with fewer will be skipped)
config.saveRecurrencePlots = false;  % Set to true to compute and store recurrence plots (uses a lot of memory)
config.usePerWindowPCA = true;  % Set to true to perform PCA on each window (addresses representational drift)

if strcmp(sessionType, 'naturalistic')
    config.behaviorNumeratorIDs = 5:10;
    config.behaviorDenominatorIDs = [config.behaviorNumeratorIDs, 0:2, 15:17];
end

% Run analysis
if runParallel
% Check if parpool is already running, start one if not
currentPool = gcp('nocreate');
if isempty(currentPool)
    NumWorkers = length(dataStruct.areas);
    parpool('local', NumWorkers);
    fprintf('Started parallel pool with %d workers\n', NumWorkers);
else
    fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
end
end

results = rqa_sliding_analysis(dataStruct, config);


fprintf('\n=== Analysis Complete ===\n');

