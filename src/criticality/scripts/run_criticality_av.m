%%
% Criticality Avalanche Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture
%
% This script maintains compatibility with the old workflow while using
% the new modular functions.

% Add paths (check if directories exist first to avoid warnings)
basePath = fileparts(mfilename('fullpath'));  % criticality/scripts
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
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = opts.collectStart + 2*60*60;
opts.collectEnd = [];
if exist('sessionType', 'var') && (strcmp(sessionType, 'reach') || strcmp(sessionType, 'hong'))
    opts.collectEnd = [];
end
opts.minFiringRate = .1;
opts.maxFiringRate = 100;

% Try to load data using new function
if exist('sessionType', 'var') && exist('dataSource', 'var')
    fprintf('Loading data using load_sliding_window_data...\n');
    dataStruct = load_sliding_window_data(sessionType, dataSource, ...
        'sessionName', sessionName, 'opts', opts);
else
    error('sessionType and dataSource must be defined, or data must be pre-loaded in workspace');
end

% Set up configuration from workspace variables
% (These should be set before running this script)
config = struct();

% Window geometry
config.slidingWindowSize = 180;  % Base window size (s). If useOptimalBinWindowFunction is false
                                 % this can be scalar or per-area vector.
config.avStepSize = 20;          % Step size between sliding windows (s)

% Parameters for automatic optimal bin/window search
config.minSpikesPerBin = 3;
config.minBinsPerWindow = 1000;

% Analysis flags (set these before running)
config.analyzeDcc = true;
config.analyzeKappa = true;
config.pcaFlag = 0;
config.pcaFirstFlag = 1;  % Use first nDim if 1, last nDim if 0
config.nDim = 5;  % Number of PCA dimensions
config.enablePermutations = true;
config.nShuffles = 5;
config.makePlots = true;
config.saveData = true;              % Set to false to skip saving results

% If true (default), binSize and slidingWindowSize are found automatically
% per area using firing rate heuristics (recommended).
% If false, user must provide:
%   - config.binSize (scalar or per-area vector, in seconds)
%   - config.slidingWindowSize (scalar or per-area vector, in seconds)
config.useOptimalBinWindowFunction = true;

% Example manual settings (uncomment and set useOptimalBinWindowFunction = false):
% config.useOptimalBinWindowFunction = false;
% config.binSize = 0.02;            % 20 ms bins for all areas
% config.slidingWindowSize = 180;   % 180 s windows for all areas

% Additional parameters
config.thresholdFlag = 1;  % Use threshold method
config.thresholdPct = 1;  % Threshold as percentage of median
config.nMinNeurons = 15;  % Minimum number of neurons required per area
config.includeM2356 = false;  % Set to true to include combined M23+M56 area
% saveDir will be obtained from dataStruct.saveDir in the analysis function
% (set by load_sliding_window_data)

% Run analysis
results = criticality_av_analysis(dataStruct, config);

fprintf('\n=== Analysis Complete ===\n');

