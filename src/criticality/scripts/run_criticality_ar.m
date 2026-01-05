%%
% Criticality AR Analysis - Wrapper Script
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
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = 30*60;
opts.minFiringRate = .05;
opts.maxFiringRate = 100;



slidingWindowSize = 6;  % Default window size
binSize = 0.02;  % Default bin size
stepSize = 0.1;  % Step size in seconds (common for all areas)

analyzeD2 = true;

analyzeMrBr = false;
pcaFlag = 0;
enablePermutations = true;
nShuffles = 3;
analyzeModulation = false;
makePlots = true;
useOptimalBinWindowFunction = true;

pOrder = 10;
critType = 2;


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
config.slidingWindowSize = 10; % Default window size
config.binSize = .02; % Default bin size
config.stepSize = .1; % Default step size
config.minSpikesPerBin = 2.5;
config.minBinsPerWindow = 1000;

% Analysis flags (set these before running)
config.analyzeD2 = true;
config.analyzeMrBr = false;
config.pcaFlag = 0;
config.enablePermutations = true;
config.nShuffles = 3;
config.analyzeModulation = false;
config.makePlots = true;
config.useOptimalBinWindowFunction = true;

% Additional parameters
config.pOrder = 10;
config.critType = 2;
% saveDir will be obtained from dataStruct.saveDir in the analysis function
% (set by load_sliding_window_data)


% Run analysis
results = criticality_ar_analysis(dataStruct, config);


fprintf('\n=== Analysis Complete ===\n');

