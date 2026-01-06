%%
% Criticality AR Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture
%
% This script maintains compatibility with the old workflow while using
% the new modular functions.

% Set to 1 to load and plot existing results instead of running analysis
loadAndPlot = 0;

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

% Load and plot existing results if requested
if loadAndPlot
    % Require sessionType to be defined
    if ~exist('sessionType', 'var')
        error('sessionType must be defined to load and plot results');
    end
    
    % Load dataStruct (needed for plotting)
    fprintf('Loading data using load_sliding_window_data...\n');
    dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
        'sessionName', sessionName, 'opts', opts);
    
    % Find results file
    sessionNameForPath = '';
    if exist('sessionName', 'var') && ~isempty(sessionName)
        sessionNameForPath = sessionName;
    end
    
    % Build filename suffix from config if available
    filenameSuffix = '';
    if exist('config', 'var') && isfield(config, 'pcaFlag') && config.pcaFlag
        filenameSuffix = '_pca';
    end
    
    resultsPath = create_results_path('criticality_ar', sessionType, sessionNameForPath, ...
        dataStruct.saveDir, 'filenameSuffix', filenameSuffix, 'createDir', false);
    
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
    if isfield(results.params, 'pOrder')
        config.pOrder = results.params.pOrder;
    end
    if isfield(results.params, 'critType')
        config.critType = results.params.critType;
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
    criticality_ar_plot(results, plotConfig, config, dataStruct, filenameSuffix);
    
    fprintf('\n=== Plotting Complete ===\n');
    return;
end

% Normal analysis flow
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

