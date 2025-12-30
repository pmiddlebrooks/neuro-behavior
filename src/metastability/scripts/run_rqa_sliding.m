%%
% RQA Sliding Window Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture
%
% This script maintains compatibility with the old workflow while using
% the new modular functions. You can still set variables in workspace
% and run this script, or use rqa_sliding_analysis() function directly.

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
opts.collectStart = 0;
opts.collectEnd = 20 * 60;
opts.minFiringRate = .25;
opts.maxFiringRate = 200;

% Check if data is already loaded (workspace variables)
% If not, try to load using new function
if ~exist('dataMat', 'var')
    % Try to load data using new function
    if exist('sessionType', 'var') && exist('dataSource', 'var')
        fprintf('Loading data using load_sliding_window_data...\n');
        dataStruct = load_sliding_window_data(sessionType, dataSource, ...
            'sessionName', sessionName, 'opts', opts);
        
        % Convert structure to workspace variables for backward compatibility
        dataMat = dataStruct.dataMat;
        areas = dataStruct.areas;
        idMatIdx = dataStruct.idMatIdx;
        idLabel = dataStruct.idLabel;
        opts = dataStruct.opts;
        saveDir = dataStruct.saveDir;
        sessionType = dataStruct.sessionType;
        dataSource = dataStruct.dataSource;
        areasToTest = dataStruct.areasToTest;
        
        if isfield(dataStruct, 'sessionName')
            sessionName = dataStruct.sessionName;
        end
        if isfield(dataStruct, 'dataR')
            dataR = dataStruct.dataR;
        end
        if isfield(dataStruct, 'reachStart')
            reachStart = dataStruct.reachStart;
        end
        if isfield(dataStruct, 'startBlock2')
            startBlock2 = dataStruct.startBlock2;
        end
    else
        error('sessionType and dataSource must be defined, or data must be pre-loaded in workspace');
    end
end

% Set up configuration
if ~exist('slidingWindowSize', 'var')
    slidingWindowSize = 60;  % Default window size
end
if ~exist('binSize', 'var')
    binSize = 0.015;  % Default bin size
end
if ~exist('stepSize', 'var')
    stepSize = 30;  % Default step size
end
if ~exist('nShuffles', 'var')
    nShuffles = 3;  % Default number of shuffles
end
if ~exist('nPCADim', 'var')
    nPCADim = 3;  % Default number of PCA dimensions
end

config = struct();
config.slidingWindowSize = slidingWindowSize;
config.stepSize = stepSize;
config.binSize = binSize;
config.nShuffles = nShuffles;
config.nPCADim = nPCADim;
config.recurrenceThreshold = 'mean';  % Can be 'mean', 'median', or numeric
config.makePlots = true;

if exist('saveDir', 'var')
    config.saveDir = saveDir;
end

% Create data structure from workspace variables
dataStruct = struct();
dataStruct.sessionType = sessionType;
dataStruct.dataSource = dataSource;
dataStruct.areas = areas;
dataStruct.opts = opts;
if exist('dataMat', 'var')
    dataStruct.dataMat = dataMat;
end
if exist('idMatIdx', 'var')
    dataStruct.idMatIdx = idMatIdx;
end
if exist('idLabel', 'var')
    dataStruct.idLabel = idLabel;
end
if exist('sessionName', 'var')
    dataStruct.sessionName = sessionName;
end
if exist('dataBaseName', 'var')
    dataStruct.dataBaseName = dataBaseName;
end
if exist('areasToTest', 'var')
    dataStruct.areasToTest = areasToTest;
end
if exist('dataR', 'var')
    dataStruct.dataR = dataR;
end
if exist('reachStart', 'var')
    dataStruct.reachStart = reachStart;
end
if exist('startBlock2', 'var')
    dataStruct.startBlock2 = startBlock2;
end
if exist('saveDir', 'var')
    dataStruct.saveDir = saveDir;
end

% Run analysis
results = rqa_sliding_analysis(dataStruct, config);

% Store results in workspace for backward compatibility
recurrenceRate = results.recurrenceRate;
determinism = results.determinism;
laminarity = results.laminarity;
trappingTime = results.trappingTime;
recurrenceRateNormalized = results.recurrenceRateNormalized;
determinismNormalized = results.determinismNormalized;
laminarityNormalized = results.laminarityNormalized;
trappingTimeNormalized = results.trappingTimeNormalized;
recurrenceRateNormalizedBernoulli = results.recurrenceRateNormalizedBernoulli;
determinismNormalizedBernoulli = results.determinismNormalizedBernoulli;
laminarityNormalizedBernoulli = results.laminarityNormalizedBernoulli;
trappingTimeNormalizedBernoulli = results.trappingTimeNormalizedBernoulli;
recurrencePlots = results.recurrencePlots;
startS = results.startS;

fprintf('\n=== Analysis Complete ===\n');

