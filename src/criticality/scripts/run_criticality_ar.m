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
        opts.collectEnd = 10*60;
        opts.minFiringRate = .05;
        opts.maxFiringRate = 200;



    slidingWindowSize = 6;  % Default window size
    binSize = 0.02;  % Default bin size

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
    if exist('dataType', 'var') && exist('dataSource', 'var')
        fprintf('Loading data using load_sliding_window_data...\n');
        dataStruct = load_sliding_window_data(dataType, dataSource, ...
            'sessionName', sessionName, 'opts', opts);
        
        % Convert structure to workspace variables for backward compatibility
        dataMat = dataStruct.dataMat;
        areas = dataStruct.areas;
        idMatIdx = dataStruct.idMatIdx;
        idLabel = dataStruct.idLabel;
        opts = dataStruct.opts;
        saveDir = dataStruct.saveDir;
        dataType = dataStruct.dataType;
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
        error('dataType and dataSource must be defined, or data must be pre-loaded in workspace');
    end

% Set up configuration from workspace variables
% (These should be set before running this script)
if ~exist('slidingWindowSize', 'var')
    slidingWindowSize = 2;  % Default window size
end
if ~exist('binSize', 'var')
    binSize = 0.02;  % Default bin size
end

config = struct();
config.slidingWindowSize = slidingWindowSize;
config.binSize = binSize;

% Analysis flags (set these before running)
if exist('analyzeD2', 'var')
    config.analyzeD2 = analyzeD2;
else
    config.analyzeD2 = true;
end

if exist('analyzeMrBr', 'var')
    config.analyzeMrBr = analyzeMrBr;
else
    config.analyzeMrBr = false;
end

if exist('pcaFlag', 'var')
    config.pcaFlag = pcaFlag;
else
    config.pcaFlag = 0;
end

if exist('enablePermutations', 'var')
    config.enablePermutations = enablePermutations;
else
    config.enablePermutations = true;
end

if exist('nShuffles', 'var')
    config.nShuffles = nShuffles;
else
    config.nShuffles = 3;
end

if exist('analyzeModulation', 'var')
    config.analyzeModulation = analyzeModulation;
else
    config.analyzeModulation = false;
end

if exist('makePlots', 'var')
    config.makePlots = makePlots;
else
    config.makePlots = true;
end

if exist('useOptimalBinWindowFunction', 'var')
    config.useOptimalBinWindowFunction = useOptimalBinWindowFunction;
else
    config.useOptimalBinWindowFunction = true;
end

% Additional parameters
if exist('pOrder', 'var')
    config.pOrder = pOrder;
else
    config.pOrder = 10;
end

if exist('critType', 'var')
    config.critType = critType;
else
    config.critType = 2;
end

if exist('saveDir', 'var')
    config.saveDir = saveDir;
end

% Create data structure from workspace variables
dataStruct = struct();
dataStruct.dataType = dataType;
dataStruct.dataSource = 'spikes';  % AR analysis is for spikes
dataStruct.areas = areas;
dataStruct.opts = opts;
dataStruct.dataMat = dataMat;
dataStruct.idMatIdx = idMatIdx;
dataStruct.idLabel = idLabel;
if exist('areasToTest', 'var')
    dataStruct.areasToTest = areasToTest;
end
if exist('sessionName', 'var')
    dataStruct.sessionName = sessionName;
end
if exist('dataBaseName', 'var')
    dataStruct.dataBaseName = dataBaseName;
end
if exist('dataR', 'var')
    dataStruct.dataR = dataR;
end
if exist('spikeData', 'var')
    dataStruct.spikeData = spikeData;
end
if exist('saveDir', 'var')
    dataStruct.saveDir = saveDir;
end

% Run analysis
dataStruct.areasToTest = 2:3;
results = criticality_ar_analysis(dataStruct, config);

% Store results in workspace for backward compatibility
mrBr = results.mrBr;
d2 = results.d2;
startS = results.startS;
popActivity = results.popActivity;
optimalBinSize = results.optimalBinSize;
optimalWindowSize = results.optimalWindowSize;
d2StepSize = results.d2StepSize;
d2WindowSize = results.d2WindowSize;

if config.enablePermutations
    d2Permuted = results.d2Permuted;
    mrBrPermuted = results.mrBrPermuted;
    d2PermutedMean = results.d2PermutedMean;
    d2PermutedSEM = results.d2PermutedSEM;
    mrBrPermutedMean = results.mrBrPermutedMean;
    mrBrPermutedSEM = results.mrBrPermutedSEM;
end

fprintf('\n=== Analysis Complete ===\n');

