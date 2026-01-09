%%
% Criticality Avalanche Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture

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

% Check if data is already loaded
if ~exist('dataMat', 'var')
    if exist('dataType', 'var') && exist('dataSource', 'var')
        fprintf('Loading data using load_sliding_window_data...\n');
        dataStruct = load_sliding_window_data(dataType, dataSource, ...
            'sessionName', sessionName, 'opts', opts);
        
        % Convert structure to workspace variables
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
    else
        error('dataType and dataSource must be defined, or data must be pre-loaded');
    end
end

% Set up configuration
if ~exist('slidingWindowSize', 'var')
    slidingWindowSize = 180;
end
if ~exist('avStepSize', 'var')
    avStepSize = 20;
end

config = struct();
config.slidingWindowSize = slidingWindowSize;
config.avStepSize = avStepSize;

% Analysis flags
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

if exist('makePlots', 'var')
    config.makePlots = makePlots;
else
    config.makePlots = true;
end

if exist('saveDir', 'var')
    config.saveDir = saveDir;
end

% M2356 combined area option
if exist('includeM2356', 'var')
    config.includeM2356 = includeM2356;
else
    config.includeM2356 = false;  % Set to true to include combined M23+M56 area
end

% Create data structure
dataStruct = struct();
dataStruct.dataType = dataType;
dataStruct.dataSource = 'spikes';
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
if exist('saveDir', 'var')
    dataStruct.saveDir = saveDir;
end

% Run analysis
results = criticality_av_analysis(dataStruct, config);

% Store results in workspace
dcc = results.dcc;
kappa = results.kappa;
decades = results.decades;
tau = results.tau;
alpha = results.alpha;
paramSD = results.paramSD;
startS = results.startS;
optimalBinSize = results.optimalBinSize;
optimalWindowSize = results.optimalWindowSize;

if config.enablePermutations
    dccPermuted = results.dccPermuted;
    kappaPermuted = results.kappaPermuted;
    decadesPermuted = results.decadesPermuted;
    tauPermuted = results.tauPermuted;
    alphaPermuted = results.alphaPermuted;
    paramSDPermuted = results.paramSDPermuted;
end

fprintf('\n=== Analysis Complete ===\n');

