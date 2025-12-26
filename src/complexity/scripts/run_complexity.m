%%
% Complexity Sliding Window Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture
%
% This script maintains compatibility with the old workflow while using
% the new modular functions. You can still set variables in workspace
% and run this script, or use complexity_analysis() function directly.

% Add paths
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data_prep'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'data_prep'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'analyses'));

% Check if data is already loaded (workspace variables)
% If not, try to load using new function
if ~exist('dataMat', 'var') && ~exist('lfpPerArea', 'var')
    % Try to load data using new function
    if exist('dataType', 'var') && exist('dataSource', 'var')
        fprintf('Loading data using load_sliding_window_data...\n');
        dataStruct = load_sliding_window_data(dataType, dataSource, ...
            'sessionName', sessionName);
        
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
        
        if isfield(dataStruct, 'lfpPerArea')
            lfpPerArea = dataStruct.lfpPerArea;
        end
        if isfield(dataStruct, 'binnedEnvelopes')
            binnedEnvelopes = dataStruct.binnedEnvelopes;
            bands = dataStruct.bands;
            bandBinSizes = dataStruct.bandBinSizes;
        end
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
end

% Set up configuration
if ~exist('slidingWindowSize', 'var')
    slidingWindowSize = 20;  % Default window size
end
if ~exist('stepSize', 'var')
    stepSize = 1;  % Default step size
end
if ~exist('nShuffles', 'var')
    nShuffles = 3;  % Default number of shuffles
end

config = struct();
config.slidingWindowSize = slidingWindowSize;
config.stepSize = stepSize;
config.nShuffles = nShuffles;
config.makePlots = true;

if strcmp(dataSource, 'spikes')
    if ~exist('binSize', 'var')
        error('binSize must be defined for spike data analysis');
    end
    config.binSize = binSize;
elseif strcmp(dataSource, 'lfp')
    if ~exist('lfpLowpassFreq', 'var')
        config.lfpLowpassFreq = 80;
    else
        config.lfpLowpassFreq = lfpLowpassFreq;
    end
end

if exist('saveDir', 'var')
    config.saveDir = saveDir;
end

% Create data structure from workspace variables
dataStruct = struct();
dataStruct.dataType = dataType;
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
if exist('lfpPerArea', 'var')
    dataStruct.lfpPerArea = lfpPerArea;
end
if exist('binnedEnvelopes', 'var')
    dataStruct.binnedEnvelopes = binnedEnvelopes;
    dataStruct.bands = bands;
    dataStruct.bandBinSizes = bandBinSizes;
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
results = complexity_analysis(dataStruct, config);

% Store results in workspace for backward compatibility
lzComplexity = results.lzComplexity;
lzComplexityNormalized = results.lzComplexityNormalized;
lzComplexityNormalizedBernoulli = results.lzComplexityNormalizedBernoulli;
startS = results.startS;

fprintf('\n=== Analysis Complete ===\n');

