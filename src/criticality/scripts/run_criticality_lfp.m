%%
% Criticality LFP Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture

% Add paths
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data_prep'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'data_prep'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'analyses'));

% Check if data is already loaded
if ~exist('binnedEnvelopes', 'var')
    if exist('dataType', 'var') && exist('dataSource', 'var')
        fprintf('Loading data using load_sliding_window_data...\n');
        dataStruct = load_sliding_window_data(dataType, dataSource, ...
            'sessionName', sessionName);
        
        % Convert structure to workspace variables
        binnedEnvelopes = dataStruct.binnedEnvelopes;
        bands = dataStruct.bands;
        bandBinSizes = dataStruct.bandBinSizes;
        areas = dataStruct.areas;
        opts = dataStruct.opts;
        saveDir = dataStruct.saveDir;
        dataType = dataStruct.dataType;
        dataSource = dataStruct.dataSource;
        areasToTest = dataStruct.areasToTest;
        
        if isfield(dataStruct, 'lfpPerArea')
            lfpPerArea = dataStruct.lfpPerArea;
        end
        if isfield(dataStruct, 'lfpBinSize')
            lfpBinSize = dataStruct.lfpBinSize;
        end
        if isfield(dataStruct, 'sessionName')
            sessionName = dataStruct.sessionName;
        end
        if isfield(dataStruct, 'dataBaseName')
            dataBaseName = dataStruct.dataBaseName;
        end
    else
        error('dataType and dataSource must be defined, or data must be pre-loaded');
    end
end

% Set up configuration
if ~exist('slidingWindowSize', 'var')
    slidingWindowSize = 10;
end

config = struct();
config.slidingWindowSize = slidingWindowSize;

% Analysis flags
if exist('analyzeD2', 'var')
    config.analyzeD2 = analyzeD2;
else
    config.analyzeD2 = true;
end

if exist('analyzeDFA', 'var')
    config.analyzeDFA = analyzeDFA;
else
    config.analyzeDFA = true;
end

if exist('makePlots', 'var')
    config.makePlots = makePlots;
else
    config.makePlots = true;
end

if exist('saveDir', 'var')
    config.saveDir = saveDir;
end

% Create data structure
dataStruct = struct();
dataStruct.dataType = dataType;
dataStruct.dataSource = 'lfp';
dataStruct.areas = areas;
dataStruct.opts = opts;
dataStruct.binnedEnvelopes = binnedEnvelopes;
dataStruct.bands = bands;
dataStruct.bandBinSizes = bandBinSizes;
if exist('lfpPerArea', 'var')
    dataStruct.lfpPerArea = lfpPerArea;
end
if exist('lfpBinSize', 'var')
    dataStruct.lfpBinSize = lfpBinSize;
end
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
results = criticality_lfp_analysis(dataStruct, config);

% Store results in workspace
d2 = results.d2;
dfa = results.dfa;
startS = results.startS;
stepSize = results.stepSize;
d2WindowSize = results.d2WindowSize;
dfaEnvWinSize = results.dfaEnvWinSize;
bandBinSizes = results.bandBinSizes;

if isfield(results, 'd2Lfp')
    d2Lfp = results.d2Lfp;
    dfaLfp = results.dfaLfp;
    lfpBinSize = results.lfpBinSize;
end

fprintf('\n=== Analysis Complete ===\n');

