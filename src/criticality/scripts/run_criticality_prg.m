%%
% Criticality PRG Analysis - Wrapper Script
%
% Momentum-space phenomenological renormalization group (PRG) kurtosis analysis
% following Cambrainha et al. 2025 PRX Life / Bradde & Bialek 2017.
%
% Uses non-overlapping 30 s windows and 50 ms spike bins (paper defaults).
% Real-space paired-correlation coarse-graining and scaling exponents are not run.

% Set to 1 to load and plot saved results instead of running analysis
loadAndPlot = 0;

basePath = fileparts(mfilename('fullpath'));
srcPath = fullfile(basePath, '..', '..');

sessionDataPrepPath = fullfile(srcPath, 'session_prep', 'data_prep');
sessionUtilsPath = fullfile(srcPath, 'session_prep', 'utils');
dataPrepPath = fullfile(srcPath, 'data_prep');
analysesPath = fullfile(basePath, '..', 'analyses');
criticalityPath = fullfile(basePath, '..');

if exist(sessionDataPrepPath, 'dir')
    addpath(sessionDataPrepPath);
end
if exist(sessionUtilsPath, 'dir')
    addpath(sessionUtilsPath);
end
if exist(dataPrepPath, 'dir')
    addpath(dataPrepPath);
end
if exist(analysesPath, 'dir')
    addpath(analysesPath);
end
if exist(criticalityPath, 'dir')
    addpath(criticalityPath);
end

opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.collectEnd = 40*60;
if exist('sessionType', 'var') && (strcmp(sessionType, 'reach') || strcmp(sessionType, 'hong'))
    opts.collectEnd = [];
end
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;

if loadAndPlot
    if ~exist('sessionType', 'var')
        error('sessionType must be defined to load and plot results');
    end

    loadArgs = {'sessionName', sessionName, 'opts', opts};
    if exist('subjectName', 'var') && ~isempty(subjectName)
        loadArgs = [loadArgs, {'subjectName', subjectName}];
    end
    dataStruct = load_session_data(sessionType, 'spikes', loadArgs{:});

    sessionNameForPath = '';
    if exist('sessionName', 'var') && ~isempty(sessionName)
        sessionNameForPath = sessionName;
    end

    resultsPath = create_results_path('criticality_prg', sessionType, sessionNameForPath, ...
        dataStruct.saveDir, 'createDir', false);

    if ~exist(resultsPath, 'file')
        error('Results file not found: %s', resultsPath);
    end

    load(resultsPath, 'results');
    config = struct();
    config.saveDir = dataStruct.saveDir;
    if isfield(results, 'params')
        config.blockWindowSize = results.params.blockWindowSize;
        config.binSize = results.params.binSize;
        config.finalCutoffDivisor = results.params.finalCutoffDivisor;
    end

    plotArgs = {};
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        plotArgs = [plotArgs, {'sessionName', dataStruct.sessionName}];
    end
    plotConfig = setup_plotting(dataStruct.saveDir, plotArgs{:});
    criticality_prg_plot(results, plotConfig, config, dataStruct);
    fprintf('\n=== Plotting Complete ===\n');
    return;
end

if exist('sessionType', 'var') && exist('dataSource', 'var')
    fprintf('Loading data using load_session_data...\n');
    loadArgs = {'sessionName', sessionName, 'opts', opts};
    if exist('subjectName', 'var') && ~isempty(subjectName)
        loadArgs = [loadArgs, {'subjectName', subjectName}];
    end
    dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});
else
    error('sessionType and dataSource must be defined, or data must be pre-loaded in workspace');
end

config = struct();

% Cambrainha et al. 2025 defaults: 50 ms bins, 30 s non-overlapping windows
config.blockWindowSize = 30;
config.binSize = 0.05;
config.cvThreshold = 5;
config.cutoffDivisors = [1, 2, 4, 8, 16];
config.finalCutoffDivisor = 16;

config.enableSurrogates = true;
config.nSurrogates = 1;
config.makePlots = true;
config.saveData = false;
config.nMinNeurons = 20;
config.includeM2356 = false;

% Optional: restrict areas, e.g. config.brainAreas = {'M23', 'M56'};
config.brainAreas = {};

results = criticality_prg_analysis(dataStruct, config);

fprintf('\n=== PRG Analysis Complete ===\n');
