%% 
% Criticality AR Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture
%
% This script maintains compatibility with the old workflow while using
% the new modular functions.

% Want to parallelize the area-wise analysis?
runParallel = 0;

% Set to 1 to load and plot existing results instead of running analysis
loadAndPlot = 0;
% When loadAndPlot = 1, you can optionally specify a time range to plot:
%   config.plotTimeRange = [startTime, endTime];  % in seconds
%   Example: config.plotTimeRange = [100, 500];  % Plot from 100s to 500s
%   If not specified or empty, all data will be plotted

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
% opts.collectEnd = 60*60;
opts.collectEnd = [];
if strcmp(sessionType, 'reach') || strcmp(sessionType, 'hong')
opts.collectEnd = [];
end
opts.minFiringRate = .1;
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
    
    % Add saveDir from dataStruct (needed for plotting)
    config.saveDir = dataStruct.saveDir;
    % Optional: if PNG export crashes, use lighter plotting (defaults in plot fn):
    % config.plotResolution = 100; config.maxPlotPoints = 500;
    % config.useSoftwareRenderer = true; config.saveEps = false;
    
    % Allow user to specify time range for plotting
    % Example: config.plotTimeRange = [100, 500];  % Plot from 100s to 500s
    if ~isfield(config, 'plotTimeRange') || isempty(config.plotTimeRange)
        config.plotTimeRange = [];  % Empty means plot all data
    end
          config.plotTimeRange = [0 48*60];  % Empty means plot all data
  
    % Filter results by time range if specified
    if ~isempty(config.plotTimeRange) && length(config.plotTimeRange) == 2
        timeStart = config.plotTimeRange(1);
        timeEnd = config.plotTimeRange(2);
        fprintf('Filtering results to time range [%.1f, %.1f] s\n', timeStart, timeEnd);
        
        % Filter each area's data
        numAreas = length(results.areas);
        for a = 1:numAreas
            if ~isempty(results.startS{a})
                % Find indices within time range
                timeMask = results.startS{a} >= timeStart & results.startS{a} <= timeEnd;
                
                % Filter all time-series data for this area
                results.startS{a} = results.startS{a}(timeMask);
                
                if ~isempty(results.d2{a})
                    results.d2{a} = results.d2{a}(timeMask);
                end
                
                if isfield(results, 'd2Normalized') && ~isempty(results.d2Normalized{a})
                    results.d2Normalized{a} = results.d2Normalized{a}(timeMask);
                end
                
                % Filter subsample matrices, keeping rows aligned with d2 / startS
                if isfield(results, 'd2Subsamples') && ~isempty(results.d2Subsamples{a})
                    results.d2Subsamples{a} = results.d2Subsamples{a}(timeMask, :);
                end
                if isfield(results, 'd2NormalizedSubsamples') && ~isempty(results.d2NormalizedSubsamples{a})
                    results.d2NormalizedSubsamples{a} = results.d2NormalizedSubsamples{a}(timeMask, :);
                end
                
                if ~isempty(results.mrBr{a})
                    results.mrBr{a} = results.mrBr{a}(timeMask);
                end
                
                if isfield(results, 'popActivityWindows') && ~isempty(results.popActivityWindows{a})
                    results.popActivityWindows{a} = results.popActivityWindows{a}(timeMask);
                end
                
                if isfield(results, 'popActivityFull') && ~isempty(results.popActivityFull{a})
                    results.popActivityFull{a} = results.popActivityFull{a}(timeMask);
                end
                
                % Filter permutation data if present
                if isfield(results, 'd2Permuted') && ~isempty(results.d2Permuted{a})
                    results.d2Permuted{a} = results.d2Permuted{a}(timeMask, :);
                end
                
                if isfield(results, 'mrBrPermuted') && ~isempty(results.mrBrPermuted{a})
                    results.mrBrPermuted{a} = results.mrBrPermuted{a}(timeMask, :);
                end
                
                if isfield(results, 'd2PermutedMean') && ~isempty(results.d2PermutedMean{a})
                    results.d2PermutedMean{a} = results.d2PermutedMean{a}(timeMask);
                end
                
                if isfield(results, 'd2PermutedSEM') && ~isempty(results.d2PermutedSEM{a})
                    results.d2PermutedSEM{a} = results.d2PermutedSEM{a}(timeMask);
                end
                
                if isfield(results, 'mrBrPermutedMean') && ~isempty(results.mrBrPermutedMean{a})
                    results.mrBrPermutedMean{a} = results.mrBrPermutedMean{a}(timeMask);
                end
                
                if isfield(results, 'mrBrPermutedSEM') && ~isempty(results.mrBrPermutedSEM{a})
                    results.mrBrPermutedSEM{a} = results.mrBrPermutedSEM{a}(timeMask);
                end
                
                if isfield(results, 'behaviorProportion') && ~isempty(results.behaviorProportion{a})
                    results.behaviorProportion{a} = results.behaviorProportion{a}(timeMask);
                end
            end
        end
        fprintf('Filtered results: %d areas processed\n', numAreas);
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



% ============================================================================================================

% Normal analysis flow


% Try to load data using new function
if exist('sessionType', 'var') && exist('dataSource', 'var')
    fprintf('Loading data using load_sliding_window_data...\n');
    dataStruct = load_sliding_window_data(sessionType, dataSource, ...
        'sessionName', sessionName, 'opts', opts);
else
    error('sessionType and dataSource must be defined, or data must be pre-loaded in workspace');
end

% Set up configuration from workspace variables
% (These should be set before running this script)config = struct();
config.slidingWindowSize = 10; % Default window size
config.binSize = .05; % Default bin size
config.stepSize = .2; % Default step size
config.minSpikesPerBin = 2.5;
config.minBinsPerWindow = 1000;

% Analysis flags (set these before running)
config.analyzeD2 = true;
config.analyzeMrBr = false;
config.pcaFlag = 0;
config.pcaFirstFlag = 1;  % Use first nDim if 1, last nDim if 0
config.nDim = 4;  % Number of PCA dimensions
config.enablePermutations = true;
config.nShuffles = 20;
config.analyzeModulation = false;
config.makePlots = true;
config.saveData = true;  % Set to false to skip saving results
% Optional: lighter plotting to avoid export crash (plotResolution, maxPlotPoints, useSoftwareRenderer, saveEps)
config.useOptimalBinWindowFunction = false;

% Additional parameters
config.pOrder = 10;
config.critType = 2;
config.normalizeD2 = true;  % Normalize d2 by shuffled d2 values
config.maxSpikesPerBin = 50;  % Maximum spikes per bin for filtering
config.nMinNeurons = 10;  % Minimum number of neurons required per area (no subsampling)
config.includeM2356 = true;  % Set to true to include combined M23+M56 area

if strcmp(sessionType, 'spontaneous')
    config.behaviorNumeratorIDs = 5:10;
    config.behaviorDenominatorIDs = [config.behaviorNumeratorIDs, 0:2, 15:17];
end

% Optional list of brain areas (by name) to analyze.
% Example: {'M23', 'M56'}; leave empty to analyze all available areas.
config.brainAreas = {'M23', 'M56'};
config.brainAreas = [];

% Optional neural subsampling configuration
% When useSubsampling is true:
%   - nSubsamples controls how many random neuron subsets are analyzed
%   - nNeuronsSubsample is the number of neurons per subset
%   - Minimum neurons required per area becomes round(nNeuronsSubsample * minNeuronsMultiple)
config.useSubsampling = true;       % Enable neural subsampling across windows
config.nSubsamples = 20;             % Number of subsampling iterations
config.nNeuronsSubsample = 20;       % Neurons per subsample
config.minNeuronsMultiple = 1.25;     % Multiplier for minimum neuron requirement

% Modulation analysis parameters (used if analyzeModulation = true)
config.modulationThreshold = 2;  % Standard deviations for modulation detection
config.modulationBinSize = nan;  % Bin size for modulation analysis
config.modulationBaseWindow = [-3, -2];  % Baseline time range [min max] in seconds relative to reach onset
config.modulationEventWindow = [-0.2, 0.6];  % Event time range [min max] in seconds relative to reach onset
config.modulationPlotFlag = false;  % Set to true to generate modulation analysis plots
% saveDir will be obtained from dataStruct.saveDir in the analysis function
% (set by load_sliding_window_data)

% Run analysis
% Optional parallel pool for area-wise processing (similar to LZC sliding analysis)
if runParallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        NumWorkers = min(3, length(dataStruct.areas));
        parpool('local', NumWorkers);
        fprintf('Started parallel pool with %d workers\n', NumWorkers);
    else
        fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
    end
end

results = criticality_ar_analysis(dataStruct, config);


fprintf('\n=== Analysis Complete ===\n');

