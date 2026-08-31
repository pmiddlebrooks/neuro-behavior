%% Criticality AR Compare - Euclidean d2 vs KL-rate d2 (sandbox)
% Wrapper modeled on run_criticality_ar.m. Computes the current Euclidean
% d2 and the Sooter et al. KL-rate d2 (prox_crit_toolkit) on the same
% windows, then overlays both (plus KL error bars) on one plot.
%
% Workspace variables (from choose_task_and_session.m or similar):
%   sessionType, sessionName, dataSource
%   subjectName - required for spontaneous and interval; omit for reach/schall/hong
%
% KL-rate error bars require MaxLikelihood AR fitting and are slow
% (one extra KL minimization per AR coefficient). Use a larger stepSize
% for a first pass; set config.klErrBars = false to skip uncertainty.

runParallel = 1;

% Set to 1 to load and plot existing results instead of running analysis
loadAndPlot = 0;

basePath = fileparts(mfilename('fullpath'));  % criticality/sandbox
srcPath = fullfile(basePath, '..', '..');     % src
critPath = fullfile(basePath, '..');          % criticality
swDataPrepPath = fullfile(srcPath, 'sliding_window_prep', 'data_prep');
swUtilsPath = fullfile(srcPath, 'sliding_window_prep', 'utils');
analysesPath = fullfile(critPath, 'analyses');
scriptsPath = fullfile(critPath, 'scripts');
manuscriptPath = fullfile(srcPath, 'criticality_manuscript');

opts = neuro_behavior_options;
opts.firingRateCheckTime = 20 * 60;
opts.firingRateCheckTime = [];
opts.collectStart = 0;
opts.collectEnd = 15*60;
% opts.collectEnd = [];
opts.minFiringRate = .25;
opts.maxFiringRate = 150;

subjectNameForLoad = '';
if exist('subjectName', 'var') && ~isempty(subjectName)
    subjectNameForLoad = subjectName;
end

% Single or merged area (e.g. 'M56', 'M23M56', 'M2356'); '' = all loaded areas
brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();

pcaFlag = 0;
sdfFlag = true;
sdfSigmaMs = 50;

if loadAndPlot
    if ~exist('sessionType', 'var')
        error('sessionType must be defined to load and plot results');
    end

    fprintf('Loading data using load_sliding_window_data...\n');
    loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
    dataStruct = load_sliding_window_data(sessionType, 'spikes', loadArgs{:});
    dataStruct = apply_run_ar_brain_area_selection(dataStruct, brainArea, brainAreaCombinations);

    sessionNameForPath = '';
    if exist('sessionName', 'var') && ~isempty(sessionName)
        sessionNameForPath = sessionName;
    end

    filenameSuffix = format_ar_file_suffix(struct( ...
        'pcaFlag', pcaFlag, 'sdfFlag', sdfFlag, 'sdfSigmaMs', sdfSigmaMs));
    filenameSuffix = [filenameSuffix, '_klcompare'];

    resultsPath = create_results_path('criticality_ar_compare', sessionType, sessionNameForPath, ...
        dataStruct.saveDir, 'filenameSuffix', filenameSuffix, 'createDir', false);

    if ~exist(resultsPath, 'file')
        error('Results file not found: %s', resultsPath);
    end

    fprintf('Loading results from: %s\n', resultsPath);
    load(resultsPath, 'results');

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
    if isfield(results.params, 'useLog10D2')
        config.useLog10D2 = results.params.useLog10D2;
    else
        config.useLog10D2 = false;
    end
    if isfield(results.params, 'brainAreas') && ~isempty(results.params.brainAreas)
        config.brainAreas = results.params.brainAreas;
    elseif isfield(results.params, 'areasToTest') && ~isempty(results.params.areasToTest)
        config.brainAreas = results.areas(results.params.areasToTest);
    end
    if isfield(results.params, 'sdfFlag')
        config.sdfFlag = results.params.sdfFlag;
    end
    if isfield(results.params, 'sdfSigmaMs')
        config.sdfSigmaMs = results.params.sdfSigmaMs;
    end
    if isfield(results.params, 'pcaFlag')
        config.pcaFlag = results.params.pcaFlag;
    end
    if isfield(results.params, 'klFitMethod')
        config.klFitMethod = results.params.klFitMethod;
    end
    if isfield(results.params, 'klErrBars')
        config.klErrBars = results.params.klErrBars;
    end
    if isfield(results.params, 'klParallel')
        config.klParallel = results.params.klParallel;
    end

    config.saveDir = dataStruct.saveDir;

    if ~isfield(config, 'plotTimeRange') || isempty(config.plotTimeRange)
        config.plotTimeRange = [];
    end
    config.plotTimeRange = [0 48*60];

    if ~isempty(config.plotTimeRange) && length(config.plotTimeRange) == 2
        timeStart = config.plotTimeRange(1);
        timeEnd = config.plotTimeRange(2);
        fprintf('Filtering results to time range [%.1f, %.1f] s\n', timeStart, timeEnd);

        numAreas = length(results.areas);
        for a = 1:numAreas
            if isempty(results.startS{a})
                continue;
            end
            timeMask = results.startS{a} >= timeStart & results.startS{a} <= timeEnd;
            results.startS{a} = results.startS{a}(timeMask);
            if ~isempty(results.d2{a})
                results.d2{a} = results.d2{a}(timeMask);
            end
            if isfield(results, 'd2Kl') && ~isempty(results.d2Kl{a})
                results.d2Kl{a} = results.d2Kl{a}(timeMask);
            end
            if isfield(results, 'd2KlErr') && ~isempty(results.d2KlErr{a})
                results.d2KlErr{a} = results.d2KlErr{a}(timeMask);
            end
            if isfield(results, 'd2KlExit') && ~isempty(results.d2KlExit{a})
                results.d2KlExit{a} = results.d2KlExit{a}(timeMask);
            end
            if isfield(results, 'popActivityWindows') && ~isempty(results.popActivityWindows{a})
                results.popActivityWindows{a} = results.popActivityWindows{a}(timeMask);
            end
            if isfield(results, 'popActivityFull') && ~isempty(results.popActivityFull{a})
                results.popActivityFull{a} = results.popActivityFull{a}(timeMask);
            end
            if isfield(results, 'behaviorProportion') && ~isempty(results.behaviorProportion{a})
                results.behaviorProportion{a} = results.behaviorProportion{a}(timeMask);
            end
        end
        fprintf('Filtered results: %d areas processed\n', numAreas);
    end

    plotArgs = {};
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        plotArgs = [plotArgs, {'sessionName', dataStruct.sessionName}];
    end
    if isfield(dataStruct, 'dataBaseName') && ~isempty(dataStruct.dataBaseName)
        plotArgs = [plotArgs, {'dataBaseName', dataStruct.dataBaseName}];
    end
    plotConfig = setup_plotting(dataStruct.saveDir, plotArgs{:});

    fprintf('Plotting results...\n');
    criticality_ar_compare_plot(results, plotConfig, config, dataStruct, filenameSuffix);

    fprintf('\n=== Plotting Complete ===\n');
    return;
end

% ============================================================================================================

if exist('sessionType', 'var') && exist('dataSource', 'var')
    fprintf('Loading data using load_sliding_window_data...\n');
    loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
    dataStruct = load_sliding_window_data(sessionType, dataSource, loadArgs{:});
    dataStruct = apply_run_ar_brain_area_selection(dataStruct, brainArea, brainAreaCombinations);
else
    error('sessionType and dataSource must be defined, or data must be pre-loaded in workspace');
end

config = struct();
config.slidingWindowSize = 45;
config.binSize = .04;
config.stepSize = 2;  % KL+error bars are slow; use 0.5 to match run_criticality_ar
config.minSpikesPerBin = 2.5;
config.minBinsPerWindow = 1000;

config.pcaFlag = pcaFlag;
config.pcaFirstFlag = 1;
config.nDim = 4;
config.sdfFlag = sdfFlag;
config.sdfSigmaMs = sdfSigmaMs;
config.makePlots = true;
config.saveData = true;
config.useOptimalBinWindowFunction = false;

config.pOrder = 10;
config.critType = 2;
config.useLog10D2 = false;
config.maxSpikesPerBin = 50;
config.nMinNeurons = 10;

% prox_crit_toolkit / Sooter et al. S2.5
config.klFitMethod = 'MaxLikelihood';  % required for error bars
config.klErrBars = true;
config.klParallel = runParallel;

if strcmp(sessionType, 'spontaneous')
    config.behaviorNumeratorIDs = 5:10;
    config.behaviorDenominatorIDs = [config.behaviorNumeratorIDs, 0:2, 15:17];
end

if ~isempty(brainArea)
    config.brainAreas = {char(brainArea)};
else
    config.brainAreas = [];
end

if runParallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        parpool('local', min(3, feature('numcores')));
        fprintf('Started parallel pool for calc_db gradient error bars\n');
    else
        fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
    end
end

results = criticality_ar_compare_analysis(dataStruct, config);

fprintf('\n=== Compare Analysis Complete ===\n');

function dataStruct = apply_run_ar_brain_area_selection(dataStruct, brainArea, brainAreaCombinations)
% APPLY_RUN_AR_BRAIN_AREA_SELECTION Merge/restrict areas like session_d2_distributions
%
% Variables:
%   dataStruct              - Loaded session struct
%   brainArea               - '' = all; single name or combined name (e.g. M23M56)
%   brainAreaCombinations   - Cell of structs with .name and .areas
%
% Goal:
%   Create combined areas (M23+M56) when requested and set areasToTest.

[dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
    dataStruct, brainArea, brainAreaCombinations);
if ~isempty(brainArea) && ~areaOk
    error('Brain area "%s" not available in this session.', brainArea);
end
end
