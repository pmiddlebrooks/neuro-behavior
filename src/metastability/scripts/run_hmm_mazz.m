%% HMM Mazzucato Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture:
%   - hmm_mazz_analysis.m (analysis and saving)
%   - hmm_mazz_plot.m     (basic plotting)
%
% This script mirrors the style of run_rqa_sliding.m. You can:
%   - Set variables in the workspace and run this script, or
%   - Call hmm_mazz_analysis() directly with a dataStruct/HmmParam.
% Synthetic pipeline test: set hmmMazzTestMode = true below (after addpath). Uses
% hmm_mazz_unit_test_generate with hmmMazzTestSynthParams; data file under
% dropboxMetastabilityData. Delete the .mat there to regenerate after changing params.

% Toggle: set to 1 to load and plot existing results instead of running analysis
loadAndPlot = 0;

% When loadAndPlot == 1, specify how to select saved HMM results:
%   sessionType = 'spontaneous' or 'reach'
%   Optional: config.brainArea = 'M23' | 'M56' | 'DS' | 'VS';
%             binSize, minDur to target specific files (see hmm_load_saved_model)
%             If analysis used nonempty opts.collectEnd, also pass collectStart, collectEnd
%             (seconds) so the loader finds ..._start_XX_end_XX.mat files.
if loadAndPlot
    if ~exist('sessionType', 'var') || isempty(sessionType)
        sessionType = 'spontaneous';
    end

    % Optional selection parameters
    brainArea = 'M23';      % e.g., 'M56'
    binSizeLoad = .002;     % e.g., 0.01
    minDurLoad = .04;       % e.g., 0.04
    % Set these for windowed analyses saved as ..._start_XX_end_XX.mat
    collectStartLoad = [];  % e.g., 0
    collectEndLoad = 120*60;    % e.g., 3600

    loadArgs = {};
    if ~isempty(brainArea)
        loadArgs = [loadArgs, {'brainArea'}, {brainArea}]; %#ok<AGROW>
    end
    if ~isempty(binSizeLoad)
        loadArgs = [loadArgs, {'binSize'}, {binSizeLoad}]; %#ok<AGROW>
    end
    if ~isempty(minDurLoad)
        loadArgs = [loadArgs, {'minDur'}, {minDurLoad}]; %#ok<AGROW>
    end
    if ~isempty(collectStartLoad)
        loadArgs = [loadArgs, {'collectStart'}, {collectStartLoad}]; %#ok<AGROW>
    end
    if ~isempty(collectEndLoad)
        loadArgs = [loadArgs, {'collectEnd'}, {collectEndLoad}]; %#ok<AGROW>
    end

    fprintf('Loading saved HMM model via hmm_load_saved_model...\n');
    hmmRes = hmm_load_saved_model(sessionType, loadArgs{:});

    % Add analyses path for new plotting function
    basePath = fileparts(mfilename('fullpath'));   % metastability/scripts
    analysesPath = fullfile(basePath, '..', 'analyses');
    if exist(analysesPath, 'dir')
        addpath(analysesPath);
    end

    configPlot = struct();
    if exist('brainArea', 'var') && ~isempty(brainArea)
        configPlot.brainArea = brainArea;
    end

    fprintf('Plotting HMM results...\n');
    hmm_mazz_plot(hmmRes, configPlot);
    fprintf('\n=== HMM Plotting Complete ===\n');
    return;
end

% =====================================================================
% Normal analysis flow: run HMM on a single session
% =====================================================================

% Add analyses path so we can call hmm_mazz_analysis / hmm_mazz_plot
paths = get_paths;
basePath = fileparts(mfilename('fullpath'));   % metastability/scripts
srcPath = fullfile(basePath, '..', '..');
analysesPath = fullfile(basePath, '..', 'analyses');
dataPrepPath = fullfile(srcPath, 'data_prep');
if exist(analysesPath, 'dir')
    addpath(analysesPath);
end
if exist(srcPath, 'dir')
    addpath(srcPath);
end
if exist(dataPrepPath, 'dir')
    addpath(dataPrepPath);
end
hmmToolboxPathLocal = fullfile(basePath, '..', 'hmm_mazz_toolbox');
if exist(hmmToolboxPathLocal, 'dir')
    addpath(hmmToolboxPathLocal);
end


% ---------------------------------------------------------------------
% User configuration
% ---------------------------------------------------------------------

% sessionType and sessionName should be set in the workspace before running:

% ---------------------------------------------------------------------
% Build opts and load spikes per area (match sliding-window loaders)
% ---------------------------------------------------------------------

opts = neuro_behavior_options;
opts.minActTime = 0.16;
opts.minFiringRate = .2; % 0.7;
opts.frameSize = 0.001;
opts.firingRateCheckTime = 5 * 60;
opts.maxFiringRate = 100;
opts.collectStart = 0;
opts.collectEnd = 2*60*60;
opts.collectEnd = [];
% When collectEnd is nonempty after load_spike_times, saved .mat names include
% _start_SEC_end_SEC so different time windows from the same session do not overwrite.
opts.removeSome = true;

% HMM fitting uses a conceptual "trial duration" (seconds)
trialDur = 30;

areas = {'M23', 'M56', 'DS', 'VS'};

% Load spike times using the same infrastructure as sliding-window analyses
fprintf('Loading spike times for %s session: %s\n', sessionType, sessionName);
spikeDataStruct = load_spike_times(sessionType, paths, sessionName, opts);

% Update opts with actual collection window used by loader
if isfield(spikeDataStruct, 'collectStart')
    opts.collectStart = spikeDataStruct.collectStart;
end
if isfield(spikeDataStruct, 'collectEnd')
    opts.collectEnd = spikeDataStruct.collectEnd;
end

% Convert spikeDataStruct to [time, neuronId, areaIdx] table expected by hmm_mazz_analysis
spikeTimes = spikeDataStruct.spikeTimes(:);
spikeClusters = spikeDataStruct.spikeClusters(:);
neuronIDs = spikeDataStruct.neuronIDs(:);
neuronAreas = spikeDataStruct.neuronAreas(:);

areaMapping = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
areaIdxPerNeuron = zeros(numel(neuronIDs), 1);
for n = 1:numel(neuronIDs)
    label = neuronAreas{n};
    if isKey(areaMapping, label)
        areaIdxPerNeuron(n) = areaMapping(label);
    else
        areaIdxPerNeuron(n) = 0;
    end
end

[~, loc] = ismember(spikeClusters, neuronIDs);
areaIdxPerSpike = zeros(size(spikeClusters));
validMask = loc > 0;
areaIdxPerSpike(validMask) = areaIdxPerNeuron(loc(validMask));

% Force double so concatenation never promotes the matrix to int32 (loses sub-ms times).
spikeData = [double(spikeTimes), double(spikeClusters), double(areaIdxPerSpike)];

% Build neuron id lists per area (area indices encoded in column 3)
idM23 = unique(spikeData(spikeData(:, 3) == 1, 2));
idM56 = unique(spikeData(spikeData(:, 3) == 2, 2));
idDS = unique(spikeData(spikeData(:, 3) == 3, 2));
idVS = unique(spikeData(spikeData(:, 3) == 4, 2));
idList = {idM23, idM56, idDS, idVS};

fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', ...
    numel(idM23), numel(idM56), numel(idDS), numel(idVS));

% ---------------------------------------------------------------------
% Assemble dataStruct and HmmParam, then run analysis
% ---------------------------------------------------------------------

dataStruct = struct();
dataStruct.sessionType = sessionType;
dataStruct.paths = paths;
dataStruct.opts = opts;
dataStruct.areas = areas;
dataStruct.idList = idList;
dataStruct.spikeData = spikeData;
dataStruct.trialDur = trialDur;
dataStruct.sessionName = sessionName;
% For reach sessions, hmm_mazz_analysis will infer saving location from paths and sessionName

% Unified analysis + HMM fit configuration (single struct)
HmmParam = struct();
HmmParam.modelSelectionMethod = 'XVAL';  % XVAL, BIC, AIC
HmmParam.areasToTest = 1:4;     % Area IDs to analyze: M23=1, M56=2, DS=3, VS=4
HmmParam.minNumNeurons = 15;
HmmParam.saveData = true;
HmmParam.useParallel = true;
HmmParam.useOptimalBinSize = false; % true: per-area bin sweep; false: use fixed BinSize below
HmmParam.maxAllowedMultiSpikeProportion = 0.0; % Used when useOptimalBinSize=true
HmmParam.hiddenMin = 3;
HmmParam.hiddenMax = 32;
HmmParam.AdjustT = 0.0;        % Interval to skip at trial start (s)
HmmParam.BinSize = 0.002;      % Fixed bin size when useOptimalBinSize=false
HmmParam.MinDur = 0.04;        % Minimum admissible state duration in decoding (s)
HmmParam.MinP = 0.8;           % Minimum posterior probability for state assignment
HmmParam.NumSteps = 10;        % Number of independent EM runs at fixed parameters
HmmParam.NumRuns = 30;         % Maximum iterations per EM run
HmmParam.singleSeqXval.K = 10;  % Cross-validation folds

fprintf('\nRunning hmm_mazz_analysis...\n');
results = hmm_mazz_analysis(dataStruct, HmmParam);

fprintf('\n=== HMM Analysis Complete ===\n');

%% Penalized -2LL / criterion vs number of states (toolbox-style curve + chosen K marker)
% Uses results.llCriterionAcrossStates plus hmm_data-derived SEM ribbons when available.
makeCriterionLlPlots = true;
if makeCriterionLlPlots
    fprintf('Plotting penalized LL / model-selection criterion curves...\n');
    configLlCurve = struct();
    % Restrict with e.g. configLlCurve.areasToPlot = {'M56','M23'}; ([] = every nonempty area)
    hmm_mazz_plot_penalized_ll_curve(results, configLlCurve);
    fprintf('=== Penalized LL plots complete ===\n');
end

%% Optional immediate plotting from in-memory results
makePlots = true;
checkArea = 'M56';
configPlot = struct('brainArea', checkArea);
if makePlots
    fprintf('Creating basic HMM plots from in-memory results...\n');
    hmm_mazz_plot(results, configPlot);
    fprintf('=== HMM Plotting Complete ===\n');
end

%% Optional HMM state + behavior ethogram (spontaneous: behavior_labels*.csv via sessionName)

    %% Load a model
brainArea = 'VS';

    % Optional selection parameters
    binSizeLoad = .002;     % e.g., 0.01
    minDurLoad = .04;       % e.g., 0.04
    % Set these for windowed analyses saved as ..._start_XX_end_XX.mat
    collectStartLoad = [];  % e.g., 0
    collectEndLoad = 120*60;    % e.g., 3600

    loadArgs = {};
        loadArgs = [loadArgs, {'brainArea'}, {brainArea}]; %#ok<AGROW>
        loadArgs = [loadArgs, {'binSize'}, {binSizeLoad}]; %#ok<AGROW>
        loadArgs = [loadArgs, {'minDur'}, {minDurLoad}]; %#ok<AGROW>
        loadArgs = [loadArgs, {'collectStart'}, {collectStartLoad}]; %#ok<AGROW>
        loadArgs = [loadArgs, {'collectEnd'}, {collectEndLoad}]; %#ok<AGROW>

    fprintf('Loading saved HMM model via hmm_load_saved_model...\n');
    hmmRes = hmm_load_saved_model(sessionType, loadArgs{:});

% Ethogram plot
makeEthogramPlots = true;
ethogramArea = brainArea;
if makeEthogramPlots
    if ~exist('sessionType', 'var') || isempty(sessionType) || ~strcmpi(sessionType, 'spontaneous')
        fprintf('Skipping HMM + behavior ethogram (only wired for spontaneous + behavior_labels CSV).\n');
    else
        fprintf('Creating HMM + behavior ethogram...\n');
        configEthogram = struct();
        configEthogram.brainArea = ethogramArea;
        configEthogram.paths = paths;
        configEthogram.sessionName = sessionName;
        configEthogram.exportEthogramCsvDir = fullfile(paths.dropPath, 'temp_data');
        collectStartSec = 0;
        if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
            collectStartSec = opts.collectStart;
        end
        if isfield(opts, 'collectEnd') && ~isempty(opts.collectEnd)
            configEthogram.timeRangeSec = [0, opts.collectEnd - collectStartSec];
        else
            areaMapEth = containers.Map({'M23', 'M56', 'DS', 'VS'}, {1, 2, 3, 4});
            ethIdx = areaMapEth(ethogramArea);
            hmmSingleEth = results.hmm_results{ethIdx};
            if ~isempty(hmmSingleEth) && isfield(hmmSingleEth, 'continuous_results') ...
                    && isfield(hmmSingleEth.continuous_results, 'totalTime')
                configEthogram.timeRangeSec = [0, hmmSingleEth.continuous_results.totalTime];
            else
                configEthogram.timeRangeSec = [0, 600];
            end
        end
        hmm_mazz_plot_states_ethogram(hmmRes, configEthogram);
        fprintf('=== HMM ethogram plotting complete ===\n');
    end
end

%% Optional debug figure to verify state-selection behavior
makeModelSelectionDebugPlots = true;
if makeModelSelectionDebugPlots
    fprintf('Creating HMM model-selection debug plots...\n');
    debugConfig = struct();
    debugConfig.showDiffElbow = true;
    debugConfig.areasToPlot = {'M23','M56','DS','VS'};
    hmm_mazz_debug_model_selection_plot(results, debugConfig);
    fprintf('=== HMM Model-Selection Debug Plots Complete ===\n');
end

