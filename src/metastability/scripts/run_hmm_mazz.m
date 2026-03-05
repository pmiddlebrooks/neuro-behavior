%% HMM Mazzucato Analysis - Wrapper Script
% Backward-compatible wrapper that uses the new function-based architecture:
%   - hmm_mazz_analysis.m (analysis and saving)
%   - hmm_mazz_plot.m     (basic plotting)
%
% This script mirrors the style of run_rqa_sliding.m. You can:
%   - Set variables in the workspace and run this script, or
%   - Call hmm_mazz_analysis() directly with a dataStruct/config.

% Toggle: set to 1 to load and plot existing results instead of running analysis
loadAndPlot = 0;

% When loadAndPlot == 1, specify how to select saved HMM results:
%   sessionType = 'spontaneous' or 'reach'
%   Optional: config.brainArea = 'M23' | 'M56' | 'DS' | 'VS';
%             binSize, minDur to target specific files (see hmm_load_saved_model)
if loadAndPlot
    if ~exist('sessionType', 'var') || isempty(sessionType)
        sessionType = 'spontaneous';
    end

    % Optional selection parameters
    brainArea = [];      % e.g., 'M56'
    binSizeLoad = [];    % e.g., 0.01
    minDurLoad = [];     % e.g., 0.04

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
srcPath = fullfile(basePath, '..', '..');      %#ok<NASGU>
analysesPath = fullfile(basePath, '..', 'analyses');
if exist(analysesPath, 'dir')
    addpath(analysesPath);
end

% ---------------------------------------------------------------------
% User configuration
% ---------------------------------------------------------------------

% sessionType and sessionName should be set in the workspace before running:

% ---------------------------------------------------------------------
% Build opts and load spikes per area
% ---------------------------------------------------------------------

opts = neuro_behavior_options;
opts.minActTime = 0.16;
opts.minFiringRate = 0.1;
opts.frameSize = 0.001;
opts.firingRateCheckTime = 5 * 60;

% HMM fitting uses a conceptual "trial duration" (seconds)
trialDur = 30;

areas = {'M23', 'M56', 'DS', 'VS'};

switch lower(sessionType)
    case 'spontaneous'
        % Spontaneous data
        opts.collectStart = 0;
        opts.collectEnd = [];

        freeDataPath = fullfile(paths.freeDataPath, ['animal_', animal], sessionNrn, 'recording1');
        opts.dataPath = freeDataPath;

        fprintf('Loading Nat spikes via spike_times_per_area for %s (%s)...\n', ...
            animal, sessionNrn);
        spikeData = spike_times_per_area(opts);

    case 'reach'
        % Reach data
        reachDataFile = fullfile(paths.reachDataPath, sessionName);
        opts.collectStart = 0;
        opts.collectEnd = [];
        opts.dataPath = reachDataFile;

        fprintf('Loading Reach spikes via spike_times_per_area_reach for %s...\n', ...
            sessionName);
        spikeData = spike_times_per_area_reach(opts);
end

% Build neuron id lists per area (area indices encoded in column 3)
idM23 = unique(spikeData(spikeData(:, 3) == 1, 2));
idM56 = unique(spikeData(spikeData(:, 3) == 2, 2));
idDS = unique(spikeData(spikeData(:, 3) == 3, 2));
idVS = unique(spikeData(spikeData(:, 3) == 4, 2));
idList = {idM23, idM56, idDS, idVS};

fprintf('%d M23\n%d M56\n%d DS\n%d VS\n', ...
    numel(idM23), numel(idM56), numel(idDS), numel(idVS));

% ---------------------------------------------------------------------
% Assemble dataStruct and config, then run analysis
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
if strcmpi(sessionType, 'reach')
    dataStruct.reachDataFile = reachDataFile;
end

if ~exist('config', 'var') || isempty(config)
    config = struct();
end
if ~isfield(config, 'modelSelectionMethod')
    config.modelSelectionMethod = 'XVAL';
end
if ~isfield(config, 'minNumNeurons')
    config.minNumNeurons = 15;
end
if ~isfield(config, 'saveData')
    config.saveData = true;
end
if ~isfield(config, 'useParallel')
    config.useParallel = true;
end

% Set HMM parameter configuration (can be overridden by caller)
if ~isfield(config, 'HmmParam') || isempty(config.HmmParam)
    hmmParam = struct();
    hmmParam.AdjustT = 0.0;        % Interval to skip at trial start (s)
    hmmParam.BinSize = 0.01;       % Markov chain time step (s)
    hmmParam.MinDur = 0.04;        % Minimum admissible state duration in decoding (s)
    hmmParam.MinP = 0.8;           % Minimum posterior probability for state assignment
    hmmParam.NumSteps = 8;         % Number of independent EM runs at fixed parameters
    hmmParam.NumRuns = 33;         % Maximum iterations per EM run
    hmmParam.singleSeqXval.K = 5;  % Cross-validation folds
    config.HmmParam = hmmParam;
end

fprintf('\nRunning hmm_mazz_analysis...\n');
results = hmm_mazz_analysis(dataStruct, config);

fprintf('\n=== HMM Analysis Complete ===\n');

% Optional immediate plotting from in-memory results
makePlots = true;
if makePlots
    fprintf('Creating basic HMM plots from in-memory results...\n');
    hmm_mazz_plot(results, struct());
    fprintf('=== HMM Plotting Complete ===\n');
end

