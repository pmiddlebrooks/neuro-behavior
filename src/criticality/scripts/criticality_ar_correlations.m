%% 
% Criticality AR Correlations
% Run criticality_ar_analysis without saving or plotting and visualize
% the relationship between d2 window values and behaviorProportion, and
% between d2 and behavior label switching.
%
% Assumes the following workspace variables are set before running:
%   sessionType  - e.g., 'spontaneous', 'reach', etc.
%   sessionName  - recording/session identifier (string)
%
% This script will:
%   1) Load spike data using load_sliding_window_data
%   2) Configure and run criticality_ar_analysis with saveData=false and makePlots=false
%   3) Extract d2 (normalized if available) and behaviorProportion
%   4) Create scatter plots of d2 vs. behaviorProportion for each area
%   5) Optionally remap behavior labels into user-defined groups and compute
%      a behavior label switching rate per window
%   6) Create scatter plots of d2 vs. behavior switching rate for each area

%% Parallelization option
runParallel = 0;

%% Optional behavior label grouping for switching metric
% You can define behaviorLabelGroups in the workspace before running this
% script. It should be a cell array of structs with fields:
%   .name - descriptive name of the new label (string)
%   .ids  - vector of original behavior IDs to group together
%
% Example:
%   behaviorLabelGroups = { ...
%       struct('name', 'Locomotor', 'ids', 5:10), ...
%       struct('name', 'Other',     'ids', [0:2 15:17]) ...
%   };
%
% If behaviorLabelGroups is empty or not defined, the switching metric is
% computed on the original behavior IDs in dataStruct.bhvID.
  behaviorLabelGroups = { ...
      struct('name', 'Locomotor', 'ids', [0:2, 13:15]), ...
      struct('name', 'Groom',     'ids', 5:10) ...
      struct('name', 'Itch',     'ids', [11 12]) ...
      struct('name', 'Rear-Dive-Other',     'ids', [3 4]) ...
  };
  behaviorLabelGroups = {};

%% Add paths (check if directories exist first to avoid warnings)
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

%% Configure data loading options
opts = neuro_behavior_options;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
if strcmp(sessionType, 'reach') || strcmp(sessionType, 'hong')
    opts.collectEnd = [];
end
opts.minFiringRate = .1;
opts.maxFiringRate = 100;

%% Load data
dataSource = 'spikes';
if ~(exist('sessionType', 'var') && exist('sessionName', 'var'))
    error('sessionType and sessionName must be defined in the workspace before running this script.');
end

fprintf('Loading data using load_sliding_window_data...\n');
dataStruct = load_sliding_window_data(sessionType, dataSource, ...
    'sessionName', sessionName, 'opts', opts);

%% Set up configuration (based on run_criticality_ar, but without saving/plotting)
config = struct();
config.slidingWindowSize = 30; % Default window size (seconds)
config.binSize = .05;          % Default bin size (seconds)
config.stepSize = config.slidingWindowSize;           % Step size (seconds); use same as window for fewer windows, or e.g. 0.2 for finer steps
config.minSpikesPerBin = 2.5;
config.minBinsPerWindow = 1000;

% Analysis flags
config.analyzeD2 = true;
config.analyzeMrBr = false;
config.pcaFlag = 0;
config.pcaFirstFlag = 1;  % Use first nDim if 1, last nDim if 0
config.nDim = 4;          % Number of PCA dimensions
config.enablePermutations = true;
config.nShuffles = 20;
config.analyzeModulation = false;
config.makePlots = false; % Disable plotting inside analysis
config.saveData = false;  % Disable saving inside analysis
config.useOptimalBinWindowFunction = false;

% Additional parameters
config.pOrder = 10;
config.critType = 2;
config.normalizeD2 = false;    % Normalize d2 by shuffled d2 values
config.maxSpikesPerBin = 50;  % Maximum spikes per bin for filtering
config.nMinNeurons = 10;      % Minimum number of neurons required per area (no subsampling)
config.includeM2356 = true;   % Include combined M23+M56 area

% Optional list of brain areas (by name) to analyze.
% Example: {'M23', 'M56'}; leave empty to analyze all available areas.
config.brainAreas = [];

% Optional neural subsampling configuration (match typical AR defaults)
config.useSubsampling = false;        % Enable neural subsampling across windows
config.nSubsamples = 20;             % Number of subsampling iterations
config.nNeuronsSubsample = 20;       % Neurons per subsample
config.minNeuronsMultiple = 1.25;    % Multiplier for minimum neuron requirement

% Modulation analysis parameters (unused when analyzeModulation = false)
config.modulationThreshold = 2;
config.modulationBinSize = nan;
config.modulationBaseWindow = [-3, -2];
config.modulationEventWindow = [-0.2, 0.6];
config.modulationPlotFlag = false;

% Behavior configuration for spontaneous sessions (needed for behaviorProportion)
if strcmp(sessionType, 'spontaneous')
    config.behaviorNumeratorIDs = 5:10;
    config.behaviorDenominatorIDs = [config.behaviorNumeratorIDs, 0:2, 15:17];
end

%% Optional parallel pool for area-wise processing
if runParallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        numWorkers = min(3, length(dataStruct.areas));
        parpool('local', numWorkers);
        fprintf('Started parallel pool with %d workers\n', numWorkers);
    else
        fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
    end
end

%% Run analysis (no saving or internal plotting)
results = criticality_ar_analysis(dataStruct, config);

fprintf('\n=== Analysis Complete (results only, no save/plot) ===\n');

%% Extract d2 (normalized if available) and behaviorProportion
if isfield(results, 'd2Normalized') && isfield(results, 'params') && ...
        isfield(results.params, 'normalizeD2') && results.params.normalizeD2
    d2Cell = results.d2Normalized;
    d2Label = 'd2 (normalized)';
else
    d2Cell = results.d2;
    d2Label = 'd2';
end

if ~isfield(results, 'behaviorProportion')
    error('results.behaviorProportion not found. Ensure sessionType is ''spontaneous'' and behavior IDs are configured.');
end

behaviorCell = results.behaviorProportion;
areas = results.areas;
numAreas = numel(areas);

%% Prepare per-area correlation scatter plots: d2 vs behaviorProportion
validAreas = [];
for a = 1:numAreas
    if ~isempty(d2Cell{a}) && ~isempty(behaviorCell{a})
        validAreas(end+1) = a; %#ok<AGROW>
    end
end

if isempty(validAreas)
    error('No areas with both d2 and behaviorProportion data available for correlation.');
end

nValid = numel(validAreas);

figure(945); clf;
set(gcf, 'Name', 'Criticality AR: d2 vs Behavior Proportion', 'NumberTitle', 'off');

    ha = tight_subplot(1, nValid, [0.02 0.02], [0.12 0.08], [0.06 0.03]);

for iArea = 1:nValid
    a = validAreas(iArea);
    d2Vals = d2Cell{a};
    bhvVals = behaviorCell{a};
    
    if isempty(d2Vals) || isempty(bhvVals)
        continue;
    end
    
    validMask = ~isnan(d2Vals) & ~isnan(bhvVals);
    d2Vals = d2Vals(validMask);
    bhvVals = bhvVals(validMask);
    
    if isempty(d2Vals)
        continue;
    end
    
    % Compute Pearson correlation
    if numel(d2Vals) > 1
        cMat = corrcoef(d2Vals, bhvVals);
        rVal = cMat(1, 2);
    else
        rVal = NaN;
    end
    
        axes(ha(iArea));
    scatter(bhvVals, d2Vals, 12, 'filled', 'MarkerFaceAlpha', 0.5);
    grid on;
    xlabel('Behavior Proportion');
    if iArea == 1
        ylabel(d2Label);
    end
    title(sprintf('%s: %s vs BehaviorProportion (r = %.3f, n = %d)', ...
        areas{a}, d2Label, rVal, numel(d2Vals)), 'Interpreter', 'none');
    set(gca, 'YTickLabel', []);
end

fprintf('Generated correlation scatter plot(s) of %s vs behaviorProportion.\n', d2Label);

%% Per-area scatter plots: d2 vs popActivity (mean pop activity per window)
if ~isfield(results, 'popActivityWindows')
    warning('results.popActivityWindows not found; skipping d2 vs popActivity figure.');
else
    popActivityWindows = results.popActivityWindows;
    validAreasPop = [];
    for a = 1:numAreas
        if ~isempty(d2Cell{a}) && ~isempty(popActivityWindows{a})
            validAreasPop(end+1) = a; %#ok<AGROW>
        end
    end
    if ~isempty(validAreasPop)
        nValidPop = numel(validAreasPop);
        figure(947); clf;
        set(gcf, 'Name', 'Criticality AR: d2 vs Pop Activity', 'NumberTitle', 'off');
        haPop = tight_subplot(1, nValidPop, [0.02 0.02], [0.12 0.08], [0.06 0.03]);
        for iArea = 1:nValidPop
            a = validAreasPop(iArea);
            d2Vals = d2Cell{a};
            popVals = popActivityWindows{a};
            validMask = ~isnan(d2Vals) & ~isnan(popVals);
            d2Vals = d2Vals(validMask);
            popVals = popVals(validMask);
            if isempty(d2Vals)
                continue;
            end
            if numel(d2Vals) > 1
                cMat = corrcoef(d2Vals, popVals);
                rVal = cMat(1, 2);
            else
                rVal = NaN;
            end
            axes(haPop(iArea));
            scatter(popVals, d2Vals, 12, 'filled', 'MarkerFaceAlpha', 0.5);
            grid on;
            xlabel('Pop Activity');
            if iArea == 1
                ylabel(d2Label);
            end
            title(sprintf('%s: %s vs PopActivity (r = %.3f, n = %d)', ...
                areas{a}, d2Label, rVal, numel(d2Vals)), 'Interpreter', 'none');
            set(gca, 'YTickLabel', []);
        end
        fprintf('Generated correlation scatter plot(s) of %s vs popActivity.\n', d2Label);
    end
end

%% Compute behavior label switching rate per window (using optional regrouped labels)
if ~isfield(dataStruct, 'bhvID') || isempty(dataStruct.bhvID)
    error('dataStruct.bhvID not found or empty. Behavior time series is required for switching analysis.');
end
if ~isfield(dataStruct, 'fsBhv') || isempty(dataStruct.fsBhv)
    error('dataStruct.fsBhv must be defined in dataStruct for behavior switching analysis.');
end

bhvID = dataStruct.bhvID(:);
bhvID(bhvID == -1) = nan;
fsBhv = dataStruct.fsBhv;
bhvBinSize = 1 / fsBhv;

% Map original behavior IDs to new grouped labels if requested
if ~isempty(behaviorLabelGroups)
    newLabels = nan(size(bhvID));
    for g = 1:numel(behaviorLabelGroups)
        idsThisGroup = behaviorLabelGroups{g}.ids;
        newLabels(ismember(bhvID, idsThisGroup)) = g;
    end
else
    % Use original labels directly (treat each bhvID as its own label)
    newLabels = double(bhvID);
end

% Use first area with non-empty startS as reference for window centers
refAreaIdx = find(~cellfun(@isempty, results.startS), 1);
if isempty(refAreaIdx)
    error('No non-empty startS found in results; cannot compute window-based switching.');
end

centerTimes = results.startS{refAreaIdx};
if isfield(results, 'params') && isfield(results.params, 'slidingWindowSize')
    winSize = results.params.slidingWindowSize;
else
    winSize = config.slidingWindowSize;
end

numWindows = numel(centerTimes);
switchingRate = nan(numWindows, 1);

for w = 1:numWindows
    centerTime = centerTimes(w);
    winStartTime = centerTime - winSize / 2;
    winEndTime = centerTime + winSize / 2;
    
    bhvStartIdx = round(winStartTime / bhvBinSize) + 1;
    bhvEndIdx = round(winEndTime / bhvBinSize);
    bhvStartIdx = max(1, bhvStartIdx);
    bhvEndIdx = min(length(newLabels), bhvEndIdx);
    
    if bhvStartIdx > bhvEndIdx
        continue;
    end
    
    winLabels = newLabels(bhvStartIdx:bhvEndIdx);
    validMask = ~isnan(winLabels);
    winLabels = winLabels(validMask);
    
    if numel(winLabels) < 2
        continue;
    end
    
    labelChanges = winLabels(1:end-1) ~= winLabels(2:end);
    nTransitions = sum(labelChanges);
    winDuration = (bhvEndIdx - bhvStartIdx + 1) * bhvBinSize;
    if winDuration > 0
        switchingRate(w) = nTransitions / winDuration;  % transitions per second
    end
end

%% Per-area scatter plots: d2 vs behavior switching rate
validAreasSwitch = [];
for a = 1:numAreas
    if ~isempty(d2Cell{a})
        validAreasSwitch(end+1) = a; %#ok<AGROW>
    end
end

if isempty(validAreasSwitch)
    error('No areas with d2 data available for d2 vs behavior switching correlation.');
end

nValidSwitch = numel(validAreasSwitch);

figure(946); clf;
set(gcf, 'Name', 'Criticality AR: d2 vs Behavior Switching Rate', 'NumberTitle', 'off');

    haSwitch = tight_subplot(1, nValidSwitch, [0.02 0.02], [0.12 0.08], [0.06 0.03]);

for iArea = 1:nValidSwitch
    a = validAreasSwitch(iArea);
    d2Vals = d2Cell{a};
    
    if isempty(d2Vals) || numel(d2Vals) ~= numel(switchingRate)
        continue;
    end
    
    bhvSwitchVals = switchingRate;
    
    validMask = ~isnan(d2Vals(:)) & ~isnan(bhvSwitchVals(:));
    d2Vals = d2Vals(validMask);
    bhvSwitchVals = bhvSwitchVals(validMask);
    
    if isempty(d2Vals)
        continue;
    end
    
    % Compute Pearson correlation
    if numel(d2Vals) > 1
        cMat = corrcoef(d2Vals, bhvSwitchVals);
        rVal = cMat(1, 2);
    else
        rVal = NaN;
    end
    
        axes(haSwitch(iArea));
    scatter(bhvSwitchVals, d2Vals, 12, 'filled', 'MarkerFaceAlpha', 0.5);
    grid on;
    xlabel('Behavior Switching Rate (transitions/s)');
    ylabel(d2Label);
    title(sprintf('%s: %s vs Behavior Switching (r = %.3f, n = %d)', ...
        areas{a}, d2Label, rVal, numel(d2Vals)), 'Interpreter', 'none');
    set(gca, 'YTickLabel', []);
end

fprintf('Generated correlation scatter plot(s) of %s vs behavior switching rate.\n', d2Label);

