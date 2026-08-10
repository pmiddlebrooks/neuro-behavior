%%
% d2ForEric - Sliding-window d2 sweeps for all reach sessions
%
% For every reach session, runs criticality_ar_analysis over a grid of
% d2 window sizes and spike bin sizes (fixed step size), then packs each
% run into a struct and saves to d2Results.mat.
%
% Variables (configure in this section):
%   windowsToTest   - d2 sliding-window lengths (s)
%   binSizesToTest  - Spike bin widths (s)
%   stepSize        - Window step / hop (s); timestamps are window centers
%   brainArea       - Single or merged area (e.g. 'M2356')
%   collectStart    - Analysis window start (s from session onset)
%   collectEnd      - Analysis window end (s); [] = session end
%   saveFile        - Output .mat path (default: dropPath/.../d2Results.mat)
%
% Goal:
%   Provide Eric with per-session sliding-window d2 time series across the
%   requested window/bin grid, with timestamps (window centers), d2, binSize,
%   stepSize, and windowSize stored for each run.

%% Paths
setup_criticality_manuscript_paths('d2ForEric');
paths = get_paths();

% Configuration
windowsToTest = [30 45 60];
binSizesToTest = [.025 .04 .07];
stepSize = 0.05;

dataSource = 'spikes';
collectStart = 0;
collectEnd = [];
brainArea = 'M2356';
brainAreaCombinations = default_manuscript_brain_area_combinations();

useLog10D2 = true;
useSubsampling = true;
enablePermutations = false;
nShuffles = 0;
nMinNeurons = 45;

saveFile = fullfile(paths.dropPath, 'criticality_manuscript', 'd2Results.mat');
saveIncrementally = true;

% Session list
reachSessions = reach_session_list();
nSessions = numel(reachSessions);
nWindows = numel(windowsToTest);
nBins = numel(binSizesToTest);
nRunsPerSession = nWindows * nBins;

fprintf('\n=== d2ForEric: sliding-window d2 for reach sessions ===\n');
fprintf('Sessions: %d\n', nSessions);
fprintf('Windows (s): %s\n', mat2str(windowsToTest));
fprintf('Bin sizes (s): %s\n', mat2str(binSizesToTest));
fprintf('Step size (s): %.4f\n', stepSize);
fprintf('Brain area: %s\n', brainArea);
fprintf('Runs per session: %d (total planned: %d)\n', ...
  nRunsPerSession, nSessions * nRunsPerSession);

% Load options
loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = [];
loadOpts.collectStart = collectStart;
loadOpts.collectEnd = collectEnd;
loadOpts.minFiringRate = 0.05;
loadOpts.maxFiringRate = 150;

% Shared AR analysis config (window / bin filled in the loop)
analysisConfig = struct();
analysisConfig.stepSize = stepSize;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = enablePermutations;
analysisConfig.nShuffles = nShuffles;
analysisConfig.normalizeD2 = false;
analysisConfig.useLog10D2 = useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 100;
analysisConfig.maxSpikesPerBin = 100;
analysisConfig.nMinNeurons = nMinNeurons;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = 25;
analysisConfig.nNeuronsSubsample = 45;
analysisConfig.minNeuronsMultiple = 1.1;

% Output container
d2Results = struct();
d2Results.windowsToTest = windowsToTest;
d2Results.binSizesToTest = binSizesToTest;
d2Results.stepSize = stepSize;
d2Results.brainArea = brainArea;
d2Results.collectStart = collectStart;
d2Results.collectEnd = collectEnd;
d2Results.useLog10D2 = useLog10D2;
d2Results.sessions = repmat(struct( ...
  'sessionName', '', ...
  'success', false, ...
  'skipReason', '', ...
  'runs', []), nSessions, 1);

emptyRun = struct( ...
  'windowSize', [], ...
  'binSize', [], ...
  'stepSize', [], ...
  'timestamps', [], ...
  'd2', [], ...
  'areaName', '', ...
  'success', false, ...
  'skipReason', '');

% Main loop: sessions x windows x bin sizes
for iSess = 1:nSessions
  sessionName = reachSessions{iSess};
  fprintf('\n%s\n', repmat('=', 1, 80));
  fprintf('Session %d/%d: %s\n', iSess, nSessions, sessionName);

  d2Results.sessions(iSess).sessionName = sessionName;
  d2Results.sessions(iSess).success = false;
  d2Results.sessions(iSess).skipReason = '';
  d2Results.sessions(iSess).runs = repmat(emptyRun, nRunsPerSession, 1);

  try
    loadArgs = build_session_load_args('reach', sessionName, loadOpts, '');
    dataStruct = load_session_data('reach', dataSource, loadArgs{:});
    [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
      dataStruct, brainArea, brainAreaCombinations);
    if ~areaOk
      reason = sprintf('Brain area "%s" not available', brainArea);
      fprintf('  %s; skipping session.\n', reason);
      d2Results.sessions(iSess).skipReason = reason;
      if saveIncrementally
        save_d2_for_eric(saveFile, d2Results);
      end
      continue;
    end

    runIdx = 0;
    sessionHadSuccess = false;
    for iWin = 1:nWindows
      for iBin = 1:nBins
        runIdx = runIdx + 1;
        windowSize = windowsToTest(iWin);
        binSize = binSizesToTest(iBin);

        d2Results.sessions(iSess).runs(runIdx).windowSize = windowSize;
        d2Results.sessions(iSess).runs(runIdx).binSize = binSize;
        d2Results.sessions(iSess).runs(runIdx).stepSize = stepSize;
        d2Results.sessions(iSess).runs(runIdx).timestamps = [];
        d2Results.sessions(iSess).runs(runIdx).d2 = [];
        d2Results.sessions(iSess).runs(runIdx).areaName = brainArea;
        d2Results.sessions(iSess).runs(runIdx).success = false;
        d2Results.sessions(iSess).runs(runIdx).skipReason = '';

        fprintf('\n  Run %d/%d: window=%.0f s, bin=%.3f s, step=%.4f s\n', ...
          runIdx, nRunsPerSession, windowSize, binSize, stepSize);

        try
          runConfig = analysisConfig;
          runConfig.slidingWindowSize = windowSize;
          runConfig.binSize = binSize;
          runConfig.stepSize = stepSize;

          arResults = criticality_ar_analysis(dataStruct, runConfig);
          arResults = filter_ar_results_to_brain_area(arResults, brainArea);
          if isempty(arResults.areas)
            reason = sprintf('No AR results for area "%s"', brainArea);
            fprintf('    %s\n', reason);
            d2Results.sessions(iSess).runs(runIdx).skipReason = reason;
            continue;
          end

          [timestamps, d2Vals] = extract_d2_run_vectors(arResults, useLog10D2);
          if isempty(timestamps)
            reason = 'Empty d2 / timestamp vectors';
            fprintf('    %s\n', reason);
            d2Results.sessions(iSess).runs(runIdx).skipReason = reason;
            continue;
          end

          d2Results.sessions(iSess).runs(runIdx).timestamps = timestamps;
          d2Results.sessions(iSess).runs(runIdx).d2 = d2Vals;
          d2Results.sessions(iSess).runs(runIdx).success = true;
          sessionHadSuccess = true;
          fprintf('    Stored %d windows (t = %.1f .. %.1f s)\n', ...
            numel(timestamps), timestamps(1), timestamps(end));
        catch MERun
          if is_skippable_session_analysis_error(MERun)
            fprintf('    Skipping run: %s\n', MERun.message);
            d2Results.sessions(iSess).runs(runIdx).skipReason = MERun.message;
          else
            fprintf('    Error: %s\n', MERun.message);
            d2Results.sessions(iSess).runs(runIdx).skipReason = MERun.message;
            for st = 1:min(5, numel(MERun.stack))
              fprintf('      %s (line %d)\n', MERun.stack(st).name, MERun.stack(st).line);
            end
          end
        end
      end
    end

    d2Results.sessions(iSess).success = sessionHadSuccess;
    if ~sessionHadSuccess
      d2Results.sessions(iSess).skipReason = 'No successful runs';
    end
  catch ME
    fprintf('  Session error: %s\n', ME.message);
    d2Results.sessions(iSess).skipReason = ME.message;
    for st = 1:min(5, numel(ME.stack))
      fprintf('    %s (line %d)\n', ME.stack(st).name, ME.stack(st).line);
    end
  end

  if saveIncrementally
    save_d2_for_eric(saveFile, d2Results);
  end
end

%% Final save
save_d2_for_eric(saveFile, d2Results);
fprintf('\n=== Done. Saved: %s ===\n', saveFile);

%% Local functions

function save_d2_for_eric(saveFile, d2Results)
% SAVE_D2_FOR_ERIC - Write results struct to .mat as d2ForEric
%
% Variables:
%   saveFile  - Absolute path for output .mat
%   d2Results - Packed results struct (script workspace name)
%
% Goal:
%   Persist incremental / final batch results for Eric. Saved variable name
%   remains d2ForEric so it does not collide with the script name.

saveDir = fileparts(saveFile);
if ~isempty(saveDir) && ~exist(saveDir, 'dir')
  mkdir(saveDir);
end
savePayload = struct('d2ForEric', d2Results);
save(saveFile, '-struct', 'savePayload', '-v7.3');
fprintf('  Saved: %s\n', saveFile);
end

function [timestamps, d2Vals] = extract_d2_run_vectors(results, useLog10D2)
% EXTRACT_D2_RUN_VECTORS - Window-center times and d2 for one AR result
%
% Variables:
%   results     - criticality_ar_analysis output (single area preferred)
%   useLog10D2  - If true, store log10(d2); non-positive -> NaN
%
% Goal:
%   Return aligned timestamp (window center, s) and d2 vectors.
%
% Returns:
%   timestamps - Column vector of window centers (absolute session time, s)
%   d2Vals     - Column vector of d2 values

timestamps = [];
d2Vals = [];
if ~isfield(results, 'd2') || isempty(results.d2) || isempty(results.d2{1})
  return;
end
if ~isfield(results, 'startS') || isempty(results.startS) || isempty(results.startS{1})
  return;
end

d2Vals = results.d2{1}(:);
timestamps = results.startS{1}(:);
n = min(numel(d2Vals), numel(timestamps));
d2Vals = d2Vals(1:n);
timestamps = timestamps(1:n);

if useLog10D2
  d2Vals = log10_safe_numeric(d2Vals);
end
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with non-positive values set to NaN

y = nan(size(x));
ok = isfinite(x) & x > 0;
y(ok) = log10(x(ok));
end

function results = filter_ar_results_to_brain_area(results, brainArea)
% FILTER_AR_RESULTS_TO_BRAIN_AREA - Keep one area in AR results struct
%
% Variables:
%   results   - criticality_ar_analysis output
%   brainArea - Area name to retain
%
% Goal:
%   Restrict multi-area AR results to a single brain area.

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end

areaNames = results.areas;
if ischar(areaNames)
  areaNames = {areaNames};
elseif isstring(areaNames)
  areaNames = cellstr(areaNames(:));
end

areaIdx = find(strcmp(areaNames, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end

cellFields = {'d2', 'd2Normalized', 'startS', 'd2Permuted', 'mrBrPermuted', ...
  'd2PermutedMean', 'd2PermutedSEM', 'popActivityWindows', 'popActivityFull', ...
  'd2Subsamples', 'd2NormalizedSubsamples'};
results.areas = areaNames(areaIdx);
for f = 1:numel(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
if isfield(results, 'binSize') && numel(results.binSize) >= areaIdx
  results.binSize = results.binSize(areaIdx);
end
if isfield(results, 'slidingWindowSize') && numel(results.slidingWindowSize) >= areaIdx
  results.slidingWindowSize = results.slidingWindowSize(areaIdx);
end
if isfield(results, 'nNeurons') && numel(results.nNeurons) >= areaIdx
  results.nNeurons = results.nNeurons(areaIdx);
end
end

function tf = is_skippable_session_analysis_error(ME)
% IS_SKIPPABLE_SESSION_ANALYSIS_ERROR - Expected per-session / per-run skips

tf = contains(ME.message, 'No valid areas to process') ...
  || contains(ME.message, 'insufficient neurons') ...
  || contains(ME.message, 'No valid windows found');
end
