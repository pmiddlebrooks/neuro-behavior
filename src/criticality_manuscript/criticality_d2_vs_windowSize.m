%%
% Criticality: d2 vs window size (Manuscript)
%
% Sweeps window durations in windowsToTest for three example sessions
% (one spontaneous, one interval, one reach). At each size W, places
% non-overlapping tiled windows of length W within the collect range,
% computes population d2 and CV per tile, then averages across valid tiles.
%
% Variables (configure in this section):
%   exampleSessions  - 3x1 struct array with fields:
%                      .sessionType, .sessionName, .subjectName ('' for reach),
%                      .displayLabel
%   dataSource       - 'spikes' (required for this analysis)
%   collectStart     - Analysis window start (s)
%   collectEnd       - Analysis window end (s); [] = session end
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' = all
%   brainAreaCombinations  - Merged areas from default_manuscript_brain_area_combinations
%   windowsToTest    - Vector of window durations (s)
%   binSizeManual    - Fixed bin width (s) across window sizes
%   pOrder, critType - AR / d2 parameters
%   meanSubtract     - If true, subtract mean pop activity within each window
%   useLog10D2       - Log y-axis for mean d2 when all plotted means are positive
%   enablePermutations - If true, circularly permute neurons over the full
%                        collect range (once per shuffle), then re-tile; avoids
%                        per-window circshifts for speed
%   nShuffles        - Number of whole-session circular permutations
%   makePlots        - Plot mean d2 / mean CV (data + shuffled if enabled)
%   saveFigure       - Export PNG to dropPath/criticality_manuscript
%
% Coefficient of variation (CV): for each popActivity trace,
% CV = nanstd(x) / |nanmean(x)|. Mean CV at each W is nanmean across tiles.
%
% Goal:
%   Characterize how mean d2 and mean pop-activity CV depend on analysis
%   window length across task types, using the same tiling logic so
%   spontaneous / interval / reach are directly comparable (1x3 figure).

%% Paths
setup_criticality_manuscript_paths('criticality_d2_vs_windowSize');
paths = get_paths();

%% Configuration — three example sessions (one per condition)
exampleSessions(1) = struct( ...
  'sessionType', 'spontaneous', ...
  'subjectName', 'ag25290', ...
  'sessionName', 'ag112321_1', ...
  'displayLabel', 'spontaneous');
exampleSessions(2) = struct( ...
  'sessionType', 'interval', ...
  'subjectName', 'ey9166', ...
  'sessionName', 'ey9166_2026_04_03', ...
  'displayLabel', 'interval');
exampleSessions(3) = struct( ...
  'sessionType', 'reach', ...
  'subjectName', '', ...
  'sessionName', 'Y16_23-Dec-2025 16_07_49_NeuroBeh', ...
  'displayLabel', 'reach');

dataSource = 'spikes';
collectStart = 0;
collectEnd = [];

brainArea = 'M2356';
brainAreaCombinations = default_manuscript_brain_area_combinations();

windowsToTest = 2:1:45;   % seconds
binSizeManual = 0.025;      % seconds (fixed across window sizes)
pOrder = 10;
critType = 2;
meanSubtract = false;
useLog10D2 = false;
enablePermutations = true;
nShuffles = 10;
makePlots = true;
saveFigure = false;

fprintf('\n=== criticality_d2_vs_windowSize ===\n');
fprintf('Windows (s): %s\n', mat2str(windowsToTest([1, end]), 3));
fprintf('binSize = %.3f s; pOrder = %d; brainArea = %s\n', ...
  binSizeManual, pOrder, brainArea);
if enablePermutations
  fprintf('Circular permutations: %d (whole-session neuron circshift)\n', nShuffles);
else
  fprintf('Circular permutations: off\n');
end

validate_example_sessions(exampleSessions);

%% Analysis — one example session per task type
numExamples = numel(exampleSessions);
numW = numel(windowsToTest);
exampleResults = repmat(struct(), numExamples, 1);

for e = 1:numExamples
  ex = exampleSessions(e);
  fprintf('\n--- Example %d/%d [%s]: %s ---\n', e, numExamples, ex.sessionType, ex.sessionName);

  opts = build_example_load_opts(collectStart, collectEnd);
  subjectNameForLoad = ex.subjectName;
  loadArgs = build_session_load_args(ex.sessionType, ex.sessionName, opts, subjectNameForLoad);
  dataStruct = load_sliding_window_data(ex.sessionType, dataSource, loadArgs{:});

  [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
    dataStruct, brainArea, brainAreaCombinations);
  if ~areaOk
    warning('criticality_d2_vs_windowSize:MissingArea', ...
      'Brain area "%s" not available for %s; skipping.', brainArea, ex.sessionName);
    continue;
  end

  areas = dataStruct.areas;
  numAreas = numel(areas);
  if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areasToTest = dataStruct.areasToTest;
  else
    areasToTest = 1:numAreas;
  end
  a = areasToTest(1);

  if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart') ...
      && isfield(dataStruct.spikeData, 'collectEnd') ...
      && ~isempty(dataStruct.spikeData.collectEnd)
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
  elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
      && ~isempty(dataStruct.opts.collectEnd)
    timeRange = [dataStruct.opts.collectStart, dataStruct.opts.collectEnd];
  else
    timeRange = [0, max(dataStruct.spikeTimes)];
  end
  fprintf('  Collect / analysis range: [%.1f, %.1f] s (%.1f min); area %s\n', ...
    timeRange(1), timeRange(2), (timeRange(2) - timeRange(1)) / 60, areas{a});

  neuronIDs = dataStruct.idLabel{a};
  aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
    neuronIDs, timeRange, binSizeManual);
  binSizeSec = binSizeManual;

  [meanD2, semD2, nValid, meanCv, semCv, nCvValid] = sweep_tiled_d2_cv( ...
    aDataMat, windowsToTest, binSizeSec, pOrder, critType, meanSubtract);

  for iW = 1:numW
    fprintf('  W = %5.1f s: nValid = %4d, mean d2 = %.4g, mean CV = %.3g (n_CV = %d)\n', ...
      windowsToTest(iW), nValid(iW), meanD2(iW), meanCv(iW), nCvValid(iW));
  end

  meanD2Perm = nan(1, numW);
  semD2Perm = nan(1, numW);
  meanCvPerm = nan(1, numW);
  semCvPerm = nan(1, numW);
  d2PermPerShuffle = [];
  cvPermPerShuffle = [];
  if enablePermutations && nShuffles > 0
    fprintf('  Running %d whole-session circular permutations...\n', nShuffles);
    ticPerm = tic;
    d2PermPerShuffle = nan(nShuffles, numW);
    cvPermPerShuffle = nan(nShuffles, numW);
    for s = 1:nShuffles
      permutedDataMat = circular_permute_neurons(aDataMat);
      [meanD2Shuf, ~, ~, meanCvShuf, ~, ~] = sweep_tiled_d2_cv( ...
        permutedDataMat, windowsToTest, binSizeSec, pOrder, critType, meanSubtract);
      d2PermPerShuffle(s, :) = meanD2Shuf;
      cvPermPerShuffle(s, :) = meanCvShuf;
    end
    meanD2Perm = nanmean(d2PermPerShuffle, 1);
    meanCvPerm = nanmean(cvPermPerShuffle, 1);
    nShufValidD2 = sum(~isnan(d2PermPerShuffle), 1);
    nShufValidCv = sum(~isnan(cvPermPerShuffle), 1);
    for iW = 1:numW
      if nShufValidD2(iW) > 1
        semD2Perm(iW) = nanstd(d2PermPerShuffle(:, iW)) / sqrt(nShufValidD2(iW));
      end
      if nShufValidCv(iW) > 1
        semCvPerm(iW) = nanstd(cvPermPerShuffle(:, iW)) / sqrt(nShufValidCv(iW));
      end
    end
    fprintf('  Permutations done in %.1f s\n', toc(ticPerm));
    for iW = 1:numW
      fprintf(['  W = %5.1f s: mean shuffled d2 = %.4g +/- %.4g ', ...
        '(SEM over %d shuffles); mean shuffled CV = %.3g +/- %.3g ', ...
        '(SEM over %d shuffles)\n'], ...
        windowsToTest(iW), meanD2Perm(iW), semD2Perm(iW), nShufValidD2(iW), ...
        meanCvPerm(iW), semCvPerm(iW), nShufValidCv(iW));
    end
  end

  exampleResults(e).example = ex;
  exampleResults(e).areaName = areas{a};
  exampleResults(e).timeRange = timeRange;
  exampleResults(e).binSize = binSizeSec;
  exampleResults(e).meanD2 = meanD2;
  exampleResults(e).semD2 = semD2;
  exampleResults(e).nValid = nValid;
  exampleResults(e).meanCv = meanCv;
  exampleResults(e).semCv = semCv;
  exampleResults(e).nCvValid = nCvValid;
  exampleResults(e).meanD2Permuted = meanD2Perm;
  exampleResults(e).semD2Permuted = semD2Perm;
  exampleResults(e).d2PermutedMeanPerShuffle = d2PermPerShuffle;
  exampleResults(e).meanCvPermuted = meanCvPerm;
  exampleResults(e).semCvPermuted = semCvPerm;
  exampleResults(e).cvPermutedMeanPerShuffle = cvPermPerShuffle;
end

% Pack results
results = struct();
results.exampleSessions = exampleSessions;
results.windowsToTest = windowsToTest;
results.binSizeManual = binSizeManual;
results.pOrder = pOrder;
results.critType = critType;
results.meanSubtract = meanSubtract;
results.brainArea = brainArea;
results.enablePermutations = enablePermutations;
results.nShuffles = nShuffles;
results.exampleResults = exampleResults;

% Plot — 1x3: spontaneous | interval | reach (shared y-axes)
if makePlots
  saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
  if saveFigure && ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  areaTag = matlab.lang.makeValidName(char(string(brainArea)));
  if isempty(areaTag)
    areaTag = 'allAreas';
  end

  plotOrder = {'spontaneous', 'interval', 'reach'};
  [d2YLim, cvYLim] = shared_d2_cv_ylims(exampleResults, plotOrder, enablePermutations);

  figMain = figure('Color', 'w', 'Name', 'd2 vs window size');
  set(figMain, 'WindowState', 'maximized');
  tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
  for k = 1:numel(plotOrder)
    e = find(arrayfun(@(r) isfield(r, 'example') && ~isempty(r.example) ...
      && strcmpi(r.example.sessionType, plotOrder{k}), exampleResults), 1);
    nexttile;
    if isempty(e) || ~isfield(exampleResults(e), 'meanD2') || isempty(exampleResults(e).meanD2)
      title(sprintf('%s (no data)', plotOrder{k}), 'Interpreter', 'none');
      continue;
    end
    hold on;
    er = exampleResults(e);
    meanD2 = er.meanD2;
    semD2 = er.semD2;
    meanCv = er.meanCv;
    semCvPlot = er.semCv;
    semCvPlot(isnan(semCvPlot)) = 0;

    yyaxis left
    semD2Plot = semD2;
    semD2Plot(isnan(semD2Plot)) = 0;
    errorbar(windowsToTest, meanD2, semD2Plot, '-o', 'LineWidth', 1.5, ...
      'CapSize', 4, 'Color', [0 0.45 0.74], 'DisplayName', 'd2 data');
    if enablePermutations && ~isempty(er.meanD2Permuted) && any(isfinite(er.meanD2Permuted))
      semD2PermPlot = er.semD2Permuted;
      semD2PermPlot(isnan(semD2PermPlot)) = 0;
      errorbar(windowsToTest, er.meanD2Permuted, semD2PermPlot, '--s', 'LineWidth', 1.4, ...
        'CapSize', 4, 'Color', [0.45 0.45 0.45], 'DisplayName', 'd2 shuffled');
    end
    if useLog10D2 && all(meanD2(~isnan(meanD2)) > 0)
      set(gca, 'YScale', 'log');
    end
    if ~isempty(d2YLim)
      ylim(d2YLim);
    end
    if k == 1
      ylabel('mean d2');
    end

    yyaxis right
    errorbar(windowsToTest, meanCv, semCvPlot, '-s', 'LineWidth', 1.5, ...
      'CapSize', 4, 'Color', [0.85 0.33 0.1], 'DisplayName', 'CV data');
    if enablePermutations && ~isempty(er.meanCvPermuted) && any(isfinite(er.meanCvPermuted))
      semCvPermPlot = er.semCvPermuted;
      semCvPermPlot(isnan(semCvPermPlot)) = 0;
      errorbar(windowsToTest, er.meanCvPermuted, semCvPermPlot, '--d', 'LineWidth', 1.4, ...
        'CapSize', 4, 'Color', [0.65 0.45 0.35], 'DisplayName', 'CV shuffled');
    end
    if ~isempty(cvYLim)
      ylim(cvYLim);
    end
    if k == numel(plotOrder)
      ylabel('mean CV');
    end
    set(gca, 'YColor', [0.85 0.33 0.1]);

    yyaxis left
    set(gca, 'YColor', [0 0.45 0.74]);
    grid on;
    xlabel('Window size (s)');
    title(sprintf('%s: %s (%s, bin %.3f s)', er.example.displayLabel, ...
      er.example.sessionName, er.areaName, er.binSize), 'Interpreter', 'none');
    legend('Location', 'best');
  end
  if enablePermutations
    titleSuffix = sprintf('non-overlapping tiles; %d circular shuffles', nShuffles);
  else
    titleSuffix = 'non-overlapping tiles';
  end
  sgtitle(tl, sprintf('d2 vs window size across tasks (%s; %s)', ...
    brainArea, titleSuffix), 'Interpreter', 'none', 'FontSize', 12);

  if saveFigure
    if enablePermutations
      shuffleTag = '_shuffle';
    else
      shuffleTag = '';
    end
    outPng = fullfile(saveDir, sprintf( ...
      'criticality_d2_vs_windowSize_%s%s.png', areaTag, shuffleTag));
    exportgraphics(figMain, outPng, 'Resolution', 300);
    fprintf('Saved figure: %s\n', outPng);
  end
end

fprintf('\n=== criticality_d2_vs_windowSize: done ===\n');

%% Local functions

function validate_example_sessions(exampleSessions)
% VALIDATE_EXAMPLE_SESSIONS - Require spontaneous, interval, and reach examples

requiredTypes = {'spontaneous', 'interval', 'reach'};
if numel(exampleSessions) ~= 3
  error('exampleSessions must contain exactly three entries.');
end
foundTypes = {exampleSessions.sessionType};
for t = 1:numel(requiredTypes)
  if ~any(strcmpi(foundTypes, requiredTypes{t}))
    error('exampleSessions must include one %s session.', requiredTypes{t});
  end
end
for e = 1:numel(exampleSessions)
  if isempty(exampleSessions(e).sessionName)
    error('exampleSessions(%d).sessionName is required.', e);
  end
  if any(strcmpi(exampleSessions(e).sessionType, {'spontaneous', 'interval'})) ...
      && isempty(exampleSessions(e).subjectName)
    error('exampleSessions(%d).subjectName is required for %s sessions.', ...
      e, exampleSessions(e).sessionType);
  end
end
end

function opts = build_example_load_opts(collectStart, collectEnd)
% BUILD_EXAMPLE_LOAD_OPTS - Loader opts for example sessions
%
% Variables:
%   collectStart - Analysis window start (s)
%   collectEnd   - Analysis window end (s); [] = full session
%
% Goal:
%   Apply the same collectStart/collectEnd to spontaneous, interval, and reach.

opts = neuro_behavior_options();
opts.frameSize = 0.001;
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.1;
opts.maxFiringRate = 150;
end

function [d2YLim, cvYLim] = shared_d2_cv_ylims(exampleResults, plotOrder, enablePermutations)
% SHARED_D2_CV_YLIMS - Common y-limits for d2 and CV across task panels
%
% Variables:
%   exampleResults     - Per-example analysis structs
%   plotOrder          - Task types in panel order
%   enablePermutations - Include shuffled mean +/- SEM in range
%
% Goal:
%   Return [ymin ymax] for left (d2) and right (CV) axes spanning all panels.

d2Vals = [];
cvVals = [];
for k = 1:numel(plotOrder)
  e = find(arrayfun(@(r) isfield(r, 'example') && ~isempty(r.example) ...
    && strcmpi(r.example.sessionType, plotOrder{k}), exampleResults), 1);
  if isempty(e) || ~isfield(exampleResults(e), 'meanD2') || isempty(exampleResults(e).meanD2)
    continue;
  end
  er = exampleResults(e);
  semD2 = er.semD2;
  semD2(isnan(semD2)) = 0;
  d2Vals = [d2Vals, er.meanD2 - semD2, er.meanD2 + semD2]; %#ok<AGROW>
  semCv = er.semCv;
  semCv(isnan(semCv)) = 0;
  cvVals = [cvVals, er.meanCv - semCv, er.meanCv + semCv]; %#ok<AGROW>
  if enablePermutations
    if ~isempty(er.meanD2Permuted) && any(isfinite(er.meanD2Permuted))
      semD2Perm = er.semD2Permuted;
      semD2Perm(isnan(semD2Perm)) = 0;
      d2Vals = [d2Vals, er.meanD2Permuted - semD2Perm, ...
        er.meanD2Permuted + semD2Perm]; %#ok<AGROW>
    end
    if ~isempty(er.meanCvPermuted) && any(isfinite(er.meanCvPermuted))
      semCvPerm = er.semCvPermuted;
      semCvPerm(isnan(semCvPerm)) = 0;
      cvVals = [cvVals, er.meanCvPermuted - semCvPerm, ...
        er.meanCvPermuted + semCvPerm]; %#ok<AGROW>
    end
  end
end

d2YLim = padded_ylim(d2Vals);
cvYLim = padded_ylim(cvVals);
end

function yLim = padded_ylim(vals)
% PADDED_YLIM - [ymin ymax] with small padding; [] if no finite values

yLim = [];
vals = vals(isfinite(vals));
if isempty(vals)
  return;
end
yMin = min(vals);
yMax = max(vals);
if yMax > yMin
  pad = 0.05 * (yMax - yMin);
else
  pad = max(0.05 * abs(yMax), 1e-6);
end
yLim = [yMin - pad, yMax + pad];
end

function [meanD2, semD2, nValid, meanCv, semCv, nCvValid] = sweep_tiled_d2_cv( ...
  aDataMat, windowsToTest, binSizeSec, pOrder, critType, meanSubtract)
% SWEEP_TILED_D2_CV - Mean d2 and CV across non-overlapping tiles per window size
%
% Variables:
%   aDataMat       - Binned spikes [timeBins x neurons]
%   windowsToTest  - Window durations (s)
%   binSizeSec     - Bin width (s)
%   pOrder, critType, meanSubtract - Analysis options
%
% Goal:
%   For each W, tile the session with non-overlapping windows and summarize
%   d2 and pop-activity CV across tiles.

numW = numel(windowsToTest);
meanD2 = nan(1, numW);
semD2 = nan(1, numW);
nValid = zeros(1, numW);
meanCv = nan(1, numW);
semCv = nan(1, numW);
nCvValid = zeros(1, numW);

for iW = 1:numW
  [d2PerWindow, cvPerWindow] = tiled_window_d2_cv( ...
    aDataMat, windowsToTest(iW), binSizeSec, pOrder, critType, meanSubtract);

  validMask = ~isnan(d2PerWindow);
  nValid(iW) = sum(validMask);
  if nValid(iW) > 0
    meanD2(iW) = nanmean(d2PerWindow(validMask));
    if nValid(iW) > 1
      semD2(iW) = nanstd(d2PerWindow(validMask)) / sqrt(nValid(iW));
    end
  end

  cvMask = ~isnan(cvPerWindow);
  nCvValid(iW) = sum(cvMask);
  if nCvValid(iW) > 0
    meanCv(iW) = nanmean(cvPerWindow(cvMask));
    if nCvValid(iW) > 1
      semCv(iW) = nanstd(cvPerWindow(cvMask)) / sqrt(nCvValid(iW));
    end
  end
end
end

function [d2PerWindow, cvPerWindow] = tiled_window_d2_cv( ...
  aDataMat, winSize, binSizeSec, pOrder, critType, meanSubtract)
% TILED_WINDOW_D2_CV - Per-tile d2 and CV for one window size
%
% Variables:
%   aDataMat     - Binned spikes [timeBins x neurons]
%   winSize      - Window duration (s)
%   binSizeSec   - Bin width (s)
%   pOrder, critType, meanSubtract - AR / preprocessing options
%
% Goal:
%   Place non-overlapping tiles of length winSize and return per-tile metrics.

numTimePoints = size(aDataMat, 1);
winSamples = round(winSize / binSizeSec);
if winSamples < 1
  winSamples = 1;
end
stepSamples = winSamples;
numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
d2PerWindow = nan(1, max(numWindows, 0));
cvPerWindow = nan(1, max(numWindows, 0));
if numWindows < 1
  return;
end

for w = 1:numWindows
  [startIdx, endIdx] = calculate_window_indices( ...
    w, numTimePoints, winSamples, stepSamples, binSizeSec);
  if startIdx < 1 || endIdx > numTimePoints || endIdx <= startIdx
    continue;
  end

  wPopActivity = sum(aDataMat(startIdx:endIdx, :), 2);
  if meanSubtract
    wPopActivity = wPopActivity - nanmean(wPopActivity);
  end
  d2PerWindow(w) = compute_d2_from_pop_activity(wPopActivity, pOrder, critType);
  cvPerWindow(w) = population_trace_cv(wPopActivity);
end
end

function permutedDataMat = circular_permute_neurons(aDataMat)
% CIRCULAR_PERMUTE_NEURONS - Independent circshift of each neuron over all bins
%
% Variables:
%   aDataMat - Binned spikes [timeBins x neurons]
%
% Goal:
%   Destroy cross-neuron synchrony while preserving each neuron's marginal
%   statistics by circularly shifting each column independently over the
%   full collect range (one shift per neuron).

numTimePoints = size(aDataMat, 1);
numNeurons = size(aDataMat, 2);
permutedDataMat = zeros(size(aDataMat));
for n = 1:numNeurons
  shiftAmount = randi(numTimePoints);
  permutedDataMat(:, n) = circshift(aDataMat(:, n), shiftAmount);
end
end

function d2Val = compute_d2_from_pop_activity(wPopActivity, pOrder, critType)
% COMPUTE_D2_FROM_POP_ACTIVITY - Scalar d2 for one pop-activity window
%
% Variables:
%   wPopActivity - Population spike-count vector [timeBins x 1]
%   pOrder       - AR order for myYuleWalker3
%   critType     - Criticality type for getFixedPointDistance2
%
% Goal:
%   Return scalar d2 for one window; NaN on failure.

d2Val = nan;
if isempty(wPopActivity)
  return;
end
v = double(wPopActivity(:));
if numel(v) <= pOrder
  return;
end
try
  [varphi, ~] = myYuleWalker3(v, pOrder);
  d2Val = getFixedPointDistance2(pOrder, critType, varphi);
catch
  d2Val = nan;
end
end

function cvProp = population_trace_cv(wPopActivity)
% POPULATION_TRACE_CV - Coefficient of variation for a popActivity trace
%
% Variables:
%   wPopActivity - Population spike-count vector over time bins
%
% Goal:
%   Return CV = nanstd(x) / |nanmean(x)| (dimensionless). NaN if the mean
%   is undefined or too close to zero (|mean| < 1e-12).

cvProp = nan;
if isempty(wPopActivity)
  return;
end
x = double(wPopActivity(:));
meanVal = nanmean(x);
if isnan(meanVal) || abs(meanVal) < 1e-12
  return;
end
cvProp = nanstd(x) / abs(meanVal);
end
