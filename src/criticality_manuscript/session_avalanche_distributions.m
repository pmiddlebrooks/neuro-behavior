%%
% Session Avalanche Size and Duration Distributions (Manuscript)
%
% For one session and brain area, runs the same avalanche pipeline as
% criticality_av_across_tasks.m (single full collect window) and plots
% complementary CCDFs on log-log axes with power-law fits over the viable
% cutoff range (tau for size, alpha for duration).
%
% Variables (configure in this section):
%   sessionType        - 'spontaneous', 'interval', 'reach', 'schall'
%   sessionName        - Session identifier
%   subjectName        - Required for spontaneous/interval; '' for reach
%   dataSource         - 'spikes' or 'lfp'
%   collectStart       - Window start (seconds from session onset)
%   collectEnd         - Window end (seconds)
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' uses all valid areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   powerLawFitMethod  - 'clauset', 'plfit2023', or 'hybrid'
%   avalancheDetectionMode - 'fixedBinMedian' or 'meanIsiZero'
%   clausetPlfitPath   - Path to .../Power-Law-Fit-Distribution-MATLAB-main/MATLAB Code
%   plfit2023Path      - Path to folder containing plfit2023.m
%   runClausetPlpva    - If true and method is 'clauset', run plpva (slow)
%   saveFigure         - Export PNG/EPS to dropPath/criticality_manuscript
%   useSubsampling     - If true, pool avalanches from neuron subsamples in the window
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings
%   splitExcitatoryInhibitory - If true, run combined (E+I), excitatory, and inhibitory;
%                               also plots summary of tau and alpha fits
%   widthCutoff        - Peak-to-trough width threshold in ms (narrow <= cutoff = I)
%
% Goal:
%   Visualize avalanche size and duration distributions for one session
%   with log-scaled x-axes and fitted slopes within the fitted cutoff range.

%% Configuration
% sessionType = 'interval';
% subjectName = 'ey9166';
% sessionName = 'ey9166_2026_04_03';
% dataSource = 'spikes';

collectStart = 0;
collectEnd = 45 * 60;
windowDurationSec = collectEnd - collectStart;

brainArea = 'M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
saveFigure = false;

% Power-law fitting: 'clauset', 'plfit2023', 'hybrid' = plfit2023 xmax, then Clauset plfit on x <= xmax
powerLawFitMethod = 'hybrid';
runClausetPlpva = false;

gofThreshold = 0.8;  % used for 'plfit2023' and 'hybrid'

% 'fixedBinMedian': analysisConfig.binSize + median cutoff
% 'meanIsiZero': mean population ISI bin size + zero cutoff
avalancheDetectionMode = 'fixedBinMedian';

useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.25;

splitExcitatoryInhibitory = false;
widthCutoff = 0.035;  % ms; peak-to-trough width (narrow <= cutoff = inhibitory)

opts = neuro_behavior_options();
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;

analysisConfig = struct();
analysisConfig.slidingWindowSize = windowDurationSec;
analysisConfig.avStepSize = windowDurationSec;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.avalancheDetectionMode = avalancheDetectionMode;
if strcmpi(avalancheDetectionMode, 'meanIsiZero')
  % binSize resolved per area from spikes in collect window
else
  analysisConfig.binSize = 0.05;
end
analysisConfig.thresholdFlag = 1;
analysisConfig.thresholdPct = 1;
analysisConfig.nMinNeurons = 20;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;
analysisConfig.pcaFlag = 0;
analysisConfig.gofThreshold = gofThreshold;
analysisConfig.powerLawFitMethod = powerLawFitMethod;
analysisConfig.runClausetPlpva = runClausetPlpva;

%% Paths
paths = get_paths();
scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  scriptDir = fileparts(which('session_avalanche_distributions'));
end
srcPath = fullfile(scriptDir, '..');
addpath(srcPath);
addpath(fullfile(srcPath, 'reach_task'));
addpath(fullfile(srcPath, 'schall'));
addpath(fullfile(srcPath, 'spontaneous'));
addpath(fullfile(srcPath, 'interval_timing_task'));
addpath(fullfile(srcPath, 'criticality', 'scripts'));
addpath(fullfile(srcPath, 'criticality', 'analyses'));
addpath(fullfile(srcPath, 'session_prep', 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'utils'));
addpath(fullfile(srcPath, 'data_prep'));
addpath(fullfile(srcPath, 'sliding_window_prep', 'utils'));
addpath(fullfile(srcPath, 'criticality'));

[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();
analysisConfig.clausetPlfitPath = clausetPlfitPath;
analysisConfig.plfit2023Path = plfit2023Path;

fprintf('\n=== Session Avalanche Distributions ===\n');
fprintf('Power-law fit method: %s\n', powerLawFitMethod);
fprintf('Avalanche detection mode: %s\n', avalancheDetectionMode);
if useSubsampling
  fprintf('Subsampling: %d subsets x %d neurons\n', nSubsamples, nNeuronsSubsample);
end
if splitExcitatoryInhibitory
  fprintf('E/I split: on (widthCutoff = %.3f ms)\n', widthCutoff);
end
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, windowDurationSec / 60);

%% Load session
subjectNameForLoad = subjectName;
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

[dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations, false);
if ~areaOk
  error('Brain area "%s" not available in this session.', brainArea);
end

cellTypesToRun = get_session_cell_types_to_run(splitExcitatoryInhibitory);
if splitExcitatoryInhibitory
  eiSummary = init_session_ei_summary({'tau', 'alpha'}, {'\tau', '\alpha'});
end

for iCellRun = 1:numel(cellTypesToRun)
  cellType = cellTypesToRun{iCellRun};
  dataStructRun = prepare_session_data_for_cell_type(dataStruct, paths, cellType, widthCutoff, splitExcitatoryInhibitory);

  [dataStructRun, ~] = apply_manuscript_brain_area_selection(dataStructRun, brainArea, brainAreaCombinations);

  areasToAnalyze = resolve_areas_to_analyze(dataStructRun, brainArea, analysisConfig.nMinNeurons);
  if isempty(areasToAnalyze)
    error('No areas meet minimum neuron count (%d) for %s.', ...
      analysisConfig.nMinNeurons, cell_type_label(cellType));
  end

  %% Avalanche extraction and figure
  fig = figure('Name', sprintf('Session avalanche distributions%s', cell_type_file_tag(cellType)));
  clf(fig);
  tiledlayout(fig, 1, numel(areasToAnalyze) * 2, 'TileSpacing', 'compact', 'Padding', 'compact');

  avSummaryMetrics = struct('tau', nan, 'alpha', nan);
  for aIdx = 1:numel(areasToAnalyze)
    areaIndex = areasToAnalyze(aIdx);
    areaName = dataStructRun.areas{areaIndex};

    fprintf('\nArea %s (%s)...\n', areaName, cell_type_label(cellType));
    avData = extract_area_avalanches(dataStructRun, areaIndex, analysisConfig, collectStart, collectEnd);

    if ~avData.hasAvalanches
      warning('No avalanches detected for area %s (%s); skipping plots.', areaName, cell_type_label(cellType));
      continue;
    end

    fprintf('  Size:  tau = %.3f, x in [%.3g, %.3g]', ...
      avData.tau, avData.minSizeFit, avData.maxSizeFit);
    print_fit_diagnostics(avData.sizeFitInfo);
    print_hybrid_fit_diagnostics(avData.sizeFitInfo);
    fprintf('\n');
    fprintf('  Dur:   alpha = %.3f, x in [%.3g, %.3g]', ...
      avData.alpha, avData.minDurFit, avData.maxDurFit);
    print_fit_diagnostics(avData.durFitInfo);
    print_hybrid_fit_diagnostics(avData.durFitInfo);
    fprintf('\n');
    fprintf('  Scaling relation (alpha-1)/(tau-1): ');
    print_scaling_relation(avData.tau, avData.alpha);
    fprintf('\n');
    fprintf('  n = %d avalanches\n', avData.nAvalanches);

    tileSize = (aIdx - 1) * 2 + 1;
    tileDur = tileSize + 1;

    axSize = nexttile(tileSize);
    plot_avalanche_ccdf_with_fit(axSize, avData.sizes, avData.tau, ...
      avData.minSizeFit, avData.maxSizeFit, 'Avalanche size', '\tau', avData.sizeFitInfo);

    axDur = nexttile(tileDur);
    plot_avalanche_ccdf_with_fit(axDur, avData.durations, avData.alpha, ...
      avData.minDurFit, avData.maxDurFit, 'Avalanche duration (bins)', '\alpha', avData.durFitInfo);

    if ~isfinite(avSummaryMetrics.tau)
      avSummaryMetrics.tau = avData.tau;
      avSummaryMetrics.alpha = avData.alpha;
    end
  end

  if splitExcitatoryInhibitory && isfinite(avSummaryMetrics.tau)
    eiSummary = set_session_ei_summary_population(eiSummary, cellType, avSummaryMetrics);
  end

  titleSuffix = '';
  if splitExcitatoryInhibitory
    titleSuffix = sprintf(' | %s | width cutoff %.3f ms', cell_type_label(cellType), widthCutoff);
  end
  sgtitle(fig, sprintf('%s — %s [%.0f–%.0f s]%s', sessionName, ...
    format_areas_label(dataStructRun.areas(areasToAnalyze)), collectStart, collectEnd, titleSuffix), ...
    'FontWeight', 'bold', 'Interpreter', 'none');

  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    areaTag = format_areas_label(dataStructRun.areas(areasToAnalyze));
    plotBase = sprintf('session_avalanche_distributions_%s_%s_%s_%.0f-%.0fs%s', ...
      sessionName, areaTag, powerLawFitMethod, collectStart, collectEnd, cell_type_file_tag(cellType));
    exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
    exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved figure: %s\n', fullfile(saveDir, plotBase));
  end
end

if splitExcitatoryInhibitory
  areaTag = format_areas_label(brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  summaryTitle = sprintf('%s | %s | avalanche exponents [%.0f–%.0f s]', ...
    sessionName, areaTag, collectStart, collectEnd);
  figEiSummary = plot_session_ei_summary(eiSummary, summaryTitle, 'Exponent');
  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    plotBase = sprintf('session_avalanche_ei_summary_%s_%s_%s_%.0f-%.0fs%s', ...
      sessionName, areaTag, powerLawFitMethod, collectStart, collectEnd, session_ei_summary_file_tag());
    exportgraphics(figEiSummary, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
    exportgraphics(figEiSummary, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved E/I summary figure: %s\n', fullfile(saveDir, plotBase));
  end
end

fprintf('\n=== Done ===\n');

%% Local functions

function areasToAnalyze = resolve_areas_to_analyze(dataStruct, brainArea, nMinNeurons)
% RESOLVE_AREAS_TO_ANALYZE - Area indices to process

if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
  candidateAreas = dataStruct.areasToTest(:)';
elseif ~isempty(brainArea)
  areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
  if isempty(areaIdx)
    areasToAnalyze = [];
    return;
  end
  candidateAreas = areaIdx;
else
  candidateAreas = 1:numel(dataStruct.areas);
end

areasToAnalyze = [];
for areaIndex = candidateAreas
  if numel(dataStruct.idMatIdx{areaIndex}) >= nMinNeurons
    areasToAnalyze(end+1) = areaIndex; %#ok<AGROW>
  end
end
end

function avData = extract_area_avalanches(dataStruct, areaIndex, analysisConfig, collectStart, collectEnd)
% EXTRACT_AREA_AVALANCHES - Bin, threshold, and fit one collect window
%
% Variables:
%   dataStruct      - From load_session_data
%   areaIndex       - Index into dataStruct.areas
%   analysisConfig  - Bin size and threshold settings
%   collectStart    - Window start (s)
%   collectEnd      - Window end (s)
%
% Goal:
%   Match criticality_av_analysis single-window avalanche extraction.
%
% Returns:
%   avData - Struct with sizes, durations, tau, alpha, fit bounds, flags

avData = struct('hasAvalanches', false, 'sizes', [], 'durations', [], ...
  'tau', nan, 'alpha', nan, 'minSizeFit', nan, 'maxSizeFit', nan, ...
  'minDurFit', nan, 'maxDurFit', nan, 'sizeFitInfo', struct(), ...
  'durFitInfo', struct(), 'nAvalanches', 0);

timeRange = [collectStart, collectEnd];
neuronIds = dataStruct.idLabel{areaIndex};
binSizeVec = resolve_avalanche_bin_sizes(dataStruct, areaIndex, timeRange, analysisConfig);
binSize = binSizeVec(areaIndex);

aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
  neuronIds, timeRange, binSize);

useSubsampling = isfield(analysisConfig, 'useSubsampling') && analysisConfig.useSubsampling;
sizes = [];
durations = [];
if useSubsampling
  numNeuronsArea = size(aDataMat, 2);
  nSubsamplesArea = analysisConfig.nSubsamples;
  nNeuronsSubsampleArea = min(analysisConfig.nNeuronsSubsample, numNeuronsArea);
  for s = 1:nSubsamplesArea
    if nNeuronsSubsampleArea == numNeuronsArea
      colIdx = 1:numNeuronsArea;
    else
      colIdx = randperm(numNeuronsArea, nNeuronsSubsampleArea);
    end
    wPopActivity = sum(aDataMat(:, colIdx), 2);
    avMetrics = compute_av_metrics_from_pop_activity(wPopActivity, analysisConfig);
    if ~isfinite(avMetrics.kappa)
      continue;
    end
    wPopActivity = apply_avalanche_population_threshold(wPopActivity, analysisConfig);
    zeroBins = find(wPopActivity == 0);
    if ~(numel(zeroBins) > 1 && any(diff(zeroBins) > 1))
      continue;
    end
    [sizesSub, dursSub] = getAvalanches(wPopActivity', 0.5, 1);
    sizes = [sizes; sizesSub(:)]; %#ok<AGROW>
    durations = [durations; dursSub(:)]; %#ok<AGROW>
  end
else
  wPopActivity = sum(aDataMat, 2);
  wPopActivity = apply_avalanche_population_threshold(wPopActivity, analysisConfig);
  zeroBins = find(wPopActivity == 0);
  if ~(numel(zeroBins) > 1 && any(diff(zeroBins) > 1))
    return;
  end
  [sizes, durations] = getAvalanches(wPopActivity', 0.5, 1);
end

sizes = sizes(:);
durations = durations(:);
if isempty(sizes) || isempty(durations)
  return;
end

sizeFit = fit_avalanche_power_law(sizes, analysisConfig);
durFit = fit_avalanche_power_law(durations, analysisConfig);

avData.hasAvalanches = true;
avData.sizes = sizes;
avData.durations = durations;
avData.tau = sizeFit.exponent;
avData.alpha = durFit.exponent;
avData.minSizeFit = sizeFit.fitMin;
avData.maxSizeFit = sizeFit.fitMax;
avData.minDurFit = durFit.fitMin;
avData.maxDurFit = durFit.fitMax;
avData.sizeFitInfo = sizeFit;
avData.durFitInfo = durFit;
avData.nAvalanches = numel(sizes);
end

function print_scaling_relation(tau, alpha)
% PRINT_SCALING_RELATION - Size-duration scaling exponent (alpha-1)/(tau-1)
%
% Variables:
%   tau   - Avalanche size power-law exponent
%   alpha - Avalanche duration power-law exponent
%
% Goal:
%   Print gamma_predicted from exponent relation (Ma et al. 2019 / distance_to_criticality).

if ~isfinite(tau) || ~isfinite(alpha)
  fprintf('nan (non-finite tau or alpha)');
  return;
end
if tau <= 1
  fprintf('nan (tau must be > 1)');
  return;
end

scalingRelation = (alpha - 1) / (tau - 1);
fprintf('%.4f', scalingRelation);
end

function print_hybrid_fit_diagnostics(fitInfo)
% PRINT_HYBRID_FIT_DIAGNOSTICS - plfit2023 bounds used for hybrid xmax

if isempty(fitInfo) || ~isstruct(fitInfo) || ~isfield(fitInfo, 'method') ...
    || ~strcmpi(fitInfo.method, 'hybrid')
  return;
end
if isfield(fitInfo, 'fitMaxPlfit2023') && isfinite(fitInfo.fitMaxPlfit2023)
  fprintf(' [hybrid xmax=%.3g', fitInfo.fitMaxPlfit2023);
  if isfield(fitInfo, 'exponentPlfit2023') && isfinite(fitInfo.exponentPlfit2023)
    fprintf(', plfit2023 exp=%.2f', fitInfo.exponentPlfit2023);
  end
  fprintf(']');
end
end

function print_fit_diagnostics(fitInfo)
% PRINT_FIT_DIAGNOSTICS - Optional p-value / tail count for command window

if isempty(fitInfo) || ~isstruct(fitInfo)
  return;
end
extras = {};
if isfield(fitInfo, 'nTail') && fitInfo.nTail > 0
  extras{end+1} = sprintf('nTail=%d', fitInfo.nTail); %#ok<AGROW>
end
if isfield(fitInfo, 'pValue') && isfinite(fitInfo.pValue)
  extras{end+1} = sprintf('p=%.3f', fitInfo.pValue); %#ok<AGROW>
end
if ~isempty(extras)
  fprintf(' (%s)', strjoin(extras, ', '));
end
end

function plot_avalanche_ccdf_with_fit(ax, values, exponent, fitMin, fitMax, xLabelText, exponentLabel, fitInfo)
% PLOT_AVALANCHE_CCDF_WITH_FIT - Log-log CCDF with power-law fit segment
%
% Variables:
%   ax            - Axes handle
%   values        - Avalanche sizes or durations
%   exponent      - Power-law exponent (tau or alpha)
%   fitMin, fitMax - Fitted scaling range (xmin to xmax in tail)
%   xLabelText    - X-axis label
%   exponentLabel - Display name for exponent ('\tau' or '\alpha')
%   fitInfo       - Optional struct from fit_avalanche_power_law (method, pValue)
%
% Goal:
%   Plot empirical CCDF with log x-axis and overlay fit slope on [fitMin, fitMax].

if nargin < 8
  fitInfo = struct();
end

values = values(values > 0 & isfinite(values));
if isempty(values)
  cla(ax);
  title(ax, sprintf('%s (no data)', xLabelText));
  return;
end

uniqueVals = unique(values);
cdfY = arrayfun(@(x) mean(values >= x), uniqueVals);

hold(ax, 'on');
plot(ax, uniqueVals, cdfY, 'o', 'Color', [0.15, 0.35, 0.75], ...
  'MarkerFaceColor', [0.15, 0.35, 0.75], 'MarkerSize', 5, ...
  'DisplayName', 'Empirical CCDF');

fitPlotted = false;
if isfinite(exponent) && exponent > 1 && isfinite(fitMin) && isfinite(fitMax) ...
    && fitMin > 0 && fitMax > fitMin
  xFit = logspace(log10(fitMin), log10(fitMax), 100);
  yAtMin = mean(values >= fitMin);
  % CCDF for p(x) ~ x^{-exponent}: P(X >= x) ~ (x / xMin)^{-(exponent - 1)}
  yFit = (xFit / fitMin) .^ (-(exponent - 1)) * yAtMin;
  methodTag = '';
  if isstruct(fitInfo) && isfield(fitInfo, 'method') && ~isempty(fitInfo.method)
    methodTag = sprintf(', %s', fitInfo.method);
  end
  plot(ax, xFit, yFit, '-', 'Color', [0.85, 0.2, 0.15], 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Power-law fit (%s=%.2f%s)', exponentLabel, exponent, methodTag));
  fitPlotted = true;
end

set(ax, 'XScale', 'log', 'YScale', 'log');
xlabel(ax, xLabelText);
ylabel(ax, 'P(X \geq x)');
grid(ax, 'on');
legend(ax, 'Location', 'southwest');

pText = '';
if isstruct(fitInfo) && isfield(fitInfo, 'pValue') && isfinite(fitInfo.pValue)
  pText = sprintf(', p=%.2f', fitInfo.pValue);
end
if fitPlotted
  title(ax, sprintf('%s (%s = %.2f, x_{min}=%.3g, x_{max}=%.3g%s)', ...
    xLabelText, exponentLabel, exponent, fitMin, fitMax, pText));
else
  title(ax, sprintf('%s (%s = %.2f, fit range n/a%s)', xLabelText, exponentLabel, exponent, pText));
end

hold(ax, 'off');
end

function label = format_areas_label(areaNames)
% FORMAT_AREAS_LABEL - Underscore-safe tag for filenames/titles

if iscell(areaNames)
  areaNames = areaNames(:)';
  label = strjoin(areaNames, '_');
else
  label = char(areaNames);
end
label = matlab.lang.makeValidName(label);
end
