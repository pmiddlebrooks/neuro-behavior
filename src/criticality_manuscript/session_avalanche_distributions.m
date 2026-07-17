%%
% Session Avalanche Size and Duration Distributions (Manuscript)
%
% For one session and brain area, runs the same avalanche pipeline as
% criticality_av_across_tasks.m (single full collect window) and plots
% complementary CCDFs on log-log axes with power-law fits over the viable
% cutoff range (tau for size, alpha for duration), plus ⟨S⟩(T) with the
% crackling WLS slope (paramSD = 1/σνz) and printed dcc.
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
%   enableCircularPermutations - If true, circular-shift each neuron and overlay shuffle CCDFs
%   nShuffles          - Number of independent circular permutations per area
%   saveAnalysisResults - If true, save sessionResults after %% Analysis
%   analysisResultsFile - Path to .mat cache; '' = default under dropPath/criticality_manuscript
%
% Sections:
%   %% Analysis  - load session, extract avalanches, optionally save sessionResults
%   %% Plotting  - figures from sessionResults (loads .mat if not in workspace)
%
% Goal:
%   Visualize avalanche size and duration distributions for one session
%   with log-scaled x-axes, fitted slopes, and the crackling ⟨S⟩(T) relation.

%% Configuration
% sessionType = 'interval';
% subjectName = 'ey9166';
% sessionName = 'ey9166_2026_04_03';
% dataSource = 'spikes';

collectStart = 0;
collectEnd = 45 * 60;
collectEnd = [];
windowDurationSec = collectEnd - collectStart;

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
saveFigure = false;

% Power-law fitting: 'clauset', 'plfit2023', 'hybrid' = plfit2023 xmax, then Clauset plfit on x <= xmax
powerLawFitMethod = 'plfit2023';
runClausetPlpva = false;

gofThreshold = 0.8;  % used for 'plfit2023' and 'hybrid'

% 'fixedBinMedian': analysisConfig.binSize + median cutoff
% 'meanIsiZero': mean population ISI bin size + zero cutoff
avalancheDetectionMode = 'fixedBinMedian';

useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.25;

splitExcitatoryInhibitory = true;
widthCutoff = 0.35;  % ms; peak-to-trough width (narrow <= cutoff = inhibitory)

enableCircularPermutations = true;
nShuffles = 5;

saveAnalysisResults = false;
analysisResultsFile = '';  % default: dropPath/criticality_manuscript/session_avalanche_results_<session>.mat

% Plot formatting (edit and re-run %% Plotting)
plotConfig = struct();
plotConfig.observedMarkerSize = 5;
plotConfig.shuffleMarkerSize = 4;
plotConfig.fitLineWidth = 2.5;
plotConfig.tileSpacing = 'compact';
plotConfig.tilePadding = 'compact';
plotConfig.legendLocation = 'southwest';
plotConfig.axisLabelFontSize = 14;
plotConfig.tickLabelFontSize = 12;
plotConfig.axesLineWidth = 1.5;
plotConfig.observedMarkerFaceAlpha = 0.35;
plotConfig.figureWidthInches = 6.5;  % fits portrait PDF (A4/Letter) with margins

opts = neuro_behavior_options();
opts.firingRateCheckTime = []; %5 * 60;
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
analysisConfig.enableCircularPermutations = enableCircularPermutations;
analysisConfig.nShuffles = nShuffles;

% Paths

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
if enableCircularPermutations
  fprintf('Circular permutations: %d shuffles per area\n', nShuffles);
end
fprintf('Session [%s]: %s\n', sessionType, sessionName);

%% Analysis — load session, extract avalanches, optionally cache results
subjectNameForLoad = '';
if exist('subjectName', 'var') && ~isempty(subjectName)
  subjectNameForLoad = subjectName;
end
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

% Resolve collectEnd to full session when empty (uses loaded spike times)
if isempty(collectEnd)
  collectEnd = resolve_session_collect_end(dataStruct, collectStart);
end
windowDurationSec = collectEnd - collectStart;
analysisConfig.slidingWindowSize = windowDurationSec;
analysisConfig.avStepSize = windowDurationSec;
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, windowDurationSec / 60);

[dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations, false);
if ~areaOk
  error('Brain area "%s" not available in this session.', brainArea);
end

if splitExcitatoryInhibitory
  eiCheck = check_session_ei_neuron_counts(dataStruct, paths, widthCutoff, brainArea, ...
    brainAreaCombinations, analysisConfig.nMinNeurons);
  if ~eiCheck.isOk
    return;
  end
end

runMeta = struct( ...
  'sessionType', sessionType, ...
  'sessionName', sessionName, ...
  'subjectName', subjectNameForLoad, ...
  'dataSource', dataSource, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'powerLawFitMethod', powerLawFitMethod, ...
  'avalancheDetectionMode', avalancheDetectionMode, ...
  'enableCircularPermutations', enableCircularPermutations, ...
  'nShuffles', nShuffles, ...
  'useSubsampling', useSubsampling, ...
  'splitExcitatoryInhibitory', splitExcitatoryInhibitory, ...
  'widthCutoff', widthCutoff, ...
  'saveFigure', saveFigure);
sessionResults = build_session_avalanche_results(dataStruct, paths, analysisConfig, runMeta, plotConfig);

if saveAnalysisResults
  resultsFile = resolve_avalanche_results_file(paths, sessionName, analysisResultsFile);
  save(resultsFile, 'sessionResults', '-v7.3');
  fprintf('\nSaved analysis results: %s\n', resultsFile);
end

%% Plotting — figures from sessionResults (re-run this section to tweak formatting)
if ~exist('sessionResults', 'var') || isempty(sessionResults)
  resultsFile = resolve_avalanche_results_file(paths, sessionName, analysisResultsFile);
  if ~isfile(resultsFile)
    error(['sessionResults not in workspace and results file not found: %s. ', ...
      'Run %% Analysis first or set analysisResultsFile.'], resultsFile);
  end
  loaded = load(resultsFile, 'sessionResults');
  sessionResults = loaded.sessionResults;
  fprintf('\nLoaded analysis results: %s\n', resultsFile);
end

sessionResults.plotConfig = plotConfig;
plot_session_avalanche_results(sessionResults, paths, plotConfig);

fprintf('\n=== Done ===\n');

%% Local functions

function collectEnd = resolve_session_collect_end(dataStruct, collectStart)
% RESOLVE_SESSION_COLLECT_END - Full-session end time (s) when collectEnd is empty
%
% Variables:
%   dataStruct   - Session struct from load_session_data
%   collectStart - Window start (s)
%
% Goal:
%   Mirror the loaders: use the loaded spike extent (falling back to opts.collectEnd)
%   so an empty collectEnd analyzes the entire session.

collectEnd = [];
if isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
  collectEnd = max(dataStruct.spikeTimes);
elseif isfield(dataStruct, 'opts') && isstruct(dataStruct.opts) ...
    && isfield(dataStruct.opts, 'collectEnd') && ~isempty(dataStruct.opts.collectEnd)
  collectEnd = dataStruct.opts.collectEnd;
end

if isempty(collectEnd) || ~isfinite(collectEnd) || collectEnd <= collectStart
  error(['Could not resolve collectEnd for the full session. ', ...
    'Set collectEnd explicitly in the configuration.']);
end
end

function resultsFile = resolve_avalanche_results_file(paths, sessionName, analysisResultsFile)
% RESOLVE_AVALANCHE_RESULTS_FILE - Default path for cached session avalanche results

if ~isempty(analysisResultsFile)
  resultsFile = analysisResultsFile;
  return;
end

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
sessionTag = matlab.lang.makeValidName(sessionName);
resultsFile = fullfile(saveDir, sprintf('session_avalanche_results_%s.mat', sessionTag));
end

function sessionResults = build_session_avalanche_results(dataStruct, paths, analysisConfig, runMeta, plotConfig)
% BUILD_SESSION_AVALANCHE_RESULTS - Run avalanche extraction for all cell-type runs

runMeta = runMeta(1);
cellTypesToRun = get_session_cell_types_to_run(runMeta.splitExcitatoryInhibitory);
sessionResults = struct();
sessionResults.meta = runMeta;
sessionResults.plotConfig = plotConfig;
sessionResults.runs = cell(1, numel(cellTypesToRun));
sessionResults.eiSummary = [];

if runMeta.splitExcitatoryInhibitory
  sessionResults.eiSummary = init_session_ei_summary({'tau', 'alpha'}, {'\tau', '\alpha'});
end

for iCellRun = 1:numel(cellTypesToRun)
  cellType = cellTypesToRun{iCellRun};
  dataStructRun = prepare_session_data_for_cell_type(dataStruct, paths, cellType, ...
    runMeta.widthCutoff, runMeta.splitExcitatoryInhibitory);
  [dataStructRun, ~] = apply_manuscript_brain_area_selection(dataStructRun, ...
    runMeta.brainArea, runMeta.brainAreaCombinations);

  areasToAnalyze = resolve_areas_to_analyze(dataStructRun, runMeta.brainArea, analysisConfig.nMinNeurons);
  if isempty(areasToAnalyze)
    error('No areas meet minimum neuron count (%d) for %s.', ...
      analysisConfig.nMinNeurons, cell_type_label(cellType));
  end

  runResult = struct();
  runResult.cellType = cellType;
  runResult.areasToAnalyze = areasToAnalyze;
  runResult.areaNames = dataStructRun.areas(areasToAnalyze);
  runResult.areaResults = {};
  runResult.eiSummaryMetrics = struct('tau', nan, 'alpha', nan);

  for aIdx = 1:numel(areasToAnalyze)
    areaIndex = areasToAnalyze(aIdx);
    areaName = dataStructRun.areas{areaIndex};

    fprintf('\nArea %s (%s)...\n', areaName, cell_type_label(cellType));
    avData = extract_area_avalanches(dataStructRun, areaIndex, analysisConfig, ...
      runMeta.collectStart, runMeta.collectEnd);
    avData.areaName = areaName;

    if ~avData.hasAvalanches
      warning('No avalanches detected for area %s (%s).', areaName, cell_type_label(cellType));
      continue;
    end

    print_area_avalanche_summary(avData, runMeta.enableCircularPermutations);
    runResult.areaResults{end+1} = avData; %#ok<AGROW>

    if ~isfinite(runResult.eiSummaryMetrics.tau)
      runResult.eiSummaryMetrics.tau = avData.tau;
      runResult.eiSummaryMetrics.alpha = avData.alpha;
    end
  end

  if runMeta.splitExcitatoryInhibitory && isfinite(runResult.eiSummaryMetrics.tau)
    sessionResults.eiSummary = set_session_ei_summary_population( ...
      sessionResults.eiSummary, cellType, runResult.eiSummaryMetrics);
  end

  sessionResults.runs{iCellRun} = runResult;
end
end

function plot_session_avalanche_results(sessionResults, paths, plotConfig)
% PLOT_SESSION_AVALANCHE_RESULTS - Figures from cached avalanche extraction results

runMeta = sessionResults.meta(1);
if nargin < 3 || isempty(plotConfig)
  if isfield(sessionResults, 'plotConfig') && ~isempty(sessionResults.plotConfig)
    plotConfig = sessionResults.plotConfig;
  else
    plotConfig = struct();
  end
end
if ~isfield(plotConfig, 'sessionType') || isempty(plotConfig.sessionType)
  if isfield(runMeta, 'sessionType') && ~isempty(runMeta.sessionType)
    plotConfig.sessionType = runMeta.sessionType;
  end
end
plotConfig = fill_default_avalanche_plot_config(plotConfig);

for iCellRun = 1:numel(sessionResults.runs)
  runResult = sessionResults.runs{iCellRun};
  if isempty(runResult.areaResults)
    warning('No avalanche results to plot for cell type "%s".', cell_type_label(runResult.cellType));
    continue;
  end

  figName = sprintf('Session avalanche distributions%s', cell_type_file_tag(runResult.cellType));
  fig = get_task_figure_handle(plotConfig.sessionType, 'distributions', runResult.cellType, figName);
  nAreas = numel(runResult.areaResults);
  tiledlayout(fig, nAreas, 3, ...
    'TileSpacing', plotConfig.tileSpacing, 'Padding', plotConfig.tilePadding);

  for aIdx = 1:numel(runResult.areaResults)
    avData = runResult.areaResults{aIdx};

    axSize = nexttile((aIdx - 1) * 3 + 1);
    plot_avalanche_ccdf_with_fit(axSize, avData.sizes, avData.tau, ...
      avData.minSizeFit, avData.maxSizeFit, 'Sizes', '\tau', avData.sizeFitInfo, ...
      avData.shuffleSizes, plotConfig);

    axDur = nexttile((aIdx - 1) * 3 + 2);
    binSize = resolve_avalanche_duration_bin_size(avData);
    plot_avalanche_ccdf_with_fit(axDur, avData.durations * binSize * 1000, avData.alpha, ...
      avData.minDurFit * binSize, avData.maxDurFit * binSize, 'Durations (ms)', '\alpha', ...
      avData.durFitInfo, avData.shuffleDurations * binSize * 1000, plotConfig);

    axCrack = nexttile((aIdx - 1) * 3 + 3);
    plot_size_given_duration_with_fit(axCrack, avData, plotConfig);
  end

  titleSuffix = '';
  if runMeta.splitExcitatoryInhibitory
    titleSuffix = sprintf(' | %s | width cutoff %.3f ms', ...
      cell_type_label(runResult.cellType), runMeta.widthCutoff);
  end
  sgtitle(fig, sprintf('%s — %s [%.0f–%.0f s]%s', runMeta.sessionName, ...
    format_areas_label(runResult.areaNames), runMeta.collectStart, runMeta.collectEnd, titleSuffix), ...
    'FontWeight', 'bold', 'Interpreter', 'none');

  apply_portrait_figure_size(fig, plotConfig.figureWidthInches, nAreas, 3);

  if runMeta.saveFigure
    save_session_avalanche_distribution_figure(fig, paths, runMeta, runResult);
  end
end

if runMeta.splitExcitatoryInhibitory && ~isempty(sessionResults.eiSummary)
  areaTag = format_areas_label(runMeta.brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  summaryTitle = sprintf('%s | %s | avalanche exponents [%.0f–%.0f s]', ...
    runMeta.sessionName, areaTag, runMeta.collectStart, runMeta.collectEnd);
  figEiSummary = get_task_figure_handle(plotConfig.sessionType, 'ei_summary', '', ...
    'Session avalanche E/I summary');
  figEiSummary = plot_session_ei_summary(sessionResults.eiSummary, summaryTitle, 'Exponent', [], figEiSummary);
  apply_portrait_figure_size(figEiSummary, plotConfig.figureWidthInches, 1, 1);
  if runMeta.saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    plotBase = sprintf('session_avalanche_ei_summary_%s_%s_%s_%.0f-%.0fs%s', ...
      runMeta.sessionName, areaTag, runMeta.powerLawFitMethod, ...
      runMeta.collectStart, runMeta.collectEnd, session_ei_summary_file_tag());
    exportgraphics(figEiSummary, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
    exportgraphics(figEiSummary, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved E/I summary figure: %s\n', fullfile(saveDir, plotBase));
  end
end
end

function save_session_avalanche_distribution_figure(fig, paths, runMeta, runResult)
saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end
areaTag = format_areas_label(runResult.areaNames);
plotBase = sprintf('session_avalanche_distributions_%s_%s_%s_%.0f-%.0fs%s', ...
  runMeta.sessionName, areaTag, runMeta.powerLawFitMethod, ...
  runMeta.collectStart, runMeta.collectEnd, cell_type_file_tag(runResult.cellType));
if runMeta.enableCircularPermutations
  plotBase = sprintf('%s_circ%d', plotBase, runMeta.nShuffles);
end
exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
fprintf('\nSaved figure: %s\n', fullfile(saveDir, plotBase));
end

function print_area_avalanche_summary(avData, enableCircularPermutations)
fprintf('  Size:  tau = %.3f, x in [%.3g, %.3g]', ...
  avData.tau, avData.minSizeFit, avData.maxSizeFit);
print_fit_diagnostics(avData.sizeFitInfo);
print_hybrid_fit_diagnostics(avData.sizeFitInfo);
fprintf('\n');
binSize = resolve_avalanche_duration_bin_size(avData);
fprintf('  Dur:   alpha = %.3f, x in [%.3g, %.3g] s', ...
  avData.alpha, avData.minDurFit * binSize, avData.maxDurFit * binSize);
print_fit_diagnostics(avData.durFitInfo);
print_hybrid_fit_diagnostics(avData.durFitInfo);
fprintf('\n');
fprintf('  Scaling relation (alpha-1)/(tau-1): ');
print_scaling_relation(avData.tau, avData.alpha);
fprintf('\n');
if isfield(avData, 'paramSD') && isfinite(avData.paramSD)
  fprintf('  Crackling 1/σνz (paramSD from ⟨S⟩|T): %.4f\n', avData.paramSD);
else
  fprintf('  Crackling 1/σνz (paramSD from ⟨S⟩|T): nan\n');
end
if isfield(avData, 'dcc') && isfinite(avData.dcc)
  fprintf('  dcc = |γ_pred - paramSD|: %.4f\n', avData.dcc);
else
  fprintf('  dcc = |γ_pred - paramSD|: nan\n');
end
fprintf('  n = %d avalanches\n', avData.nAvalanches);
if enableCircularPermutations && ~isempty(avData.shuffleSizes)
  fprintf('  Shuffle (pooled): n = %d avalanches over %d shuffles\n', ...
    numel(avData.shuffleSizes), avData.nShufflesCompleted);
  if isfinite(avData.shuffleTau)
    fprintf('  Shuffle size:  tau = %.3f\n', avData.shuffleTau);
  end
  if isfinite(avData.shuffleAlpha)
    fprintf('  Shuffle dur:   alpha = %.3f\n', avData.shuffleAlpha);
  end
end
end

function plotConfig = fill_default_avalanche_plot_config(plotConfig)
if ~isfield(plotConfig, 'observedMarkerSize') || isempty(plotConfig.observedMarkerSize)
  plotConfig.observedMarkerSize = 5;
end
if ~isfield(plotConfig, 'shuffleMarkerSize') || isempty(plotConfig.shuffleMarkerSize)
  plotConfig.shuffleMarkerSize = 4;
end
if ~isfield(plotConfig, 'fitLineWidth') || isempty(plotConfig.fitLineWidth)
  plotConfig.fitLineWidth = 2.5;
end
if ~isfield(plotConfig, 'tileSpacing') || isempty(plotConfig.tileSpacing)
  plotConfig.tileSpacing = 'compact';
end
if ~isfield(plotConfig, 'tilePadding') || isempty(plotConfig.tilePadding)
  plotConfig.tilePadding = 'compact';
end
if ~isfield(plotConfig, 'legendLocation') || isempty(plotConfig.legendLocation)
  plotConfig.legendLocation = 'southwest';
end
if ~isfield(plotConfig, 'axisLabelFontSize') || isempty(plotConfig.axisLabelFontSize)
  plotConfig.axisLabelFontSize = 14;
end
if ~isfield(plotConfig, 'tickLabelFontSize') || isempty(plotConfig.tickLabelFontSize)
  plotConfig.tickLabelFontSize = 12;
end
if ~isfield(plotConfig, 'axesLineWidth') || isempty(plotConfig.axesLineWidth)
  plotConfig.axesLineWidth = 1.5;
end
if ~isfield(plotConfig, 'observedMarkerFaceAlpha') || isempty(plotConfig.observedMarkerFaceAlpha)
  plotConfig.observedMarkerFaceAlpha = 0.35;
end
if ~isfield(plotConfig, 'sessionType')
  plotConfig.sessionType = '';
end
if ~isfield(plotConfig, 'figureWidthInches') || isempty(plotConfig.figureWidthInches)
  plotConfig.figureWidthInches = 6.5;
end
end

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
  'tau', nan, 'alpha', nan, 'paramSD', nan, 'dcc', nan, 'scalingRelation', nan, ...
  'minSizeFit', nan, 'maxSizeFit', nan, ...
  'minDurFit', nan, 'maxDurFit', nan, 'sizeFitInfo', struct(), ...
  'durFitInfo', struct(), 'nAvalanches', 0, 'binSize', nan, ...
  'shuffleSizes', [], 'shuffleDurations', [], 'shuffleTau', nan, 'shuffleAlpha', nan, ...
  'nShufflesCompleted', 0);

timeRange = [collectStart, collectEnd];
neuronIds = dataStruct.idLabel{areaIndex};
binSizeVec = resolve_avalanche_bin_sizes(dataStruct, areaIndex, timeRange, analysisConfig);
binSize = binSizeVec(areaIndex);
avData.binSize = binSize;

aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
  neuronIds, timeRange, binSize);

[sizes, durations, hasAvalanches] = compute_avalanche_sizes_durations_from_binned( ...
  aDataMat, analysisConfig);
if ~hasAvalanches
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

% Crackling / size-given-duration exponent (1/σνz) and DCC vs (α-1)/(τ-1)
if isfinite(durFit.fitMin) && isfinite(durFit.fitMax) && durFit.fitMin <= durFit.fitMax ...
    && numel(sizes) >= 2 && numel(durations) >= 2
  [avData.paramSD, ~, ~] = size_given_duration(sizes(:), durations(:), ...
    'durmin', durFit.fitMin, 'durmax', durFit.fitMax);
end
if isfinite(avData.tau) && isfinite(avData.alpha) && avData.tau > 1
  avData.scalingRelation = (avData.alpha - 1) / (avData.tau - 1);
end
if isfinite(avData.paramSD) && isfinite(avData.scalingRelation)
  avData.dcc = abs(avData.scalingRelation - avData.paramSD);
end

if isfield(analysisConfig, 'enableCircularPermutations') && analysisConfig.enableCircularPermutations
  nShufflesArea = 10;
  if isfield(analysisConfig, 'nShuffles') && ~isempty(analysisConfig.nShuffles)
    nShufflesArea = analysisConfig.nShuffles;
  end
  [avData.shuffleSizes, avData.shuffleDurations, avData.nShufflesCompleted] = ...
    pooled_circular_shuffle_avalanches(aDataMat, analysisConfig, nShufflesArea);
  if ~isempty(avData.shuffleSizes)
    shuffleSizeFit = fit_avalanche_power_law(avData.shuffleSizes, analysisConfig);
    avData.shuffleTau = shuffleSizeFit.exponent;
  end
  if ~isempty(avData.shuffleDurations)
    shuffleDurFit = fit_avalanche_power_law(avData.shuffleDurations, analysisConfig);
    avData.shuffleAlpha = shuffleDurFit.exponent;
  end
end
end

function [sizes, durations, hasAvalanches] = compute_avalanche_sizes_durations_from_binned(aDataMat, analysisConfig)
% COMPUTE_AVALANCHE_SIZES_DURATIONS_FROM_BINNED - Avalanche vectors from binned spike matrix

sizes = [];
durations = [];
hasAvalanches = false;

useSubsampling = isfield(analysisConfig, 'useSubsampling') && analysisConfig.useSubsampling;
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
hasAvalanches = ~isempty(sizes) && ~isempty(durations);
end

function [shuffleSizes, shuffleDurations, nCompleted] = pooled_circular_shuffle_avalanches(aDataMat, analysisConfig, nShuffles)
% POOLED_CIRCULAR_SHUFFLE_AVALANCHES - Pool avalanches across circular neuron shuffles
%
% Variables:
%   aDataMat        - [timeBins x neurons] binned activity for one collect window
%   analysisConfig  - Avalanche detection settings
%   nShuffles       - Number of independent circular permutations
%
% Goal:
%   Circularly shift each neuron's binned activity, then extract avalanches using
%   the same pipeline as the observed data.

shuffleSizes = [];
shuffleDurations = [];
nCompleted = 0;

for shuffleIdx = 1:nShuffles
  permutedMat = circular_shuffle_binned_matrix(aDataMat);
  [sizesSub, durationsSub, hasAvalanches] = compute_avalanche_sizes_durations_from_binned( ...
    permutedMat, analysisConfig);
  if ~hasAvalanches
    continue;
  end
  shuffleSizes = [shuffleSizes; sizesSub(:)]; %#ok<AGROW>
  shuffleDurations = [shuffleDurations; durationsSub(:)]; %#ok<AGROW>
  nCompleted = nCompleted + 1;
end
end

function permutedMat = circular_shuffle_binned_matrix(aDataMat)
% CIRCULAR_SHUFFLE_BINNED_MATRIX - Independent circular shift per neuron column

permutedMat = aDataMat;
winSamples = size(aDataMat, 1);
if winSamples < 1
  return;
end

for neuronIdx = 1:size(aDataMat, 2)
  shiftAmount = randi([1, winSamples]);
  permutedMat(:, neuronIdx) = circshift(aDataMat(:, neuronIdx), shiftAmount);
end
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

function plot_avalanche_ccdf_with_fit(ax, values, exponent, fitMin, fitMax, xLabelText, exponentLabel, fitInfo, shuffleValues, plotConfig)
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
%   shuffleValues - Optional pooled shuffle avalanches for overlay CCDF
%   plotConfig    - Optional struct with observedMarkerSize, shuffleMarkerSize, etc.
%
% Goal:
%   Plot empirical CCDF with log x-axis and overlay fit slope on [fitMin, fitMax].

if nargin < 8 || isempty(fitInfo)
  fitInfo = struct();
end
if nargin < 9
  shuffleValues = [];
end
if nargin < 10 || isempty(plotConfig)
  plotConfig = struct();
end
plotConfig = fill_default_avalanche_plot_config(plotConfig);

values = values(values > 0 & isfinite(values));
if isempty(values)
  cla(ax);
  title(ax, sprintf('%s (no data)', xLabelText));
  return;
end

uniqueVals = unique(values);
cdfY = arrayfun(@(x) mean(values >= x), uniqueVals);

observedColor = colors_for_tasks(plotConfig.sessionType);
markerArea = plotConfig.observedMarkerSize .^ 2;

hold(ax, 'on');
scatter(ax, uniqueVals, cdfY, markerArea, 'filled', ...
  'MarkerEdgeColor', observedColor, 'MarkerFaceColor', observedColor, ...
  'MarkerFaceAlpha', plotConfig.observedMarkerFaceAlpha, ...
  'DisplayName', 'Obs');

shufflePlotted = false;
shuffleValues = shuffleValues(shuffleValues > 0 & isfinite(shuffleValues));
if ~isempty(shuffleValues)
  uniqueShuffle = unique(shuffleValues);
  cdfShuffle = arrayfun(@(x) mean(shuffleValues >= x), uniqueShuffle);
  plot(ax, uniqueShuffle, cdfShuffle, 'o', 'Color', [0.55, 0.55, 0.55], ...
    'MarkerFaceColor', [0.75, 0.75, 0.75], 'MarkerSize', plotConfig.shuffleMarkerSize, ...
    'DisplayName', 'Shuff');
  shufflePlotted = true;
end

fitPlotted = false;
if isfinite(exponent) && exponent > 1 && isfinite(fitMin) && isfinite(fitMax) ...
    && fitMin > 0 && fitMax > fitMin
  xFit = logspace(log10(fitMin), log10(fitMax), 100);
  yAtMin = mean(values >= fitMin);
  yFit = (xFit / fitMin) .^ (-(exponent - 1)) * yAtMin;
  methodTag = '';
  if isstruct(fitInfo) && isfield(fitInfo, 'method') && ~isempty(fitInfo.method)
    methodTag = sprintf(', %s', fitInfo.method);
  end
  % plot(ax, xFit, yFit, '-', 'Color', [0.85, 0.2, 0.15], 'LineWidth', plotConfig.fitLineWidth, ...
  %   'DisplayName', sprintf('Observed fit (%s=%.2f%s)', exponentLabel, exponent, methodTag));
  % fitPlotted = true;
end

set(ax, 'XScale', 'log', 'YScale', 'log', 'FontSize', plotConfig.tickLabelFontSize, ...
  'LineWidth', plotConfig.axesLineWidth);
axis(ax, 'square');
xlabel(ax, xLabelText, 'FontSize', plotConfig.axisLabelFontSize);
ylabel(ax, 'P(X \geq x)', 'FontSize', plotConfig.axisLabelFontSize);
% grid(ax, 'on');
if shufflePlotted || fitPlotted
  legend(ax, 'Location', plotConfig.legendLocation, 'FontSize', plotConfig.tickLabelFontSize);
end

pText = '';
if isstruct(fitInfo) && isfield(fitInfo, 'pValue') && isfinite(fitInfo.pValue)
  pText = sprintf(', p=%.2f', fitInfo.pValue);
end
if fitPlotted
  title(ax, sprintf('%s (%s = %.2f, x_{min}=%.3g, x_{max}=%.3g%s)', ...
    xLabelText, exponentLabel, exponent, fitMin, fitMax, pText));
else
  title(ax, sprintf('%s (%s = %.2f %s)', xLabelText, exponentLabel, exponent, pText));
end

hold(ax, 'off');
end

function plot_size_given_duration_with_fit(ax, avData, plotConfig)
% PLOT_SIZE_GIVEN_DURATION_WITH_FIT - log⟨S⟩ vs log T with crackling WLS slope
%
% Variables:
%   ax         - Axes handle
%   avData     - Area avalanche struct (sizes, durations, fit range, paramSD)
%   plotConfig - Marker / font options
%
% Goal:
%   Show mean size given duration and the fitted crackling exponent 1/σνz.

if nargin < 3 || isempty(plotConfig)
  plotConfig = struct();
end
plotConfig = fill_default_avalanche_plot_config(plotConfig);

cla(ax);
if ~isstruct(avData) || ~isfield(avData, 'hasAvalanches') || ~avData.hasAvalanches
  title(ax, '⟨S⟩(T) (no data)');
  return;
end

sizes = avData.sizes(:);
durations = avData.durations(:);
valid = isfinite(sizes) & isfinite(durations) & sizes > 0 & durations > 0;
sizes = sizes(valid);
durations = durations(valid);
if numel(sizes) < 2
  title(ax, '⟨S⟩(T) (insufficient data)');
  return;
end

unqDurations = unique(durations);
meanSize = nan(size(unqDurations));
for iDur = 1:numel(unqDurations)
  meanSize(iDur) = mean(sizes(durations == unqDurations(iDur)));
end
validMean = isfinite(meanSize) & meanSize > 0;
unqDurations = unqDurations(validMean);
meanSize = meanSize(validMean);

observedColor = colors_for_tasks(plotConfig.sessionType);
markerArea = plotConfig.observedMarkerSize .^ 2;

hold(ax, 'on');
scatter(ax, unqDurations, meanSize, markerArea, 'filled', ...
  'MarkerEdgeColor', observedColor, 'MarkerFaceColor', observedColor, ...
  'MarkerFaceAlpha', plotConfig.observedMarkerFaceAlpha, ...
  'DisplayName', '⟨S⟩|T');

paramSD = nan;
logCoeff = nan;
durMin = nan;
durMax = nan;
if isfield(avData, 'minDurFit')
  durMin = avData.minDurFit;
end
if isfield(avData, 'maxDurFit')
  durMax = avData.maxDurFit;
end
if isfinite(durMin) && isfinite(durMax) && durMin <= durMax
  [paramSD, ~, logCoeff] = size_given_duration(sizes, durations, ...
    'durmin', durMin, 'durmax', durMax);
elseif isfield(avData, 'paramSD') && isfinite(avData.paramSD)
  paramSD = avData.paramSD;
end

fitPlotted = false;
if isfinite(paramSD) && isfinite(logCoeff) && isfinite(durMin) && isfinite(durMax) ...
    && durMin > 0 && durMax > durMin
  xFit = logspace(log10(durMin), log10(durMax), 80);
  yFit = 10 .^ (paramSD * log10(xFit) + logCoeff);
  plot(ax, xFit, yFit, '-', 'Color', [0.85, 0.2, 0.15], ...
    'LineWidth', plotConfig.fitLineWidth, ...
    'DisplayName', sprintf('1/\\sigma\\nu z=%.2f', paramSD));
  fitPlotted = true;
end

set(ax, 'XScale', 'log', 'YScale', 'log', 'FontSize', plotConfig.tickLabelFontSize, ...
  'LineWidth', plotConfig.axesLineWidth);
axis(ax, 'square');
xlabel(ax, 'Duration (bins)', 'FontSize', plotConfig.axisLabelFontSize);
ylabel(ax, '\langleS\rangle(T)', 'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', 'tex');
if fitPlotted
  legend(ax, 'Location', plotConfig.legendLocation, 'FontSize', plotConfig.tickLabelFontSize, ...
    'Interpreter', 'tex');
end

dccText = '';
if isfield(avData, 'dcc') && isfinite(avData.dcc)
  dccText = sprintf(', dcc=%.3f', avData.dcc);
end
if isfinite(paramSD)
  title(ax, sprintf('\\langleS\\rangle(T) (1/\\sigma\\nu z = %.2f%s)', paramSD, dccText), ...
    'Interpreter', 'tex');
else
  title(ax, '⟨S⟩(T)');
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

function binSize = resolve_avalanche_duration_bin_size(avData)
% RESOLVE_AVALANCHE_DURATION_BIN_SIZE - Bin width (s) for duration conversion
%
% Variables:
%   avData - Area avalanche result struct with optional .binSize
%
% Returns:
%   binSize - Spike bin width in seconds

if isfield(avData, 'binSize') && isscalar(avData.binSize) && isfinite(avData.binSize) && avData.binSize > 0
  binSize = avData.binSize;
else
  binSize = 0.05;
  warning('session_avalanche_distributions:missingBinSize', ...
    'avData.binSize missing; assuming %.3f s for duration conversion.', binSize);
end
end

function fig = get_task_figure_handle(sessionType, plotKind, cellType, figName)
% GET_TASK_FIGURE_HANDLE - Reuse or create a task-specific figure window
%
% Variables:
%   sessionType - Task name for figure_number_for_task
%   plotKind    - 'distributions' or 'ei_summary'
%   cellType    - Population tag for distributions figures
%   figName     - Figure window title
%
% Returns:
%   fig - Figure handle (cleared and ready for new content)

figNumber = figure_number_for_task(sessionType, plotKind, cellType);
fig = figure(figNumber);
set(fig, 'Color', 'w', 'Name', figName);
clf(fig);
end

function apply_portrait_figure_size(fig, figureWidthInches, nRows, nCols)
% APPLY_PORTRAIT_FIGURE_SIZE - Size figure for portrait PDF export
%
% Variables:
%   fig               - Figure handle
%   figureWidthInches - Target width in inches (portrait page with margins)
%   nRows, nCols      - tiledlayout grid for height estimate
%
% Goal:
%   Keep exported figure width within a portrait PDF page.

layoutPadIn = 0.55;
titlePadIn = 0.45;
usableWidth = max(figureWidthInches - layoutPadIn, 1);
panelWidth = usableWidth / max(nCols, 1);
panelHeight = panelWidth;
figHeight = nRows * panelHeight + layoutPadIn + titlePadIn;
maxPortraitHeightIn = 9.5;
figHeight = min(figHeight, maxPortraitHeightIn);

set(fig, 'Units', 'inches', 'Position', [1, 1, figureWidthInches, figHeight]);
set(fig, 'PaperUnits', 'inches', 'PaperSize', [figureWidthInches, figHeight], ...
  'PaperPositionMode', 'auto');
end
