function out = session_engagement_criticality(subjectName, sessionName, windowEngaged, windowNotEngaged, varargin)
% SESSION_ENGAGEMENT_CRITICALITY - Compare criticality metrics across two session windows
%
% Variables:
%   subjectName      - Subject folder (e.g. 'ey9166')
%   sessionName      - Session identifier (e.g. 'ey9166_2026_04_03')
%   windowEngaged    - [startSec, endSec] or struct with .start, .end, optional .label
%   windowNotEngaged - Same format for the comparison window (e.g. disengaged / passive)
%   varargin         - Name-value analysis options (see below)
%
% Goal:
%   For one session and brain area, run avalanche, d2 (AR), and PRG analyses on each
%   broad time window and plot three figures (one per approach), overlaying both segments.
%
% Name-value options:
%   SessionType, DataSource, BrainArea
%   SaveFigure (default false), OutputDir (default dropPath/interval_timing_task)
%   D2Window, PrgWindow, UseLog10D2, PrgMethod, SurrogateMethod
%   PowerLawFitMethod, AvalancheDetectionMode, UseSubsampling, ...
%
% Returns:
%   out - Struct with segments, avResults, arResults, prgResults, figHandles, summaryTable, areas

setup_engagement_criticality_paths();

if nargin < 4
  error('session_engagement_criticality:NotEnoughInputs', ...
    'Usage: session_engagement_criticality(subjectName, sessionName, windowEngaged, windowNotEngaged, ...)');
end
if isempty(subjectName) || isempty(sessionName)
  error('session_engagement_criticality:MissingSession', ...
    'subjectName and sessionName are required.');
end

p = inputParser;
p.addParameter('SessionType', 'interval', @ischar);
p.addParameter('DataSource', 'spikes', @ischar);
p.addParameter('BrainArea', 'VS', @ischar);
p.addParameter('SaveFigure', false, @islogical);
p.addParameter('OutputDir', '', @ischar);
p.addParameter('D2Window', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('PrgWindow', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('UseLog10D2', true, @islogical);
p.addParameter('PrgMethod', 'pca', @ischar);
p.addParameter('SurrogateMethod', 'isi', @ischar);
p.addParameter('PowerLawFitMethod', 'hybrid', @ischar);
p.addParameter('RunClausetPlpva', false, @islogical);
p.addParameter('AvalancheDetectionMode', 'fixedBinMedian', @ischar);
p.addParameter('UseSubsampling', false, @islogical);
p.addParameter('NSubsamples', 20, @isnumeric);
p.addParameter('NNeuronsSubsample', 20, @isnumeric);
p.addParameter('MinNeuronsMultiple', 1.25, @isnumeric);
p.addParameter('EngagedLabel', '', @ischar);
p.addParameter('NotEngagedLabel', '', @ischar);
p.parse(varargin{:});
cfg = p.Results;
cfg.SubjectName = subjectName;
cfg.SessionName = sessionName;

segments(1) = parse_engagement_window(windowEngaged, cfg.EngagedLabel, 'Engaged');
segments(2) = parse_engagement_window(windowNotEngaged, cfg.NotEngagedLabel, 'Not engaged');

if segments(1).start >= segments(1).end || segments(2).start >= segments(2).end
  error('session_engagement_criticality:InvalidWindow', ...
    'Each window must have start < end (seconds).');
end

collectStart = min([segments.start]);
collectEnd = max([segments.end]);

fprintf('\n=== Session engagement criticality ===\n');
fprintf('Session [%s]: %s\n', cfg.SessionType, cfg.SessionName);
fprintf('Segment A (%s): [%.1f, %.1f] s (%.1f min)\n', segments(1).label, ...
  segments(1).start, segments(1).end, segments(1).duration / 60);
fprintf('Segment B (%s): [%.1f, %.1f] s (%.1f min)\n', segments(2).label, ...
  segments(2).start, segments(2).end, segments(2).duration / 60);
fprintf('Load span: [%.1f, %.1f] s\n', collectStart, collectEnd);

opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 150;

subjectNameForLoad = cfg.SubjectName;
loadArgs = build_session_load_args(cfg.SessionType, cfg.SessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(cfg.SessionType, cfg.DataSource, loadArgs{:});

[dataStruct, areaOk] = apply_brain_area_selection(dataStruct, cfg.BrainArea);
if ~areaOk
  error('Brain area "%s" not available in this session.', cfg.BrainArea);
end

[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();
avConfig = build_av_config(cfg, clausetPlfitPath, plfit2023Path);
arConfig = build_ar_config(cfg);
prgConfig = build_prg_config(cfg);

if config_include_m2356(avConfig)
  dataStruct = maybe_add_m2356_area(dataStruct, avConfig.includeM2356);
end

areasToAnalyze = resolve_areas_to_analyze(dataStruct, cfg.BrainArea, avConfig.nMinNeurons);
if isempty(areasToAnalyze)
  error('No areas meet minimum neuron count (%d).', avConfig.nMinNeurons);
end
areaNames = dataStruct.areas(areasToAnalyze);

avResults = cell(1, 2);
arResults = cell(1, 2);
prgResults = cell(1, 2);

for sIdx = 1:2
  seg = segments(sIdx);
  fprintf('\n--- %s [%.1f, %.1f] s ---\n', seg.label, seg.start, seg.end);

  dataStructSeg = restrict_collect_window(dataStruct, seg.start, seg.end);

  avResults{sIdx} = run_segment_avalanche_analysis( ...
    dataStructSeg, areasToAnalyze, avConfig, seg);
  arResults{sIdx} = run_segment_d2_analysis( ...
    dataStructSeg, arConfig, cfg.BrainArea, seg, cfg.UseLog10D2);
  prgResults{sIdx} = run_segment_prg_analysis( ...
    dataStructSeg, prgConfig, cfg.BrainArea, seg);
end

figAv = plot_engagement_avalanche_comparison( ...
  avResults, segments, areaNames, cfg.SessionName);
figD2 = plot_engagement_d2_comparison( ...
  arResults, segments, cfg.UseLog10D2, cfg.SessionType, cfg.SessionName, cfg.D2Window);
figPrg = plot_engagement_prg_comparison( ...
  prgResults, segments, prgConfig.finalCutoffDivisor, cfg.SessionType, ...
  cfg.SessionName, cfg.PrgWindow, prgConfig.prgMethod);

figHandles = struct('avalanche', figAv, 'd2', figD2, 'prg', figPrg);

if cfg.SaveFigure
  saveDir = cfg.OutputDir;
  if isempty(saveDir)
    paths = get_paths();
    saveDir = fullfile(paths.dropPath, 'interval_timing_task');
  end
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  areaTag = format_areas_label(areaNames);
  segTag = sprintf('%.0f-%.0fs_vs_%.0f-%.0fs', ...
    segments(1).start, segments(1).end, segments(2).start, segments(2).end);
  baseName = sprintf('session_engagement_criticality_%s_%s_%s', ...
    cfg.SessionName, areaTag, segTag);
  exportgraphics(figAv, fullfile(saveDir, [baseName, '_avalanche.png']), 'Resolution', 300);
  exportgraphics(figD2, fullfile(saveDir, [baseName, '_d2.png']), 'Resolution', 300);
  exportgraphics(figPrg, fullfile(saveDir, [baseName, '_prg.png']), 'Resolution', 300);
  fprintf('\nSaved figures to %s (%s_*)\n', saveDir, baseName);
end

summaryTable = build_engagement_summary_table( ...
  avResults, arResults, prgResults, segments, areaNames, cfg.UseLog10D2, ...
  prgConfig.finalCutoffDivisor);
print_engagement_summary_table(summaryTable, segments);

out = struct();
out.segments = segments;
out.areas = areaNames;
out.avResults = avResults;
out.arResults = arResults;
out.prgResults = prgResults;
out.figHandles = figHandles;
out.summaryTable = summaryTable;
out.config = cfg;

fprintf('\n=== Done ===\n');
end

%% Path setup and configuration

function setup_engagement_criticality_paths()
% SETUP_ENGAGEMENT_CRITICALITY_PATHS - Add neuro-behavior paths for this analysis

scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  scriptDir = fileparts(which('session_engagement_criticality'));
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
end

function seg = parse_engagement_window(windowSpec, labelOverride, defaultLabel)
% PARSE_ENGAGEMENT_WINDOW - Normalize window input to struct with start, end, label

if isnumeric(windowSpec)
  windowSpec = windowSpec(:)';
  if numel(windowSpec) ~= 2
    error('Numeric window must be [startSec, endSec].');
  end
  seg.start = windowSpec(1);
  seg.end = windowSpec(2);
elseif isstruct(windowSpec)
  if ~isfield(windowSpec, 'start') || ~isfield(windowSpec, 'end')
    error('Window struct must include .start and .end (seconds).');
  end
  seg.start = windowSpec.start;
  seg.end = windowSpec.end;
else
  error('Window must be [start, end] or a struct with .start and .end.');
end

if ~isempty(labelOverride)
  seg.label = labelOverride;
elseif isstruct(windowSpec) && isfield(windowSpec, 'label') && ~isempty(windowSpec.label)
  seg.label = windowSpec.label;
else
  seg.label = defaultLabel;
end

seg.duration = seg.end - seg.start;
end

function avConfig = build_av_config(cfg, clausetPlfitPath, plfit2023Path)
% BUILD_AV_CONFIG - Avalanche analysis settings (single window per segment)

avConfig = struct();
avConfig.useOptimalBinWindowFunction = false;
avConfig.avalancheDetectionMode = cfg.AvalancheDetectionMode;
if strcmpi(cfg.AvalancheDetectionMode, 'fixedBinMedian')
  avConfig.binSize = 0.05;
end
avConfig.thresholdFlag = 1;
avConfig.thresholdPct = 1;
avConfig.nMinNeurons = 20;
avConfig.useSubsampling = cfg.UseSubsampling;
avConfig.nSubsamples = cfg.NSubsamples;
avConfig.nNeuronsSubsample = cfg.NNeuronsSubsample;
avConfig.minNeuronsMultiple = cfg.MinNeuronsMultiple;
avConfig.pcaFlag = 0;
avConfig.gofThreshold = 0.8;
avConfig.powerLawFitMethod = cfg.PowerLawFitMethod;
avConfig.runClausetPlpva = cfg.RunClausetPlpva;
avConfig.clausetPlfitPath = clausetPlfitPath;
avConfig.plfit2023Path = plfit2023Path;
avConfig.includeM2356 = false;
if ~isempty(cfg.BrainArea) && strcmpi(cfg.BrainArea, 'M2356')
  avConfig.includeM2356 = true;
end
end

function arConfig = build_ar_config(cfg)
% BUILD_AR_CONFIG - d2 / AR analysis settings (non-overlapping windows within segment)

arConfig = struct();
arConfig.slidingWindowSize = cfg.D2Window;
arConfig.stepSize = cfg.D2Window;
arConfig.binSize = 0.025;
arConfig.useOptimalBinWindowFunction = false;
arConfig.analyzeD2 = true;
arConfig.analyzeMrBr = false;
arConfig.pcaFlag = 0;
arConfig.pcaFirstFlag = 1;
arConfig.nDim = 4;
arConfig.enablePermutations = true;
arConfig.nShuffles = 50;
arConfig.normalizeD2 = true;
arConfig.useLog10D2 = cfg.UseLog10D2;
arConfig.makePlots = false;
arConfig.saveData = false;
arConfig.pOrder = 10;
arConfig.critType = 2;
arConfig.minSpikesPerBin = 2.5;
arConfig.minBinsPerWindow = 1000;
arConfig.maxSpikesPerBin = 100;
arConfig.nMinNeurons = 25;
arConfig.useSubsampling = cfg.UseSubsampling;
arConfig.nSubsamples = cfg.NSubsamples;
arConfig.nNeuronsSubsample = cfg.NNeuronsSubsample;
arConfig.minNeuronsMultiple = cfg.MinNeuronsMultiple;
arConfig.includeM2356 = false;
if ~isempty(cfg.BrainArea) && strcmpi(cfg.BrainArea, 'M2356')
  arConfig.includeM2356 = true;
end
end

function prgConfig = build_prg_config(cfg)
% BUILD_PRG_CONFIG - PRG kurtosis settings (non-overlapping blocks within segment)

prgConfig = struct();
prgConfig.prgMethod = cfg.PrgMethod;
prgConfig.blockWindowSize = cfg.PrgWindow;
prgConfig.binSize = 0.05;
prgConfig.cvThreshold = 5;
prgConfig.cutoffDivisors = [1, 2, 4, 8, 16];
prgConfig.finalCutoffDivisor = 16;
prgConfig.kappaAxisMax = 20;
prgConfig.enableSurrogates = true;
prgConfig.nSurrogates = 1;
prgConfig.surrogateMethod = cfg.SurrogateMethod;
prgConfig.makePlots = false;
prgConfig.saveData = false;
prgConfig.plotTimeSeries = false;
prgConfig.nMinNeurons = 20;
prgConfig.useSubsampling = cfg.UseSubsampling;
prgConfig.nSubsamples = cfg.NSubsamples;
prgConfig.nNeuronsSubsample = cfg.NNeuronsSubsample;
prgConfig.minNeuronsMultiple = cfg.MinNeuronsMultiple;
prgConfig.includeM2356 = false;
if ~isempty(cfg.BrainArea) && strcmpi(cfg.BrainArea, 'M2356')
  prgConfig.includeM2356 = true;
end
end

function dataStructSeg = restrict_collect_window(dataStruct, collectStart, collectEnd)
% RESTRICT_COLLECT_WINDOW - Set spikeData collect bounds for one segment

dataStructSeg = dataStruct;
if isfield(dataStructSeg, 'spikeData') && isstruct(dataStructSeg.spikeData)
  dataStructSeg.spikeData.collectStart = collectStart;
  dataStructSeg.spikeData.collectEnd = collectEnd;
else
  dataStructSeg.spikeData = struct('collectStart', collectStart, 'collectEnd', collectEnd);
end
end

function effectiveWindow = effective_block_window(requestedWindow, segmentDuration, labelText)
% EFFECTIVE_BLOCK_WINDOW - Cap block/window size to segment length

effectiveWindow = min(requestedWindow, segmentDuration);
if effectiveWindow < requestedWindow
  warning('session_engagement_criticality:ShortSegment', ...
    '%s duration (%.1f s) < requested window (%.1f s); using %.1f s.', ...
    labelText, segmentDuration, requestedWindow, effectiveWindow);
end
if effectiveWindow < 5
  warning('session_engagement_criticality:VeryShortSegment', ...
    '%s effective window is only %.1f s; results may be sparse.', ...
    labelText, effectiveWindow);
end
end

%% Per-segment analyses

function segAv = run_segment_avalanche_analysis(dataStruct, areasToAnalyze, avConfig, seg)
% RUN_SEGMENT_AVALANCHE_ANALYSIS - Avalanche CCDF data for one segment

segAv = struct('segment', seg, 'areas', {{}}, 'byArea', {{}});
windowDurationSec = seg.duration;
localConfig = avConfig;
localConfig.slidingWindowSize = windowDurationSec;
localConfig.avStepSize = windowDurationSec;

for aIdx = 1:numel(areasToAnalyze)
  areaIndex = areasToAnalyze(aIdx);
  areaName = dataStruct.areas{areaIndex};
  avData = extract_area_avalanches(dataStruct, areaIndex, localConfig, seg.start, seg.end);
  segAv.areas{end+1} = areaName; %#ok<AGROW>
  segAv.byArea{end+1} = avData; %#ok<AGROW>

  if avData.hasAvalanches
    fprintf('  AV %s: n=%d, tau=%.2f, alpha=%.2f, (alpha-1)/(tau-1)=%.3f\n', ...
      areaName, avData.nAvalanches, avData.tau, avData.alpha, avData.scalingRelation);
  else
    fprintf('  AV %s: no avalanches\n', areaName);
  end
end
end

function segAr = run_segment_d2_analysis(dataStruct, arConfig, brainArea, seg, useLog10D2)
% RUN_SEGMENT_D2_ANALYSIS - d2 pipeline for one segment

localConfig = arConfig;
localConfig.slidingWindowSize = effective_block_window(arConfig.slidingWindowSize, seg.duration, seg.label);
localConfig.stepSize = localConfig.slidingWindowSize;

results = criticality_ar_analysis(dataStruct, localConfig);
if ~isempty(brainArea)
  results = filter_ar_results_to_brain_area(results, brainArea);
end

print_segment_d2_summary(results, seg.label, useLog10D2);

segAr = struct('segment', seg, 'results', results, 'd2Window', localConfig.slidingWindowSize);
end

function segPrg = run_segment_prg_analysis(dataStruct, prgConfig, brainArea, seg)
% RUN_SEGMENT_PRG_ANALYSIS - PRG kurtosis pipeline for one segment

localConfig = prgConfig;
localConfig.blockWindowSize = effective_block_window(prgConfig.blockWindowSize, seg.duration, seg.label);

results = criticality_prg_analysis(dataStruct, localConfig);
if ~isempty(brainArea)
  results = filter_prg_results_to_brain_area(results, brainArea);
end

print_segment_kappa_summary(results, seg.label);

segPrg = struct('segment', seg, 'results', results, 'prgWindow', localConfig.blockWindowSize);
end

%% Comparison plots

function fig = plot_engagement_avalanche_comparison(avResults, segments, areaNames, sessionName)
% PLOT_ENGAGEMENT_AVALANCHE_COMPARISON - CCDF size and duration for both segments

segmentColors = engagement_segment_colors();
nAreas = numel(areaNames);
fig = figure('Name', 'Engagement avalanche comparison', 'Color', 'w', ...
  'Position', [80 80 max(900, 360 * nAreas) 420]);
tiledlayout(fig, nAreas, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for aIdx = 1:nAreas
  axSize = nexttile((aIdx - 1) * 2 + 1);
  axDur = nexttile((aIdx - 1) * 2 + 2);
  hold(axSize, 'on');
  hold(axDur, 'on');

  for sIdx = 1:2
    avData = avResults{sIdx}.byArea{aIdx};
    if ~avData.hasAvalanches
      continue;
    end
    plot_empirical_ccdf(axSize, avData.sizes, segmentColors(sIdx, :), segments(sIdx).label);
    plot_empirical_ccdf(axDur, avData.durations, segmentColors(sIdx, :), segments(sIdx).label);
  end

  set(axSize, 'XScale', 'log', 'YScale', 'log');
  set(axDur, 'XScale', 'log', 'YScale', 'log');
  xlabel(axSize, 'Avalanche size');
  xlabel(axDur, 'Avalanche duration (bins)');
  ylabel(axSize, 'P(X \geq x)');
  ylabel(axDur, 'P(X \geq x)');
  title(axSize, sprintf('%s — size', areaNames{aIdx}), 'Interpreter', 'none');
  title(axDur, sprintf('%s — duration', areaNames{aIdx}), 'Interpreter', 'none');
  grid(axSize, 'on');
  grid(axDur, 'on');
  legend(axSize, 'Location', 'southwest');
  legend(axDur, 'Location', 'southwest');
  hold(axSize, 'off');
  hold(axDur, 'off');
end

sgtitle(fig, sprintf('%s — avalanche CCDFs by engagement window', sessionName), ...
  'FontWeight', 'bold', 'Interpreter', 'none');
end

function fig = plot_engagement_d2_comparison(arResults, segments, useLog10D2, sessionType, sessionName, d2Window)
% PLOT_ENGAGEMENT_D2_COMPARISON - Raw and normalized d2 PDFs for both segments

segmentColors = engagement_segment_colors();
plotData = build_engagement_d2_plot_data(arResults, useLog10D2);
numAreas = numel(plotData.areas);
if numAreas == 0
  error('No finite d2 values for either segment.');
end

allRaw = collect_plot_data_values(plotData.segD2, true);
allNorm = collect_plot_data_values(plotData.segD2Norm, false);
[rawBinEdges, rawXMin, rawXMax] = build_shared_bin_edges(allRaw, 28);
[normBinEdges, normXMin, normXMax] = build_shared_bin_edges(allNorm, 28, [0.5, 2]);

fig = figure('Color', 'w', 'Position', [120 120 1200 260 * numAreas], ...
  'Name', 'Engagement d2 comparison');
tileLayout = tiledlayout(numAreas, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:numAreas
  axRaw = nexttile(tileLayout, (a - 1) * 2 + 1);
  plot_engagement_metric_histogram(axRaw, plotData.segD2, a, segments, segmentColors, ...
    rawBinEdges, rawXMin, rawXMax, useLog10D2, nan);
  title(axRaw, sprintf('%s — raw', plotData.areas{a}), 'Interpreter', 'none');
  ylabel(axRaw, 'Probability density');
  if useLog10D2
    xlabel(axRaw, 'log_{10}(d2)');
  else
    xlabel(axRaw, 'd2');
  end

  axNorm = nexttile(tileLayout, (a - 1) * 2 + 2);
  normRef = 1;
  if useLog10D2
    plot_engagement_metric_histogram(axNorm, plotData.segD2Norm, a, segments, segmentColors, ...
      normBinEdges, normXMin, normXMax, true, nan);
    xlabel(axNorm, 'log_{10}(d2 / shuffle)');
  else
    plot_engagement_metric_histogram(axNorm, plotData.segD2Norm, a, segments, segmentColors, ...
      normBinEdges, normXMin, normXMax, false, normRef);
    xlabel(axNorm, 'd2 / shuffle');
  end
  title(axNorm, sprintf('%s — normalized', plotData.areas{a}), 'Interpreter', 'none');
  ylabel(axNorm, 'Probability density');
end

sgtitle(tileLayout, sprintf( ...
  'Distribution of d2 by engagement window | %s | %.0fs windows | %s', ...
  sessionType, d2Window, sessionName), 'FontSize', 12, 'Interpreter', 'none');
end

function fig = plot_engagement_prg_comparison(prgResults, segments, finalCutoffDivisor, ...
    sessionType, sessionName, prgWindow, prgMethod)
% PLOT_ENGAGEMENT_PRG_COMPARISON - Kappa and D_JS PDFs (raw and normalized) for both segments

segmentColors = engagement_segment_colors();
plotData = build_engagement_prg_plot_data(prgResults);
numAreas = numel(plotData.areas);
if numAreas == 0
  error('No valid PRG kappa values for either segment.');
end

allKappa = collect_plot_data_values(plotData.segKappa, true);
allKappaNorm = collect_plot_data_values(plotData.segKappaNorm, false);
allDjs = collect_plot_data_values(plotData.segDjs, false);
allDjsNorm = collect_plot_data_values(plotData.segDjsNorm, false);

[kappaBinEdges, kappaXMin, kappaXMax] = build_shared_bin_edges(allKappa, 28);
[kappaNormBinEdges, kappaNormXMin, kappaNormXMax] = build_shared_bin_edges(allKappaNorm, 28, [0.5, 2]);
[djsBinEdges, djsXMin, djsXMax] = build_shared_bin_edges(allDjs, 28, [0, 1]);
[djsNormBinEdges, djsNormXMin, djsNormXMax] = build_shared_bin_edges(allDjsNorm, 28, [0.5, 2]);

fig = figure('Color', 'w', 'Position', [140 140 1200 520 * numAreas], ...
  'Name', 'Engagement PRG comparison');
tileLayout = tiledlayout(2 * numAreas, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:numAreas
  rowKappa = (a - 1) * 2 + 1;
  rowDjs = rowKappa + 1;

  axKappaRaw = nexttile(tileLayout, (rowKappa - 1) * 2 + 1);
  plot_engagement_metric_histogram(axKappaRaw, plotData.segKappa, a, segments, segmentColors, ...
    kappaBinEdges, kappaXMin, kappaXMax, false, 3);
  title(axKappaRaw, sprintf('%s — \\kappa raw', plotData.areas{a}), 'Interpreter', 'tex');

  axKappaNorm = nexttile(tileLayout, (rowKappa - 1) * 2 + 2);
  plot_engagement_metric_histogram(axKappaNorm, plotData.segKappaNorm, a, segments, segmentColors, ...
    kappaNormBinEdges, kappaNormXMin, kappaNormXMax, false, 1);
  title(axKappaNorm, sprintf('%s — \\kappa / surrogate', plotData.areas{a}), 'Interpreter', 'tex');

  axDjsRaw = nexttile(tileLayout, (rowDjs - 1) * 2 + 1);
  plot_engagement_metric_histogram(axDjsRaw, plotData.segDjs, a, segments, segmentColors, ...
    djsBinEdges, djsXMin, djsXMax, false, nan);
  title(axDjsRaw, sprintf('%s — D_{JS} raw', plotData.areas{a}), 'Interpreter', 'tex');

  axDjsNorm = nexttile(tileLayout, (rowDjs - 1) * 2 + 2);
  plot_engagement_metric_histogram(axDjsNorm, plotData.segDjsNorm, a, segments, segmentColors, ...
    djsNormBinEdges, djsNormXMin, djsNormXMax, false, 1);
  title(axDjsNorm, sprintf('%s — D_{JS} / surrogate', plotData.areas{a}), 'Interpreter', 'tex');

  ylabel(axKappaRaw, 'Probability density');
  ylabel(axKappaNorm, 'Probability density');
  ylabel(axDjsRaw, 'Probability density');
  ylabel(axDjsNorm, 'Probability density');
  xlabel(axKappaRaw, sprintf('Kurtosis \\kappa (N/%d)', finalCutoffDivisor));
  xlabel(axKappaNorm, sprintf('\\kappa / surrogate (N/%d)', finalCutoffDivisor));
  xlabel(axDjsRaw, 'D_{JS}');
  xlabel(axDjsNorm, 'D_{JS} / surrogate');
end

sgtitle(tileLayout, sprintf( ...
  'PRG (%s) kurtosis and D_{JS} by engagement window | %s | %.0fs blocks | %s', ...
  prgMethod, sessionType, prgWindow, sessionName), 'FontSize', 12, 'Interpreter', 'none');
end

function plot_engagement_metric_histogram(ax, segMetric, areaIdx, segments, segmentColors, ...
    binEdges, xMin, xMax, drawZeroRef, refValue)
% PLOT_ENGAGEMENT_METRIC_HISTOGRAM - Overlapping segment PDFs on one axes

hold(ax, 'on');
for sIdx = 1:2
  vals = segMetric{sIdx}{areaIdx};
  if isempty(vals)
    continue;
  end
  histogram(ax, vals, binEdges, 'Normalization', 'pdf', ...
    'FaceColor', segmentColors(sIdx, :), 'FaceAlpha', 0.4, 'EdgeColor', 'none', ...
    'DisplayName', segments(sIdx).label);
end
if drawZeroRef
  xline(ax, 0, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1, 'HandleVisibility', 'off');
end
if isfinite(refValue)
  xline(ax, refValue, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1, 'HandleVisibility', 'off');
end
xlim(ax, [xMin, xMax]);
grid(ax, 'on');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');
end

function allVals = collect_plot_data_values(segMetric, requireData)
% COLLECT_PLOT_DATA_VALUES - Pool finite values across segments and areas
%
% Variables:
%   requireData - If true, error when no finite values are found

if nargin < 2
  requireData = true;
end

allVals = [];
for sIdx = 1:2
  for a = 1:numel(segMetric{sIdx})
    allVals = [allVals; segMetric{sIdx}{a}(:)]; %#ok<AGROW>
  end
end
allVals = allVals(isfinite(allVals));
if requireData && isempty(allVals)
  error('No finite values available for histogram binning.');
end
end

function colors = engagement_segment_colors()
% ENGAGEMENT_SEGMENT_COLORS - Consistent colors for segment A and B

colors = [0.15, 0.45, 0.75; 0.85, 0.35, 0.15];
end

function plot_empirical_ccdf(ax, values, lineColor, displayName)
% PLOT_EMPIRICAL_CCDF - Log-log CCDF line for one segment

values = values(values > 0 & isfinite(values));
if isempty(values)
  return;
end
uniqueVals = unique(values);
cdfY = arrayfun(@(x) mean(values >= x), uniqueVals);
plot(ax, uniqueVals, cdfY, 'o-', 'Color', lineColor, ...
  'MarkerFaceColor', lineColor, 'MarkerSize', 4, 'LineWidth', 1.5, ...
  'DisplayName', displayName);
end

function plotData = build_engagement_d2_plot_data(arResults, useLog10D2)
% BUILD_ENGAGEMENT_D2_PLOT_DATA - Per-area raw and normalized d2 vectors per segment

plotData = struct();
plotData.areas = {};
plotData.segD2 = { {}, {} };
plotData.segD2Norm = { {}, {} };

for sIdx = 1:2
  results = arResults{sIdx}.results;
  areaNames = normalize_results_area_list(results.areas);
  for a = 1:numel(areaNames)
    areaName = areaNames{a};
    if sIdx == 1
      plotData.areas{end+1} = areaName; %#ok<AGROW>
      plotData.segD2{1}{end+1} = []; %#ok<AGROW>
      plotData.segD2{2}{end+1} = []; %#ok<AGROW>
      plotData.segD2Norm{1}{end+1} = []; %#ok<AGROW>
      plotData.segD2Norm{2}{end+1} = []; %#ok<AGROW>
    end
    areaPlotIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaPlotIdx)
      continue;
    end
    resultsAreaIdx = find(strcmp(normalize_results_area_list(results.areas), areaName), 1);
    plotData.segD2{sIdx}{areaPlotIdx} = extract_finite_d2(results, resultsAreaIdx, useLog10D2);
    plotData.segD2Norm{sIdx}{areaPlotIdx} = extract_finite_d2_normalized(results, resultsAreaIdx, useLog10D2);
  end
end
end

function plotData = build_engagement_prg_plot_data(prgResults)
% BUILD_ENGAGEMENT_PRG_PLOT_DATA - Per-area kappa and D_JS vectors (raw and normalized)

plotData = struct();
plotData.areas = {};
plotData.segKappa = { {}, {} };
plotData.segKappaNorm = { {}, {} };
plotData.segDjs = { {}, {} };
plotData.segDjsNorm = { {}, {} };

for sIdx = 1:2
  results = prgResults{sIdx}.results;
  areaNames = normalize_results_area_list(results.areas);
  for a = 1:numel(areaNames)
    areaName = areaNames{a};
    if sIdx == 1
      plotData.areas{end+1} = areaName; %#ok<AGROW>
      plotData.segKappa{1}{end+1} = []; %#ok<AGROW>
      plotData.segKappa{2}{end+1} = []; %#ok<AGROW>
      plotData.segKappaNorm{1}{end+1} = []; %#ok<AGROW>
      plotData.segKappaNorm{2}{end+1} = []; %#ok<AGROW>
      plotData.segDjs{1}{end+1} = []; %#ok<AGROW>
      plotData.segDjs{2}{end+1} = []; %#ok<AGROW>
      plotData.segDjsNorm{1}{end+1} = []; %#ok<AGROW>
      plotData.segDjsNorm{2}{end+1} = []; %#ok<AGROW>
    end
    areaPlotIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaPlotIdx)
      continue;
    end
    resultsAreaIdx = find(strcmp(normalize_results_area_list(results.areas), areaName), 1);
    plotData.segKappa{sIdx}{areaPlotIdx} = extract_valid_prg_kappa(results, resultsAreaIdx);
    plotData.segKappaNorm{sIdx}{areaPlotIdx} = extract_valid_prg_kappa_normalized(results, resultsAreaIdx);
    plotData.segDjs{sIdx}{areaPlotIdx} = extract_valid_prg_djs(results, resultsAreaIdx);
    plotData.segDjsNorm{sIdx}{areaPlotIdx} = extract_valid_prg_djs_normalized(results, resultsAreaIdx);
  end
end
end

function areaNames = normalize_results_area_list(areasField)
% NORMALIZE_RESULTS_AREA_LIST - Flatten results.areas to a cell of area name strings
%
% Variables:
%   areasField - results.areas from AR/PRG output (cell, char, or nested cell)
%
% Goal:
%   Avoid comma-separated-list errors when a single cell holds multiple names.

areaNames = {};
if isempty(areasField)
  return;
end

if ischar(areasField)
  areaNames = {areasField};
  return;
end

if isstring(areasField)
  areaNames = cellstr(areasField(:));
  return;
end

if ~iscell(areasField)
  areaNames = {char(areasField)};
  return;
end

rawAreas = areasField(:);
for k = 1:numel(rawAreas)
  entry = rawAreas{k};
  if iscell(entry)
    for j = 1:numel(entry)
      if ischar(entry{j}) || isstring(entry{j})
        areaNames{end+1} = char(string(entry{j})); %#ok<AGROW>
      end
    end
  elseif ischar(entry) || isstring(entry)
    areaNames{end+1} = char(string(entry)); %#ok<AGROW>
  end
end
end

function d2Vec = extract_finite_d2(results, areaIdx, useLog10D2)
% EXTRACT_FINITE_D2 - Window-wise d2 values for one area

d2Vec = [];
if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
  return;
end
d2Vec = results.d2{areaIdx}(:);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
d2Vec = d2Vec(isfinite(d2Vec));
end

function d2Vec = extract_finite_d2_normalized(results, areaIdx, useLog10D2)
% EXTRACT_FINITE_D2_NORMALIZED - Window-wise d2/shuffle values for one area

d2Vec = [];
if ~isfield(results, 'd2Normalized') || areaIdx > numel(results.d2Normalized) ...
    || isempty(results.d2Normalized{areaIdx})
  return;
end
d2Vec = results.d2Normalized{areaIdx}(:);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
d2Vec = d2Vec(isfinite(d2Vec));
end

function kappaVec = extract_valid_prg_kappa(results, areaIdx)
% EXTRACT_VALID_PRG_KAPPA - CV-valid kappa values for one area

kappaVec = [];
validMask = get_prg_valid_window_mask(results, areaIdx);
if isempty(validMask)
  return;
end
kappaVec = results.kappa{areaIdx}(:);
kappaVec = kappaVec(validMask);
kappaVec = kappaVec(isfinite(kappaVec));
end

function kappaNormVec = extract_valid_prg_kappa_normalized(results, areaIdx)
% EXTRACT_VALID_PRG_KAPPA_NORMALIZED - CV-valid kappa/surrogate values for one area

kappaNormVec = extract_valid_prg_metric_normalized(results, areaIdx, 'kappa', 'kappaSurrogate');
end

function djsVec = extract_valid_prg_djs(results, areaIdx)
% EXTRACT_VALID_PRG_DJS - CV-valid D_JS values for one area

validMask = get_prg_valid_window_mask(results, areaIdx);
if isempty(validMask) || ~isfield(results, 'djs') || areaIdx > numel(results.djs) ...
    || isempty(results.djs{areaIdx})
  djsVec = [];
  return;
end
djsVec = results.djs{areaIdx}(:);
djsVec = djsVec(validMask);
djsVec = djsVec(isfinite(djsVec));
end

function djsNormVec = extract_valid_prg_djs_normalized(results, areaIdx)
% EXTRACT_VALID_PRG_DJS_NORMALIZED - CV-valid D_JS/surrogate values for one area

djsNormVec = extract_valid_prg_metric_normalized(results, areaIdx, 'djs', 'djsSurrogate');
end

function validMask = get_prg_valid_window_mask(results, areaIdx)
% GET_PRG_VALID_WINDOW_MASK - Finite, non-CV-excluded windows for one area
%
% Goal:
%   Return an n x 1 logical mask aligned with window-wise PRG vectors.

validMask = [];
if areaIdx > numel(results.kappa) || isempty(results.kappa{areaIdx})
  return;
end

validMask = isfinite(results.kappa{areaIdx}(:));
if isfield(results, 'windowExcluded') && areaIdx <= numel(results.windowExcluded) ...
    && ~isempty(results.windowExcluded{areaIdx})
  excluded = results.windowExcluded{areaIdx}(:);
  if numel(excluded) == numel(validMask)
    validMask = validMask & ~excluded;
  end
end
end

function normVec = extract_valid_prg_metric_normalized(results, areaIdx, dataField, surrField)
% EXTRACT_VALID_PRG_METRIC_NORMALIZED - Valid-window metric / mean(surrogate)

normVec = [];
validMask = get_prg_valid_window_mask(results, areaIdx);
if isempty(validMask) || ~isfield(results, dataField) || areaIdx > numel(results.(dataField)) ...
    || isempty(results.(dataField){areaIdx}) || ~isfield(results, surrField) ...
    || areaIdx > numel(results.(surrField)) || isempty(results.(surrField){areaIdx})
  return;
end

dataVec = results.(dataField){areaIdx}(:);
surrMat = results.(surrField){areaIdx};
if size(surrMat, 1) ~= numel(dataVec)
  return;
end
surrMeanVec = mean(surrMat, 2, 'omitnan');
normFull = normalize_prg_by_surrogate(dataVec, surrMeanVec);
normVec = normFull(validMask);
normVec = normVec(isfinite(normVec));
end

function normVec = normalize_prg_by_surrogate(dataVec, surrogateMeanVec)
% NORMALIZE_PRG_BY_SURROGATE - Per-window ratio to mean surrogate value

dataVec = dataVec(:);
surrogateMeanVec = surrogateMeanVec(:);
normVec = nan(size(dataVec));
if numel(dataVec) ~= numel(surrogateMeanVec)
  return;
end
validDenom = isfinite(dataVec) & isfinite(surrogateMeanVec) & surrogateMeanVec > 0;
normVec(validDenom) = dataVec(validDenom) ./ surrogateMeanVec(validDenom);
end

%% Shared helpers (aligned with criticality_manuscript session scripts)

function tf = config_include_m2356(analysisConfig)
tf = isfield(analysisConfig, 'includeM2356') && analysisConfig.includeM2356;
end

function [dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea)
areaOk = true;
if isempty(brainArea)
  return;
end
if strcmpi(brainArea, 'M2356')
  areaOk = any(strcmp(dataStruct.areas, 'M23')) && any(strcmp(dataStruct.areas, 'M56'));
  return;
end
areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
  areaOk = false;
  return;
end
dataStruct.areasToTest = areaIdx;
fprintf('  Restricting analysis to area: %s\n', brainArea);
end

function dataStruct = maybe_add_m2356_area(dataStruct, includeM2356)
if ~includeM2356
  return;
end
areas = dataStruct.areas;
idxM23 = find(strcmp(areas, 'M23'), 1);
idxM56 = find(strcmp(areas, 'M56'), 1);
if isempty(idxM23) || isempty(idxM56) || any(strcmp(areas, 'M2356'))
  return;
end
areas{end+1} = 'M2356';
dataStruct.areas = areas;
dataStruct.idMatIdx{end+1} = [dataStruct.idMatIdx{idxM23}(:); dataStruct.idMatIdx{idxM56}(:)];
if isfield(dataStruct, 'idLabel')
  dataStruct.idLabel{end+1} = [dataStruct.idLabel{idxM23}(:); dataStruct.idLabel{idxM56}(:)];
end
fprintf('  Added combined M2356 area\n');
end

function areasToAnalyze = resolve_areas_to_analyze(dataStruct, brainArea, nMinNeurons)
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
avData = struct('hasAvalanches', false, 'sizes', [], 'durations', [], ...
  'tau', nan, 'alpha', nan, 'scalingRelation', nan, ...
  'minSizeFit', nan, 'maxSizeFit', nan, ...
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
avData.scalingRelation = compute_avalanche_scaling_relation(avData.tau, avData.alpha);
avData.minSizeFit = sizeFit.fitMin;
avData.maxSizeFit = sizeFit.fitMax;
avData.minDurFit = durFit.fitMin;
avData.maxDurFit = durFit.fitMax;
avData.sizeFitInfo = sizeFit;
avData.durFitInfo = durFit;
avData.nAvalanches = numel(sizes);
end

function results = filter_ar_results_to_brain_area(results, brainArea)
if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end
areaNames = normalize_results_area_list(results.areas);
areaIdx = find(strcmp(areaNames, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end
cellFields = {'d2', 'd2Normalized', 'startS', 'd2Permuted', 'mrBrPermuted', ...
  'd2PermutedMean', 'd2PermutedSEM', 'popActivityWindows', 'popActivityFull'};
results.areas = {areaNames{areaIdx}};
for f = 1:length(cellFields)
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
end

function results = filter_prg_results_to_brain_area(results, brainArea)
if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end
areaNames = normalize_results_area_list(results.areas);
areaIdx = find(strcmp(areaNames, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end
cellFields = {'kappa', 'kappaByCutoff', 'windowStartS', 'popCv', 'windowExcluded', ...
  'nNeuronsPerWindow', 'kappaSurrogate', 'djs', 'djsSurrogate', 'nCutoffList'};
results.areas = {areaNames{areaIdx}};
for f = 1:length(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
end

function print_segment_d2_summary(results, segmentLabel, useLog10D2)
areaNames = normalize_results_area_list(results.areas);
for a = 1:numel(areaNames)
  if a > numel(results.d2) || isempty(results.d2{a})
    fprintf('  d2 %s | %s: no data\n', segmentLabel, areaNames{a});
    continue;
  end
  d2Vec = extract_finite_d2(results, a, useLog10D2);
  if isempty(d2Vec)
    fprintf('  d2 %s | %s: no finite values\n', segmentLabel, areaNames{a});
  else
    fprintf('  d2 %s | %s: %d windows, mean = %.4f\n', ...
      segmentLabel, areaNames{a}, numel(d2Vec), mean(d2Vec));
  end
end
end

function print_segment_kappa_summary(results, segmentLabel)
areaNames = normalize_results_area_list(results.areas);
for a = 1:numel(areaNames)
  if a > numel(results.kappa) || isempty(results.kappa{a})
    fprintf('  PRG %s | %s: no kappa data\n', segmentLabel, areaNames{a});
    continue;
  end
  kappaVec = extract_valid_prg_kappa(results, a);
  kappaNormVec = extract_valid_prg_kappa_normalized(results, a);
  djsVec = extract_valid_prg_djs(results, a);
  djsNormVec = extract_valid_prg_djs_normalized(results, a);
  nExcluded = sum(results.windowExcluded{a}(:));
  if isempty(kappaVec)
    fprintf('  PRG %s | %s: no valid windows (%d CV-excluded)\n', ...
      segmentLabel, areaNames{a}, nExcluded);
  else
    fprintf('  PRG %s | %s: %d valid windows (%d excluded), mean \\kappa = %.3f', ...
      segmentLabel, areaNames{a}, numel(kappaVec), nExcluded, mean(kappaVec));
    if ~isempty(kappaNormVec)
      fprintf(', mean \\kappa/surrogate = %.3f', mean(kappaNormVec));
    end
    if ~isempty(djsVec)
      fprintf(', mean D_{JS} = %.4f', mean(djsVec));
    end
    if ~isempty(djsNormVec)
      fprintf(', mean D_{JS}/surrogate = %.4f', mean(djsNormVec));
    end
    fprintf('\n');
  end
end
end

function summaryTable = build_engagement_summary_table( ...
    avResults, arResults, prgResults, segments, areaNames, useLog10D2, finalCutoffDivisor)
% BUILD_ENGAGEMENT_SUMMARY_TABLE - Engaged vs not-engaged metrics across analyses
%
% Variables:
%   avResults, arResults, prgResults - Per-segment result structs
%   segments                         - Two segment descriptors with .label
%   areaNames                        - Brain areas analyzed
%   useLog10D2                       - Whether d2 means use log10 transform
%   finalCutoffDivisor               - PRG cutoff label (N/divisor)
%
% Returns:
%   summaryTable - Table with Area, Metric, and one column per segment

segColNames = { ...
  matlab.lang.makeValidName(segments(1).label, 'ReplacementStyle', 'underscore'), ...
  matlab.lang.makeValidName(segments(2).label, 'ReplacementStyle', 'underscore') ...
  };

rowArea = strings(0, 1);
rowMetric = strings(0, 1);
rowSeg1 = [];
rowSeg2 = [];

if useLog10D2
  d2RawLabel = 'mean log_{10}(d2)';
  d2NormLabel = 'mean log_{10}(d2 / shuffle)';
else
  d2RawLabel = 'mean d2';
  d2NormLabel = 'mean d2 / shuffle';
end

kappaLabel = sprintf('mean \\kappa (N/%d)', finalCutoffDivisor);
kappaNormLabel = sprintf('mean \\kappa / surrogate (N/%d)', finalCutoffDivisor);
djsLabel = 'mean D_{JS}';
djsNormLabel = 'mean D_{JS} / surrogate';

for aIdx = 1:numel(areaNames)
  areaName = char(areaNames{aIdx});

  av1 = get_av_summary_metrics(avResults{1}, aIdx);
  av2 = get_av_summary_metrics(avResults{2}, aIdx);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, 'AV tau', av1.tau, av2.tau);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, 'AV alpha', av1.alpha, av2.alpha);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, 'AV (alpha-1)/(tau-1)', ...
    av1.scalingRelation, av2.scalingRelation);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, 'AV n avalanches', ...
    av1.nAvalanches, av2.nAvalanches);

  ar1 = get_ar_summary_metrics(arResults{1}.results, aIdx, useLog10D2);
  ar2 = get_ar_summary_metrics(arResults{2}.results, aIdx, useLog10D2);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, d2RawLabel, ar1.d2Mean, ar2.d2Mean);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, d2NormLabel, ar1.d2NormMean, ar2.d2NormMean);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, 'd2 n windows', ar1.nWindows, ar2.nWindows);

  prg1 = get_prg_summary_metrics(prgResults{1}.results, aIdx);
  prg2 = get_prg_summary_metrics(prgResults{2}.results, aIdx);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, kappaLabel, prg1.kappaMean, prg2.kappaMean);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, kappaNormLabel, ...
    prg1.kappaNormMean, prg2.kappaNormMean);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, djsLabel, prg1.djsMean, prg2.djsMean);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, djsNormLabel, ...
    prg1.djsNormMean, prg2.djsNormMean);
  [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, 'PRG n valid windows', ...
    prg1.nWindows, prg2.nWindows);
end

summaryTable = table(rowArea, rowMetric, rowSeg1, rowSeg2, ...
  'VariableNames', {'Area', 'Metric', segColNames{1}, segColNames{2}});
summaryTable.Properties.UserData.segmentLabels = {segments(1).label, segments(2).label};
end

function print_engagement_summary_table(summaryTable, segments)
% PRINT_ENGAGEMENT_SUMMARY_TABLE - Command-window table of segment comparisons

if nargin < 2 || isempty(segments)
  if isfield(summaryTable.Properties, 'UserData') ...
      && isfield(summaryTable.Properties.UserData, 'segmentLabels')
    segLabels = summaryTable.Properties.UserData.segmentLabels;
  else
    segLabels = summaryTable.Properties.VariableNames(3:4);
  end
else
  segLabels = {segments(1).label, segments(2).label};
end

varNames = summaryTable.Properties.VariableNames;
segCol1 = varNames{3};
segCol2 = varNames{4};

fprintf('\n=== Engagement summary: %s vs %s ===\n', segLabels{1}, segLabels{2});
fprintf('%-10s %-36s %16s %16s\n', 'Area', 'Metric', segLabels{1}, segLabels{2});
fprintf('%s\n', repmat('-', 1, 82));

for r = 1:height(summaryTable)
  metricName = char(summaryTable.Metric(r));
  fprintf('%-10s %-36s %16s %16s\n', ...
    char(summaryTable.Area(r)), metricName, ...
    format_summary_table_value(summaryTable.(segCol1)(r), metricName), ...
    format_summary_table_value(summaryTable.(segCol2)(r), metricName));
end
fprintf('\n');
end

function [rowArea, rowMetric, rowSeg1, rowSeg2] = append_summary_row( ...
    rowArea, rowMetric, rowSeg1, rowSeg2, areaName, metricName, val1, val2)
% APPEND_SUMMARY_ROW - Append one summary-table row

rowArea(end+1, 1) = string(areaName);
rowMetric(end+1, 1) = string(metricName);
rowSeg1(end+1, 1) = val1;
rowSeg2(end+1, 1) = val2;
end

function metrics = get_av_summary_metrics(segAv, areaIdx)
% GET_AV_SUMMARY_METRICS - Avalanche fit summary for one area and segment

metrics = struct('tau', nan, 'alpha', nan, 'scalingRelation', nan, 'nAvalanches', nan);
if areaIdx > numel(segAv.byArea)
  return;
end
avData = segAv.byArea{areaIdx};
if ~avData.hasAvalanches
  metrics.nAvalanches = 0;
  return;
end
metrics.tau = avData.tau;
metrics.alpha = avData.alpha;
metrics.scalingRelation = avData.scalingRelation;
metrics.nAvalanches = avData.nAvalanches;
end

function metrics = get_ar_summary_metrics(results, areaIdx, useLog10D2)
% GET_AR_SUMMARY_METRICS - Mean raw and normalized d2 for one segment and area

metrics = struct('d2Mean', nan, 'd2NormMean', nan, 'nWindows', nan);
if isempty(results) || ~isfield(results, 'areas') || isempty(results.areas)
  return;
end
if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
  return;
end
d2Vec = extract_finite_d2(results, areaIdx, useLog10D2);
d2NormVec = extract_finite_d2_normalized(results, areaIdx, useLog10D2);
if ~isempty(d2Vec)
  metrics.d2Mean = mean(d2Vec);
  metrics.nWindows = numel(d2Vec);
end
if ~isempty(d2NormVec)
  metrics.d2NormMean = mean(d2NormVec);
end
end

function metrics = get_prg_summary_metrics(results, areaIdx)
% GET_PRG_SUMMARY_METRICS - Mean PRG metrics for one segment and area

metrics = struct('kappaMean', nan, 'kappaNormMean', nan, ...
  'djsMean', nan, 'djsNormMean', nan, 'nWindows', nan);
if isempty(results) || ~isfield(results, 'areas') || isempty(results.areas)
  return;
end
if areaIdx > numel(results.kappa) || isempty(results.kappa{areaIdx})
  return;
end
kappaVec = extract_valid_prg_kappa(results, areaIdx);
kappaNormVec = extract_valid_prg_kappa_normalized(results, areaIdx);
djsVec = extract_valid_prg_djs(results, areaIdx);
djsNormVec = extract_valid_prg_djs_normalized(results, areaIdx);
if ~isempty(kappaVec)
  metrics.kappaMean = mean(kappaVec);
  metrics.nWindows = numel(kappaVec);
end
if ~isempty(kappaNormVec)
  metrics.kappaNormMean = mean(kappaNormVec);
end
if ~isempty(djsVec)
  metrics.djsMean = mean(djsVec);
end
if ~isempty(djsNormVec)
  metrics.djsNormMean = mean(djsNormVec);
end
end

function textOut = format_summary_table_value(value, metricName)
% FORMAT_SUMMARY_TABLE_VALUE - Pretty-print one table cell

if isnan(value)
  textOut = '—';
  return;
end
if contains(metricName, 'n avalanches') || contains(metricName, 'n windows')
  textOut = sprintf('%d', round(value));
else
  textOut = sprintf('%.4f', value);
end
end

function scalingRelation = compute_avalanche_scaling_relation(tau, alpha)
% COMPUTE_AVALANCHE_SCALING_RELATION - Size-duration exponent (alpha-1)/(tau-1)
%
% Goal:
%   Match session_avalanche_distributions scaling relation (Ma et al. 2019).

if ~isfinite(tau) || ~isfinite(alpha) || tau <= 1
  scalingRelation = nan;
  return;
end
scalingRelation = (alpha - 1) / (tau - 1);
end

function [binEdges, xMin, xMax] = build_shared_bin_edges(allVals, nBinsTarget, defaultRange)
% BUILD_SHARED_BIN_EDGES - Shared histogram bin edges and x-limits

if nargin < 3
  defaultRange = [0, 1];
end
if isempty(allVals)
  xMin = defaultRange(1);
  xMax = defaultRange(2);
  nBins = max(8, round(nBinsTarget));
  binEdges = linspace(xMin, xMax, nBins + 1);
  return;
end

xMin = min(allVals);
xMax = max(allVals);
xSpan = xMax - xMin;
if xSpan <= 0 || ~isfinite(xSpan)
  pad = max(0.5, abs(xMin) * 0.05 + eps);
  xMin = xMin - pad;
  xMax = xMax + pad;
else
  pad = 0.03 * xSpan;
  xMin = xMin - pad;
  xMax = xMax + pad;
end
nBins = max(8, round(nBinsTarget));
binEdges = linspace(xMin, xMax, nBins + 1);
end

function y = log10_safe_numeric(x)
validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function label = format_areas_label(areaNames)
if iscell(areaNames)
  areaNames = areaNames(:)';
  label = strjoin(areaNames, '_');
else
  label = char(areaNames);
end
label = matlab.lang.makeValidName(label);
end
