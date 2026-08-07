function out = criticality_per_subsample_size(sessionType, sessionName, subjectName, opts)
% CRITICALITY_PER_SUBSAMPLE_SIZE - Metric dependence on nNeuronsSubsample (one session)
%
% Variables:
%   sessionType - 'spontaneous' | 'interval' | 'reach' | 'semicircle' | 'schall'
%   sessionName - Session identifier
%   subjectName - Subject ('' for reach / some lists)
%   opts        - Optional struct; see defaults below
%
% Goal:
%   Load the session once, then for each nNeuronsSubsample in
%   opts.nNeuronsSubsampleList run selected analyses (d2 / avalanche / PRG)
%   with useSubsampling=true. When plotting, make separate figures for each
%   analysis family so users can enable only the metrics they care about.
%
% Example:
%   opts = struct('analyses', {{'d2','av','prg'}}, 'plotResults', true);
%   out = criticality_per_subsample_size('reach', '20220325', '', opts);

if nargin < 4 || isempty(opts)
  opts = struct();
end
setup_criticality_manuscript_paths('criticality_per_subsample_size');
opts = fill_criticality_per_subsample_size_opts(opts);
paths = get_paths();
[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();
nList = opts.nNeuronsSubsampleList(:);
nSizes = numel(nList);

fprintf('\n=== Subsample-size sweep: %s / %s ===\n', sessionType, sessionName);
fprintf('  nNeuronsSubsample = [%s]\n', strjoin(string(nList), ' '));
fprintf('  analyses: %s\n', strjoin(opts.analyses, ', '));

% --- Load session once ----------------------------------------------------
loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = opts.firingRateCheckTime;
loadOpts.collectStart = opts.collectStart;
loadOpts.collectEnd = opts.collectEnd;
loadOpts.minFiringRate = opts.minFiringRate;
loadOpts.maxFiringRate = opts.maxFiringRate;

loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, subjectName);
dataStruct = load_session_data(sessionType, opts.dataSource, loadArgs{:});
[dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
  dataStruct, opts.brainArea, opts.brainAreaCombinations);
if ~areaOk
  error('criticality_per_subsample_size:MissingArea', ...
    'Brain area "%s" not available for %s / %s.', opts.brainArea, sessionType, sessionName);
end

sessionDuration = get_session_collect_duration(dataStruct, opts);
nUnits = count_session_neurons_for_brain_area(dataStruct, opts.brainArea);
fprintf('  Session duration: %.1f s | neurons in %s: %d\n', ...
  sessionDuration, ternary_str(isempty(opts.brainArea), 'all areas', opts.brainArea), nUnits);

runD2 = any(strcmpi(opts.analyses, 'd2'));
runAv = any(strcmpi(opts.analyses, 'av'));
runPrg = any(strcmpi(opts.analyses, 'prg'));

% Preallocate result tables (one row per subsample size)
sweep = struct();
sweep.nNeuronsSubsample = nList;
sweep.nUnits = nUnits;
sweep.sessionDuration = sessionDuration;
sweep.skipped = false(nSizes, 1);
sweep.skipReason = repmat({''}, nSizes, 1);

if runD2
  sweep.d2Mean = nan(nSizes, 1);
  sweep.d2Sem = nan(nSizes, 1);
end
if runAv
  sweep.tau = nan(nSizes, 1);
  sweep.tauSem = nan(nSizes, 1);
  sweep.alpha = nan(nSizes, 1);
  sweep.alphaSem = nan(nSizes, 1);
  sweep.paramSD = nan(nSizes, 1);
  sweep.paramSDSem = nan(nSizes, 1);
  sweep.dcc = nan(nSizes, 1);
  sweep.dccSem = nan(nSizes, 1);
  sweep.decades = nan(nSizes, 1);
  sweep.decadesSem = nan(nSizes, 1);
end
if runPrg
  sweep.kappaMean = nan(nSizes, 1);
  sweep.kappaSem = nan(nSizes, 1);
  sweep.djsMean = nan(nSizes, 1);
  sweep.djsSem = nan(nSizes, 1);
end

arBase = [];
avBase = [];
prgBase = [];
if runD2
  arBase = build_ar_analysis_config(opts);
  if isempty(opts.d2Window)
    arBase.slidingWindowSize = sessionDuration;
    arBase.stepSize = sessionDuration;
  end
end
if runAv
  avBase = build_av_analysis_config(opts, sessionDuration, clausetPlfitPath, plfit2023Path);
end
if runPrg
  prgBase = build_prg_analysis_config(opts);
  if isempty(opts.prgWindow)
    prgBase.blockWindowSize = sessionDuration;
  end
end

for iSize = 1:nSizes
  nSub = nList(iSize);
  minNeeded = round(nSub * opts.minNeuronsMultiple);
  fprintf('\n--- nNeuronsSubsample = %d (need >= %d units) ---\n', nSub, minNeeded);

  if nUnits < minNeeded
    sweep.skipped(iSize) = true;
    sweep.skipReason{iSize} = sprintf('Too few neurons (%d < %d)', nUnits, minNeeded);
    fprintf('  Skip: %s\n', sweep.skipReason{iSize});
    continue;
  end

  if runD2
    arCfg = arBase;
    arCfg.nNeuronsSubsample = nSub;
    arCfg.useSubsampling = true;
    try
      arResults = criticality_ar_analysis(dataStruct, arCfg);
      arResults = filter_results_to_brain_area(arResults, opts.brainArea);
      areaIdx = 1;
      if isempty(arResults.areas)
        warning('No AR areas for n=%d', nSub);
      else
        d2Sum = summarize_session_d2(arResults, areaIdx, opts.useLog10D2);
        sweep.d2Mean(iSize) = d2Sum.d2Mean;
        sweep.d2Sem(iSize) = d2Sum.d2Sem;
        fprintf('  d2 mean = %.4g\n', sweep.d2Mean(iSize));
      end
    catch ME
      warning('AR failed at n=%d: %s', nSub, ME.message);
      sweep.skipReason{iSize} = [sweep.skipReason{iSize}, ' AR:', ME.message];
    end
  end

  if runAv
    avCfg = avBase;
    avCfg.nNeuronsSubsample = nSub;
    avCfg.useSubsampling = true;
    try
      avResults = criticality_av_analysis(dataStruct, avCfg);
      avResults = filter_results_to_brain_area(avResults, opts.brainArea);
      areaIdx = 1;
      if isempty(avResults.areas)
        warning('No AV areas for n=%d', nSub);
      else
        useSub = isfield(avResults, 'useSubsampling') && avResults.useSubsampling;
        [sweep.tau(iSize), sweep.tauSem(iSize)] = summarize_av_metric(avResults, areaIdx, 'tau', useSub);
        [sweep.alpha(iSize), sweep.alphaSem(iSize)] = summarize_av_metric(avResults, areaIdx, 'alpha', useSub);
        [sweep.paramSD(iSize), sweep.paramSDSem(iSize)] = summarize_av_metric(avResults, areaIdx, 'paramSD', useSub);
        [sweep.dcc(iSize), sweep.dccSem(iSize)] = summarize_av_metric(avResults, areaIdx, 'dcc', useSub);
        [sweep.decades(iSize), sweep.decadesSem(iSize)] = summarize_av_metric(avResults, areaIdx, 'decades', useSub);
        fprintf('  AV tau=%.3g alpha=%.3g dcc=%.3g\n', ...
          sweep.tau(iSize), sweep.alpha(iSize), sweep.dcc(iSize));
      end
    catch ME
      warning('AV failed at n=%d: %s', nSub, ME.message);
      sweep.skipReason{iSize} = [sweep.skipReason{iSize}, ' AV:', ME.message];
    end
  end

  if runPrg
    prgCfg = prgBase;
    prgCfg.nNeuronsSubsample = nSub;
    prgCfg.useSubsampling = true;
    try
      prgResults = criticality_prg_analysis(dataStruct, prgCfg);
      prgResults = filter_results_to_brain_area(prgResults, opts.brainArea);
      areaIdx = 1;
      if isempty(prgResults.areas)
        warning('No PRG areas for n=%d', nSub);
      else
        prgSum = summarize_session_prg(prgResults, areaIdx);
        sweep.kappaMean(iSize) = prgSum.kappaMean;
        sweep.kappaSem(iSize) = prgSum.kappaSem;
        sweep.djsMean(iSize) = prgSum.djsMean;
        sweep.djsSem(iSize) = prgSum.djsSem;
        fprintf('  PRG kappa=%.3g djs=%.3g\n', sweep.kappaMean(iSize), sweep.djsMean(iSize));
      end
    catch ME
      warning('PRG failed at n=%d: %s', nSub, ME.message);
      sweep.skipReason{iSize} = [sweep.skipReason{iSize}, ' PRG:', ME.message];
    end
  end
end

% Per-metric slopes (metric ~ nNeuronsSubsample) for this session
trend = compute_subsample_trends(sweep, opts.analyses);

if opts.plotResults
  plot_session_subsample_sweep(sweep, opts, sessionType, sessionName);
end

out = struct();
out.sweep = sweep;
out.trend = trend;
out.opts = opts;
out.paths = paths;
out.sessionType = sessionType;
out.sessionName = sessionName;
out.subjectName = subjectName;
out.nUnits = nUnits;
end

%% ========================================================================
function opts = fill_criticality_per_subsample_size_opts(opts)
defaults = struct();
defaults.analyses = {'d2', 'av', 'prg'};  % subset any of these
defaults.nNeuronsSubsampleList = 20:5:60;
defaults.dataSource = 'spikes';
defaults.collectStart = 0;
defaults.collectEnd = 60 * 60;
defaults.d2Window = [];   % [] = full collect window
defaults.prgWindow = [];  % [] = full collect window
defaults.brainArea = 'M23M56';
defaults.brainAreaCombinations = default_manuscript_brain_area_combinations();
defaults.plotResults = true;
defaults.useLog10D2 = true;
defaults.nSubsamples = 25;
defaults.minNeuronsMultiple = 1.1;
defaults.nMinNeurons = 20;
defaults.enablePermutations = false;
defaults.nShuffles = 0;
defaults.avalancheDetectionMode = 'fixedBinMedian';
defaults.powerLawFitMethod = 'plfit2023';
defaults.gofThreshold = 0.1;
defaults.prgMethod = 'pca';
defaults.binSize = 0.05;
defaults.cvThreshold = 5;
defaults.cutoffDivisors = [1, 2, 4, 8, 16, 32];
defaults.finalCutoffDivisor = 16;
defaults.kappaAxisMax = 20;
defaults.enableSurrogates = false;
defaults.nSurrogates = 0;
defaults.surrogateMethod = 'isi';
defaults.firingRateCheckTime = [];
defaults.minFiringRate = 0.05;
defaults.maxFiringRate = 100;

preserveCollectEndEmpty = isfield(opts, 'collectEnd') && isempty(opts.collectEnd);
preserveD2WindowEmpty = isfield(opts, 'd2Window') && isempty(opts.d2Window);
preservePrgWindowEmpty = isfield(opts, 'prgWindow') && isempty(opts.prgWindow);
opts = merge_structs(defaults, opts);
if preserveCollectEndEmpty
  opts.collectEnd = [];
end
if preserveD2WindowEmpty
  opts.d2Window = [];
end
if preservePrgWindowEmpty
  opts.prgWindow = [];
end
if ischar(opts.analyses) || isstring(opts.analyses)
  opts.analyses = cellstr(opts.analyses);
end
end

function opts = merge_structs(defaults, opts)
fn = fieldnames(defaults);
for i = 1:numel(fn)
  if ~isfield(opts, fn{i}) || isempty(opts.(fn{i}))
    opts.(fn{i}) = defaults.(fn{i});
  end
end
end

function s = ternary_str(cond, a, b)
if cond
  s = a;
else
  s = b;
end
end

%% --- Config builders -----------------------------------------------------
function analysisConfig = build_ar_analysis_config(opts)
analysisConfig = struct();
if isempty(opts.d2Window)
  analysisConfig.slidingWindowSize = 1;
  analysisConfig.stepSize = 1;
else
  analysisConfig.slidingWindowSize = opts.d2Window;
  analysisConfig.stepSize = opts.d2Window;
end
analysisConfig.binSize = 0.05;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = opts.enablePermutations;
analysisConfig.nShuffles = opts.nShuffles;
analysisConfig.normalizeD2 = opts.enablePermutations;
analysisConfig.useLog10D2 = opts.useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 100;
analysisConfig.nMinNeurons = opts.nMinNeurons;
analysisConfig.useSubsampling = true;
analysisConfig.nSubsamples = opts.nSubsamples;
analysisConfig.nNeuronsSubsample = opts.nNeuronsSubsampleList(1);
analysisConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
end

function analysisConfig = build_av_analysis_config(opts, windowDurationSec, clausetPlfitPath, plfit2023Path)
analysisConfig = struct();
if isempty(windowDurationSec)
  analysisConfig.slidingWindowSize = 1;
  analysisConfig.avStepSize = 1;
else
  analysisConfig.slidingWindowSize = windowDurationSec;
  analysisConfig.avStepSize = windowDurationSec;
end
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.avalancheDetectionMode = opts.avalancheDetectionMode;
if ~strcmpi(opts.avalancheDetectionMode, 'meanIsiZero')
  analysisConfig.binSize = 0.05;
end
analysisConfig.analyzeDcc = true;
analysisConfig.analyzeKappa = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 5;
analysisConfig.enablePermutations = opts.enablePermutations;
analysisConfig.nShuffles = opts.nShuffles;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.thresholdFlag = 1;
analysisConfig.thresholdPct = 1;
analysisConfig.nMinNeurons = opts.nMinNeurons;
analysisConfig.useSubsampling = true;
analysisConfig.nSubsamples = opts.nSubsamples;
analysisConfig.nNeuronsSubsample = opts.nNeuronsSubsampleList(1);
analysisConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
analysisConfig.normalizeMetrics = opts.enablePermutations;
analysisConfig.powerLawFitMethod = opts.powerLawFitMethod;
analysisConfig.gofThreshold = opts.gofThreshold;
analysisConfig.runClausetPlpva = false;
analysisConfig.clausetPlfitPath = clausetPlfitPath;
analysisConfig.plfit2023Path = plfit2023Path;
end

function analysisConfig = build_prg_analysis_config(opts)
analysisConfig = struct();
analysisConfig.prgMethod = opts.prgMethod;
if isempty(opts.prgWindow)
  analysisConfig.blockWindowSize = 1;
else
  analysisConfig.blockWindowSize = opts.prgWindow;
end
analysisConfig.binSize = opts.binSize;
analysisConfig.cvThreshold = opts.cvThreshold;
analysisConfig.cutoffDivisors = opts.cutoffDivisors;
analysisConfig.finalCutoffDivisor = opts.finalCutoffDivisor;
analysisConfig.kappaAxisMax = opts.kappaAxisMax;
analysisConfig.enableSurrogates = opts.enableSurrogates;
analysisConfig.nSurrogates = opts.nSurrogates;
analysisConfig.surrogateMethod = opts.surrogateMethod;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.nMinNeurons = opts.nMinNeurons;
analysisConfig.useSubsampling = true;
analysisConfig.nSubsamples = opts.nSubsamples;
analysisConfig.nNeuronsSubsample = opts.nNeuronsSubsampleList(1);
analysisConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
end

%% --- Session helpers -----------------------------------------------------
function sessionDuration = get_session_collect_duration(dataStruct, opts)
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
    && isfield(dataStruct.spikeData, 'collectStart')
  sessionDuration = dataStruct.spikeData.collectEnd - dataStruct.spikeData.collectStart;
elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
    && ~isempty(dataStruct.opts.collectEnd)
  collectStart = 0;
  if isfield(dataStruct.opts, 'collectStart') && ~isempty(dataStruct.opts.collectStart)
    collectStart = dataStruct.opts.collectStart;
  end
  sessionDuration = dataStruct.opts.collectEnd - collectStart;
elseif isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
  collectStart = opts.collectStart;
  if isempty(collectStart)
    collectStart = 0;
  end
  sessionDuration = max(dataStruct.spikeTimes) - collectStart;
else
  sessionDuration = opts.collectEnd - opts.collectStart;
  if isempty(sessionDuration)
    error('criticality_per_subsample_size:UnknownSessionDuration', ...
      'Could not determine session collect duration.');
  end
end
end

function nUnits = count_session_neurons_for_brain_area(dataStruct, brainArea)
nUnits = 0;
if ~isfield(dataStruct, 'idMatIdx') || isempty(dataStruct.idMatIdx)
  return;
end
if isempty(brainArea)
  for areaIdx = 1:numel(dataStruct.idMatIdx)
    nUnits = nUnits + numel(dataStruct.idMatIdx{areaIdx});
  end
  return;
end
areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
  return;
end
nUnits = numel(dataStruct.idMatIdx{areaIdx});
end

function results = filter_results_to_brain_area(results, brainArea)
% FILTER_RESULTS_TO_BRAIN_AREA - Keep one area's cell/vector fields
%
% Save areaIdx first; do not subset results.areas until other fields are
% indexed (subsetting areas mid-loop makes a later areas(areaIdx) fail when
% areaIdx > 1).

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end
areaIdx = find(strcmp(results.areas, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end

nAreas = numel(results.areas);
fn = fieldnames(results);
for i = 1:numel(fn)
  if strcmp(fn{i}, 'areas')
    continue;
  end
  val = results.(fn{i});
  if iscell(val) && numel(val) == nAreas
    results.(fn{i}) = val(areaIdx);
  elseif isnumeric(val) && isvector(val) && numel(val) == nAreas
    results.(fn{i}) = val(areaIdx);
  end
end
results.areas = results.areas(areaIdx);
end

function summary = summarize_session_d2(results, areaIdx, useLog10D2)
summary = struct('d2Mean', nan, 'd2Sem', nan);
if areaIdx > length(results.d2) || isempty(results.d2{areaIdx})
  return;
end
useSubsampling = isfield(results, 'useSubsampling') && results.useSubsampling;
d2Vec = results.d2{areaIdx}(:);
d2SubMat = [];
if useSubsampling && isfield(results, 'd2Subsamples') && areaIdx <= numel(results.d2Subsamples) ...
    && ~isempty(results.d2Subsamples{areaIdx})
  d2SubMat = results.d2Subsamples{areaIdx};
  if useLog10D2
    d2SubMat = log10_safe_numeric(d2SubMat);
  end
end
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
[summary.d2Mean, summary.d2Sem] = mean_sem_across_windows_or_subsamples( ...
  d2Vec, d2SubMat, useSubsampling);
end

function y = log10_safe_numeric(x)
y = nan(size(x));
ok = isfinite(x) & x > 0;
y(ok) = log10(x(ok));
end

function [meanVal, semVal] = summarize_av_metric(results, areaIdx, metricName, useSubsampling)
meanVal = nan;
semVal = nan;
if ~isfield(results, metricName) || areaIdx > numel(results.(metricName)) ...
    || isempty(results.(metricName){areaIdx})
  return;
end
windowVec = results.(metricName){areaIdx};
subField = [metricName, 'Subsamples'];
subMat = [];
if useSubsampling && isfield(results, subField) && areaIdx <= numel(results.(subField)) ...
    && ~isempty(results.(subField){areaIdx})
  subMat = results.(subField){areaIdx};
end
[meanVal, semVal] = mean_sem_across_windows_or_subsamples(windowVec, subMat, useSubsampling);
end

function summary = summarize_session_prg(results, areaIdx)
summary = struct('kappaMean', nan, 'kappaSem', nan, 'djsMean', nan, 'djsSem', nan);
if areaIdx > length(results.kappa) || isempty(results.kappa{areaIdx})
  return;
end
useSubsampling = isfield(results, 'useSubsampling') && results.useSubsampling;
kappaVec = results.kappa{areaIdx}(:);
nWin = numel(kappaVec);
excluded = false(nWin, 1);
if isfield(results, 'windowExcluded') && areaIdx <= length(results.windowExcluded) ...
    && ~isempty(results.windowExcluded{areaIdx})
  excluded = results.windowExcluded{areaIdx}(:);
  if numel(excluded) ~= nWin
    excluded = false(nWin, 1);
  end
end
validMask = isfinite(kappaVec) & ~excluded;
kappaValid = kappaVec;
kappaValid(~validMask) = nan;
kappaSubMat = [];
if useSubsampling && isfield(results, 'kappaSubsamples') && areaIdx <= numel(results.kappaSubsamples) ...
    && ~isempty(results.kappaSubsamples{areaIdx})
  kappaSubMat = results.kappaSubsamples{areaIdx};
end
[summary.kappaMean, summary.kappaSem] = mean_sem_across_windows_or_subsamples( ...
  kappaValid, kappaSubMat, useSubsampling);

if isfield(results, 'djs') && areaIdx <= length(results.djs) && ~isempty(results.djs{areaIdx})
  djsVec = results.djs{areaIdx}(:);
  djsValid = djsVec;
  djsValid(~validMask) = nan;
  djsSubMat = [];
  if useSubsampling && isfield(results, 'djsSubsamples') && areaIdx <= numel(results.djsSubsamples) ...
      && ~isempty(results.djsSubsamples{areaIdx})
    djsSubMat = results.djsSubsamples{areaIdx};
  end
  [summary.djsMean, summary.djsSem] = mean_sem_across_windows_or_subsamples( ...
    djsValid, djsSubMat, useSubsampling);
end
end

%% --- Trends & plotting ---------------------------------------------------
function trend = compute_subsample_trends(sweep, analyses)
% COMPUTE_SUBSAMPLE_TRENDS - OLS slope and relative change vs nNeuronsSubsample
%
% For each metric y(N): slope from polyfit(N, y, 1) on finite points;
% relativeChange = (y_last - y_first) / max(|y_first|, eps) using first/last
% finite sizes (scale-free sensitivity complementary to raw slope).

trend = struct();
n = sweep.nNeuronsSubsample(:);
specs = metric_specs_for_analyses(analyses);
for i = 1:numel(specs)
  name = specs(i).field;
  if ~isfield(sweep, name)
    continue;
  end
  y = sweep.(name)(:);
  ok = isfinite(n) & isfinite(y);
  entry = struct('slope', nan, 'intercept', nan, 'rhoSpearman', nan, ...
    'relativeChange', nan, 'nPoints', sum(ok));
  if sum(ok) >= 2
    p = polyfit(n(ok), y(ok), 1);
    entry.slope = p(1);
    entry.intercept = p(2);
    entry.rhoSpearman = corr(n(ok), y(ok), 'Type', 'Spearman', 'Rows', 'complete');
    idx = find(ok);
    y0 = y(idx(1));
    y1 = y(idx(end));
    entry.relativeChange = (y1 - y0) / max(abs(y0), eps);
  end
  trend.(name) = entry;
end
end

function specs = metric_specs_for_analyses(analyses)
specs = struct('field', {}, 'label', {}, 'family', {});
if any(strcmpi(analyses, 'd2'))
  specs(end+1) = struct('field', 'd2Mean', 'label', 'd_2', 'family', 'd2'); %#ok<AGROW>
end
if any(strcmpi(analyses, 'av'))
  specs(end+1) = struct('field', 'tau', 'label', '\tau', 'family', 'av'); %#ok<AGROW>
  specs(end+1) = struct('field', 'alpha', 'label', '\alpha', 'family', 'av'); %#ok<AGROW>
  specs(end+1) = struct('field', 'paramSD', 'label', '1/(\sigma\nu z)', 'family', 'av'); %#ok<AGROW>
  specs(end+1) = struct('field', 'dcc', 'label', 'DCC', 'family', 'av'); %#ok<AGROW>
  specs(end+1) = struct('field', 'decades', 'label', 'decades', 'family', 'av'); %#ok<AGROW>
end
if any(strcmpi(analyses, 'prg'))
  specs(end+1) = struct('field', 'kappaMean', 'label', '\kappa', 'family', 'prg'); %#ok<AGROW>
  specs(end+1) = struct('field', 'djsMean', 'label', 'D_{JS}', 'family', 'prg'); %#ok<AGROW>
end
end

function plot_session_subsample_sweep(sweep, opts, sessionType, sessionName)
n = sweep.nNeuronsSubsample(:);
titleBase = sprintf('%s / %s', sessionType, sessionName);
families = unique({metric_specs_for_analyses(opts.analyses).family}, 'stable');

for f = 1:numel(families)
  fam = families{f};
  famSpecs = metric_specs_for_analyses(opts.analyses);
  famSpecs = famSpecs(strcmp({famSpecs.family}, fam));
  if isempty(famSpecs)
    continue;
  end
  nMet = numel(famSpecs);
  nCol = min(3, nMet);
  nRow = ceil(nMet / nCol);
  figure('Name', sprintf('Subsample size — %s — %s', upper(fam), titleBase), ...
    'Color', 'w', 'Position', [80 80 420*nCol 320*nRow]);
  for m = 1:nMet
    subplot(nRow, nCol, m);
    y = sweep.(famSpecs(m).field)(:);
    ySem = [];
    semField = [famSpecs(m).field, 'Sem'];
    if strcmp(famSpecs(m).field, 'd2Mean') && isfield(sweep, 'd2Sem')
      ySem = sweep.d2Sem(:);
    elseif strcmp(famSpecs(m).field, 'kappaMean') && isfield(sweep, 'kappaSem')
      ySem = sweep.kappaSem(:);
    elseif strcmp(famSpecs(m).field, 'djsMean') && isfield(sweep, 'djsSem')
      ySem = sweep.djsSem(:);
    elseif isfield(sweep, [famSpecs(m).field, 'Sem'])
      ySem = sweep.([famSpecs(m).field, 'Sem'])(:);
    elseif isfield(sweep, semField)
      ySem = sweep.(semField)(:);
    end
    ok = isfinite(n) & isfinite(y);
    if any(ok)
      if ~isempty(ySem) && any(isfinite(ySem(ok)))
        errorbar(n(ok), y(ok), ySem(ok), '-o', 'LineWidth', 1.5, 'MarkerFaceColor', [0.2 0.4 0.7]);
      else
        plot(n(ok), y(ok), '-o', 'LineWidth', 1.5, 'MarkerFaceColor', [0.2 0.4 0.7]);
      end
    end
    hold on;
    if sum(ok) >= 2
      p = polyfit(n(ok), y(ok), 1);
      xx = linspace(min(n(ok)), max(n(ok)), 50);
      plot(xx, polyval(p, xx), '--', 'Color', [0.6 0.2 0.2], 'LineWidth', 1.2);
    end
    hold off;
    grid on;
    xlabel('nNeuronsSubsample');
    ylabel(famSpecs(m).label, 'Interpreter', 'tex');
    title(famSpecs(m).label, 'Interpreter', 'tex');
    xlim([min(n)-2, max(n)+2]);
  end
  sgtitle(sprintf('%s metrics vs subsample size — %s', upper(fam), titleBase), ...
    'Interpreter', 'none');
end
end
