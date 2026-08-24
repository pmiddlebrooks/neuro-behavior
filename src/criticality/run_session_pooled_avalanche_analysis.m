function results = run_session_pooled_avalanche_analysis(dataStruct, analysisConfig, ...
    collectStart, collectEnd)
% RUN_SESSION_POOLED_AVALANCHE_ANALYSIS - Session-level AV: tile, pool, fit once
%
% Variables:
%   dataStruct      - Loaded session
%   analysisConfig  - AV config (binSize, thresholdMethod, avWindow, ...)
%   collectStart    - Collect range start (s)
%   collectEnd      - Collect range end (s)
%
% Goal:
%   Manuscript session-level avalanche scalars. When avWindow is set, tile the
%   collect range, compute a threshold per tile from that tile's pop activity,
%   pool avalanches, and fit once (do not average per-window exponents).
%   Returns a criticality_av_analysis-like results struct (one value per area).

if nargin < 4 || isempty(collectEnd) || isempty(collectStart)
  error('run_session_pooled_avalanche_analysis:MissingCollectRange', ...
    'collectStart and collectEnd are required.');
end

areas = dataStruct.areas;
numAreas = numel(areas);
nMinNeurons = 10;
if isfield(analysisConfig, 'nMinNeurons') && ~isempty(analysisConfig.nMinNeurons)
  nMinNeurons = analysisConfig.nMinNeurons;
end
if isfield(analysisConfig, 'useSubsampling') && analysisConfig.useSubsampling ...
    && isfield(analysisConfig, 'nNeuronsSubsample') ...
    && isfield(analysisConfig, 'minNeuronsMultiple')
  nMinNeurons = round(analysisConfig.nNeuronsSubsample * analysisConfig.minNeuronsMultiple);
  analysisConfig.nMinNeurons = nMinNeurons;
end

if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
  candidateAreas = dataStruct.areasToTest(:)';
else
  candidateAreas = 1:numAreas;
end
candidateAreas = candidateAreas(candidateAreas >= 1 & candidateAreas <= numAreas);

areasToAnalyze = [];
for a = candidateAreas
  if numel(dataStruct.idMatIdx{a}) >= nMinNeurons
    areasToAnalyze(end + 1) = a; %#ok<AGROW>
  end
end
if isempty(areasToAnalyze)
  error('run_session_pooled_avalanche_analysis:NoValidAreas', ...
    'No valid areas to process (insufficient neurons).');
end

analysisConfig.sharedThresholdByArea = prepare_shared_av_thresholds_by_area( ...
  dataStruct, areasToAnalyze, analysisConfig, collectStart, collectEnd);

computeShuffles = false;
if isfield(analysisConfig, 'enablePermutations') && analysisConfig.enablePermutations
  computeShuffles = true;
elseif isfield(analysisConfig, 'enableCircularPermutations') ...
    && analysisConfig.enableCircularPermutations
  computeShuffles = true;
end

segments = struct('start', collectStart, 'end', collectEnd);

metricNames = {'dcc', 'kappa', 'decades', 'tau', 'alpha', 'paramSD'};
results = struct();
results.sessionType = '';
if isfield(dataStruct, 'sessionType')
  results.sessionType = dataStruct.sessionType;
end
results.areas = areas;
results.useSubsampling = isfield(analysisConfig, 'useSubsampling') && analysisConfig.useSubsampling;
for m = 1:numel(metricNames)
  results.(metricNames{m}) = cell(1, numAreas);
  results.([metricNames{m}, 'Normalized']) = cell(1, numAreas);
  results.([metricNames{m}, 'PermutedMean']) = cell(1, numAreas);
end
results.startS = cell(1, numAreas);
results.binSize = nan(1, numAreas);
results.slidingWindowSize = nan(1, numAreas);
avWindowUsed = collectEnd - collectStart;
if use_local_av_window_thresholds(analysisConfig)
  avWindowUsed = analysisConfig.avWindow;
end

for a = 1:numAreas
  for m = 1:numel(metricNames)
    results.(metricNames{m}){a} = nan;
    results.([metricNames{m}, 'Normalized']){a} = nan;
    results.([metricNames{m}, 'PermutedMean']){a} = nan;
  end
  results.startS{a} = collectStart;
  results.slidingWindowSize(a) = avWindowUsed;
end

for aIdx = 1:numel(areasToAnalyze)
  a = areasToAnalyze(aIdx);
  avData = extract_pooled_area_avalanches(dataStruct, a, analysisConfig, segments, ...
    computeShuffles);
  results.binSize(a) = avData.binSize;
  if ~avData.hasAvalanches
    fprintf('  AV %s: no avalanches\n', areas{a});
    continue;
  end
  results.dcc{a} = avData.dcc;
  results.kappa{a} = avData.kappa;
  results.decades{a} = avData.decades;
  results.tau{a} = avData.tau;
  results.alpha{a} = avData.alpha;
  results.paramSD{a} = avData.paramSD;
  if isfield(avData, 'shuffleDcc') && isfinite(avData.shuffleDcc)
    results.dccPermutedMean{a} = avData.shuffleDcc;
    results.kappaPermutedMean{a} = avData.shuffleKappa;
    results.decadesPermutedMean{a} = avData.shuffleDecades;
    results.tauPermutedMean{a} = avData.shuffleTau;
    results.alphaPermutedMean{a} = avData.shuffleAlpha;
    results.paramSDPermutedMean{a} = avData.shuffleParamSD;
  end
  fprintf(['  AV %s: n=%d, tau=%.2f, alpha=%.2f, paramSD=%.3f, ', ...
    'decades=%.2f, dcc=%.3f (%d windows)\n'], ...
    areas{a}, avData.nAvalanches, avData.tau, avData.alpha, avData.paramSD, ...
    avData.decades, avData.dcc, avData.nSegments);
  if isfield(avData, 'sizeFitInfo') && isstruct(avData.sizeFitInfo) ...
      && isfield(avData.sizeFitInfo, 'tailComparison')
    print_avalanche_tail_comparison(avData.sizeFitInfo.tailComparison, 'Size');
  end
  if isfield(avData, 'durFitInfo') && isstruct(avData.durFitInfo) ...
      && isfield(avData.durFitInfo, 'tailComparison')
    print_avalanche_tail_comparison(avData.durFitInfo.tailComparison, 'Duration');
  end
end

results.params = struct();
results.params.slidingWindowSize = avWindowUsed;
results.params.avWindow = [];
if isfield(analysisConfig, 'avWindow')
  results.params.avWindow = analysisConfig.avWindow;
end
results.params.collectStart = collectStart;
results.params.collectEnd = collectEnd;
if isfield(analysisConfig, 'thresholdMethod')
  results.params.thresholdMethod = analysisConfig.thresholdMethod;
end
if isfield(analysisConfig, 'avalancheDetectionMode')
  results.params.avalancheDetectionMode = analysisConfig.avalancheDetectionMode;
end
results.params.pooledAcrossWindows = true;
end
