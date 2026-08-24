function sharedByArea = prepare_segment_class_av_thresholds_by_area(dataStruct, ...
    areasToAnalyze, analysisConfig, segments, baseSharedByArea, classLabel)
% PREPARE_SEGMENT_CLASS_AV_THRESHOLDS_BY_AREA - Thresholds from one engagement class
%
% Variables:
%   dataStruct         - Session data
%   areasToAnalyze     - Area indices
%   analysisConfig     - AV config (thresholdMethod, useSubsampling, ...)
%   segments           - Struct array (.start, .end) for this class (s)
%   baseSharedByArea   - Collect-range prepare_shared_av_thresholds_by_area output
%                        (reuses .neuronIdxSubsamples and .binSize per area)
%   classLabel         - Optional string for fprintf (e.g. 'engaged')
%
% Goal:
%   When avWindow is empty, engaged and non-engaged each need a population
%   cutoff from that class's own pop activity (not the full collect range),
%   while keeping the same neuron subsample indices as the collect-range prep.
%   Concatenate binned activity across discontinuous segments, then compute
%   fixed / per-subsample thresholds.
%
% Returns:
%   sharedByArea - Cell indexed by areaIndex (same layout as baseSharedByArea)

if nargin < 6 || isempty(classLabel)
  classLabel = 'class';
end
if nargin < 5 || isempty(baseSharedByArea)
  baseSharedByArea = {};
end

numAreas = numel(dataStruct.areas);
sharedByArea = cell(1, numAreas);

if isempty(segments)
  sharedByArea = baseSharedByArea;
  return;
end

% With avWindow tiles, cutoffs are recomputed per tile; only neuron subsets matter
if use_local_av_window_thresholds(analysisConfig)
  sharedByArea = baseSharedByArea;
  return;
end

for aIdx = 1:numel(areasToAnalyze)
  areaIndex = areasToAnalyze(aIdx);
  baseInfo = [];
  if numel(baseSharedByArea) >= areaIndex && ~isempty(baseSharedByArea{areaIndex})
    baseInfo = baseSharedByArea{areaIndex};
  end

  if ~isempty(baseInfo) && isfield(baseInfo, 'binSize') && isfinite(baseInfo.binSize)
    binSize = baseInfo.binSize;
  else
    timeRange = [segments(1).start, segments(1).end];
    binSizeVec = resolve_avalanche_bin_sizes(dataStruct, areaIndex, timeRange, analysisConfig);
    binSize = binSizeVec(areaIndex);
  end

  neuronIds = dataStruct.idLabel{areaIndex};
  aDataMat = concatenate_segment_binned_activity( ...
    dataStruct, neuronIds, segments, binSize);
  if isempty(aDataMat)
    if ~isempty(baseInfo)
      sharedByArea{areaIndex} = baseInfo;
    end
    continue;
  end
  aDataMat = apply_config_pca_reconstruction(aDataMat, analysisConfig);

  % Keep collect-range neuron subsets; recompute cutoffs on class activity only
  if ~isempty(baseInfo) && isfield(baseInfo, 'useSubsampling') && baseInfo.useSubsampling ...
      && isfield(baseInfo, 'neuronIdxSubsamples') ...
      && ~isempty(baseInfo.neuronIdxSubsamples)
    nSubsamplesArea = numel(baseInfo.neuronIdxSubsamples);
    thresholdPerSubsample = nan(1, nSubsamplesArea);
    numNeuronsArea = size(aDataMat, 2);
    computeThreshold = ~is_mean_isi_zero_avalanche_mode(analysisConfig);
    if isfield(analysisConfig, 'thresholdFlag') && ~analysisConfig.thresholdFlag
      computeThreshold = false;
    end
    for s = 1:nSubsamplesArea
      colIdx = baseInfo.neuronIdxSubsamples{s};
      colIdx = colIdx(colIdx >= 1 & colIdx <= numNeuronsArea);
      if isempty(colIdx) || ~computeThreshold
        continue;
      end
      popClass = sum(aDataMat(:, colIdx), 2);
      thresholdPerSubsample(s) = compute_avalanche_population_threshold( ...
        popClass, analysisConfig);
    end
    threshInfo = struct( ...
      'useSubsampling', true, ...
      'neuronIdxSubsamples', [], ...
      'thresholdPerSubsample', thresholdPerSubsample, ...
      'fixedThreshold', nan);
    threshInfo.neuronIdxSubsamples = baseInfo.neuronIdxSubsamples;
  else
    threshInfo = prepare_shared_avalanche_threshold_info(aDataMat, analysisConfig);
  end

  threshInfo.binSize = binSize;
  threshInfo.areaIndex = areaIndex;
  sharedByArea{areaIndex} = threshInfo;

  if isfield(threshInfo, 'useSubsampling') && threshInfo.useSubsampling
    fprintf('  %s AV thresholds (%s): %d subsamples from %d segment(s)\n', ...
      classLabel, dataStruct.areas{areaIndex}, ...
      numel(threshInfo.thresholdPerSubsample), numel(segments));
  elseif isfield(threshInfo, 'fixedThreshold') && isfinite(threshInfo.fixedThreshold)
    fprintf('  %s AV threshold (%s): %.4g from %d segment(s)\n', ...
      classLabel, dataStruct.areas{areaIndex}, threshInfo.fixedThreshold, numel(segments));
  end
end
end

function aDataMat = concatenate_segment_binned_activity(dataStruct, neuronIds, segments, binSize)
% CONCATENATE_SEGMENT_BINNED_ACTIVITY - Stack binned spikes across segments

aDataMat = [];
minSegDur = max(0.2, binSize * 4);
for i = 1:numel(segments)
  segStart = segments(i).start;
  segEnd = segments(i).end;
  if ~(isfinite(segStart) && isfinite(segEnd)) || (segEnd - segStart) < minSegDur
    continue;
  end
  segMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
    neuronIds, [segStart, segEnd], binSize);
  if isempty(segMat)
    continue;
  end
  aDataMat = [aDataMat; segMat]; %#ok<AGROW>
end
end
