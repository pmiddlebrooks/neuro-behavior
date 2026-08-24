function threshInfo = prepare_area_shared_av_thresholds(dataStruct, areaIndex, analysisConfig, ...
    collectStart, collectEnd)
% PREPARE_AREA_SHARED_AV_THRESHOLDS - Full-collect avalanche thresholds for one area
%
% Variables:
%   dataStruct     - Session data with spikeTimes / idLabel
%   areaIndex      - Area index into dataStruct.areas / idLabel
%   analysisConfig - AV config (binSize / detection mode / thresholdMethod / subsampling)
%   collectStart   - Collect range start (s)
%   collectEnd     - Collect range end (s)
%
% Goal:
%   Bin the full collect range once and return shared threshold info (including
%   fixed subsample neuron indices) for the total class. Engaged / non-engaged
%   cutoffs are prepared separately via prepare_segment_class_av_thresholds_by_area
%   while reusing these neuron subsets.

timeRange = [collectStart, collectEnd];
neuronIds = dataStruct.idLabel{areaIndex};
binSizeVec = resolve_avalanche_bin_sizes(dataStruct, areaIndex, timeRange, analysisConfig);
binSize = binSizeVec(areaIndex);
aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
  neuronIds, timeRange, binSize);
aDataMat = apply_config_pca_reconstruction(aDataMat, analysisConfig);
threshInfo = prepare_shared_avalanche_threshold_info(aDataMat, analysisConfig);
threshInfo.binSize = binSize;
threshInfo.areaIndex = areaIndex;
end
