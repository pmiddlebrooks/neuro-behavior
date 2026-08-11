function avData = extract_pooled_area_avalanches_core(dataStruct, areaIndex, analysisConfig, ...
    segments, computeShuffles)
% EXTRACT_POOLED_AREA_AVALANCHES_CORE - Path-visible alias (avoids local shadows)

if nargin < 5
  computeShuffles = false;
end
avData = extract_pooled_area_avalanches(dataStruct, areaIndex, analysisConfig, ...
  segments, computeShuffles);
end
