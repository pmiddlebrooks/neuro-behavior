function sharedByArea = prepare_shared_av_thresholds_by_area(dataStruct, areasToAnalyze, ...
    analysisConfig, collectStart, collectEnd)
% PREPARE_SHARED_AV_THRESHOLDS_BY_AREA - Collect-range thresholds for each area
%
% Variables:
%   dataStruct       - Session data
%   areasToAnalyze   - Area indices
%   analysisConfig   - AV config
%   collectStart/End - Full collect range (s)
%
% Returns:
%   sharedByArea - Cell indexed by areaIndex with prepare_area_shared_av_thresholds output

numAreas = numel(dataStruct.areas);
sharedByArea = cell(1, numAreas);
for aIdx = 1:numel(areasToAnalyze)
  areaIndex = areasToAnalyze(aIdx);
  sharedByArea{areaIndex} = prepare_area_shared_av_thresholds( ...
    dataStruct, areaIndex, analysisConfig, collectStart, collectEnd);
  threshInfo = sharedByArea{areaIndex};
  if isfield(threshInfo, 'useSubsampling') && threshInfo.useSubsampling
    fprintf('  Shared AV thresholds (%s): %d subsamples from collect [%.0f-%.0f s]\n', ...
      dataStruct.areas{areaIndex}, numel(threshInfo.thresholdPerSubsample), ...
      collectStart, collectEnd);
  elseif isfield(threshInfo, 'fixedThreshold') && isfinite(threshInfo.fixedThreshold)
    fprintf('  Shared AV threshold (%s): %.4g from collect [%.0f-%.0f s]\n', ...
      dataStruct.areas{areaIndex}, threshInfo.fixedThreshold, collectStart, collectEnd);
  end
end
end
