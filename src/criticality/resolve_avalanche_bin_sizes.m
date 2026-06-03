function binSize = resolve_avalanche_bin_sizes(dataStruct, areasToTest, timeRange, config)
% RESOLVE_AVALANCHE_BIN_SIZES - Per-area bin sizes for avalanche analysis
%
% Variables:
%   dataStruct    - Session data with spikeTimes, spikeClusters, idLabel, areas
%   areasToTest   - Area indices to process
%   timeRange     - [startTime, endTime] in seconds
%   config        - avalancheDetectionMode, binSize, useOptimalBinWindowFunction
%
% Goal:
%   meanIsiZero: population mean ISI per area.
%   Otherwise: scalar/vector config.binSize (required when not using optimal search).

numAreas = numel(dataStruct.areas);
binSize = nan(1, numAreas);

if is_mean_isi_zero_avalanche_mode(config)
  for a = areasToTest(:)'
    neuronIds = dataStruct.idLabel{a};
    binSize(a) = population_mean_isi_bin_size( ...
      dataStruct.spikeTimes, dataStruct.spikeClusters, neuronIds, timeRange);
    fprintf('  %s: binSize = %.4f s (mean population ISI)\n', dataStruct.areas{a}, binSize(a));
  end
  return;
end

if ~isfield(config, 'binSize') || isempty(config.binSize)
  error(['config.binSize must be provided when avalancheDetectionMode is ', ...
    'fixedBinMedian and useOptimalBinWindowFunction is false.']);
end

if isscalar(config.binSize)
  binSize(:) = config.binSize;
else
  if numel(config.binSize) ~= numAreas
    error('config.binSize must be scalar or length(numAreas).');
  end
  binSize(:) = config.binSize(:);
end
end
