function tf = is_mean_isi_zero_avalanche_mode(config)
% IS_MEAN_ISI_ZERO_AVALANCHE_MODE - True for literature mean-ISI / zero-cutoff mode
%
% Variables:
%   config - Analysis config with optional .avalancheDetectionMode
%
% Goal:
%   Detect avalancheDetectionMode == 'meanIsiZero'.

tf = false;
if nargin < 1 || isempty(config) || ~isstruct(config)
  return;
end
if isfield(config, 'avalancheDetectionMode') ...
    && strcmpi(config.avalancheDetectionMode, 'meanIsiZero')
  tf = true;
end
end
