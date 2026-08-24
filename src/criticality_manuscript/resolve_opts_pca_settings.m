function pcaOpts = resolve_opts_pca_settings(opts)
% RESOLVE_OPTS_PCA_SETTINGS - pcaFlag / pcaFirstFlag / nDim from an opts struct
%
% Variables:
%   opts - Any analysis options struct (missing fields use defaults)
%
% Defaults:
%   pcaFlag false, pcaFirstFlag true (first PCs), nDim 5
%
% Goal:
%   Shared PCA reconstruction knobs for AR, AV, engagement, and cache names.

pcaOpts = struct('pcaFlag', false, 'pcaFirstFlag', true, 'nDim', 5);
if nargin < 1 || isempty(opts) || ~isstruct(opts)
  return;
end
if isfield(opts, 'pcaFlag') && ~isempty(opts.pcaFlag)
  pcaOpts.pcaFlag = logical(opts.pcaFlag);
end
if isfield(opts, 'pcaFirstFlag') && ~isempty(opts.pcaFirstFlag)
  pcaOpts.pcaFirstFlag = logical(opts.pcaFirstFlag);
end
if isfield(opts, 'nDim') && ~isempty(opts.nDim) && isfinite(opts.nDim)
  pcaOpts.nDim = max(1, round(double(opts.nDim)));
end
end
