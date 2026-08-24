function cacheParams = make_manuscript_session_cache_params(kind, opts)
% MAKE_MANUSCRIPT_SESSION_CACHE_PARAMS - Filename / match fields for session cache
%
% Variables:
%   kind - 'ar', 'av', 'prg', 'engagement_d2', 'engagement_av', or 'engagement_prg'
%   opts - Analysis options (brainArea, collectStart/End, pipeline window,
%          bin size, useSubsampling, nSubsamples, nNeuronsSubsample)
%
% Goal:
%   Store only keys that affect that pipeline's results (d2 vs AV vs PRG).

cacheParams = struct();
cacheParams.kind = lower(strtrim(char(kind)));
kindGroup = manuscript_session_cache_kind_group(cacheParams.kind);

cacheParams.brainArea = '';
if isfield(opts, 'brainArea') && ~isempty(opts.brainArea)
  cacheParams.brainArea = char(opts.brainArea);
end
cacheParams.collectStart = 0;
if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
  cacheParams.collectStart = opts.collectStart;
end
cacheParams.collectEnd = [];
if isfield(opts, 'collectEnd')
  cacheParams.collectEnd = opts.collectEnd;
end

cacheParams.d2Window = [];
cacheParams.avWindow = [];
cacheParams.prgWindow = [];
if strcmp(kindGroup, 'd2') && isfield(opts, 'd2Window')
  cacheParams.d2Window = opts.d2Window;
elseif strcmp(kindGroup, 'av') && isfield(opts, 'avWindow')
  cacheParams.avWindow = opts.avWindow;
elseif strcmp(kindGroup, 'prg') && isfield(opts, 'prgWindow')
  cacheParams.prgWindow = opts.prgWindow;
end

cacheParams.binSizeD2 = resolve_cache_bin_size(kindGroup, opts);
cacheParams.nNeuronsSubsample = 0;
cacheParams.nSubsamples = 0;
if isfield(opts, 'useSubsampling') && opts.useSubsampling ...
    && isfield(opts, 'nNeuronsSubsample') && ~isempty(opts.nNeuronsSubsample)
  cacheParams.nNeuronsSubsample = opts.nNeuronsSubsample;
  if isfield(opts, 'nSubsamples') && ~isempty(opts.nSubsamples)
    cacheParams.nSubsamples = opts.nSubsamples;
  end
end
pcaOpts = resolve_opts_pca_settings(opts);
cacheParams.pcaFlag = false;
cacheParams.pcaFirstFlag = true;
cacheParams.nDim = 0;
% Neural PCA reconstruction changes AR/AV (and engagement d2/av) results
if ~strcmp(kindGroup, 'prg') && pcaOpts.pcaFlag
  cacheParams.pcaFlag = true;
  cacheParams.pcaFirstFlag = pcaOpts.pcaFirstFlag;
  cacheParams.nDim = pcaOpts.nDim;
end
cacheParams.analyses = '';
cacheParams.cellType = '';
if isfield(opts, 'cacheCellType') && ~isempty(opts.cacheCellType)
  cacheParams.cellType = char(opts.cacheCellType);
end
end

function binSize = resolve_cache_bin_size(kindGroup, opts)
% RESOLVE_CACHE_BIN_SIZE - Pipeline-specific spike bin for the cache filename

if strcmp(kindGroup, 'av')
  binSize = first_opts_bin(opts, {'binSizeAv', 'binSize'});
elseif strcmp(kindGroup, 'prg')
  binSize = first_opts_bin(opts, {'binSizePrg', 'binSize'});
else
  binSize = first_opts_bin(opts, {'binSizeD2', 'binSize'});
end
end

function binSize = first_opts_bin(opts, fieldNames)
binSize = [];
for i = 1:numel(fieldNames)
  name = fieldNames{i};
  if isfield(opts, name) && ~isempty(opts.(name))
    binSize = opts.(name);
    return;
  end
end
end
