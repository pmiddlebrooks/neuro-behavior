function tag = format_pca_file_tag(pcaFlag, nDim, pcaFirstFlag)
% FORMAT_PCA_FILE_TAG - Filename suffix when PCA reconstruction is on
%
% Variables:
%   pcaFlag      - If false/empty, return ''
%   nDim         - Number of PCs kept (default 4)
%   pcaFirstFlag - If true (default), first nDim PCs; else last nDim
%
% Goal:
%   Shared plot/cache stem fragment, e.g. '_pca_5' or '_pcaLast_5'.

tag = '';
if nargin < 1 || isempty(pcaFlag) || ~pcaFlag
  return;
end
if nargin < 2 || isempty(nDim) || ~isfinite(nDim)
  nDim = 4;
end
nDim = max(1, round(double(nDim)));
useFirst = true;
if nargin >= 3 && ~isempty(pcaFirstFlag)
  useFirst = logical(pcaFirstFlag);
end
if useFirst
  tag = sprintf('_pca_%d', nDim);
else
  tag = sprintf('_pcaLast_%d', nDim);
end
end
