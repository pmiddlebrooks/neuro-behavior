function dataMat = apply_config_pca_reconstruction(dataMat, config)
% APPLY_CONFIG_PCA_RECONSTRUCTION - Rebuild [time x neurons] from first/last PCs
%
% Variables:
%   dataMat - Binned activity [timeBins x neurons]
%   config  - Analysis config. Fields:
%     .pcaFlag      - If false/missing, return dataMat unchanged
%     .pcaFirstFlag - If true (default), keep first nDim PCs; else last nDim
%     .nDim         - Number of components to keep (default 4)
%
% Goal:
%   Match AR/AV PCA usage: project onto nDim components and reconstruct in
%   neuron space so population-sum analyses (d2, avalanches) can proceed.

if nargin < 2 || isempty(config) || ~isstruct(config)
  return;
end
if ~isfield(config, 'pcaFlag') || isempty(config.pcaFlag) || ~config.pcaFlag
  return;
end
if isempty(dataMat) || size(dataMat, 1) < 2 || size(dataMat, 2) < 2
  return;
end

nDim = 4;
if isfield(config, 'nDim') && ~isempty(config.nDim) && isfinite(config.nDim)
  nDim = config.nDim;
end
pcaFirstFlag = true;
if isfield(config, 'pcaFirstFlag') && ~isempty(config.pcaFirstFlag)
  pcaFirstFlag = logical(config.pcaFirstFlag);
end

dataMat = reconstruct_binned_matrix_from_pca(dataMat, nDim, pcaFirstFlag);
end

function reconstructedMat = reconstruct_binned_matrix_from_pca(dataMat, nDim, pcaFirstFlag)
% RECONSTRUCT_BINNED_MATRIX_FROM_PCA - Keep first or last nDim PCs, project back
%
% Variables:
%   dataMat      - Binned activity [timeBins x neurons]
%   nDim         - Number of PCs to keep
%   pcaFirstFlag - If true, highest-variance PCs; else lowest-variance PCs
%
% Goal:
%   Reconstruct in neuron space from only the requested PCs. Uses the
%   neuron x neuron covariance (not a full time x neuron SVD / pca) so
%   long sessions stay memory-safe.

nTime = size(dataMat, 1);
nNeurons = size(dataMat, 2);
nCompMax = min(nTime - 1, nNeurons);
if nCompMax < 1
  reconstructedMat = dataMat;
  return;
end

nDimUse = nDim;
if isempty(nDimUse) || ~isfinite(nDimUse)
  nDimUse = nCompMax;
end
nDimUse = max(1, min(round(double(nDimUse)), nCompMax));

mu = mean(dataMat, 1, 'omitnan');
xCentered = dataMat - mu;
xCentered(~isfinite(xCentered)) = 0;

covMat = (xCentered' * xCentered) / max(1, nTime - 1);
covMat = (covMat + covMat') / 2;
[eigenVec, eigenVal] = eig(covMat, 'vector');
[~, sortIdx] = sort(real(eigenVal), 'descend');
eigenVec = real(eigenVec(:, sortIdx));

if pcaFirstFlag
  dimIdx = 1:nDimUse;
else
  dimIdx = (nCompMax - nDimUse + 1):nCompMax;
end

coeffKeep = eigenVec(:, dimIdx);
scoreKeep = xCentered * coeffKeep;
reconstructedMat = scoreKeep * coeffKeep' + mu;
end
