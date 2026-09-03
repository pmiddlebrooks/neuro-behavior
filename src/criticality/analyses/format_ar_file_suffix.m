function filenameSuffix = format_ar_file_suffix(config)
% FORMAT_AR_FILE_SUFFIX - Results/plot filename tag for AR PCA and SDF options
%
% Variables:
%   config - AR config struct. Uses .pcaFlag, .sdfFlag, .sdfSigmaMs,
%            .d2Method, .klFitMethod, .klErrBars
%
% Goal:
%   Keep count-based and SDF (and PCA) caches from overwriting each other,
%   and keep Euclidean vs KL d2 results in separate files.
%   Examples: '', '_pca', '_sdf_10', '_pca_sdf_10', '_d2kl'.

filenameSuffix = '';
if nargin < 1 || isempty(config) || ~isstruct(config)
    return;
end

if isfield(config, 'pcaFlag') && ~isempty(config.pcaFlag) && config.pcaFlag
    filenameSuffix = '_pca';
end

if isfield(config, 'sdfFlag') && ~isempty(config.sdfFlag) && config.sdfFlag
    sdfSigmaMs = 10;
    if isfield(config, 'sdfSigmaMs') && ~isempty(config.sdfSigmaMs) && isfinite(config.sdfSigmaMs)
        sdfSigmaMs = double(config.sdfSigmaMs);
    end
    if abs(sdfSigmaMs - round(sdfSigmaMs)) < 1e-9
        filenameSuffix = sprintf('%s_sdf_%d', filenameSuffix, round(sdfSigmaMs));
    else
        filenameSuffix = sprintf('%s_sdf_%g', filenameSuffix, sdfSigmaMs);
    end
end

d2Method = 'euclidean';
if isfield(config, 'd2Method') && ~isempty(config.d2Method)
    d2Method = config.d2Method;
end
klFitMethod = 'MaxLikelihood';
if isfield(config, 'klFitMethod') && ~isempty(config.klFitMethod)
    klFitMethod = config.klFitMethod;
end
klErrBars = false;
if isfield(config, 'klErrBars') && ~isempty(config.klErrBars)
    klErrBars = config.klErrBars;
end
filenameSuffix = [filenameSuffix, format_d2_method_file_tag(d2Method, klFitMethod, klErrBars)];
end
