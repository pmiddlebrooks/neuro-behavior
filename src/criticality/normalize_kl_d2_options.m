function [d2Method, klFitMethod, klErrBars, klParallel] = normalize_kl_d2_options( ...
    d2Method, klFitMethod, klErrBars, klParallel)
% NORMALIZE_KL_D2_OPTIONS Validate Euclidean vs KL-rate d2 options
%
% Variables:
%   d2Method    - 'euclidean' or 'kl'
%   klFitMethod - 'MaxLikelihood' or 'YuleWalker' (KL only)
%   klErrBars   - If true, S2.5 Hessian error bars (KL + MaxLikelihood)
%   klParallel  - If true, parallelize the KL error-bar gradient
%
% Goal:
%   Shared validation for manuscript / AR wrappers. Non-KL runs force KL
%   flags off so callers can leave the KL block set.

if nargin < 1 || isempty(d2Method)
  d2Method = 'euclidean';
end
d2Method = lower(strtrim(char(d2Method)));
if ~ismember(d2Method, {'euclidean', 'kl'})
  error('d2Method must be ''euclidean'' or ''kl'' (got "%s").', d2Method);
end

if nargin < 2 || isempty(klFitMethod)
  klFitMethod = 'MaxLikelihood';
end
if nargin < 3 || isempty(klErrBars)
  klErrBars = false;
end
if nargin < 4 || isempty(klParallel)
  klParallel = false;
end

if strcmp(d2Method, 'kl')
  klFitMethod = char(strtrim(klFitMethod));
  if ~ismember(klFitMethod, {'MaxLikelihood', 'YuleWalker'})
    error('klFitMethod must be ''MaxLikelihood'' or ''YuleWalker'' (got "%s").', ...
      klFitMethod);
  end
  klErrBars = logical(klErrBars);
  klParallel = logical(klParallel);
  if klErrBars && ~strcmp(klFitMethod, 'MaxLikelihood')
    error('KL error bars require klFitMethod = ''MaxLikelihood'' (S2.5 Hessian).');
  end
else
  klFitMethod = 'MaxLikelihood';
  klErrBars = false;
  klParallel = false;
end
end
