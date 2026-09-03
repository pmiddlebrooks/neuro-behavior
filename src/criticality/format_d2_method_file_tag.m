function tag = format_d2_method_file_tag(d2Method, klFitMethod, klErrBars)
% FORMAT_D2_METHOD_FILE_TAG Filename stem suffix for Euclidean vs KL d2
%
% Variables:
%   d2Method    - 'euclidean' or 'kl'
%   klFitMethod - KL fit method (tagged only when not MaxLikelihood)
%   klErrBars   - If true, append _klerr
%
% Goal:
%   Keep Euclidean plot/cache names unchanged. KL runs get _d2kl, plus
%   fit-method / error-bar suffixes when they differ from the defaults.

tag = '';
if nargin < 1 || isempty(d2Method)
  return;
end
d2Method = lower(strtrim(char(d2Method)));
if ~strcmp(d2Method, 'kl')
  return;
end
tag = '_d2kl';
if nargin >= 2 && ~isempty(klFitMethod) && ~strcmp(char(klFitMethod), 'MaxLikelihood')
  tag = [tag, '_', matlab.lang.makeValidName(char(klFitMethod))];
end
if nargin >= 3 && ~isempty(klErrBars) && logical(klErrBars)
  tag = [tag, '_klerr'];
end
end
