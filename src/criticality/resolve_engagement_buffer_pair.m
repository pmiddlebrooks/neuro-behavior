function [bufferBefore, bufferAfter] = resolve_engagement_buffer_pair(opts, beforeField, ...
    afterField, legacyField, defaultVal)
% RESOLVE_ENGAGEMENT_BUFFER_PAIR - Before/after buffers with legacy single-buffer fallback
%
% Variables:
%   opts         - Options struct
%   beforeField  - e.g. 'reachBufferBefore' or 'eventBufferBefore'
%   afterField   - e.g. 'reachBufferAfter' or 'eventBufferAfter'
%   legacyField  - e.g. 'reachBuffer' or 'eventBuffer' (symmetric alias)
%   defaultVal   - Default when nothing is set (default 1)
%
% Goal:
%   Prefer explicit before/after fields. If only the legacy symmetric field is
%   set, use it for both sides. Missing sides default to defaultVal (or to the
%   other side when only one of before/after is provided).

if nargin < 5 || isempty(defaultVal)
  defaultVal = 1;
end

hasBefore = isfield(opts, beforeField) && ~isempty(opts.(beforeField));
hasAfter = isfield(opts, afterField) && ~isempty(opts.(afterField));
hasLegacy = isfield(opts, legacyField) && ~isempty(opts.(legacyField));

if ~hasBefore && ~hasAfter && hasLegacy
  legacyVal = opts.(legacyField);
  if numel(legacyVal) >= 2
    bufferBefore = legacyVal(1);
    bufferAfter = legacyVal(2);
  else
    bufferBefore = legacyVal(1);
    bufferAfter = legacyVal(1);
  end
elseif hasBefore || hasAfter
  if hasBefore
    bufferBefore = opts.(beforeField);
  elseif hasLegacy
    bufferBefore = opts.(legacyField);
    if numel(bufferBefore) >= 2
      bufferBefore = bufferBefore(1);
    end
  else
    bufferBefore = defaultVal;
  end
  if hasAfter
    bufferAfter = opts.(afterField);
  elseif hasLegacy
    bufferAfter = opts.(legacyField);
    if numel(bufferAfter) >= 2
      bufferAfter = bufferAfter(2);
    else
      bufferAfter = bufferAfter(1);
    end
  else
    bufferAfter = defaultVal;
  end
else
  bufferBefore = defaultVal;
  bufferAfter = defaultVal;
end

bufferBefore = max(0, double(bufferBefore(1)));
bufferAfter = max(0, double(bufferAfter(1)));
end
