function label = format_engagement_buffer_label(bufferBefore, bufferAfter)
% FORMAT_ENGAGEMENT_BUFFER_LABEL - Compact before/after buffer string for logs
%
% Variables:
%   bufferBefore - Seconds before each event counted as engaged
%   bufferAfter  - Seconds after each event counted as engaged
%
% Goal:
%   One-line label: "before=X s, after=Y s" (or "X s (symmetric)" when equal).

if nargin < 2 || isempty(bufferAfter)
  bufferAfter = bufferBefore;
end
if isequal(bufferBefore, bufferAfter)
  label = sprintf('%.3g s (symmetric)', bufferBefore);
else
  label = sprintf('before=%.3g s, after=%.3g s', bufferBefore, bufferAfter);
end
end
