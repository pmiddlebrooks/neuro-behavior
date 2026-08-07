function orderedTypes = order_manuscript_session_types(sessionTypes)
% ORDER_MANUSCRIPT_SESSION_TYPES - Canonical left-to-right task plot order
%
% Variables:
%   sessionTypes - Cell/string array of session types present in this run
%
% Goal:
%   Return the subset of sessionTypes in manuscript plot order:
%   spontaneous -> interval -> semicircle -> reach.
%   Types not in the input are omitted. Unknown types are appended at the end
%   in their original order.
%
% Returns:
%   orderedTypes - Cell array of sessionType strings

if isempty(sessionTypes)
  orderedTypes = {};
  return;
end

if isstring(sessionTypes)
  sessionTypes = cellstr(sessionTypes);
elseif ischar(sessionTypes)
  sessionTypes = {sessionTypes};
end
sessionTypes = sessionTypes(:)';

canonicalOrder = {'spontaneous', 'interval', 'semicircle', 'reach'};
orderedTypes = {};
used = false(size(sessionTypes));

for iCanon = 1:numel(canonicalOrder)
  matchIdx = find(strcmpi(sessionTypes, canonicalOrder{iCanon}) & ~used, 1);
  if ~isempty(matchIdx)
    orderedTypes{end + 1} = canonicalOrder{iCanon}; %#ok<AGROW>
    used(matchIdx) = true;
  end
end

% Keep any non-canonical types (e.g. schall) after the standard block
for iType = 1:numel(sessionTypes)
  if ~used(iType)
    orderedTypes{end + 1} = sessionTypes{iType}; %#ok<AGROW>
  end
end
end
