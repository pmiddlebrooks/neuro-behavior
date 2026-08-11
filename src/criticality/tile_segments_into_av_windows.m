function tiled = tile_segments_into_av_windows(segments, avWindow, minSegDur)
% TILE_SEGMENTS_INTO_AV_WINDOWS - Non-overlapping avWindow tiles within segments
%
% Variables:
%   segments  - Struct array with .start, .end (seconds)
%   avWindow  - Tile length (s); [] / non-positive = return segments unchanged
%   minSegDur - Drop tiles shorter than this (default 0.2 s)
%
% Goal:
%   Split each continuous segment into consecutive avWindow blocks so each
%   block can have its own population threshold. Remainder at the end of a
%   segment is kept if it meets minSegDur (short engagement bursts stay).

if nargin < 3 || isempty(minSegDur)
  minSegDur = 0.2;
end

tiled = struct('start', {}, 'end', {});
if isempty(segments)
  return;
end

useTiles = isnumeric(avWindow) && isscalar(avWindow) && isfinite(avWindow) && avWindow > 0;
if ~useTiles
  tiled = segments;
  return;
end

for i = 1:numel(segments)
  segStart = segments(i).start;
  segEnd = segments(i).end;
  if ~(isfinite(segStart) && isfinite(segEnd)) || (segEnd - segStart) < minSegDur
    continue;
  end
  tileStart = segStart;
  while tileStart < segEnd
    tileEnd = min(tileStart + avWindow, segEnd);
    if (tileEnd - tileStart) >= minSegDur
      tiled(end + 1).start = tileStart; %#ok<AGROW>
      tiled(end).end = tileEnd;
    end
    tileStart = tileStart + avWindow;
  end
end
end
