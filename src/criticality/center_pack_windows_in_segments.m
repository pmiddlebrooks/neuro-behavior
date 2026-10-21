function centerTimes = center_pack_windows_in_segments(segments, windowSize)
% CENTER_PACK_WINDOWS_IN_SEGMENTS - As many fixed windows as fit, centered in each bout
%
% Variables:
%   segments   - Struct array with .start and .end (seconds)
%   windowSize - Window length (s). Windows do not overlap and do not cross
%                a segment boundary.
%
% Goal:
%   Put floor(duration/windowSize) windows in each segment. The block is
%   centered, so leftover time is split across the two ends. That uses the
%   same count as packing from the segment start, and keeps the windows in
%   the middle of the bout.
%
% Returns:
%   centerTimes - Row vector of absolute window centers (s), sorted in time.
%                 Empty when no segment is long enough.

centerTimes = zeros(1, 0);
if isempty(segments) || ~isscalar(windowSize) || ~isfinite(windowSize) || windowSize <= 0
  return;
end

edgeTol = 1e-6;
for iSeg = 1:numel(segments)
  segStart = segments(iSeg).start;
  segEnd = segments(iSeg).end;
  if ~(isfinite(segStart) && isfinite(segEnd)) || segEnd <= segStart
    continue;
  end
  nWindows = floor((segEnd - segStart + edgeTol) / windowSize);
  if nWindows < 1
    continue;
  end
  packDuration = nWindows * windowSize;
  packStart = segStart + ((segEnd - segStart) - packDuration) / 2;
  firstCenter = packStart + windowSize / 2;
  segCenters = firstCenter + (0:nWindows - 1) * windowSize;
  centerTimes = [centerTimes, segCenters]; %#ok<AGROW>
end

centerTimes = unique(centerTimes, 'sorted');
centerTimes = centerTimes(:)';
end
