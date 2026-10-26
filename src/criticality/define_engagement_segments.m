function [engagedSegs, nonEngagedSegs] = define_engagement_segments( ...
    collectStart, collectEnd, eventTimes, minNonEngagedWindow, bufferBefore, bufferAfter, ...
    absorbSingleEvents)
% DEFINE_ENGAGEMENT_SEGMENTS - Continuous engaged and non-engaged intervals
%
% Variables:
%   collectStart, collectEnd - Session analysis bounds (s)
%   eventTimes               - Engagement event times inside the collect window (s)
%   minNonEngagedWindow      - Minimum gap without events (s)
%   bufferBefore             - Seconds before each event counted as engaged
%   bufferAfter              - Seconds after each event counted as engaged
%   absorbSingleEvents       - If true, merge an isolated single event into the
%                              flanking non-engaged gaps
%
% Goal:
%   Same segment rule as the spike engagement modules. Each event occupies
%   [event - bufferBefore, event + bufferAfter]. Non-engaged gaps are the
%   spaces between those neighborhoods that last at least minNonEngagedWindow.
%   Engaged time is the complement, including every event buffer.
%
% Returns:
%   engagedSegs, nonEngagedSegs - Struct arrays with .start and .end (s)

if nargin < 5 || isempty(bufferBefore)
  bufferBefore = 0;
end
if nargin < 6 || isempty(bufferAfter)
  bufferAfter = bufferBefore;
end
if nargin < 7 || isempty(absorbSingleEvents)
  absorbSingleEvents = true;
end
bufferBefore = max(0, bufferBefore);
bufferAfter = max(0, bufferAfter);

eventTimes = sort(eventTimes(:));
eventTimes = eventTimes(isfinite(eventTimes));
eventTimes = eventTimes(eventTimes >= collectStart & eventTimes <= collectEnd);

occupied = merge_event_buffer_intervals( ...
  eventTimes, bufferBefore, bufferAfter, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if absorbSingleEvents && ~isempty(occupied)
  absorbedMask = absorbed_single_event_mask( ...
    eventTimes, collectStart, collectEnd, minNonEngagedWindow, bufferBefore, bufferAfter);
end

nonEngagedSegs = struct('start', {}, 'end', {});
cursor = collectStart;
iOcc = 1;
while iOcc <= numel(occupied)
  if absorbedMask(iOcc)
    gapStart = cursor;
    if iOcc < numel(occupied)
      gapEnd = occupied(iOcc + 1).start;
    else
      gapEnd = collectEnd;
    end
    if (gapEnd - gapStart) >= minNonEngagedWindow
      nonEngagedSegs(end + 1).start = gapStart; %#ok<AGROW>
      nonEngagedSegs(end).end = gapEnd;
    end
    cursor = gapEnd;
  else
    gapStart = cursor;
    gapEnd = occupied(iOcc).start;
    if (gapEnd - gapStart) >= minNonEngagedWindow
      nonEngagedSegs(end + 1).start = gapStart; %#ok<AGROW>
      nonEngagedSegs(end).end = gapEnd;
    end
    cursor = occupied(iOcc).end;
  end
  iOcc = iOcc + 1;
end
if (collectEnd - cursor) >= minNonEngagedWindow
  nonEngagedSegs(end + 1).start = cursor; %#ok<AGROW>
  nonEngagedSegs(end).end = collectEnd;
end

engagedSegs = complement_engagement_segments(collectStart, collectEnd, nonEngagedSegs);
end

function absorbedMask = absorbed_single_event_mask( ...
    eventTimes, collectStart, collectEnd, minNonEngagedWindow, bufferBefore, bufferAfter)
% ABSORBED_SINGLE_EVENT_MASK - Isolated single events flanked by long gaps
occupied = merge_event_buffer_intervals( ...
  eventTimes, bufferBefore, bufferAfter, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if isempty(occupied)
  return;
end

for iOcc = 1:numel(occupied)
  nEventsInOcc = sum(eventTimes >= occupied(iOcc).start & eventTimes <= occupied(iOcc).end);
  if nEventsInOcc ~= 1
    continue;
  end
  if iOcc == 1
    gapBeforeStart = collectStart;
  else
    gapBeforeStart = occupied(iOcc - 1).end;
  end
  gapBeforeEnd = occupied(iOcc).start;
  gapAfterStart = occupied(iOcc).end;
  if iOcc == numel(occupied)
    gapAfterEnd = collectEnd;
  else
    gapAfterEnd = occupied(iOcc + 1).start;
  end
  if (gapBeforeEnd - gapBeforeStart) >= minNonEngagedWindow && ...
      (gapAfterEnd - gapAfterStart) >= minNonEngagedWindow
    absorbedMask(iOcc) = true;
  end
end
end

function occupied = merge_event_buffer_intervals(eventTimes, bufferBefore, bufferAfter, ...
    collectStart, collectEnd)
% MERGE_EVENT_BUFFER_INTERVALS - Union of event neighborhoods inside the collect window
occupied = struct('start', {}, 'end', {});
if isempty(eventTimes)
  return;
end

starts = max(collectStart, eventTimes(:) - bufferBefore);
ends = min(collectEnd, eventTimes(:) + bufferAfter);
valid = ends > starts;
starts = starts(valid);
ends = ends(valid);
if isempty(starts)
  return;
end

[starts, orderIdx] = sort(starts);
ends = ends(orderIdx);

occupied(1).start = starts(1);
occupied(1).end = ends(1);
for iEvent = 2:numel(starts)
  if starts(iEvent) <= occupied(end).end
    occupied(end).end = max(occupied(end).end, ends(iEvent));
  else
    occupied(end + 1).start = starts(iEvent); %#ok<AGROW>
    occupied(end).end = ends(iEvent);
  end
end
end

function engagedSegs = complement_engagement_segments(collectStart, collectEnd, nonEngagedSegs)
% COMPLEMENT_ENGAGEMENT_SEGMENTS - Time in the collect window outside non-engaged gaps
engagedSegs = struct('start', {}, 'end', {});
if isempty(nonEngagedSegs)
  engagedSegs(1).start = collectStart;
  engagedSegs(1).end = collectEnd;
  return;
end

starts = [nonEngagedSegs.start];
ends = [nonEngagedSegs.end];
[starts, orderIdx] = sort(starts);
ends = ends(orderIdx);

cursor = collectStart;
for iSeg = 1:numel(starts)
  if starts(iSeg) > cursor + eps
    engagedSegs(end + 1).start = cursor; %#ok<AGROW>
    engagedSegs(end).end = starts(iSeg);
  end
  cursor = max(cursor, ends(iSeg));
end
if cursor < collectEnd - eps
  engagedSegs(end + 1).start = cursor; %#ok<AGROW>
  engagedSegs(end).end = collectEnd;
end
end
