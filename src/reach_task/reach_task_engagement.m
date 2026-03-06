function segmentWindows = reach_task_engagement(fileName, opts)
% REACH_TASK_ENGAGEMENT Defines time windows for engagement segments of a session
%
% Uses configurable percentiles of inter-reach intervals to define the largest
% possible segments:
%   - Engaged: contiguous stretch whose mean is BELOW engagedBelowPercentile,
%     with at most engagedNumExcepts intervals above engagedAvoidPercentile.
%   - Not-engaged: contiguous stretch whose mean is ABOVE nonengagedAbovePercentile,
%     with at most nonengagedNumExcepts intervals below nonengagedAvoidPercentile.
% For each block (Block 1 = before Block 2 start, Block 2 = from first Block 2 reach),
% the function returns the single largest such segment of each type.
%
% Inputs:
%   fileName - Path to reach data file (.mat file containing R and Block matrices)
%   opts     - Structure with optional fields:
%       .nReachMin - Minimum number of reaches in a segment to be valid (default: 5)
%       .engagedBelowPercentile - Segment mean must be below this percentile (default: 40)
%       .engagedAvoidPercentile - At most engagedNumExcepts intervals may be above this (default: 60)
%       .engagedNumExcepts - Allowed number of intervals above engagedAvoidPercentile (default: 2)
%       .nonengagedAbovePercentile - Segment mean must be above this percentile (default: 60)
%       .nonengagedAvoidPercentile - At most nonengagedNumExcepts intervals may be below this (default: 40)
%       .nonengagedNumExcepts - Allowed number of intervals below nonengagedAvoidPercentile (default: 2)
%       .plotSegments - If true, create a diagnostic plot (default: false)
%
% Outputs:
%   segmentWindows - Structure with fields:
%       .block1EngagedWindow - [start stop] time window for Block 1 engaged (seconds) or []
%       .block1NotEngagedWindow - [start stop] time window for Block 1 not-engaged (seconds) or []
%       .block2EngagedWindow - [start stop] time window for Block 2 engaged (seconds) or []
%       .block2NotEngagedWindow - [start stop] time window for Block 2 not-engaged (seconds) or []
%
% Example:
%   opts.nReachMin = 8;
%   opts.engagedBelowPercentile = 40;
%   opts.engagedAvoidPercentile = 60;
%   opts.engagedNumExcepts = 1;
%   opts.nonengagedAbovePercentile = 60;
%   opts.nonengagedAvoidPercentile = 40;
%   opts.nonengagedNumExcepts = 1;
%   opts.plotSegments = true;  % optional: plot reaches and segments
%   segmentWindows = reach_task_engagement('Y4_100623_Spiketimes_idchan_BEH.mat', opts);

% Default options
if nargin < 2
    opts = struct();
end
if ~isfield(opts, 'nReachMin')
    opts.nReachMin = 7;
end
if ~isfield(opts, 'engagedBelowPercentile')
    opts.engagedBelowPercentile = 60;
end
if ~isfield(opts, 'engagedAvoidPercentile')
    opts.engagedAvoidPercentile = 70;
end
if ~isfield(opts, 'engagedNumExcepts')
    opts.engagedNumExcepts = 2;
end
if ~isfield(opts, 'nonengagedAbovePercentile')
    opts.nonengagedAbovePercentile = 70;
end
if ~isfield(opts, 'nonengagedAvoidPercentile')
    opts.nonengagedAvoidPercentile = 50;
end
if ~isfield(opts, 'nonengagedNumExcepts')
    opts.nonengagedNumExcepts = 1;
end
if ~isfield(opts, 'plotSegments')
    opts.plotSegments = false;
end

% Load reach data
dataR = load(fileName);

% Get reach start times (in milliseconds)
reachStart = dataR.R(:, 1);

% Get reach classification
reachClass = dataR.Block(:,3);

% Convert reach start times from ms to seconds
reachStartSec = reachStart / 1000;

% Find Block 2 engaged start time (first reach where reachClass == 3 or 4)
block2Reaches = (reachClass == 3) | (reachClass == 4);
if any(block2Reaches)
    firstBlock2Idx = find(block2Reaches, 1, 'first');
    block2EngagedStartTime = reachStartSec(firstBlock2Idx);
else
    % If no Block 2 reaches found, set to end of session
    firstBlock2Idx = length(reachStartSec);
    block2EngagedStartTime = reachStartSec(end);
    warning('No Block 2 reaches (reachClass == 3 or 4) found in data');
end

% Percentile-based engagement segmentation (configurable via opts)
if length(reachStartSec) > 1
    interReachIntervals = diff(reachStartSec); % Intervals in seconds (index i = between reach i and i+1)
    
    % Thresholds from percentiles of all inter-reach intervals
    engagedBelowThresh = prctile(interReachIntervals, opts.engagedBelowPercentile);
    engagedAvoidThresh = prctile(interReachIntervals, opts.engagedAvoidPercentile);
    nonengagedAboveThresh = prctile(interReachIntervals, opts.nonengagedAbovePercentile);
    nonengagedAvoidThresh = prctile(interReachIntervals, opts.nonengagedAvoidPercentile);
    
    nReachMin = opts.nReachMin;
    block1EndIdx = firstBlock2Idx - 1; % Last reach index in Block 1
    
    % Block 1: engaged and not-engaged (largest valid stretch in reaches 1 .. block1EndIdx)
    if block1EndIdx > 0
        block1EngagedWindow = find_largest_engaged_percentile(reachStartSec, interReachIntervals, ...
            engagedBelowThresh, engagedAvoidThresh, opts.engagedNumExcepts, 1, block1EndIdx, nReachMin);
        block1NotEngagedWindow = find_largest_not_engaged_percentile(reachStartSec, interReachIntervals, ...
            nonengagedAboveThresh, nonengagedAvoidThresh, opts.nonengagedNumExcepts, 1, block1EndIdx, nReachMin);
    else
        block1EngagedWindow = [];
        block1NotEngagedWindow = [];
    end
    
    % Block 2: engaged and not-engaged (largest valid stretch in firstBlock2Idx .. end)
    if firstBlock2Idx <= length(reachStartSec)
        block2EngagedWindow = find_largest_engaged_percentile(reachStartSec, interReachIntervals, ...
            engagedBelowThresh, engagedAvoidThresh, opts.engagedNumExcepts, firstBlock2Idx, length(reachStartSec), nReachMin);
        block2NotEngagedWindow = find_largest_not_engaged_percentile(reachStartSec, interReachIntervals, ...
            nonengagedAboveThresh, nonengagedAvoidThresh, opts.nonengagedNumExcepts, firstBlock2Idx, length(reachStartSec), nReachMin);
    else
        block2EngagedWindow = [];
        block2NotEngagedWindow = [];
    end
else
    block1EngagedWindow = [];
    block1NotEngagedWindow = [];
    block2EngagedWindow = [];
    block2NotEngagedWindow = [];
    warning('Only one reach found, cannot calculate intervals');
end

% Store results in output structure
segmentWindows.block1EngagedWindow = block1EngagedWindow;
segmentWindows.block1NotEngagedWindow = block1NotEngagedWindow;
segmentWindows.block2EngagedWindow = block2EngagedWindow;
segmentWindows.block2NotEngagedWindow = block2NotEngagedWindow;

% Optional diagnostic plot for visualizing reaches and engagement segments
if opts.plotSegments
    try
        plot_engagement_segments(reachStartSec, reachClass, segmentWindows, fileName);
    catch plotErr
        warning('reach_task_engagement:PlotFailed', ...
            'Failed to create engagement plot: %s', plotErr.message);
    end
end

end

function window = find_largest_engaged_percentile(reachStartSec, interReachIntervals, belowThresh, avoidThresh, numExcepts, startReachIdx, endReachIdx, nReachMin)
% FIND_LARGEST_ENGAGED_PERCENTILE Largest contiguous "engaged" stretch by percentile rules.
%
% Engaged: mean(intervals in segment) < belowThresh, and at most numExcepts intervals > avoidThresh.
% Returns [startTime, endTime] in seconds or [] if no valid segment.

intervalStart = startReachIdx;
intervalEnd   = endReachIdx - 1;
if intervalStart > intervalEnd
    window = [];
    return;
end

nIntervals = length(interReachIntervals);
intervalEnd = min(intervalEnd, nIntervals);
maxLen = 0;
bestA = [];
bestB = [];

for a = intervalStart:intervalEnd
    for b = a:intervalEnd
        iv = interReachIntervals(a:b);
        nReaches = length(iv) + 1;
        if nReaches < nReachMin
            continue;
        end
        nAboveAvoid = sum(iv > avoidThresh);
        if nAboveAvoid > numExcepts
            break;
        end
        if mean(iv) >= belowThresh
            continue;
        end
        if (b - a + 1) > maxLen
            maxLen = b - a + 1;
            bestA = a;
            bestB = b;
        end
    end
end

if isempty(bestA)
    window = [];
    return;
end
window = [reachStartSec(bestA), reachStartSec(bestB + 1)];
end

function window = find_largest_not_engaged_percentile(reachStartSec, interReachIntervals, aboveThresh, avoidThresh, numExcepts, startReachIdx, endReachIdx, nReachMin)
% FIND_LARGEST_NOT_ENGAGED_PERCENTILE Largest contiguous "not-engaged" stretch by percentile rules.
%
% Not-engaged: mean(intervals in segment) > aboveThresh, and at most numExcepts intervals < avoidThresh.
% Returns [startTime, endTime] in seconds or [] if no valid segment.

intervalStart = startReachIdx;
intervalEnd   = endReachIdx - 1;
if intervalStart > intervalEnd
    window = [];
    return;
end

nIntervals = length(interReachIntervals);
intervalEnd = min(intervalEnd, nIntervals);
maxLen = 0;
bestA = [];
bestB = [];

for a = intervalStart:intervalEnd
    for b = a:intervalEnd
        iv = interReachIntervals(a:b);
        nReaches = length(iv) + 1;
        if nReaches < nReachMin
            continue;
        end
        nBelowAvoid = sum(iv < avoidThresh);
        if nBelowAvoid > numExcepts
            break;
        end
        if mean(iv) <= aboveThresh
            continue;
        end
        if (b - a + 1) > maxLen
            maxLen = b - a + 1;
            bestA = a;
            bestB = b;
        end
    end
end

if isempty(bestA)
    window = [];
    return;
end
window = [reachStartSec(bestA), reachStartSec(bestB + 1)];
end

function plot_engagement_segments(reachStartSec, reachClass, segmentWindows, fileName)
% PLOT_ENGAGEMENT_SEGMENTS Plot reaches and engagement segments for inspection
%
% Inputs:
%   reachStartSec - Vector of reach start times in seconds
%   reachClass - Vector of reach class labels (same length as reachStartSec)
%   segmentWindows - Structure with fields:
%       .block1EngagedWindow
%       .block1NotEngagedWindow
%       .block2EngagedWindow
%       .block2NotEngagedWindow
%   fileName - Name of the reach data file (used for plot title)
%
% Goal:
%   Provide a quick visual sanity check of the engagement segmentation by
%   plotting all reaches as vertical lines and highlighting the engaged and
%   not-engaged windows for Block 1 and Block 2.

% Basic y-limits for schematic engagement bands and reach lines
yMin = 0;
yMax = 1;

figure;
hold on;

% Plot vertical lines at each reach time
for iReach = 1:numel(reachStartSec)
    x = reachStartSec(iReach);
    plot([x, x], [yMin, yMax], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
end

% Demarcate Block 2 start with a red vertical line
block2Reaches = (reachClass == 3) | (reachClass == 4);
if any(block2Reaches)
    firstBlock2Idx = find(block2Reaches, 1, 'first');
    block2StartTime = reachStartSec(firstBlock2Idx);
    plot([block2StartTime, block2StartTime], [yMin, yMax], 'r', 'LineWidth', 1.5);
end

segmentHandles = [];
segmentLabels = {};

% Helper function for adding a shaded window
    function h = addWindow(windowField, colorVal, labelText)
        h = [];
        if isfield(segmentWindows, windowField) && ~isempty(segmentWindows.(windowField))
            tWin = segmentWindows.(windowField);
            h = patch([tWin(1), tWin(2), tWin(2), tWin(1)], ...
                [yMin, yMin, yMax, yMax], colorVal, ...
                'FaceAlpha', 0.2, 'EdgeColor', 'none');
            segmentHandles(end+1) = h; %#ok<AGROW>
            segmentLabels{end+1} = labelText; %#ok<AGROW>
        end
    end

% Block 1 and Block 2 engaged / not-engaged windows
addWindow('block1EngagedWindow', [0, 0.6, 0], 'Block 1 Engaged');
addWindow('block1NotEngagedWindow', [0.6, 0.9, 0.6], 'Block 1 Not Engaged');
addWindow('block2EngagedWindow', [0, 0, 0.7], 'Block 2 Engaged');
addWindow('block2NotEngagedWindow', [0.7, 0.7, 1], 'Block 2 Not Engaged');

% X-limits determined from reach times and segment windows
allTimes = reachStartSec(:);
winFields = {'block1EngagedWindow', 'block1NotEngagedWindow', ...
    'block2EngagedWindow', 'block2NotEngagedWindow'};
for iField = 1:numel(winFields)
    fName = winFields{iField};
    if isfield(segmentWindows, fName) && ~isempty(segmentWindows.(fName))
        allTimes = [allTimes; segmentWindows.(fName)(:)]; %#ok<AGROW>
    end
end
allTimes = allTimes(~isnan(allTimes));
if ~isempty(allTimes)
    xlim([min(allTimes), max(allTimes)]);
end

ylim([yMin, yMax]);
xlabel('Time (s)');
ylabel('Engagement (schematic)');

% Use file name (without path) for the title if available
if nargin >= 4 && ~isempty(fileName)
    [~, sessionNameOnly, ~] = fileparts(fileName);
    titleStr = sprintf('%s - Engagement Segments', sessionNameOnly);
else
    titleStr = 'Engagement Segments';
end
title(titleStr, 'Interpreter', 'none');

if ~isempty(segmentHandles)
    legend(segmentHandles, segmentLabels, 'Location', 'best');
end

hold off;

end
