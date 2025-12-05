function segmentWindows = reach_task_engagement(fileName, opts)
% REACH_TASK_ENGAGEMENT Defines time windows for engagement segments of a session
%
% This function identifies engagement segments of a reach task session based on threshold crossings:
%   1. Block 1 engaged: largest window with at least nEngaged consecutive reaches below threshold
%   2. Block 1 not-engaged: largest window with at least nNotEngaged consecutive reaches above threshold
%   3. Block 2 engaged: largest window with at least nEngaged consecutive reaches below threshold
%   4. Block 2 not-engaged: largest window with at least nNotEngaged consecutive reaches above threshold
%
% Threshold crossings are defined as the reach before the threshold is crossed.
%
% Inputs:
%   fileName - Path to reach data file (.mat file containing R and block matrices)
%   opts     - Structure with optional fields:
%       .timesMedian - Times median for engagement threshold (default: 0.5)
%       .windowSize - Window size for running average (default: 10 reaches)
%       .nEngaged - Number of consecutive reaches below threshold required (default: 5)
%       .nNotEngaged - Number of consecutive reaches above threshold required (default: 5)
%       .nReachMin - Minimum number of reaches required for any segment (default: 5)
%
% Outputs:
%   segmentWindows - Structure with fields:
%       .block1EngagedWindow - [start stop] time window for Block 1 engaged (seconds) or []
%       .block1NotEngagedWindow - [start stop] time window for Block 1 not-engaged (seconds) or []
%       .block2EngagedWindow - [start stop] time window for Block 2 engaged (seconds) or []
%       .block2NotEngagedWindow - [start stop] time window for Block 2 not-engaged (seconds) or []
%
% Example:
%   opts.timesMedian = 0.5;
%   opts.windowSize = 10;
%   opts.nEngaged = 5;
%   opts.nNotEngaged = 5;
%   opts.nReachMin = 5;
%   segmentWindows = reach_task_engagement('Y4_100623_Spiketimes_idchan_BEH.mat', opts);

% Default options
if nargin < 2
    opts = struct();
end
if ~isfield(opts, 'timesMedian')
    opts.timesMedian = 2.5; % Default: 0.5 times median
end
if ~isfield(opts, 'windowSize')
    opts.windowSize = 3; % Default: 10 reaches for running average
end
if ~isfield(opts, 'nEngaged')
    opts.nEngaged = 5; % Default: 5 consecutive reaches below threshold
end
if ~isfield(opts, 'nNotEngaged')
    opts.nNotEngaged = 5; % Default: 5 consecutive reaches above threshold
end
if ~isfield(opts, 'nReachMin')
    opts.nReachMin = 5; % Default: minimum 5 reaches for any segment
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

% Calculate inter-reach intervals and running average
if length(reachStartSec) > 1
    interReachIntervals = diff(reachStartSec); % Intervals in seconds
    
    % Calculate median of all inter-reach intervals
    medianInterval = median(interReachIntervals);
    threshold = opts.timesMedian * medianInterval;
    
    % Calculate running average of intervals
    windowSize = opts.windowSize;
    runningAvg = zeros(length(interReachIntervals), 1);
    
    for i = 1:length(interReachIntervals)
        % Window extends from max(1, i-windowSize+1) to i
        windowStart = max(1, i - windowSize + 1);
        windowEnd = i;
        runningAvg(i) = mean(interReachIntervals(windowStart:windowEnd));
    end
    
    % Find all threshold crossings
    % A crossing occurs when runningAvg transitions from below to above or above to below
    % The crossing is defined as the reach before the threshold is crossed
    % runningAvg(i) is the running avg for interval between reach i and reach i+1
    % If runningAvg(i) crosses threshold, the crossing reach is reach i+1, and we use reach i
    
    % Determine if each interval is above or below threshold
    isAboveThreshold = runningAvg > threshold;
    
    % Find all crossings (transitions)
    % Crossing from below to above: reach before is the reach at index i (reach i+1 is first above)
    % Crossing from above to below: reach before is the reach at index i (reach i+1 is first below)
    crossings = [];
    for i = 1:length(isAboveThreshold)-1
        if isAboveThreshold(i) ~= isAboveThreshold(i+1)
            % Threshold crossed between interval i and i+1
            % The reach before the crossing is reach i+1
            crossings = [crossings; i+1, isAboveThreshold(i+1)]; % [reachIndex, isAboveAfterCrossing]
        end
    end
    
    % Find Block 1 windows: engaged and not-engaged
    % Block 1 is before Block 2 engaged starts
    block1EndIdx = firstBlock2Idx - 1; % Last reach index in Block 1
    if block1EndIdx > 0
        % Block 1 engaged: largest window with at least nEngaged consecutive reaches below threshold
        block1EngagedWindow = findLargestWindowBelowThreshold(reachStartSec, runningAvg, threshold, ...
            1, block1EndIdx, opts.nEngaged, opts.nReachMin);
        % Block 1 not-engaged: largest window with at least nNotEngaged consecutive reaches above threshold
        block1NotEngagedWindow = findLargestWindowAboveThreshold(reachStartSec, runningAvg, threshold, ...
            1, block1EndIdx, opts.nNotEngaged, opts.nReachMin);
    else
        block1EngagedWindow = [];
        block1NotEngagedWindow = [];
    end
    
    % Find Block 2 engaged window: largest window with at least nEngaged consecutive reaches below threshold
    % Block 2 engaged is after Block 2 engaged starts
    if firstBlock2Idx <= length(reachStartSec)
        block2EngagedWindow = findLargestWindowBelowThreshold(reachStartSec, runningAvg, threshold, ...
            firstBlock2Idx, length(reachStartSec), opts.nEngaged, opts.nReachMin);
    else
        block2EngagedWindow = [];
    end
    
    % Find Block 2 not-engaged window: largest window with at least nNotEngaged consecutive reaches above threshold
    if firstBlock2Idx <= length(reachStartSec)
        block2NotEngagedWindow = findLargestWindowAboveThreshold(reachStartSec, runningAvg, threshold, ...
            firstBlock2Idx, length(reachStartSec), opts.nNotEngaged, opts.nReachMin);
    else
        block2NotEngagedWindow = [];
    end
    
else
    % If only one reach, cannot calculate intervals - return empty windows
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

end

function window = findLargestWindowBelowThreshold(reachStartSec, runningAvg, threshold, startReachIdx, endReachIdx, nConsecutive, nReachMin)
% FINDLARGESTWINDOWBELOWTHRESHOLD Finds largest window with nConsecutive reaches below threshold
%
% Inputs:
%   reachStartSec - Vector of reach start times in seconds
%   runningAvg - Vector of running averages for intervals
%   threshold - Threshold value
%   startReachIdx - Starting reach index to search from
%   endReachIdx - Ending reach index to search to
%   nConsecutive - Number of consecutive reaches below threshold required
%   nReachMin - Minimum number of reaches required for segment to be valid
%
% Outputs:
%   window - [start stop] time window in seconds, or [] if no valid segment found

% Find all intervals in the range that are below threshold
% Interval i is between reach i and reach i+1
validIntervalStart = startReachIdx;
validIntervalEnd = min(endReachIdx - 1, length(runningAvg));

if validIntervalStart > validIntervalEnd
    % No valid intervals, return empty
    window = [];
    return;
end

validIntervals = validIntervalStart:validIntervalEnd;
belowThreshold = runningAvg(validIntervals) <= threshold;

% Find all segments with at least nConsecutive consecutive reaches below threshold
% nConsecutive reaches means nConsecutive-1 consecutive intervals below threshold
maxWindow = []; % Default to empty
maxWindowSize = 0;

% Need at least nConsecutive-1 intervals to have nConsecutive reaches
if length(belowThreshold) >= nConsecutive - 1
    i = 1;
    while i <= length(belowThreshold) - (nConsecutive - 2)
        % Check if we have nConsecutive-1 consecutive intervals below threshold starting at i
        if i + nConsecutive - 2 <= length(belowThreshold) && all(belowThreshold(i:i+nConsecutive-2))
            % Found a valid segment starting at interval i
            % Find the full extent of this segment
            segmentStartInterval = i;
            segmentEndInterval = i;
            
            % Extend forward to find the end of the segment
            for j = i:length(belowThreshold)
                if belowThreshold(j)
                    segmentEndInterval = j;
                else
                    break;
                end
            end
            
            % Convert interval indices to reach indices
            % Interval segmentStartInterval corresponds to interval validIntervals(segmentStartInterval)
            % This interval is between reach validIntervals(segmentStartInterval) and reach validIntervals(segmentStartInterval)+1
            % The crossing INTO below threshold happens at the transition to interval segmentStartInterval
            % The reach before this crossing is reach validIntervals(segmentStartInterval)
            % The crossing OUT OF below threshold happens at the transition from interval segmentEndInterval to the next
            % The reach before this crossing is reach validIntervals(segmentEndInterval)+1
            windowStartReach = validIntervals(segmentStartInterval);
            windowEndReach = min(validIntervals(segmentEndInterval) + 1, endReachIdx);
            windowSize = windowEndReach - windowStartReach + 1;
            
            % Check that window meets both nConsecutive and nReachMin requirements
            if windowSize >= nConsecutive && windowSize >= nReachMin && windowSize > maxWindowSize
                maxWindowSize = windowSize;
                maxWindow = [reachStartSec(windowStartReach), reachStartSec(windowEndReach)];
            end
            
            % Move to after this segment
            i = segmentEndInterval + 1;
        else
            i = i + 1;
        end
    end
end

window = maxWindow;
end

function window = findLargestWindowAboveThreshold(reachStartSec, runningAvg, threshold, startReachIdx, endReachIdx, nConsecutive, nReachMin)
% FINDLARGESTWINDOWABOVETHRESHOLD Finds largest window with nConsecutive reaches above threshold
%
% Inputs:
%   reachStartSec - Vector of reach start times in seconds
%   runningAvg - Vector of running averages for intervals
%   threshold - Threshold value
%   startReachIdx - Starting reach index to search from
%   endReachIdx - Ending reach index to search to
%   nConsecutive - Number of consecutive reaches above threshold required
%   nReachMin - Minimum number of reaches required for segment to be valid
%
% Outputs:
%   window - [start stop] time window in seconds, or [] if no valid segment found

% Find all intervals in the range that are above threshold
% Interval i is between reach i and reach i+1
validIntervalStart = startReachIdx;
validIntervalEnd = min(endReachIdx - 1, length(runningAvg));

if validIntervalStart > validIntervalEnd
    % No valid intervals, return empty
    window = [];
    return;
end

validIntervals = validIntervalStart:validIntervalEnd;
aboveThreshold = runningAvg(validIntervals) > threshold;

% Find all segments with at least nConsecutive consecutive reaches above threshold
% nConsecutive reaches means nConsecutive-1 consecutive intervals above threshold
maxWindow = []; % Default to empty
maxWindowSize = 0;

% Need at least nConsecutive-1 intervals to have nConsecutive reaches
if length(aboveThreshold) >= nConsecutive - 1
    i = 1;
    while i <= length(aboveThreshold) - (nConsecutive - 2)
        % Check if we have nConsecutive-1 consecutive intervals above threshold starting at i
        if i + nConsecutive - 2 <= length(aboveThreshold) && all(aboveThreshold(i:i+nConsecutive-2))
            % Found a valid segment starting at interval i
            % Find the full extent of this segment
            segmentStartInterval = i;
            segmentEndInterval = i;
            
            % Extend forward to find the end of the segment
            for j = i:length(aboveThreshold)
                if aboveThreshold(j)
                    segmentEndInterval = j;
                else
                    break;
                end
            end
            
            % Convert interval indices to reach indices
            % Interval segmentStartInterval corresponds to interval validIntervals(segmentStartInterval)
            % This interval is between reach validIntervals(segmentStartInterval) and reach validIntervals(segmentStartInterval)+1
            % The crossing INTO above threshold happens at the transition to interval segmentStartInterval
            % The reach before this crossing is reach validIntervals(segmentStartInterval)
            % The crossing OUT OF above threshold happens at the transition from interval segmentEndInterval to the next
            % The reach before this crossing is reach validIntervals(segmentEndInterval)+1
            windowStartReach = validIntervals(segmentStartInterval);
            windowEndReach = min(validIntervals(segmentEndInterval) + 1, endReachIdx);
            windowSize = windowEndReach - windowStartReach + 1;
            
            % Check that window meets both nConsecutive and nReachMin requirements
            if windowSize >= nConsecutive && windowSize >= nReachMin && windowSize > maxWindowSize
                maxWindowSize = windowSize;
                maxWindow = [reachStartSec(windowStartReach), reachStartSec(windowEndReach)];
            end
            
            % Move to after this segment
            i = segmentEndInterval + 1;
        else
            i = i + 1;
        end
    end
end

window = maxWindow;
end
