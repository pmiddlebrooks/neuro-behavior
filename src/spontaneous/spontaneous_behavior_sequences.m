function [sequences, times] = spontaneous_behavior_sequences(dataFull, opts)
% SPONTANEOUS_BEHAVIOR_SEQUENCES Find non-overlapping behavior sequences meeting duration and proportion criteria.
%
% Variables:
%   dataFull    - table with behavior data (e.g. behavior_labels*.csv) with at least
%                 columns Code (behavior IDs) and Time (timestamps).
%   opts        - struct with fields:
%                 .behaviorIds   - vector of behavior Codes to count (integer IDs).
%                 .minDur        - minimum sequence duration in seconds (default 5).
%                 .propThreshold - proportion of frames in window that must be in behaviorIds, 0–1 (default 0.99).
%                 .bufferSec     - minimum seconds between end of one sequence and start of next (default 5).
%                 .fsBhv         - behavior sampling rate in Hz (required).
%                 .nMinUniqueBhv - minimum number of unique labels from behaviorIds required in each sequence
%                                  (default 2).
%
% Goal:
%   Given a behavior table dataFull, find contiguous sequences where a sliding window of
%   length minDur has at least propThreshold proportion of frames with Code in
%   behaviorIds. Sequences are merged, extended to the last valid frame, and separated by
%   at least bufferSec. Each sequence must begin and end with a frame that is in behaviorIds.
%
% Returns:
%   sequences - cell array of Code vectors (one per sequence) at native frame rate.
%   times     - cell array of Time vectors for each sequence.
%
% See also: neuro_behavior_options

    % Validate required inputs
    if nargin < 2 || isempty(opts) || ~isstruct(opts)
        error('spontaneous_behavior_sequences:opts', 'opts struct is required.');
    end
    if isempty(dataFull) || ~istable(dataFull)
        error('spontaneous_behavior_sequences:dataFull', 'dataFull must be a non-empty table.');
    end
    if ~isfield(opts, 'behaviorIds') || isempty(opts.behaviorIds)
        error('spontaneous_behavior_sequences:opts', 'opts.behaviorIds (vector of behavior IDs) is required.');
    end
    if ~isfield(opts, 'fsBhv') || isempty(opts.fsBhv)
        error('spontaneous_behavior_sequences:opts', 'opts.fsBhv is required.');
    end

    behaviorIds = opts.behaviorIds(:)';
    minDur = opts.minDur;
    propThreshold = opts.propThreshold;
    bufferSec = opts.bufferSec;
    fsBhv = opts.fsBhv;

    % Apply defaults for optional criteria
    if isempty(minDur) || ~isscalar(minDur), minDur = 5; end
    if isempty(propThreshold) || ~isscalar(propThreshold), propThreshold = 0.99; end
    if isempty(bufferSec) || ~isscalar(bufferSec), bufferSec = 5; end
    if ~isfield(opts, 'nMinUniqueBhv') || isempty(opts.nMinUniqueBhv)
        nMinUniqueBhv = 2;
    else
        nMinUniqueBhv = opts.nMinUniqueBhv;
    end


    % Frames that belong to any of the requested behaviors
    isBehavior = ismember(dataFull.Code, behaviorIds);
    nFrames = height(dataFull);
    nWin = round(minDur * fsBhv);
    if nWin < 1
        nWin = 1;
    end
    minBehaviorFrames = ceil(nWin * propThreshold);
    nBuffer = round(bufferSec * fsBhv);

    % Candidate windows: fixed-length windows with enough behavior frames
    candidateStart = [];
    candidateEnd = [];
    for startIdx = 1 : (nFrames - nWin + 1)
        endIdx = startIdx + nWin - 1;
        if sum(isBehavior(startIdx:endIdx)) >= minBehaviorFrames
            candidateStart = [candidateStart; startIdx]; %#ok<AGROW>
            candidateEnd = [candidateEnd; endIdx]; %#ok<AGROW>
        end
    end

    % Merge overlapping windows into non-overlapping intervals
    intervalStartIdx = [];
    intervalEndIdx = [];
    if ~isempty(candidateStart)
        [sortedStart, sortOrder] = sort(candidateStart);
        sortedEnd = candidateEnd(sortOrder);
        intervalStartIdx = sortedStart(1);
        intervalEndIdx = sortedEnd(1);
        for k = 2:length(sortedStart)
            if sortedStart(k) <= intervalEndIdx(end)
                intervalEndIdx(end) = max(intervalEndIdx(end), sortedEnd(k));
            else
                intervalStartIdx(end+1) = sortedStart(k); %#ok<AGROW>
                intervalEndIdx(end+1) = sortedEnd(k); %#ok<AGROW>
            end
        end
    end

    % Extend each sequence to last behavior frame where criteria hold, with buffer between sequences
    for s = 1:length(intervalStartIdx)
        a = intervalStartIdx(s);
        b = intervalEndIdx(s);
        if s < length(intervalStartIdx)
            maxEnd = intervalStartIdx(s + 1) - nBuffer - 1;
        else
            maxEnd = nFrames;
        end
        if maxEnd < a
            intervalEndIdx(s) = a - 1;
            continue;
        end
        inSegment = isBehavior(a:min(b, maxEnd));
        segIdx = find(inSegment);
        if ~isempty(segIdx)
            endIdx = a - 1 + segIdx(end);
        else
            endIdx = min(b, maxEnd);
        end
        for e = (endIdx + 1):maxEnd
            if e - nWin + 1 < 1
                break;
            end
            if ~isBehavior(e)
                continue;
            end
            if sum(isBehavior((e - nWin + 1):e)) >= minBehaviorFrames
                endIdx = e;
            end
        end
        intervalEndIdx(s) = endIdx;
    end

    % Ensure all sequences start and end with a label in behaviorIds
    for s = 1:length(intervalStartIdx)
        a = intervalStartIdx(s);
        b = intervalEndIdx(s);
        
        % Find first frame in behaviorIds from the start
        while a <= b && ~ismember(dataFull.Code(a), behaviorIds)
            a = a + 1;
        end
        
        % Find last frame in behaviorIds from the end
        while b >= a && ~ismember(dataFull.Code(b), behaviorIds)
            b = b - 1;
        end
        
        intervalStartIdx(s) = a;
        intervalEndIdx(s) = b;
    end

    % Drop invalid or too-short sequences
    keep = (intervalEndIdx >= intervalStartIdx) & ((intervalEndIdx - intervalStartIdx + 1) >= nWin);
    intervalStartIdx = intervalStartIdx(keep);
    intervalEndIdx = intervalEndIdx(keep);

    % Build output
    sequences = cell(length(intervalStartIdx), 1);
    times = cell(length(intervalStartIdx), 1);
    for s = 1:length(intervalStartIdx)
        a = intervalStartIdx(s);
        b = intervalEndIdx(s);
        sequences{s} = dataFull.Code(a:b);
        times{s} = dataFull.Time(a:b);
    end

    % Filter sequences by minimum number of unique behavior labels
    if nMinUniqueBhv > 0
        keepSeq = false(1, length(sequences));
        for s = 1:length(sequences)
            uniqueBhvInSeq = unique(sequences{s});
            uniqueBhvInBehaviorIds = uniqueBhvInSeq(ismember(uniqueBhvInSeq, behaviorIds));
            if length(uniqueBhvInBehaviorIds) >= nMinUniqueBhv
                keepSeq(s) = true;
            end
        end
        sequences = sequences(keepSeq);
        times = times(keepSeq);
    end
end
