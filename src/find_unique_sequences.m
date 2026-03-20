function [uniqueSequences, sequenceTimes] = find_unique_sequences(bhvMat, opts)
% FIND_UNIQUE_SEQUENCES Find unique run-length compressed behavior patterns of given length(s).
%
% Variables:
%   bhvMat  - Table with behavior data with fields Time and Code.
%               Code is the behavior ID per frame (column vector or table column).
%   opts      - Struct with fields:
%       patternLength - Length(s) of patterns (number of distinct behaviors after run-length
%                       compression). Scalar or vector of positive integers. All requested
%                       lengths are processed; output contains sequences of various lengths.
%                       Required.
%       nMin          - Minimum number of occurrences for a pattern to be included in
%                       the output (default 1).
%       includeBhv    - (Optional) Vector of behavior codes. Only sequences containing at
%                       least one of these behaviors are valid. If empty, include all.
%       includeIsFirst- (Optional) If true, valid sequences must begin with one of the
%                       labels in includeBhv and the previous preBufferSec seconds must not
%                       contain any label in includeBhv or in the pattern itself. Only
%                       applied when includeBhv is non-empty. Default false.
%       preBufferSec   - (Optional) When includeIsFirst is true, time window (in seconds)
%                       before the start of the pattern to check; none of those frames may
%                       contain includeBhv or pattern labels (default 1 second).
%       excludeBhv    - (Optional) Vector of behavior codes. Exclude any sequence that
%                       contains any of these behaviors. If empty, exclude none.
%       noRepeat      - (Optional) If true, only patterns with unique labels are valid
%                       (no label appears more than once in the pattern). Default false.
%       firstBhv      - (Optional) Vector of behavior codes. Only sequences whose first
%                       element (in the run-length compressed pattern) is in this set are
%                       valid. If empty, no constraint. Default empty.
%
% Goal:
%   Compress the Code sequence into the behavior pattern (run-length sequence), e.g.
%   [5 5 5 6 6 4] -> [5 6 4]. Then find all unique patterns for each length in
%   patternLength, ranked by frequency (over all lengths combined). Only patterns with at least nMin
%   occurrences are returned. Optionally filter by includeBhv (pattern must contain at
%   least one), includeIsFirst (pattern must start with one of includeBhv; the previous
%   preBufferSec seconds must not contain includeBhv or pattern labels), excludeBhv (pattern must
%   contain none), noRepeat (pattern must have all unique labels), and firstBhv (pattern
%   must start with one of these codes). All frames are treated as valid.
%
% Returns:
%   uniqueSequences - Cell array of unique patterns (each a 1 x L numeric vector for some
%                     L in patternLength), ranked by most frequent first.
%   sequenceTimes   - Cell array (same order as uniqueSequences). Each element is a cell
%                     array of occurrences for that pattern; each occurrence is a 1 x L
%                     vector of start times (seconds) for each label in the pattern
%                     (from bhvMat.Time at the first frame of each label).
%
% Example:
%   opts.patternLength = 3;
%   opts.nMin = 2;
%   [uniqueSeqs, startTimes] = find_unique_sequences(bhvMat, opts);
%   % Multiple lengths:
%   opts.patternLength = [2, 3, 4];
%   [uniqueSeqs, startTimes] = find_unique_sequences(bhvMat, opts);

    if nargin < 2 || isempty(opts) || ~isstruct(opts)
        error('find_unique_sequences:opts', 'opts struct is required.');
    end
    if ~isfield(opts, 'patternLength') || isempty(opts.patternLength)
        error('find_unique_sequences:opts', 'opts.patternLength is required.');
    end
    patternLengthVec = unique(round(opts.patternLength(:)));
    patternLengthVec = patternLengthVec(patternLengthVec >= 1);
    patternLengthVec = patternLengthVec(:)';  % row so "for L = patternLengthVec" gives scalar L
    if isempty(patternLengthVec)
        error('find_unique_sequences:opts', 'opts.patternLength must contain at least one positive integer.');
    end
    if isfield(opts, 'nMin') && ~isempty(opts.nMin)
        nMin = max(1, round(opts.nMin));
    else
        nMin = 1;
    end
    includeBhv = [];
    if isfield(opts, 'includeBhv') && ~isempty(opts.includeBhv)
        includeBhv = opts.includeBhv(:)';
    end
    excludeBhv = [];
    if isfield(opts, 'excludeBhv') && ~isempty(opts.excludeBhv)
        excludeBhv = opts.excludeBhv(:)';
    end
    includeIsFirst = false;
    if isfield(opts, 'includeIsFirst') && ~isempty(opts.includeIsFirst)
        includeIsFirst = logical(opts.includeIsFirst(1));
    end
    preBufferSec = 1;
    if isfield(opts, 'preBufferSec') && ~isempty(opts.preBufferSec)
        preBufferSec = max(0, opts.preBufferSec(1));
    end
    noRepeat = false;
    if isfield(opts, 'noRepeat') && ~isempty(opts.noRepeat)
        noRepeat = logical(opts.noRepeat(1));
    end
    firstBhv = [];
    if isfield(opts, 'firstBhv') && ~isempty(opts.firstBhv)
        firstBhv = opts.firstBhv(:)';
    end
    if isempty(bhvMat) || ~istable(bhvMat)
        error('find_unique_sequences:bhvMat', 'bhvMat must be a non-empty table with Time, Behavior, Code.');
    end
    if ~ismember('Code', bhvMat.Properties.VariableNames)
        error('find_unique_sequences:bhvMat', 'bhvMat must contain a Code column.');
    end
    if ~ismember('Time', bhvMat.Properties.VariableNames)
        error('find_unique_sequences:bhvMat', 'bhvMat must contain a Time column.');
    end

    codeVec = bhvMat.Code(:);
    timeVec = bhvMat.Time(:);
    nFrames = numel(codeVec);

    % Run-length compress Code into behavior pattern
    changeIdx = [true; diff(codeVec) ~= 0];
    compressedLabels = codeVec(changeIdx);
    runStartFrameIdx = find(changeIdx);
    nCompressed = numel(compressedLabels);

    allSequences = {};
    allTimes = {};

    for L = patternLengthVec
        for i = 1 : (nCompressed - L + 1)
            currentPattern = compressedLabels(i : i + L - 1);

            if any(isnan(currentPattern))
                continue;
            end
            % includeBhv: pattern must contain at least one of these behaviors
            if ~isempty(includeBhv) && ~any(ismember(currentPattern, includeBhv))
                continue;
            end
            % includeIsFirst: pattern must begin with one of includeBhv; the previous
            % preBufferSec seconds must not contain any label that appears in the pattern
            if includeIsFirst && ~isempty(includeBhv)
                if ~ismember(currentPattern(1), includeBhv)
                    continue;
                end
                % All frames within preBufferSec seconds before the pattern start
                % must not contain labels that are part of the current pattern.
                firstFrameOfPattern = runStartFrameIdx(i);
                t0 = timeVec(firstFrameOfPattern);
                if preBufferSec > 0
                    preMask = timeVec >= (t0 - preBufferSec) & timeVec < t0;
                    if any(preMask)
                        prevLabels = codeVec(preMask);
                        % Only exclude if any label from the current pattern appears
                        if any(ismember(prevLabels, currentPattern))
                            continue;
                        end
                    end
                end
            end
            % firstBhv: pattern must begin with one of these behavior codes
            if ~isempty(firstBhv) && ~ismember(currentPattern(1), firstBhv)
                continue;
            end
            % excludeBhv: pattern must not contain any of these behaviors
            if ~isempty(excludeBhv) && any(ismember(currentPattern, excludeBhv))
                continue;
            end
            % noRepeat: pattern must contain only unique labels (no label repeated)
            if noRepeat && numel(unique(currentPattern)) ~= L
                continue;
            end

            seqStr = strjoin(arrayfun(@num2str, currentPattern(:)', 'UniformOutput', false), ',');
            seqIndex = find(strcmp(allSequences, seqStr));
            patternFrameIdx = runStartFrameIdx(i : i + L - 1);
            patternTimes = timeVec(patternFrameIdx).';

            if isempty(seqIndex)
                allSequences{end + 1} = seqStr;
                allTimes{end + 1} = {patternTimes};
            else
                allTimes{seqIndex}{end + 1} = patternTimes;
            end
        end
    end

    % Convert sequence strings back to numeric vectors
    uniqueSequences = cellfun(@(x) str2num(strrep(x, ',', ' ')), allSequences, 'UniformOutput', false);

    % Sort by frequency (descending)
    allTimesSorted = allTimes;
    [~, sortOrder] = sort(cellfun(@length, allTimesSorted), 'descend');
    uniqueSequences = uniqueSequences(sortOrder);
    allTimesSorted = allTimesSorted(sortOrder);

    % Keep only patterns with at least nMin occurrences
    keepIdx = cellfun(@length, allTimesSorted) >= nMin;
    uniqueSequences = uniqueSequences(keepIdx);
    sequenceTimes = allTimesSorted(keepIdx);

end
